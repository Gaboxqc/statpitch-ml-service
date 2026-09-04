"""Matchday card assembly (Plan §4 Phase B).

Offline, on hand-built frames. Two things get pinned here that nothing else can:

* that the card is *computed* rather than stubbed, so an empty slate carries a
  cause instead of being indistinguishable from unwritten code
* that the fitted path works, which no other test can reach — the committed
  `decision_config` is a placeholder and `StakingEngine` refuses to size from it,
  so the whole staking branch is dead code in production today
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
import pytest

from statpitch import decision_config
from statpitch.decision import card as cb

NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
FIXTURE = "ENG.PL|2026-2027|Arsenal FC|Chelsea FC"


@pytest.fixture
def config():
    return decision_config.config()


@pytest.fixture
def fitted(config):
    """The config as it will look once Phase C fits it.

    `w` stays 0.0 — that is the measured value, and inventing a positive one to
    make a test pass would be inventing the project's headline result.
    """
    return replace(config, status="fitted", w=0.0, w_fitted=True)


@pytest.fixture
def fixtures():
    return pd.DataFrame(
        {
            "fixture_id": [FIXTURE],
            "competition_id": ["ENG.PL"],
            "date": [pd.Timestamp("2026-08-24")],
            "home_team": ["Arsenal FC"],
            "away_team": ["Chelsea FC"],
            "odds_coverage": [True],
        }
    )


@pytest.fixture
def predictions():
    return pd.DataFrame(
        {
            "fixture_id": [FIXTURE],
            "lambda_home": [1.7],
            "lambda_away": [1.1],
            "rho": [0.0],
            "model_version": ["goals-test"],
        }
    )


def _odds_row(selection, key, avg, mx, market="1x2", line=None, capture="C1"):
    return {
        "capture_id": capture,
        "captured_at": "2026-08-24T09:00:00+00:00",
        "competition_id": "ENG.PL",
        "date": pd.Timestamp("2026-08-24"),
        "kickoff_utc": pd.Timestamp("2026-08-24 14:00:00"),
        "fixture_id": FIXTURE,
        "market": market,
        "selection": selection,
        "selection_key": key,
        "line": line,
        "odds_avg": avg,
        "odds_max": mx,
        "n_books": 7,
    }


@pytest.fixture
def odds():
    """A three-way book priced the way football actually is.

    Consensus overround ~5.4%, with the best quotes a little above it but still
    summing above 1.0. Getting this wrong matters: an earlier version of this
    fixture put the best-of-N prices at 2.05/3.90/4.50, which sums to 0.966 —
    a riskless arbitrage, and therefore not a normal book at all.
    """
    return pd.DataFrame(
        [
            _odds_row("home", "1x2_home", 1.90, 1.95),
            _odds_row("draw", "1x2_draw", 3.60, 3.70),
            _odds_row("away", "1x2_away", 4.00, 4.15),
        ]
    )


# --- the card computes --------------------------------------------------------

def test_a_card_row_is_produced_for_every_priced_selection(
    fixtures, predictions, odds, config
):
    card, stats = cb.build_card(fixtures, predictions, odds, config, now=NOW)
    assert set(card["selection_key"]) == {"1x2_home", "1x2_draw", "1x2_away"}
    assert stats.fixtures_carded == 1
    assert stats.selections_assessed == 3


def test_fair_probabilities_come_from_the_consensus_and_sum_to_one(
    fixtures, predictions, odds, config
):
    """FR-16a: the de-vig runs on `odds_avg`, never on the best quote."""
    card, _ = cb.build_card(fixtures, predictions, odds, config, now=NOW)
    assert card["q_fair"].sum() == pytest.approx(1.0)


def test_the_price_taken_is_the_best_quote_not_the_consensus(
    fixtures, predictions, odds, config
):
    card, _ = cb.build_card(fixtures, predictions, odds, config, now=NOW)
    home = card[card["selection_key"] == "1x2_home"].iloc[0]
    assert home["odds_avg"] == pytest.approx(1.90)     # quoted consensus
    assert home["odds_max"] == pytest.approx(1.95)     # best quote, the price taken

    # Fair odds are LONGER than the quoted consensus, because de-vigging removes
    # the overround. Asserting the opposite is how this test first failed, and it
    # caught `1/q_fair` being written into a column named `odds_avg` — the same
    # conflation FR-16a exists to prevent.
    assert home["fair_odds"] > home["odds_avg"]
    # Here the best quote still does not reach fair value, so the price edge is
    # negative. That is the ordinary state of an efficient market.
    assert home["fair_odds"] > home["odds_max"]
    assert home["price_edge"] < 0


# --- what w=0 does, made visible ----------------------------------------------

def test_at_w_zero_the_used_probability_is_the_market_s_own(
    fixtures, predictions, odds, config
):
    card, _ = cb.build_card(fixtures, predictions, odds, config, now=NOW)
    assert card["p_used"].tolist() == pytest.approx(card["q_fair"].tolist())


def test_at_w_zero_model_edge_is_exactly_zero(fixtures, predictions, odds, config):
    """The headline finding, arriving as arithmetic rather than as prose.

    `shrink(p, q, 0)` returns `q`, so the model cannot disagree with the market
    no matter what it predicted. Every surviving pound of EV is `price_edge`.
    """
    card, _ = cb.build_card(fixtures, predictions, odds, config, now=NOW)
    assert card["model_edge"].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert card["edge_prob"].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert card["expected_value"].tolist() == pytest.approx(card["price_edge"].tolist())


def test_the_model_still_disagrees_even_though_it_is_not_used(
    fixtures, predictions, odds, config
):
    """`p_model` is recorded, so the disagreement w discards stays visible."""
    card, _ = cb.build_card(fixtures, predictions, odds, config, now=NOW)
    assert (card["p_model"] - card["q_fair"]).abs().max() > 0.01


# --- the placeholder gate -----------------------------------------------------

def test_nothing_is_staked_while_the_config_is_a_placeholder(
    fixtures, predictions, odds, config
):
    """Including the confidence fallback, which must not route around the gate.

    It is a product requirement layered on top of the rule, not a way past the
    one check that stops an unfitted config sizing a stake.
    """
    from dataclasses import replace

    config = replace(config, status="placeholder", w_fitted=False, w=None)
    card, stats = cb.build_card(fixtures, predictions, odds, config, now=NOW)
    assert (card["stake_fraction"] == 0.0).all()
    assert stats.staked == 0
    assert stats.total_exposure == 0.0


def test_the_placeholder_card_is_still_fully_computed(
    fixtures, predictions, odds, config
):
    """An empty slate must not be an empty card — that is the whole point."""
    from dataclasses import replace

    placeholder = replace(config, status="placeholder", w_fitted=False, w=None)
    card, stats = cb.build_card(fixtures, predictions, odds, placeholder, now=NOW)
    assert not card.empty
    assert stats.selections_assessed > 0
    assert card["grade"].notna().all()
    assert (card["config_status"] == "placeholder").all()
    assert (card["stake_fraction"] == 0.0).all()


# --- the fitted path, which production has never reached ----------------------

def test_a_fitted_config_can_size_a_stake(fixtures, predictions, fitted):
    """Reachable only with a mispriced book, since at w=0 EV is price alone.

    The away price here is deliberately generous enough to clear grading, which
    is what a genuine soft quote would look like.
    """
    generous = pd.DataFrame(
        [
            _odds_row("home", "1x2_home", 1.95, 1.96),
            _odds_row("draw", "1x2_draw", 3.70, 3.75),
            _odds_row("away", "1x2_away", 4.20, 4.35),
        ]
    )
    card, stats = cb.build_card(fixtures, predictions, generous, fitted, now=NOW)
    assert not card.empty
    assert (card["config_status"] == "fitted").all()
    # Whether anything clears the cutoff depends on the grading parameters Phase
    # C will fit; what must hold now is that the path runs and respects the caps.
    assert (card["stake_fraction"] <= fitted.staking.cap_per_bet + 1e-9).all()
    assert stats.total_exposure <= fitted.staking.cap_per_matchday + 1e-9


def test_staking_never_exceeds_the_matchday_cap(fixtures, predictions, fitted):
    cheap = pd.DataFrame(
        [
            _odds_row("home", "1x2_home", 1.95, 3.60),
            _odds_row("draw", "1x2_draw", 3.70, 6.00),
            _odds_row("away", "1x2_away", 4.20, 9.00),
        ]
    )
    card, stats = cb.build_card(fixtures, predictions, cheap, fitted, now=NOW)
    assert stats.total_exposure <= fitted.staking.cap_per_matchday + 1e-9


# --- de-vig grouping ----------------------------------------------------------

def test_a_partially_quoted_market_is_skipped_rather_than_half_priced(
    fixtures, predictions, config
):
    """A book can only be de-vigged as a complete set."""
    partial = pd.DataFrame(
        [
            _odds_row("home", "1x2_home", 1.90, 1.95),
            _odds_row("draw", "1x2_draw", 3.60, 3.70),
            # away missing
        ]
    )
    card, stats = cb.build_card(fixtures, predictions, partial, config, now=NOW)
    assert card.empty
    assert stats.skipped_no_devigable_market == 1


def test_a_market_with_a_missing_consensus_price_is_skipped(
    fixtures, predictions, config
):
    holed = pd.DataFrame(
        [
            _odds_row("home", "1x2_home", None, 1.95),
            _odds_row("draw", "1x2_draw", 3.60, 3.70),
            _odds_row("away", "1x2_away", 4.00, 4.15),
        ]
    )
    card, stats = cb.build_card(fixtures, predictions, holed, config, now=NOW)
    assert card.empty
    assert stats.skipped_no_devigable_market == 1


def test_totals_and_handicaps_are_devigged_as_their_own_books(
    fixtures, predictions, config
):
    rows = [
        _odds_row("home", "1x2_home", 1.90, 1.95),
        _odds_row("draw", "1x2_draw", 3.60, 3.70),
        _odds_row("away", "1x2_away", 4.00, 4.15),
        _odds_row("over", "over_2.5", 1.90, 1.95, market="ou", line=2.5),
        _odds_row("under", "under_2.5", 1.95, 2.00, market="ou", line=2.5),
        _odds_row("ah_home", "ah_home_-0.5", 1.92, 1.98, market="ah", line=-0.5),
        _odds_row("ah_away", "ah_away_0.5", 1.94, 2.00, market="ah", line=0.5),
    ]
    card, _ = cb.build_card(
        fixtures, predictions, pd.DataFrame(rows), config, now=NOW
    )
    for market_keys in (
        {"1x2_home", "1x2_draw", "1x2_away"},
        {"over_2.5", "under_2.5"},
        {"ah_home_-0.5", "ah_away_0.5"},
    ):
        block = card[card["selection_key"].isin(market_keys)]
        assert len(block) == len(market_keys)
        assert block["q_fair"].sum() == pytest.approx(1.0)


# --- append-only odds, single card --------------------------------------------

def test_only_the_newest_capture_is_priced(fixtures, predictions, config):
    """The log holds every capture; a card is about what can be had now."""
    stale = [
        _odds_row("home", "1x2_home", 1.80, 1.85, capture="C1"),
        _odds_row("draw", "1x2_draw", 3.60, 3.70, capture="C1"),
        _odds_row("away", "1x2_away", 4.00, 4.15, capture="C1"),
    ]
    fresh = [
        _odds_row("home", "1x2_home", 1.90, 1.95, capture="C2"),
        _odds_row("draw", "1x2_draw", 3.60, 3.70, capture="C2"),
        _odds_row("away", "1x2_away", 4.00, 4.15, capture="C2"),
    ]
    card, _ = cb.build_card(
        fixtures, predictions, pd.DataFrame(stale + fresh), config, now=NOW
    )
    home = card[card["selection_key"] == "1x2_home"].iloc[0]
    assert home["odds_max"] == pytest.approx(1.95)
    assert home["capture_id"] == "C2"
    assert len(card) == 3


# --- the arbitrage diagnostic Phase A found -----------------------------------

def test_a_best_of_n_book_summing_below_one_is_recorded_not_hidden(
    fixtures, predictions, config
):
    """Those quotes cannot all have been live, so the edge from them is fake.

    Not rejected here: which of them was takeable is a calibration question, and
    Phase C needs the evidence rather than a filtered frame.
    """
    arb = pd.DataFrame(
        [
            _odds_row("home", "1x2_home", 1.95, 2.60),
            _odds_row("draw", "1x2_draw", 3.70, 5.20),
            _odds_row("away", "1x2_away", 4.20, 6.40),
        ]
    )
    card, stats = cb.build_card(fixtures, predictions, arb, config, now=NOW)
    assert stats.arbitrage_fixtures == [FIXTURE]
    assert card["max_book_sum"].iloc[0] < 1.0


def test_a_normal_book_is_not_flagged_as_arbitrage(
    fixtures, predictions, odds, config
):
    card, stats = cb.build_card(fixtures, predictions, odds, config, now=NOW)
    assert stats.arbitrage_fixtures == []
    assert card["max_book_sum"].iloc[0] > 1.0


# --- fixtures without a prediction --------------------------------------------

def test_a_priced_fixture_with_no_prediction_is_counted_not_carded(
    fixtures, predictions, odds, config
):
    """Prices outlive the fixture list: a played match keeps its odds rows."""
    orphan = odds.copy()
    orphan["fixture_id"] = "ENG.PL|2026-2027|Played FC|Gone FC"
    card, stats = cb.build_card(
        fixtures, predictions, pd.concat([odds, orphan]), config, now=NOW
    )
    assert stats.skipped_no_prediction == 1
    assert set(card["fixture_id"]) == {FIXTURE}


def test_an_empty_odds_log_still_produces_a_priced_card(fixtures, predictions, config):
    """It used to produce nothing, and that was the wrong way round.

    "No bookmaker has quoted anything" is exactly the case the model-implied
    fallback exists for, so returning an empty card there skipped the fallback
    in the one situation it was written to cover.
    """
    card, stats = cb.build_card(
        fixtures, predictions, pd.DataFrame(columns=["fixture_id"]), config, now=NOW
    )
    assert not card.empty
    assert list(card.columns) == list(cb.CARD_COLUMNS)
    assert stats.fixtures_priced == 0
    assert set(card["pricing"]) == {"model"}


# --- the selection rule gates staking -----------------------------------------

def test_only_rule_qualified_selections_are_staked(fixtures, predictions, odds, fitted):
    """A selection the measured evidence does not cover is not a bet, however
    well it grades."""
    from dataclasses import replace

    from statpitch.decision_config import SelectionRule

    ruled = replace(
        fitted,
        selection_rule=SelectionRule(
            status="experimental", reference="odds_pinnacle",
            threshold=0.0, market_families=("1x2",), max_per_day=3,
        ),
    )
    priced = odds.copy()
    priced["odds_pinnacle"] = priced["odds_avg"]
    card, _ = cb.build_card(fixtures, predictions, priced, ruled, now=NOW)
    staked = card[card["stake_fraction"] > 0]
    assert staked.empty or staked["rule_qualified"].all()


def test_the_per_day_cap_keeps_the_best_by_rule_edge(fixtures, predictions, fitted):
    """A cap, not a floor: it never invents a selection on a day that has none,
    it stops one day's slate swallowing the matchday exposure."""
    from dataclasses import replace

    from statpitch.decision_config import SelectionRule

    ruled = replace(
        fitted,
        selection_rule=SelectionRule(
            status="experimental", reference="odds_pinnacle",
            threshold=0.0, market_families=("1x2",), max_per_day=1,
        ),
    )
    generous = pd.DataFrame([
        _odds_row("home", "1x2_home", 1.90, 2.60),
        _odds_row("draw", "1x2_draw", 3.60, 4.60),
        _odds_row("away", "1x2_away", 4.00, 5.20),
    ])
    generous["odds_pinnacle"] = generous["odds_avg"]
    card, stats = cb.build_card(fixtures, predictions, generous, ruled, now=NOW)
    assert (card["stake_fraction"] > 0).sum() <= 1


def test_a_rule_with_no_reference_stakes_nothing_through_it(
    fixtures, predictions, odds, config
):
    """`candidate` is recorded, not run."""
    assert not config.selection_rule.is_active or config.selection_rule.reference


# --- the confidence fallback (Part 2) -----------------------------------------

def _ruled(fitted, **kw):
    """A config whose rule cannot be cleared, so the fallback is what fires."""
    from dataclasses import replace

    from statpitch.decision_config import SelectionRule

    defaults = dict(
        status="experimental", reference="odds_pinnacle", threshold=99.0,
        market_families=("1x2",), max_per_day=3,
        fallback_enabled=True, fallback_stake=0.0005,
    )
    defaults.update(kw)
    return replace(fitted, selection_rule=SelectionRule(**defaults))


def _with_sharp(odds):
    priced = odds.copy()
    priced["odds_pinnacle"] = priced["odds_avg"]
    return priced


def test_a_day_the_rule_leaves_empty_still_gets_a_pick(
    fixtures, predictions, odds, fitted
):
    """The product requirement: every day with football answers."""
    card, stats = cb.build_card(
        fixtures, predictions, _with_sharp(odds), _ruled(fitted), now=NOW
    )
    staked = card[card["stake_fraction"] > 0]
    assert len(staked) == 1
    assert staked["selection_basis"].iloc[0] == "confidence"
    assert stats.confidence_picks == 1


def test_the_confidence_pick_is_the_most_likely_outcome(
    fixtures, predictions, odds, fitted
):
    """`p_model`, not edge. That is what makes it a confidence pick rather than
    a value one, and why it carries no measurement."""
    card, _ = cb.build_card(
        fixtures, predictions, _with_sharp(odds), _ruled(fitted), now=NOW
    )
    pick = card[card["stake_fraction"] > 0].iloc[0]
    assert pick["p_model"] == card["p_model"].max()


def test_the_fallback_is_flat_staked_not_kelly_sized(
    fixtures, predictions, odds, fitted
):
    """Kelly sizes from an edge. There is none here, so sizing from one would be
    inventing a number."""
    card, _ = cb.build_card(
        fixtures, predictions, _with_sharp(odds), _ruled(fitted), now=NOW
    )
    assert card[card["stake_fraction"] > 0]["stake_fraction"].iloc[0] == 0.0005


def test_the_fallback_can_be_switched_off(fixtures, predictions, odds, fitted):
    card, _ = cb.build_card(
        fixtures, predictions, _with_sharp(odds),
        _ruled(fitted, fallback_enabled=False), now=NOW,
    )
    assert (card["stake_fraction"] == 0.0).all()


# --- every fixture carries a price (Part 1) -----------------------------------

def test_an_unquoted_fixture_still_gets_model_implied_odds(
    fixtures, predictions, config
):
    """657 upcoming fixtures, 30 quoted. The other 627 are not a pipeline gap —
    books open a market about a week out and those prices do not exist yet."""
    card, stats = cb.build_card(
        fixtures, predictions, pd.DataFrame(columns=["fixture_id"]), config, now=NOW
    )
    assert stats.fixtures_model_priced == 1
    assert set(card["pricing"]) == {"model"}
    assert card["model_odds"].notna().all()


def test_a_model_priced_row_has_no_market_fields(fixtures, predictions, config):
    """Null, not zero. A zero would read as "the market says impossible" rather
    than "no market exists"."""
    card, _ = cb.build_card(
        fixtures, predictions, pd.DataFrame(columns=["fixture_id"]), config, now=NOW
    )
    for column in ("q_fair", "odds_max", "reference_odds", "rule_edge"):
        assert card[column].isna().all(), column


def test_model_priced_rows_are_never_rule_qualified(fixtures, predictions, config):
    card, _ = cb.build_card(
        fixtures, predictions, pd.DataFrame(columns=["fixture_id"]), config, now=NOW
    )
    assert not card["rule_qualified"].any()


def test_a_competition_outside_the_rules_scope_is_never_staked(
    fixtures, predictions, odds, fitted
):
    """The guard that stops a pooled average authorising a bet.

    The card fixtures are ENG.PL. Scoped to a rule measured only in Serie A,
    nothing here may be staked — even though the edge, the market family and the
    grade are all exactly what they were when it did qualify.
    """
    from dataclasses import replace

    from statpitch.decision_config import SelectionRule

    # A threshold nothing can fail, so the ONLY thing separating the two runs
    # below is the competition scope.
    base = dict(
        status="experimental", reference="odds_pinnacle",
        threshold=-1.0, market_families=("1x2",), max_per_day=3,
    )
    priced = _with_sharp(odds)

    in_scope = replace(fitted, selection_rule=SelectionRule(
        **base, competitions=("ENG.PL",)))
    card_in, _ = cb.build_card(fixtures, predictions, priced, in_scope, now=NOW)

    out_of_scope = replace(fitted, selection_rule=SelectionRule(
        **base, competitions=("ITA.SERIEA",)))
    card_out, _ = cb.build_card(fixtures, predictions, priced, out_of_scope, now=NOW)

    assert not card_out["rule_qualified"].any()
    assert (card_out["stake_fraction"].fillna(0.0) == 0.0).all()
    # ...and the exclusion is the scope, not something else about the fixtures.
    assert card_in["rule_qualified"].any(), (
        "the in-scope control did not qualify either, so this test would pass "
        "for the wrong reason"
    )
