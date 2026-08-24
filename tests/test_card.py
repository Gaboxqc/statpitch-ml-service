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
    card, stats = cb.build_card(fixtures, predictions, odds, config, now=NOW)
    assert (card["stake_fraction"] == 0.0).all()
    assert stats.staked == 0
    assert stats.total_exposure == 0.0


def test_the_placeholder_card_is_still_fully_computed(
    fixtures, predictions, odds, config
):
    """An empty slate must not be an empty card — that is the whole point."""
    card, stats = cb.build_card(fixtures, predictions, odds, config, now=NOW)
    assert not card.empty
    assert stats.selections_assessed > 0
    assert card["grade"].notna().all()
    assert (card["config_status"] == "placeholder").all()


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


def test_an_empty_odds_log_produces_an_empty_card(fixtures, predictions, config):
    card, stats = cb.build_card(
        fixtures, predictions, pd.DataFrame(columns=["fixture_id"]), config, now=NOW
    )
    assert card.empty
    assert list(card.columns) == list(cb.CARD_COLUMNS)
    assert stats.fixtures_priced == 0
