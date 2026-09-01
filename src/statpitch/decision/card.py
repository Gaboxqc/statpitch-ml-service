"""Assemble the matchday card from fixtures, predictions and live prices (Plan §4 Phase B).

Every piece of this existed before this module: the market engine derives 86
selections from a score matrix, `devig` turns a quoted book into fair
probabilities, `value` decomposes edge into price and model, `bet_grader` scores
it and `staking` sizes it. What did not exist was anything that called them in
order on a real fixture.

`/card/today` returned a hardcoded empty list. `ops.jobs.flag_card` said so
outright in its fitted branch — *"config is fitted but no fixture source is
wired in"*. This is that wiring.

What it does not do is manufacture a bet
========================================

With `w` = 0.000, `shrink(p_model, q_fair, 0)` returns `q_fair` exactly, so
`edge_prob` is zero by construction and `value.model_edge` with it. The only
surviving quantity is `price_edge` — what a bettor earns by taking the best
quote while believing precisely what the consensus believes.

That is not a bug in this module and it must not be papered over. The card is
computed in full and then reports that nothing is stakeable, naming the gate
that stopped it. An empty card with a cited reason and an empty card because
nobody wrote the code are the same JSON and completely different facts; before
this, the API emitted the second while claiming the first.

Why the price used is the consensus and the price taken is the best quote
========================================================================

FR-16a, and it is structural rather than stylistic. `q_fair` is de-vigged from
`odds_avg`; `o_avail` is `odds_max`. De-vigging the best-of-N price instead
would fabricate edge, because max-of-N sits above consensus by construction.

`odds_max` has its own problem, measured in Phase A and recorded per fixture
here rather than silently absorbed: it is a high-water mark over the quoting
period, not a simultaneously available price. On the first live capture, 3 of 38
fixtures had `Max` implied probabilities summing *below* 1.0 — a riskless
arbitrage, which means those quotes were not all live at once. `max_book_sum` is
carried on every row so the calibration Phase C owes this can be done on
evidence instead of assumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from statpitch.decision import bet_grader, devig, market_engine, staking, value
from statpitch.models.dixon_coles import score_matrix

log = logging.getLogger(__name__)

#: Selections quoted by the free sources, grouped into the books that must be
#: de-vigged together. A market is only de-viggable as a complete set — its
#: implied probabilities have to sum to something before the overround can be
#: removed — so a partially quoted book is skipped rather than half-priced.
_DEVIG_GROUPS: dict[str, tuple[str, ...]] = {
    "1x2": ("home", "draw", "away"),
    "ou": ("over", "under"),
    "ah": ("ah_home", "ah_away"),
}

CARD_COLUMNS = (
    "fixture_id", "competition_id", "date", "kickoff_utc", "home_team", "away_team",
    "selection_key", "market_family", "line", "description",
    "p_model", "q_fair", "p_used", "odds_avg", "fair_odds", "odds_max",
    "edge_prob", "expected_value", "price_edge", "model_edge",
    "grade", "composite", "reasons", "stake_fraction",
    "rule_edge", "rule_qualified", "reference_odds",
    "pricing", "model_odds", "selection_basis",
    "book_margin", "max_book_sum", "n_books", "capture_id",
    "w", "config_version", "config_status", "model_version", "generated_at",
)


@dataclass
class CardStats:
    """What a build did, in a form a job can log and a test can assert on."""

    fixtures_priced: int = 0
    fixtures_carded: int = 0
    selections_assessed: int = 0
    positive_ev: int = 0
    stakeable: int = 0
    staked: int = 0
    total_exposure: float = 0.0
    skipped_no_prediction: int = 0
    skipped_no_devigable_market: int = 0
    capped: int = 0
    #: Upcoming fixtures no bookmaker has quoted, carried at model-implied odds
    #: so a fixture list is never missing a price (Part 1).
    fixtures_model_priced: int = 0
    #: Days where nothing cleared the selection rule and the highest-confidence
    #: selection was surfaced instead (Part 2).
    confidence_picks: int = 0
    arbitrage_fixtures: list[str] = field(default_factory=list)
    grades: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "fixtures_priced": self.fixtures_priced,
            "fixtures_carded": self.fixtures_carded,
            "selections_assessed": self.selections_assessed,
            "positive_ev": self.positive_ev,
            "stakeable": self.stakeable,
            "staked": self.staked,
            "total_exposure": round(self.total_exposure, 6),
            "skipped_no_prediction": self.skipped_no_prediction,
            "fixtures_model_priced": self.fixtures_model_priced,
            "confidence_picks": self.confidence_picks,
            "skipped_no_devigable_market": self.skipped_no_devigable_market,
            "capped_per_day": self.capped,
            "arbitrage_fixtures": len(self.arbitrage_fixtures),
            "grades": dict(sorted(self.grades.items())),
        }


def latest_capture(odds: pd.DataFrame) -> pd.DataFrame:
    """The most recent quote for each fixture and selection.

    `live_odds.parquet` is append-only, so it holds every capture ever taken.
    A card is about what can be had *now*, which is the newest one — the earlier
    captures are the other half of a CLV measurement, not competing prices.
    """
    if odds.empty:
        return odds
    ordered = odds.sort_values("capture_id")
    return ordered.drop_duplicates(
        subset=["fixture_id", "selection_key"], keep="last"
    ).reset_index(drop=True)


def fair_book(
    rows: pd.DataFrame, method: str, reference: str | None = None
) -> tuple[
    dict[str, float], dict[str, float], dict[str, float],
    dict[str, float], dict[str, float], float | None, float | None,
]:
    """De-vig every complete market quoted for one fixture.

    Returns (fair probabilities, available prices, consensus prices, mean
    consensus margin, minimum best-of-N book sum) keyed by `market_engine`
    selection key.

    The consensus price is returned as well as the fair probability derived from
    it, because they are different numbers and must not share a column. The
    quoted consensus carries the overround; the fair odds do not, and are
    therefore always the longer of the two. Writing `1/q_fair` into a field named
    `odds_avg` — which is the *quoted* consensus everywhere else in this
    project — is the FR-16a conflation in miniature.

    The book sum on `odds_max` is returned alongside because a value below 1.0
    is proof the quotes were not simultaneously available, and that is a
    property of the fixture rather than of any one selection.
    """
    fair: dict[str, float] = {}
    available: dict[str, float] = {}
    consensus_price: dict[str, float] = {}
    reference_price: dict[str, float] = {}
    reference_fair: dict[str, float] = {}
    margins: list[float] = []
    max_sums: list[float] = []

    for market, members in _DEVIG_GROUPS.items():
        block = rows[rows["market"] == market]
        if block.empty:
            continue
        by_selection = {str(r["selection"]): r for _, r in block.iterrows()}
        if not all(name in by_selection for name in members):
            continue

        consensus = [by_selection[name]["odds_avg"] for name in members]
        if any(pd.isna(price) for price in consensus):
            continue

        result = devig.devig([float(p) for p in consensus], method)
        margins.append(result.margin)

        # The sharp reference, de-vigged as its own complete book. This is the
        # quantity MODEL_CARD 5's finding is defined on, and it only became
        # obtainable live when a source carrying Pinnacle was added.
        if reference:
            quotes = [by_selection[name].get(reference) for name in members]
            if not any(price is None or pd.isna(price) for price in quotes):
                sharp = devig.devig([float(p) for p in quotes], method)
                for name, probability in zip(members, sharp.probabilities, strict=True):
                    key = str(by_selection[name]["selection_key"])
                    reference_fair[key] = float(probability)
                    reference_price[key] = float(by_selection[name][reference])

        best = [by_selection[name]["odds_max"] for name in members]
        if not any(pd.isna(price) for price in best):
            max_sums.append(float(sum(1.0 / float(p) for p in best)))

        for name, probability in zip(members, result.probabilities, strict=True):
            row = by_selection[name]
            key = str(row["selection_key"])
            fair[key] = float(probability)
            consensus_price[key] = float(row["odds_avg"])
            price = row["odds_max"]
            if pd.notna(price):
                available[key] = float(price)

    return (
        fair,
        available,
        consensus_price,
        reference_fair,
        reference_price,
        float(np.mean(margins)) if margins else None,
        min(max_sums) if max_sums else None,
    )


def _grade_kwargs(config) -> dict:
    grading = config.grading
    guardrails = config.guardrails
    return {
        "e_peak": grading.e_peak,
        "sigma": grading.sigma,
        "e_ceiling": grading.e_ceiling,
        "cutoffs": dict(grading.cutoffs),
        "weights": dict(grading.subscore_weights),
        "max_p_std": guardrails.max_p_std,
        "max_margin": guardrails.max_book_margin,
        "odds_ceiling": guardrails.odds_ceiling,
    }


def build_card(
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
    config,
    *,
    model_version: str = "unknown",
    now: datetime | None = None,
) -> tuple[pd.DataFrame, CardStats]:
    """Fixtures x predictions x prices -> one row per assessed selection.

    Staking is attempted only when `decision_config` is fitted. While it is a
    placeholder every `stake_fraction` is 0.0 and the reason travels with the
    row, because `StakingEngine` refuses to size from unfitted parameters and
    inventing a number here would route around the one gate that stops a
    placeholder becoming a recommendation.
    """
    stats = CardStats()
    stamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
    engine_config = config.market_engine
    # `w` is None while unfitted. 0.0 is not a guess: it is the measured value
    # (MODEL_CARD §1), and using it keeps the placeholder card identical to the
    # fitted one in every respect except that nothing is staked.
    w = float(config.w or 0.0)

    priced = latest_capture(odds)
    # No early return on an empty capture. "Nothing is quoted" is precisely when
    # the model-implied fallback below matters most — a fixture list with no
    # prices at all should still carry a price for every match, and returning
    # here made the one case the fallback exists for the one case it skipped.
    by_fixture = (
        {str(f): g for f, g in priced.groupby("fixture_id")}
        if not priced.empty else {}
    )
    stats.fixtures_priced = len(by_fixture)

    fixture_meta = fixtures.set_index("fixture_id")
    prediction_meta = predictions.set_index("fixture_id")

    records: list[dict] = []
    slate: list[staking.SlateBet] = []
    slate_index: dict[str, int] = {}

    for fixture_id, rows in by_fixture.items():
        if fixture_id not in prediction_meta.index or fixture_id not in fixture_meta.index:
            stats.skipped_no_prediction += 1
            continue

        prediction = prediction_meta.loc[fixture_id]
        meta = fixture_meta.loc[fixture_id]
        competition_id = str(rows["competition_id"].iloc[0])

        rule = config.selection_rule
        fair, available, consensus, sharp_fair, sharp_price, margin, max_sum = fair_book(
            rows, config.devig_method(competition_id),
            reference=rule.reference if rule.is_active else None,
        )
        if not fair or not available:
            stats.skipped_no_devigable_market += 1
            continue

        if max_sum is not None and max_sum < 1.0:
            # Not rejected here: the prices are real, and which of them is
            # takeable is a calibration question this module cannot answer.
            stats.arbitrage_fixtures.append(fixture_id)

        matrix = score_matrix(
            float(prediction["lambda_home"]),
            float(prediction["lambda_away"]),
            rho=float(prediction.get("rho", 0.0) or 0.0),
            max_goals=engine_config.matrix_max_goals,
        )
        selections = market_engine.derive(
            matrix,
            totals_lines=tuple(engine_config.total_goals_lines),
            handicap_lines=tuple(engine_config.asian_handicap_lines),
            correct_score_top_n=engine_config.correct_score_top_n,
        )
        by_key = {s.key: s for s in selections}

        # p_used is the shrunk probability (Design §6.5). At w=0 it IS q_fair,
        # which is what collapses model_edge to zero — visible in the output
        # rather than asserted in a comment.
        shrunk = {
            key: staking.shrink(by_key[key].probability, q, w)
            for key, q in fair.items()
            if key in by_key
        }

        assessments = value.assess_book(selections, fair, available, shrunk)
        if not assessments:
            continue

        context = bet_grader.GradingContext(
            book_margin=margin,
            n_books=int(rows["n_books"].iloc[0]) if pd.notna(rows["n_books"].iloc[0]) else None,
            odds_coverage=bool(meta.get("odds_coverage", True)),
        )
        # "Back it when the best quote beats the sharp book's fair value."
        # Computed against the REFERENCE, never against the consensus: Phase C
        # measured the consensus-referenced version as regression to the mean,
        # and grading on it would encode an artifact as a signal.
        rule_edge = {
            key: float(available[key] * sharp_fair[key] - 1.0)
            for key in available
            if key in sharp_fair
        }
        graded, _ = bet_grader.grade_book(
            assessments, context, edges=rule_edge, **_grade_kwargs(config)
        )
        grades = {g.key: g for g in graded}

        stats.fixtures_carded += 1
        for assessment in assessments:
            bet = grades[assessment.key]
            stats.selections_assessed += 1
            stats.grades[bet.grade.value] = stats.grades.get(bet.grade.value, 0) + 1
            if assessment.expected_value > 0:
                stats.positive_ev += 1
            if bet.is_stakeable:
                stats.stakeable += 1

            selection = by_key[assessment.key]
            edge = rule_edge.get(assessment.key)
            qualified = bool(
                rule.is_active
                and edge is not None
                and edge > rule.threshold
                and rule.covers(str(selection.family))
            )
            records.append(
                {
                    "fixture_id": fixture_id,
                    "competition_id": competition_id,
                    "date": meta["date"],
                    "kickoff_utc": rows["kickoff_utc"].iloc[0],
                    "home_team": meta["home_team"],
                    "away_team": meta["away_team"],
                    "selection_key": assessment.key,
                    "market_family": str(selection.family),
                    "line": selection.line,
                    "description": selection.description,
                    "p_model": selection.probability,
                    "q_fair": assessment.q_fair,
                    "p_used": assessment.p_model,
                    # Quoted consensus and de-vigged fair odds, kept apart: the
                    # first carries the overround, the second does not.
                    "odds_avg": consensus.get(assessment.key),
                    "fair_odds": assessment.fair_odds,
                    "odds_max": assessment.o_avail,
                    "edge_prob": assessment.edge_prob,
                    "expected_value": assessment.expected_value,
                    "price_edge": assessment.price_edge,
                    "model_edge": assessment.model_edge,
                    "rule_edge": edge,
                    "rule_qualified": qualified,
                    "reference_odds": sharp_price.get(assessment.key),
                    "pricing": "market",
                    "model_odds": (
                        float(1.0 / selection.probability)
                        if selection.probability > 0 else None
                    ),
                    "selection_basis": None,
                    "grade": bet.grade.value,
                    "composite": bet.composite,
                    "reasons": "; ".join(bet.reasons),
                    "stake_fraction": 0.0,
                    "book_margin": margin,
                    "max_book_sum": max_sum,
                    "n_books": context.n_books,
                    "capture_id": str(rows["capture_id"].iloc[0]),
                    "w": w,
                    "config_version": config.config_version,
                    "config_status": config.status,
                    "model_version": model_version,
                    "generated_at": stamp,
                }
            )

            # The rule gates staking, not just ranking. A selection the measured
            # evidence does not cover is not a bet however well it grades.
            if bet.is_stakeable and not config.is_placeholder and (
                qualified or not rule.is_active
            ):
                slate_index[assessment.key + "@" + fixture_id] = len(records) - 1
                slate.append(
                    staking.SlateBet(
                        key=assessment.key + "@" + fixture_id,
                        odds=assessment.o_avail,
                        payoff=assessment.payoff,
                        p_used=assessment.p_model,
                        fixture_id=fixture_id,
                        max_fraction=config.staking.cap_per_bet
                        * bet.stake_multiplier(dict(config.grading.stake_multiplier)),
                    )
                )

    records += _model_priced_rows(
        fixtures, prediction_meta, set(by_fixture), engine_config, config,
        stamp, model_version, w, stats,
    )

    card = pd.DataFrame.from_records(records, columns=list(CARD_COLUMNS))
    if card.empty:
        return card, stats

    if slate:
        slate, slate_index = _cap_per_day(card, slate, slate_index, config, stats)
    if slate:
        _size_the_slate(card, slate, slate_index, config, stats)

    _fill_empty_days(card, config, stats)

    return card.reset_index(drop=True), stats


def _model_priced_rows(
    fixtures: pd.DataFrame,
    prediction_meta: pd.DataFrame,
    priced: set[str],
    engine_config,
    config,
    stamp: str,
    model_version: str,
    w: float,
    stats: CardStats,
) -> list[dict]:
    """Every upcoming fixture no bookmaker has quoted, at model-implied odds.

    657 upcoming fixtures currently carry a prediction and 30 carry a price. The
    other 627 are not a gap in this pipeline — the prices do not exist anywhere
    yet, because books open a market roughly a week before kick-off and 21 of
    them are eleven days out. No amount of fetching creates a quote a bookmaker
    has not published.

    So they are emitted at `1 / p_model` and marked `pricing="model"`. That is a
    real number and it is not a market price: nothing can be BET at it, because
    it is this project's own opinion rather than an offer from anyone. The field
    is what keeps the two apart — a consumer rendering a fixture list gets a
    price for every match, and a consumer looking for a bet can filter to
    `pricing == "market"` and get only what is actually obtainable.

    Restricted to 1X2. The full 86-selection engine over 627 fixtures would put
    54,000 rows in the card to say the same thing three times over, and 1X2 is
    the only family the selection rule can act on anyway.
    """
    rows: list[dict] = []
    for fixture_id, meta in fixtures.set_index("fixture_id").iterrows():
        key = str(fixture_id)
        if key in priced or key not in prediction_meta.index:
            continue
        prediction = prediction_meta.loc[key]
        matrix = score_matrix(
            float(prediction["lambda_home"]),
            float(prediction["lambda_away"]),
            rho=float(prediction.get("rho", 0.0) or 0.0),
            max_goals=engine_config.matrix_max_goals,
        )
        stats.fixtures_model_priced += 1
        for selection in market_engine.derive(matrix)[:3]:      # 1x2 home/draw/away
            probability = float(selection.probability)
            rows.append({
                "fixture_id": key,
                "competition_id": meta["competition_id"],
                "date": meta["date"],
                "kickoff_utc": pd.NaT,
                "home_team": meta["home_team"],
                "away_team": meta["away_team"],
                "selection_key": selection.key,
                "market_family": str(selection.family),
                "line": selection.line,
                "description": selection.description,
                "p_model": probability,
                # No market, so no consensus to de-vig and nothing to compare
                # against. Leaving these null is the point: a zero would read as
                # "the market says impossible" rather than "no market exists".
                "q_fair": None, "p_used": None,
                "odds_avg": None, "fair_odds": None, "odds_max": None,
                "edge_prob": None, "expected_value": None,
                "price_edge": None, "model_edge": None,
                "rule_edge": None, "rule_qualified": False,
                "reference_odds": None,
                "pricing": "model",
                "model_odds": float(1.0 / probability) if probability > 0 else None,
                "selection_basis": None,
                "grade": None, "composite": None,
                "reasons": "no bookmaker has quoted this fixture yet",
                "stake_fraction": 0.0,
                "book_margin": None, "max_book_sum": None,
                "n_books": 0, "capture_id": None,
                "w": w,
                "config_version": config.config_version,
                "config_status": config.status,
                "model_version": model_version,
                "generated_at": stamp,
            })
    return rows


def _cap_per_day(
    card: pd.DataFrame,
    slate: list[staking.SlateBet],
    slate_index: dict[str, int],
    config,
    stats: CardStats,
) -> tuple[list[staking.SlateBet], dict[str, int]]:
    """Keep only the best `max_per_day` candidates on each match date.

    Ranked by the rule edge — the sharp book's disagreement with the best
    available quote — because that is the quantity the rule was measured on.

    A cap rather than a floor, and it does not manufacture a selection on a day
    that has none: a day where nothing clears the threshold correctly produces
    nothing. What it prevents is the opposite failure, a single day's slate
    swallowing the matchday exposure cap and crowding out the days after it.
    """
    limit = config.selection_rule.max_per_day
    if limit is None:
        return slate, slate_index

    ranked: dict[object, list[tuple[float, staking.SlateBet]]] = {}
    for bet in slate:
        row = card.iloc[slate_index[bet.key]]
        edge = row["rule_edge"]
        ranked.setdefault(row["date"], []).append(
            (float(edge) if pd.notna(edge) else 0.0, bet)
        )

    kept: list[staking.SlateBet] = []
    for day in sorted(ranked):
        best = sorted(ranked[day], key=lambda pair: -pair[0])[:limit]
        kept.extend(bet for _, bet in best)
        dropped = len(ranked[day]) - len(best)
        if dropped:
            log.info(
                "%s: %d candidate(s) beyond the %d/day cap, keeping the best by "
                "rule edge", pd.Timestamp(day).date(), dropped, limit,
            )

    stats.capped = sum(len(v) for v in ranked.values()) - len(kept)
    return kept, {bet.key: slate_index[bet.key] for bet in kept}


def _fill_empty_days(card: pd.DataFrame, config, stats: CardStats) -> None:
    """Give every day with football a pick, even one the rule did not choose.

    A product requirement. The rule is a threshold and most days nothing clears
    it — measured over 48 upcoming days, 11 carried a price at all and one
    produced a bet — so without this the daily view is blank far more often
    than not.

    What is surfaced is the highest `p_model` selection of the day: the outcome
    the model is most certain about. That is a different question from the one
    the rule asks. The rule asks whether a PRICE is wrong; this asks which
    outcome is most LIKELY, and a heavy favourite at a fair price is extremely
    likely and worth nothing to back.

    Three things keep it from being mistaken for the rule's output:

    * `selection_basis="confidence"` on the row, carried into the API and the
      ledger, so a consumer can filter it out and a track record can be kept
      apart. MODEL_CARD §4 measured selection of this shape at -2.12% ROI.
    * A FLAT stake from config, not Kelly. Kelly sizes from an edge; there is no
      measured edge here, so sizing from one would be inventing a number.
    * Market-priced selections are preferred over model-priced ones, because a
      pick at `1/p_model` is a price nobody is offering — it can be shown, but
      it cannot be taken.
    """
    rule = config.selection_rule
    if not rule.fallback_enabled or config.is_placeholder or card.empty:
        return

    staked_days = set(card.loc[card["stake_fraction"] > 0, "date"])
    for day, group in card.groupby("date"):
        if day in staked_days:
            continue
        eligible = group[group["p_model"].notna()]
        if rule.market_families:
            eligible = eligible[eligible["market_family"].isin(rule.market_families)]
        if eligible.empty:
            continue
        # A takeable price first; a model-implied one only if nothing is quoted.
        preferred = eligible[eligible["pricing"] == "market"]
        if preferred.empty and not rule.fallback_stake:
            continue
        pool = preferred if not preferred.empty else eligible
        pick = pool["p_model"].idxmax()

        card.at[pick, "stake_fraction"] = float(rule.fallback_stake)
        card.at[pick, "selection_basis"] = "confidence"
        card.at[pick, "reasons"] = (
            "highest-confidence selection of the day; nothing cleared the "
            "selection rule, so this is a confidence pick and not a value one"
        )
        stats.confidence_picks += 1

    staked = card["stake_fraction"] > 0
    stats.staked = int(staked.sum())
    stats.total_exposure = float(card.loc[staked, "stake_fraction"].sum())


def _size_the_slate(
    card: pd.DataFrame,
    slate: list[staking.SlateBet],
    slate_index: dict[str, int],
    config,
    stats: CardStats,
) -> None:
    """Allocate across the whole slate at once, then write the fractions back.

    Solved jointly rather than bet by bet because sequential Kelly sizes each
    selection as though it were alone: a matchday holding three bets on the same
    favourite ends up with three times the exposure to one outcome that any
    single Kelly fraction implies (Design §6.5 step 4).
    """
    fractions = staking.allocate_slate(
        slate,
        cap_per_bet=config.staking.cap_per_bet,
        cap_matchday=config.staking.cap_per_matchday,
    )
    # `allocate_slate` solves full Kelly; lambda is the deliberate fraction of it
    # taken, and the caps are re-applied afterwards so scaling cannot lift a bet
    # back over its per-bet limit.
    lam = config.staking.kelly_lambda
    minimum = config.staking.min_stake_fraction

    for key, fraction in fractions.items():
        scaled = min(fraction * lam, config.staking.cap_per_bet)
        if scaled < minimum:
            scaled = 0.0
        index = slate_index.get(key)
        if index is None:
            continue
        card.at[index, "stake_fraction"] = scaled
        if scaled > 0:
            card.at[index, "selection_basis"] = "rule"

    staked = card["stake_fraction"] > 0
    stats.staked = int(staked.sum())
    stats.total_exposure = float(card.loc[staked, "stake_fraction"].sum())
