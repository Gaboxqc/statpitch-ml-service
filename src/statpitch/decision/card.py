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
            "skipped_no_devigable_market": self.skipped_no_devigable_market,
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
    rows: pd.DataFrame, method: str
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float | None, float | None]:
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
    if priced.empty:
        return pd.DataFrame(columns=list(CARD_COLUMNS)), stats

    by_fixture = {str(f): g for f, g in priced.groupby("fixture_id")}
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

        fair, available, consensus, margin, max_sum = fair_book(
            rows, config.devig_method(competition_id)
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
        graded, _ = bet_grader.grade_book(assessments, context, **_grade_kwargs(config))
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

            if bet.is_stakeable and not config.is_placeholder:
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

    card = pd.DataFrame.from_records(records, columns=list(CARD_COLUMNS))
    if card.empty:
        return card, stats

    if slate:
        _size_the_slate(card, slate, slate_index, config, stats)

    return card.reset_index(drop=True), stats


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

    staked = card["stake_fraction"] > 0
    stats.staked = int(staked.sum())
    stats.total_exposure = float(card.loc[staked, "stake_fraction"].sum())
