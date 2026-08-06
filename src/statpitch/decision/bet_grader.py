"""A-F bet grading with guardrails (FR-25, FR-33, Design §6.4).

Grading answers a different question from valuation. `value.py` says how much a
selection is worth if the model is right; this says how much the model should be
trusted on that selection.

The non-monotonic edge term
===========================

Design §6.4 calls this its most important single rule, and this project's own
measurements are direct evidence for it. Confidence peaks at a moderate edge —
around four probability points — and *decreases* above roughly ten, because in a
market this efficient an apparent twenty-point edge almost always means the model
is blind to something (a key injury, a dead rubber, confirmed rotation before a
midweek European tie) rather than that the market is wrong by twenty points.

That is not a theoretical worry here. Selecting the largest apparent edge across
markets returned -2.12% while committing to one market in advance returned
+0.13%, and the selector spent 54.5% of its picks on 1X2, the market with the
worst measured returns. Maximum-edge selection reliably finds the model's own
largest errors. The Gaussian shape penalises exactly that, and edges above the
ceiling are graded F and routed to review, where each one is a concrete instance
of information the feature set is missing.

Guardrails run first
====================

FR-33's suppressions are evaluated *before* any sub-score, and each forces an F
carrying its reason. A guardrail is a statement that the model's estimate is not
trustworthy for structural reasons, which is not something a good composite score
should be able to outvote.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum

from statpitch.decision.value import ValueAssessment

log = logging.getLogger(__name__)


class Grade(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"

    @property
    def is_stakeable(self) -> bool:
        return self in (Grade.A, Grade.B, Grade.C)


#: Design §6.4 defaults, overridden from decision_config in normal use.
DEFAULT_E_PEAK = 0.04
DEFAULT_SIGMA = 0.05
DEFAULT_E_CEILING = 0.12
DEFAULT_CUTOFFS = {"A": 0.80, "B": 0.65, "C": 0.50, "D": 0.35}
DEFAULT_WEIGHTS = {
    "c_edge": 0.30, "c_robust": 0.20, "c_market": 0.20,
    "c_calib": 0.15, "c_support": 0.15,
}
DEFAULT_MAX_P_STD = 0.06
DEFAULT_MAX_MARGIN = 0.08
DEFAULT_ODDS_CEILING = 8.0

#: Books quoting a selection, above which market quality stops improving.
BOOK_COUNT_SATURATION = 20


@dataclass(frozen=True, slots=True)
class GradingContext:
    """Everything grading needs beyond the value assessment itself."""

    #: Ensemble dispersion for this fixture (Design §5.1). Higher means the
    #: members disagree, so the blended probability is less trustworthy.
    p_std: float | None = None
    #: Book margin on this selection's market.
    book_margin: float | None = None
    #: How many books quoted it.
    n_books: int | None = None
    #: Historical calibration error for this competition and probability decile.
    calibration_error: float | None = None
    #: Realised CLV of historically similar bets, as a fraction.
    historical_clv: float | None = None

    # --- FR-33 guardrail inputs -----------------------------------------
    odds_coverage: bool = True
    lineup_confirmed: bool = True
    key_player_doubtful: bool = False
    dead_rubber: bool = False
    hours_to_other_competition_fixture: float | None = None


@dataclass(frozen=True, slots=True)
class SubScores:
    c_edge: float
    c_robust: float
    c_market: float
    c_calib: float
    c_support: float

    def composite(self, weights: dict[str, float]) -> float:
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("grading weights must sum to a positive number")
        return (
            self.c_edge * weights.get("c_edge", 0.0)
            + self.c_robust * weights.get("c_robust", 0.0)
            + self.c_market * weights.get("c_market", 0.0)
            + self.c_calib * weights.get("c_calib", 0.0)
            + self.c_support * weights.get("c_support", 0.0)
        ) / total


@dataclass(frozen=True, slots=True)
class GradedBet:
    key: str
    grade: Grade
    composite: float
    sub_scores: SubScores | None
    reasons: tuple[str, ...] = ()
    #: True when the edge exceeded the ceiling — the review queue, and the most
    #: valuable model-diagnostic signal the system produces.
    model_likely_blind: bool = False

    @property
    def is_stakeable(self) -> bool:
        return self.grade.is_stakeable

    def stake_multiplier(self, multipliers: dict[str, float] | None = None) -> float:
        table = multipliers or {"A": 1.0, "B": 0.5, "C": 0.25, "D": 0.0, "F": 0.0}
        return table.get(self.grade.value, 0.0)


# --- sub-scores ---------------------------------------------------------------

def edge_confidence(
    edge_prob: float,
    e_peak: float = DEFAULT_E_PEAK,
    sigma: float = DEFAULT_SIGMA,
    e_ceiling: float = DEFAULT_E_CEILING,
) -> float:
    """Non-monotonic confidence in an edge (Design §6.4).

    Peaks at `e_peak`, falls away either side, and is zero above `e_ceiling`.
    A negative edge scores zero: there is nothing to back.
    """
    if edge_prob <= 0.0:
        return 0.0
    if edge_prob > e_ceiling:
        return 0.0
    return math.exp(-(((edge_prob - e_peak) / sigma) ** 2))


def robustness(p_std: float | None, max_p_std: float = DEFAULT_MAX_P_STD) -> float:
    """Decreasing in ensemble dispersion.

    Missing dispersion scores a neutral 0.5 rather than 1.0 — an unmeasured
    quantity is not a reassuring one.
    """
    if p_std is None:
        return 0.5
    if p_std <= 0:
        return 1.0
    return max(0.0, 1.0 - (p_std / max_p_std))


def market_quality(
    book_margin: float | None,
    n_books: int | None,
    max_margin: float = DEFAULT_MAX_MARGIN,
) -> float:
    """Decreasing in margin, increasing in the number of quoting books."""
    if book_margin is None and n_books is None:
        return 0.5
    margin_score = (
        0.5 if book_margin is None else max(0.0, 1.0 - book_margin / max_margin)
    )
    depth_score = (
        0.5 if n_books is None else min(1.0, n_books / BOOK_COUNT_SATURATION)
    )
    return 0.5 * margin_score + 0.5 * depth_score


def calibration_confidence(calibration_error: float | None, scale: float = 0.05) -> float:
    """Inverted calibration error for this competition and decile (FR-16b)."""
    if calibration_error is None:
        return 0.5
    return max(0.0, 1.0 - abs(calibration_error) / scale)


def support(historical_clv: float | None, scale: float = 0.02) -> float:
    """Realised CLV of similar bets, mapped onto [0, 1].

    Zero CLV maps to 0.5 rather than 0: no evidence either way is neutral, not
    damning. This is the sub-score carrying the only signal this project has
    measured as significant.
    """
    if historical_clv is None:
        return 0.5
    return min(1.0, max(0.0, 0.5 + historical_clv / (2.0 * scale)))


# --- guardrails (FR-33) -------------------------------------------------------

def guardrail_reasons(
    assessment: ValueAssessment,
    context: GradingContext,
    *,
    max_p_std: float = DEFAULT_MAX_P_STD,
    max_margin: float = DEFAULT_MAX_MARGIN,
    odds_ceiling: float = DEFAULT_ODDS_CEILING,
    other_competition_hours: float = 72.0,
) -> list[str]:
    """Every guardrail that fires, each as a logged reason (FR-33)."""
    reasons: list[str] = []

    if not context.odds_coverage:
        reasons.append(
            "no free odds coverage for this competition, so no benchmark exists"
        )
    if not context.lineup_confirmed and context.key_player_doubtful:
        reasons.append("lineup unconfirmed with a key player doubtful")
    if context.dead_rubber:
        reasons.append("dead rubber: the result does not affect either side")
    if (
        context.hours_to_other_competition_fixture is not None
        and context.hours_to_other_competition_fixture < other_competition_hours
    ):
        reasons.append(
            f"fixture in another competition within "
            f"{context.hours_to_other_competition_fixture:.0f}h — rotation risk"
        )
    if context.p_std is not None and context.p_std > max_p_std:
        reasons.append(f"ensemble dispersion {context.p_std:.3f} above {max_p_std:.3f}")
    if context.book_margin is not None and context.book_margin > max_margin:
        reasons.append(f"book margin {context.book_margin:.3f} above {max_margin:.3f}")
    if assessment.o_avail > odds_ceiling:
        reasons.append(f"price {assessment.o_avail:.2f} above the odds ceiling {odds_ceiling}")

    return reasons


# --- grading ------------------------------------------------------------------

def grade(
    assessment: ValueAssessment,
    context: GradingContext | None = None,
    *,
    e_peak: float = DEFAULT_E_PEAK,
    sigma: float = DEFAULT_SIGMA,
    e_ceiling: float = DEFAULT_E_CEILING,
    cutoffs: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
    **guardrail_kwargs,
) -> GradedBet:
    """Grade one assessed selection A-F."""
    context = context or GradingContext()
    cutoffs = cutoffs or DEFAULT_CUTOFFS
    weights = weights or DEFAULT_WEIGHTS

    reasons = guardrail_reasons(
        assessment, context, max_p_std=guardrail_kwargs.pop("max_p_std", DEFAULT_MAX_P_STD),
        max_margin=guardrail_kwargs.pop("max_margin", DEFAULT_MAX_MARGIN),
        odds_ceiling=guardrail_kwargs.pop("odds_ceiling", DEFAULT_ODDS_CEILING),
        **guardrail_kwargs,
    )
    if reasons:
        # A guardrail says the estimate is untrustworthy for structural reasons.
        # No composite score is allowed to outvote that.
        return GradedBet(assessment.key, Grade.F, 0.0, None, tuple(reasons))

    if assessment.edge_prob > e_ceiling:
        return GradedBet(
            assessment.key, Grade.F, 0.0, None,
            (
                f"apparent edge {assessment.edge_prob:.3f} above the ceiling "
                f"{e_ceiling:.3f} — model likely blind to something; routed to review",
            ),
            model_likely_blind=True,
        )

    if assessment.expected_value <= 0:
        return GradedBet(
            assessment.key, Grade.F, 0.0, None,
            ("expected value is not positive at the available price",),
        )

    sub = SubScores(
        c_edge=edge_confidence(assessment.edge_prob, e_peak, sigma, e_ceiling),
        c_robust=robustness(context.p_std),
        c_market=market_quality(context.book_margin, context.n_books),
        c_calib=calibration_confidence(context.calibration_error),
        c_support=support(context.historical_clv),
    )
    composite = sub.composite(weights)

    for letter in ("A", "B", "C", "D"):
        if composite >= cutoffs[letter]:
            return GradedBet(assessment.key, Grade(letter), composite, sub)
    return GradedBet(
        assessment.key, Grade.F, composite, sub,
        (f"composite {composite:.3f} below the D cutoff {cutoffs['D']:.2f}",),
    )


@dataclass
class ReviewQueue:
    """F-graded large-edge bets, kept because they are diagnostic.

    Each entry is a concrete instance of information the feature set is missing —
    the most valuable by-product the grading step produces.
    """

    entries: list[GradedBet] = field(default_factory=list)

    def add(self, bet: GradedBet) -> None:
        if bet.model_likely_blind:
            self.entries.append(bet)

    def __len__(self) -> int:
        return len(self.entries)


def grade_book(
    assessments: list[ValueAssessment],
    context: GradingContext | None = None,
    **kwargs,
) -> tuple[list[GradedBet], ReviewQueue]:
    queue = ReviewQueue()
    graded = []
    for assessment in assessments:
        bet = grade(assessment, context, **kwargs)
        queue.add(bet)
        graded.append(bet)
    return graded, queue
