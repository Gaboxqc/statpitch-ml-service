"""Risk-managed Kelly staking (FR-27, FR-24, Design §6.5).

Four things happen here, in order: the model's probability is shrunk toward the
market, a Kelly fraction is solved over the selection's full payoff distribution,
the result is scaled by grade and by lambda and capped, and — for a slate of
simultaneous bets — the whole allocation is solved jointly rather than one bet at
a time.

Log-growth is the right objective for SIZING. It is not what this project
measured as the right criterion for SELECTING
=============================================

Design §6.5 is correct that a compounding bankroll should be sized by expected
logarithmic growth: full Kelly on an estimated probability is a bankruptcy
machine, and log-growth is what fractional Kelly is a fraction *of*. Nothing
below disputes that.

FR-24 goes further and argues log-growth should also *rank* candidate selections,
on the grounds that "EV always crowns the longshot". That part did not survive
measurement here. Ranking a fixture's markets by log-growth returned -3.27%
against -2.12% for ranking by EV, and both lost to committing to one market in
advance (+0.13%). The reason is that at capped quarter-Kelly stakes growth is
nearly monotone in EV, so the ranking criterion is second-order — while both
rank on model-versus-market disagreement, which is what actually does the damage.

So `rank_by_growth` exists and is correct, and `best_bet` is available because
FR-24 asks for it. Neither is recommended as a selection rule on this evidence.

Nothing stakes from an unfitted config
======================================

`w` is unfitted in the shipped configuration, and a stake sized from placeholder
parameters is indistinguishable from a real one. Every entry point that produces
a stake calls `require_fitted()` first.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from statpitch.decision.bet_grader import GradedBet
from statpitch.decision.market_engine import Payoff
from statpitch.decision.value import ValueAssessment, _rescale

log = logging.getLogger(__name__)

#: Kelly is solved on [0, this]. A fraction above one is leverage, which the
#: bankroll model does not represent.
MAX_KELLY = 1.0

#: Below this the stake is not worth placing, and rounding noise dominates.
MIN_STAKE = 5e-4

#: Hard ceiling on total slate exposure, independent of the matchday cap. A slate
#: staking the whole bankroll is one correlated loss from ruin, and the log
#: objective is undefined at zero wealth.
MAX_TOTAL_EXPOSURE = 0.95


class StakingError(RuntimeError):
    pass


# --- log growth ---------------------------------------------------------------

def log_growth(payoff: Payoff, odds: float, fraction: float) -> float:
    """Expected log bankroll multiplier for staking `fraction` at `odds`.

    Computed over the full outcome distribution rather than the two-outcome form,
    which is what makes pushes and quarter lines correct: a half-loss costs half
    the stake, and a push costs nothing. The two-outcome formula has nowhere to
    represent either, so it would misprice every Draw No Bet, whole-line total and
    quarter-line handicap in the book.
    """
    if fraction <= 0:
        return 0.0
    total = 0.0
    for probability, payout in payoff.outcomes(odds):
        if probability <= 0:
            continue
        wealth = 1.0 + fraction * payout
        if wealth <= 0:
            return -math.inf
        total += probability * math.log(wealth)
    return total


def kelly_fraction(payoff: Payoff, odds: float) -> float:
    """Growth-maximising stake fraction, solved numerically.

    A closed form exists for a two-outcome bet, but not once pushes and half
    outcomes enter, so the general case is solved directly. Returns zero when no
    positive fraction improves on not betting.
    """
    if payoff.expected_return(odds) <= 0:
        return 0.0

    result = minimize_scalar(
        lambda f: -log_growth(payoff, odds, f),
        bounds=(0.0, MAX_KELLY),
        method="bounded",
    )
    if not result.success:
        return 0.0
    return float(max(0.0, result.x))


# --- shrinkage ----------------------------------------------------------------

def shrink(p_model: float, q_fair: float, w: float) -> float:
    """p_used = w * p_model + (1 - w) * q_fair (Design §6.5).

    `w` is fitted, never assumed. In this project it fitted at zero, which means
    `p_used` collapses to the market's own probability and every Kelly fraction
    below is computed against the consensus rather than against the model.
    """
    if not 0.0 <= w <= 1.0:
        raise StakingError(f"w must lie in [0, 1], got {w}")
    return w * p_model + (1.0 - w) * q_fair


# --- single-bet staking -------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Stake:
    key: str
    fraction: float
    kelly: float
    growth: float
    p_used: float
    odds: float
    grade: str

    @property
    def is_placed(self) -> bool:
        return self.fraction >= MIN_STAKE


def stake_for(
    assessment: ValueAssessment,
    graded: GradedBet,
    *,
    w: float,
    kelly_lambda: float = 0.25,
    cap_per_bet: float = 0.02,
    odds_ceiling: float = 8.0,
    grade_multipliers: dict[str, float] | None = None,
) -> Stake:
    """Size one bet: shrink, solve Kelly, scale by grade and lambda, cap."""
    multiplier = graded.stake_multiplier(grade_multipliers)
    p_used = shrink(assessment.p_model, assessment.q_fair, w)

    if not graded.is_stakeable or multiplier <= 0 or assessment.o_avail > odds_ceiling:
        return Stake(assessment.key, 0.0, 0.0, 0.0, p_used, assessment.o_avail,
                     graded.grade.value)

    payoff = _rescale(assessment.payoff, p_used)
    kelly = kelly_fraction(payoff, assessment.o_avail)
    fraction = min(kelly * kelly_lambda * multiplier, cap_per_bet)
    if fraction < MIN_STAKE:
        fraction = 0.0

    return Stake(
        key=assessment.key,
        fraction=fraction,
        kelly=kelly,
        growth=log_growth(payoff, assessment.o_avail, fraction),
        p_used=p_used,
        odds=assessment.o_avail,
        grade=graded.grade.value,
    )


# --- ranking (FR-24) ----------------------------------------------------------

def rank_by_growth(stakes: list[Stake]) -> list[Stake]:
    """Order selections by log-growth at their Kelly-optimal stake (FR-24).

    Correct as an ordering. Not recommended as a selection rule — see the module
    docstring: ranking a fixture's markets this way measured worse than ranking by
    EV, and both worse than not selecting at all.
    """
    return sorted(stakes, key=lambda s: s.growth, reverse=True)


def best_bet(stakes: list[Stake]) -> Stake | None:
    ranked = [s for s in rank_by_growth(stakes) if s.is_placed]
    return ranked[0] if ranked else None


# --- caps ---------------------------------------------------------------------

def apply_matchday_cap(stakes: list[Stake], cap: float = 0.10) -> list[Stake]:
    """Scale a slate down proportionally if its total exposure exceeds `cap`."""
    total = sum(s.fraction for s in stakes)
    if total <= cap or total <= 0:
        return stakes
    scale = cap / total
    log.info("staking: slate exposure %.4f exceeds cap %.4f, scaling by %.3f",
             total, cap, scale)
    return [
        Stake(s.key, s.fraction * scale, s.kelly, s.growth, s.p_used, s.odds, s.grade)
        for s in stakes
    ]


# --- simultaneous allocation (Design §6.5 step 4) -----------------------------

@dataclass
class SlateBet:
    """One candidate in a simultaneous allocation."""

    key: str
    odds: float
    payoff: Payoff
    p_used: float
    fixture_id: str
    max_fraction: float = 0.02


def simulate_returns(
    bets: list[SlateBet], draws: int = 4000, seed: int = 0
) -> np.ndarray:
    """Simulated per-bet returns, correlated within a fixture.

    Bets on the same fixture are settled from ONE draw, so "home win" and "over
    2.5" move together for a strong favourite exactly as they do in reality.
    Fixtures are drawn independently of each other.

    That correlation is the entire point of allocating jointly: sequential
    single-bet Kelly treats a slate of correlated bets as if they were
    independent and over-stakes badly.
    """
    rng = np.random.default_rng(seed)
    fixtures = sorted({b.fixture_id for b in bets})
    uniforms = {f: rng.random(draws) for f in fixtures}

    out = np.zeros((draws, len(bets)))
    for j, bet in enumerate(bets):
        payoff = _rescale(bet.payoff, bet.p_used)
        outcomes = payoff.outcomes(bet.odds)
        edges = np.cumsum([p for p, _ in outcomes])
        payouts = np.array([r for _, r in outcomes])
        # One shared uniform per fixture: every bet on that fixture reads the
        # same draw, which is what couples them.
        index = np.searchsorted(edges, uniforms[bet.fixture_id], side="left")
        out[:, j] = payouts[np.clip(index, 0, len(payouts) - 1)]
    return out


def allocate_slate(
    bets: list[SlateBet],
    *,
    cap_per_bet: float = 0.02,
    cap_matchday: float = 0.10,
    draws: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """Solve max E[ln(1 + sum f_k r_k)] subject to the exposure caps.

    The joint solve is what stops a correlated slate being over-staked. Sequential
    Kelly sizes each bet as though it were the only one, so a matchday holding
    several bets on the same favourite ends up with far more exposure to that one
    outcome than any single Kelly fraction implies.
    """
    if not bets:
        return {}

    returns = simulate_returns(bets, draws=draws, seed=seed)

    def negative_growth(fractions: np.ndarray) -> float:
        # Wealth is clipped rather than rejected. Returning a flat penalty for
        # infeasible points gives the optimiser a zero gradient, so SLSQP cannot
        # tell which direction improves and simply stays at its starting vector —
        # which produced identical allocations for correlated and independent
        # slates, silently defeating the entire purpose of the joint solve.
        wealth = np.clip(1.0 + returns @ fractions, 1e-6, None)
        return -float(np.mean(np.log(wealth)))

    n = len(bets)
    # Total exposure is bounded below 1 as well as by the matchday cap: a slate
    # staking the whole bankroll can be wiped out by one correlated loss, and the
    # log objective is undefined there.
    total_cap = min(cap_matchday, MAX_TOTAL_EXPOSURE)
    start = np.full(n, min(cap_per_bet, total_cap / max(n, 1)) * 0.25)
    bounds = [(0.0, min(b.max_fraction, cap_per_bet)) for b in bets]
    constraints = [{"type": "ineq", "fun": lambda f: total_cap - float(np.sum(f))}]

    result = minimize(
        negative_growth, start, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-10},
    )
    fractions = np.clip(result.x, 0.0, None) if result.success else start * 0.0
    return {
        bet.key: float(f if f >= MIN_STAKE else 0.0)
        for bet, f in zip(bets, fractions, strict=True)
    }


# --- lambda frontier ----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FrontierPoint:
    kelly_lambda: float
    median_growth: float
    max_drawdown: float
    ruin_probability: float


def lambda_frontier(
    returns: np.ndarray,
    fractions: np.ndarray,
    lambdas: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00),
    *,
    ruin_threshold: float = -0.5,
) -> list[FrontierPoint]:
    """Growth against drawdown across Kelly fractions (Design §6.5 step 3).

    Publishing the curve is more informative than asserting a single lambda:
    quarter Kelly captures most of the long-run growth for a fraction of the
    drawdown, and that trade-off is only visible as a curve.
    """
    out = []
    for lam in lambdas:
        scaled = fractions * lam
        path_returns = returns @ scaled
        wealth = np.cumprod(1.0 + path_returns)
        peak = np.maximum.accumulate(wealth)
        drawdown = float(np.min(wealth / peak - 1.0)) if len(wealth) else 0.0
        out.append(
            FrontierPoint(
                kelly_lambda=lam,
                median_growth=float(np.median(np.log(np.clip(1 + path_returns, 1e-9, None)))),
                max_drawdown=drawdown,
                ruin_probability=float(np.mean(wealth / peak - 1.0 <= ruin_threshold)),
            )
        )
    return out


# --- the gate -----------------------------------------------------------------

@dataclass
class StakingEngine:
    """Config-bound staking, gated on the parameters actually being fitted."""

    config: object
    _w: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.config.require_fitted("size stakes")
        self._w = float(self.config.w or 0.0)

    def stake(self, assessment: ValueAssessment, graded: GradedBet) -> Stake:
        staking = self.config.staking
        return stake_for(
            assessment, graded,
            w=self._w,
            kelly_lambda=staking.kelly_lambda,
            cap_per_bet=staking.cap_per_bet,
            odds_ceiling=self.config.guardrails.odds_ceiling,
            grade_multipliers=self.config.grading.stake_multiplier,
        )
