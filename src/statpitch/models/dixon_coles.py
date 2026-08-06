"""Dixon-Coles score matrix (Design §5.1, §6.1).

Independent Poisson goal models systematically misprice low-scoring results —
0-0, 1-0, 0-1 and 1-1 occur more often than independence implies, because those
are exactly the scorelines where the two teams' goal counts are least independent.
Dixon-Coles corrects the four affected cells with a single parameter `rho`.

Why the accuracy of this matrix matters more than it looks
==========================================================

Design §6.1 makes this matrix the **single source of truth for every market**:
1X2, Over/Under at every line, Asian Handicap at every line, both-teams-to-score,
team totals and correct score are all summations over its cells. An error here
does not degrade one market — it propagates to all of them at once, and shows up
as a systematic edge against the book in whichever market the error happens to
favour.

So the matrix carries two invariants, both asserted in tests: it sums to one after
truncation and renormalisation, and its implied 1X2 agrees with a directly fitted
classifier. Design §5.1 and the Phase 3 acceptance criteria both call for that
second check before the matrix is trusted downstream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

log = logging.getLogger(__name__)

#: Truncation point, the value deferred from Design §11. At a heavy 2.5 v 2.0
#: goal rate this discards 7e-5 of the distribution, which renormalisation then
#: redistributes proportionally; at typical rates the discarded mass is far
#: smaller still.
DEFAULT_MAX_GOALS = 10

#: Upper bound on rho, which is **not a constant**: the 0-0 correction is
#: `1 - lambda_home * lambda_away * rho`, so rho must stay below
#: `1 / (lambda_home * lambda_away)` or that cell goes negative. At 1.5 v 1.2 the
#: ceiling is 0.56, but at 3.0 v 2.5 it is 0.13 — so a fixed +/-0.2 band is safe
#: for ordinary fixtures and unsafe for high-scoring ones. `rho_bounds()` derives
#: it from the rates instead.
MAX_ABS_RHO = 0.2

#: Keep a margin below the exact ceiling so the corrected cell stays strictly
#: positive rather than landing on zero.
RHO_SAFETY = 0.95


class DixonColesError(ValueError):
    pass


def rho_bounds(lambda_home: float, lambda_away: float) -> tuple[float, float]:
    """The rho range that keeps every corrected cell positive at these rates.

    Three cells constrain it:
      0-0:  1 - lambda_home * lambda_away * rho > 0  ->  rho <  1 / (lh * la)
      1-1:  1 - rho                            > 0  ->  rho <  1
      0-1:  1 + lambda_home * rho              > 0  ->  rho > -1 / lambda_home
      1-0:  1 + lambda_away * rho              > 0  ->  rho > -1 / lambda_away

    The binding upper constraint is the 0-0 cell, and it tightens as goal rates
    rise — which is why a fixed band is wrong for high-scoring fixtures.
    """
    upper = min(1.0 / (lambda_home * lambda_away), 1.0) * RHO_SAFETY
    lower = -min(1.0 / lambda_home, 1.0 / lambda_away) * RHO_SAFETY
    return (max(lower, -MAX_ABS_RHO), min(upper, MAX_ABS_RHO))


def tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> np.ndarray:
    """The Dixon-Coles correction, applied to the four low-score cells only.

    Every other cell is left at 1.0, so the correction is local: it redistributes
    probability among 0-0, 0-1, 1-0 and 1-1 without touching the rest of the grid.
    """
    correction = np.ones_like(home_goals, dtype=float)

    correction[(home_goals == 0) & (away_goals == 0)] = (
        1.0 - lambda_home * lambda_away * rho
    )
    correction[(home_goals == 0) & (away_goals == 1)] = 1.0 + lambda_home * rho
    correction[(home_goals == 1) & (away_goals == 0)] = 1.0 + lambda_away * rho
    correction[(home_goals == 1) & (away_goals == 1)] = 1.0 - rho
    return correction


@dataclass(frozen=True, slots=True)
class ScoreMatrix:
    """P[i][j] = probability of exactly i home goals and j away goals."""

    matrix: np.ndarray
    lambda_home: float
    lambda_away: float
    rho: float

    @property
    def max_goals(self) -> int:
        return self.matrix.shape[0] - 1

    # --- market derivations (Design §6.1) --------------------------------

    def home_win(self) -> float:
        return float(np.tril(self.matrix, -1).sum())

    def draw(self) -> float:
        return float(np.trace(self.matrix))

    def away_win(self) -> float:
        return float(np.triu(self.matrix, 1).sum())

    def one_x_two(self) -> tuple[float, float, float]:
        return self.home_win(), self.draw(), self.away_win()

    def over(self, line: float) -> float:
        """P(total goals > line). A whole-number line leaves a push, so the
        complement of `over` is not `under` there — both are returned separately."""
        totals = np.add.outer(
            np.arange(self.matrix.shape[0]), np.arange(self.matrix.shape[1])
        )
        return float(self.matrix[totals > line].sum())

    def under(self, line: float) -> float:
        totals = np.add.outer(
            np.arange(self.matrix.shape[0]), np.arange(self.matrix.shape[1])
        )
        return float(self.matrix[totals < line].sum())

    def both_teams_to_score(self) -> float:
        return float(self.matrix[1:, 1:].sum())

    def correct_score(self, home: int, away: int) -> float:
        return float(self.matrix[home, away])

    def top_scores(self, n: int = 10) -> list[tuple[int, int, float]]:
        flat = np.dstack(np.unravel_index(np.argsort(-self.matrix, axis=None), self.matrix.shape))
        return [
            (int(i), int(j), float(self.matrix[i, j]))
            for i, j in flat[0][:n]
        ]

    def expected_goals(self) -> tuple[float, float]:
        """Expected goals implied by the matrix, after correction and truncation.

        Not identical to the input lambdas: the tau correction moves probability
        between cells and truncation clips the tail, so the realised expectation
        drifts slightly. Reporting the matrix's own expectation keeps FR-5
        consistent with the markets derived from the same grid.
        """
        rows = np.arange(self.matrix.shape[0])
        cols = np.arange(self.matrix.shape[1])
        return (
            float((self.matrix.sum(axis=1) * rows).sum()),
            float((self.matrix.sum(axis=0) * cols).sum()),
        )


def score_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> ScoreMatrix:
    """Build the tau-corrected, truncated, renormalised score matrix."""
    if lambda_home <= 0 or lambda_away <= 0:
        raise DixonColesError(
            f"goal rates must be positive, got {lambda_home=}, {lambda_away=}"
        )
    if not np.isfinite(lambda_home) or not np.isfinite(lambda_away):
        raise DixonColesError("goal rates must be finite")
    if max_goals < 1:
        raise DixonColesError("max_goals must be at least 1")

    goals = np.arange(max_goals + 1)
    home_probabilities = poisson.pmf(goals, lambda_home)
    away_probabilities = poisson.pmf(goals, lambda_away)
    grid = np.outer(home_probabilities, away_probabilities)

    home_index, away_index = np.meshgrid(goals, goals, indexing="ij")
    grid = grid * tau(home_index, away_index, lambda_home, lambda_away, rho)

    if np.any(grid < 0):
        low, high = rho_bounds(lambda_home, lambda_away)
        raise DixonColesError(
            f"rho={rho} makes a cell negative at lambdas "
            f"({lambda_home}, {lambda_away}); valid range here is "
            f"[{low:.4f}, {high:.4f}] — the ceiling is 1/(lh*la) and tightens as "
            "goal rates rise"
        )

    total = grid.sum()
    if total <= 0:
        raise DixonColesError("degenerate score matrix")
    # Renormalise: truncation discards the tail beyond max_goals, and the tau
    # correction is not probability-preserving on its own.
    return ScoreMatrix(grid / total, lambda_home, lambda_away, rho)


# --- fitting rho ---------------------------------------------------------------

def log_likelihood(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: float,
) -> float:
    """Dixon-Coles log-likelihood of observed scores given per-match rates."""
    independent = poisson.logpmf(home_goals, lambda_home) + poisson.logpmf(
        away_goals, lambda_away
    )
    # Built inline rather than via tau(): here the rates vary per match, so each
    # corrected cell needs its own lambda rather than the two scalars tau takes.
    correction = np.ones_like(independent, dtype=float)

    zero_zero = (home_goals == 0) & (away_goals == 0)
    zero_one = (home_goals == 0) & (away_goals == 1)
    one_zero = (home_goals == 1) & (away_goals == 0)
    one_one = (home_goals == 1) & (away_goals == 1)

    correction[zero_zero] = 1.0 - lambda_home[zero_zero] * lambda_away[zero_zero] * rho
    correction[zero_one] = 1.0 + lambda_home[zero_one] * rho
    correction[one_zero] = 1.0 + lambda_away[one_zero] * rho
    correction[one_one] = 1.0 - rho

    if np.any(correction <= 0):
        return -np.inf
    return float(np.sum(independent + np.log(correction)))


def fit_rho(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
) -> float:
    """Maximum-likelihood rho for one competition, given fitted goal rates.

    Fitted per competition because the low-score dependence is a property of how
    a league plays: a division where sides shut up shop at 1-0 shows a different
    rho from one that keeps attacking.
    """
    home_goals = np.asarray(home_goals)
    away_goals = np.asarray(away_goals)
    lambda_home = np.asarray(lambda_home, dtype=float)
    lambda_away = np.asarray(lambda_away, dtype=float)

    if len(home_goals) == 0:
        raise DixonColesError("cannot fit rho without matches")

    # Bound by the most constraining fixture in the sample, not by a constant:
    # one high-scoring match is enough to make a nominally reasonable rho produce
    # a negative 0-0 cell.
    bounds = rho_bounds(float(lambda_home.max()), float(lambda_away.max()))
    result = minimize_scalar(
        lambda r: -log_likelihood(home_goals, away_goals, lambda_home, lambda_away, r),
        bounds=bounds,
        method="bounded",
    )
    if not result.success:
        log.warning("rho fit did not converge; falling back to 0.0 (independent Poisson)")
        return 0.0
    return float(result.x)


def low_score_lift(rho: float, lambda_home: float, lambda_away: float) -> dict[str, float]:
    """How much each corrected cell moves relative to independent Poisson.

    Useful as a diagnostic: a negative rho should raise 0-0 and 1-1 and lower the
    1-0 and 0-1 cells, which is the empirical pattern Dixon-Coles was built for.
    """
    plain = score_matrix(lambda_home, lambda_away, rho=0.0)
    corrected = score_matrix(lambda_home, lambda_away, rho=rho)
    return {
        "0-0": corrected.matrix[0, 0] / plain.matrix[0, 0] - 1.0,
        "0-1": corrected.matrix[0, 1] / plain.matrix[0, 1] - 1.0,
        "1-0": corrected.matrix[1, 0] / plain.matrix[1, 0] - 1.0,
        "1-1": corrected.matrix[1, 1] / plain.matrix[1, 1] - 1.0,
    }
