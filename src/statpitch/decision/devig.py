"""De-vigging: recovering fair probabilities from quoted odds (FR-28, Design §6.2).

A bookmaker's quoted prices imply probabilities summing to more than one. The
excess is the margin, and removing it is the first step in every downstream
number: edge, expected value, CLV, calibration, the lot.

Why the method choice is load-bearing rather than a detail
==========================================================

Bookmakers do not spread margin evenly. They load it onto longshots — the
favourite-longshot bias. Proportional de-vigging assumes an even spread, so it
systematically **overstates** the fair probability of longshots. A model compared
against those inflated probabilities will appear to find value on draws and away
underdogs where none exists, which is precisely where losses accumulate.

Getting this wrong does not produce a slightly worse system. It produces one
whose value flags point at exactly the bets that lose money.

Three methods, all implemented, one selected empirically per competition:

* **Proportional** — divide through by the overround. Assumes uniform margin.
* **Power** — solve for `k` with `sum(p_i^k) = 1`. Compresses longshots more than
  favourites, so it bends in the direction the bias implies.
* **Shin** — models the margin as arising from a proportion `z` of insider money
  and redistributes it accordingly. The standard treatment in the literature.

`select_method` scores them against realised outcomes; nothing here assumes a
winner.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq

log = logging.getLogger(__name__)

Method = Literal["proportional", "power", "shin"]
METHODS: tuple[Method, ...] = ("proportional", "power", "shin")

#: Below this overround the prices are treated as already fair. Values at or
#: under 1.0 mean an arbitrage (or stale quotes) rather than a margin, and the
#: root-finds have no solution there.
MIN_OVERROUND = 1.0 + 1e-9


class DevigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DevigResult:
    probabilities: np.ndarray
    overround: float
    method: Method
    #: Fitted exponent (power) or insider share (Shin); None for proportional.
    parameter: float | None = None

    @property
    def margin(self) -> float:
        """Book margin as a fraction, e.g. 0.05 for a 5% book."""
        return self.overround - 1.0


def implied(odds: Sequence[float] | np.ndarray) -> np.ndarray:
    """Raw implied probabilities, 1/o, before any margin removal."""
    values = np.asarray(odds, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise DevigError("de-vigging needs at least two selections from one market")
    if not np.all(np.isfinite(values)):
        raise DevigError(f"non-finite odds: {values}")
    if np.any(values <= 1.0):
        raise DevigError(
            f"odds must exceed 1.0 (a price never returns less than the stake): {values}"
        )
    return 1.0 / values


def overround(odds: Sequence[float] | np.ndarray) -> float:
    return float(np.sum(implied(odds)))


# --- the three methods --------------------------------------------------------

def proportional(odds: Sequence[float] | np.ndarray) -> DevigResult:
    """Divide through by the overround.

    Assumes margin is spread evenly across outcomes, which is empirically false —
    see the module docstring. Retained because FR-28 requires all three be
    implemented and compared, not because it is expected to win.
    """
    p = implied(odds)
    s = float(np.sum(p))
    return DevigResult(p / s, s, "proportional")


def power(odds: Sequence[float] | np.ndarray) -> DevigResult:
    """Solve for `k` such that `sum(p_i ** k) == 1`.

    Since every `p_i < 1`, raising to a power above one shrinks small
    probabilities proportionally more than large ones — bending in the direction
    the favourite-longshot bias implies.
    """
    p = implied(odds)
    s = float(np.sum(p))
    if s <= MIN_OVERROUND:
        return DevigResult(p / s, s, "power", 1.0)

    def excess(k: float) -> float:
        return float(np.sum(np.power(p, k)) - 1.0)

    # excess(1) = S - 1 > 0, and grows negative as k rises since each p_i < 1.
    hi = 2.0
    while excess(hi) > 0 and hi < 64.0:
        hi *= 2.0
    if excess(hi) > 0:
        raise DevigError(f"power de-vig failed to bracket a root for odds {np.asarray(odds)}")

    k = float(brentq(excess, 1.0, hi, xtol=1e-12, rtol=1e-12))
    return DevigResult(np.power(p, k), s, "power", k)


def shin(odds: Sequence[float] | np.ndarray) -> DevigResult:
    """Shin's method: margin as the book's protection against insider money.

    Solves for the insider share `z` making the recovered probabilities sum to
    one, with

        q_i = [ sqrt(z^2 + 4(1-z) * p_i^2 / S) - z ] / (2(1-z))

    Unlike proportional de-vigging this removes proportionally more margin from
    longshots, which is the direction the favourite-longshot bias actually runs.
    """
    p = implied(odds)
    s = float(np.sum(p))
    if s <= MIN_OVERROUND:
        return DevigResult(p / s, s, "shin", 0.0)

    def recovered(z: float) -> np.ndarray:
        inner = np.sqrt(z * z + 4.0 * (1.0 - z) * (p * p) / s)
        return (inner - z) / (2.0 * (1.0 - z))

    def excess(z: float) -> float:
        return float(np.sum(recovered(z)) - 1.0)

    lo, hi = 0.0, 1.0 - 1e-9
    if excess(lo) <= 0:
        # Already fair at z=0; nothing for the insider term to explain.
        return DevigResult(recovered(0.0), s, "shin", 0.0)
    if excess(hi) > 0:
        log.debug("shin de-vig could not bracket a root; falling back to proportional")
        return DevigResult(p / s, s, "shin", None)

    z = float(brentq(excess, lo, hi, xtol=1e-12, rtol=1e-12))
    q = recovered(z)
    # Guard against round-off leaving the vector fractionally off one.
    return DevigResult(q / float(np.sum(q)), s, "shin", z)


_DISPATCH = {"proportional": proportional, "power": power, "shin": shin}


def devig(odds: Sequence[float] | np.ndarray, method: Method = "shin") -> DevigResult:
    """De-vig one market's odds with the named method."""
    try:
        fn = _DISPATCH[method]
    except KeyError:
        raise DevigError(f"unknown de-vig method {method!r}; known: {METHODS}") from None
    return fn(odds)


def devig_many(odds_matrix: np.ndarray, method: Method = "shin") -> np.ndarray:
    """De-vig many markets at once. Rows are markets, columns are selections."""
    matrix = np.asarray(odds_matrix, dtype=float)
    if matrix.ndim != 2:
        raise DevigError("odds_matrix must be two-dimensional (markets x selections)")
    return np.vstack([devig(row, method).probabilities for row in matrix])
