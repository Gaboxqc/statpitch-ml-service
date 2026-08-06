"""Market-shrinkage weight `w` — the project's truth serum (Design §6.5, FR-27).

    p_used = w * p_model + (1 - w) * q_fair

`w` answers one question: how much information does the model add over the
market? It is **fitted, never assumed**. Requirements §8.4 makes reporting it
mandatory, and Requirements §9 names it the project's truth serum precisely
because a value near zero is a valid, publishable finding — it means the model
adds nothing over the closing line and the Decision Layer should be scoped
accordingly.

Two criteria, reported together
===============================

* **log-loss optimal** — the `w` giving the best-calibrated blend. A strictly
  proper scoring rule, minimised only by the true probabilities.
* **log-growth optimal** — the `w` maximising realised bankroll growth under
  Kelly staking, which is what Design §6.5 specifies and what actually matters
  for a compounding bankroll.

They answer different questions and can disagree: a blend can be better
calibrated on average while being worse at the specific fixtures where a bet gets
placed. Reporting only the flattering one would be exactly the kind of
self-deception this parameter exists to prevent.

Every fit is returned with a bootstrap interval. A `w` of 0.15 whose interval
spans zero is not evidence of edge, and must not be read as such.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

#: Search grid for w. Fine enough to locate the optimum, coarse enough that the
#: bootstrap stays cheap.
DEFAULT_GRID = np.round(np.linspace(0.0, 1.0, 101), 3)

EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class ShrinkageFit:
    w: float
    criterion: str
    score: float
    ci_low: float
    ci_high: float
    n_matches: int
    #: Score at w=0, i.e. the market alone — the number `w` must beat to mean
    #: anything at all.
    market_only_score: float
    #: Score at w=1, i.e. the model alone.
    model_only_score: float

    @property
    def interval_excludes_zero(self) -> bool:
        return self.ci_low > 0.0

    @property
    def beats_market_alone(self) -> bool:
        """Whether blending improves on the market by itself."""
        if self.criterion == "log_loss":
            return self.score < self.market_only_score
        return self.score > self.market_only_score

    def verdict(self) -> str:
        """Requirements §9's reading of the fitted value."""
        if not self.interval_excludes_zero:
            return (
                f"w={self.w:.3f} but its 95% interval [{self.ci_low:.3f}, "
                f"{self.ci_high:.3f}] includes zero — no demonstrated information "
                "over the market"
            )
        if self.w < 0.1:
            return f"w={self.w:.3f} — the model adds almost nothing over the market"
        if self.w <= 0.3:
            return f"w={self.w:.3f} — modest but real information; shrink heavily"
        return (
            f"w={self.w:.3f} — substantial; audit for leakage before believing it, "
            "since a high w against closing lines is more often a bug than a discovery"
        )


def blend(p_model: np.ndarray, q_fair: np.ndarray, w: float) -> np.ndarray:
    """The shrinkage blend, renormalised so it stays a distribution."""
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"w must lie in [0, 1], got {w}")
    mixed = w * np.asarray(p_model, dtype=float) + (1.0 - w) * np.asarray(q_fair, dtype=float)
    mixed = np.clip(mixed, EPSILON, 1.0)
    return mixed / mixed.sum(axis=1, keepdims=True)


def log_loss_at(p_model: np.ndarray, q_fair: np.ndarray, outcomes: np.ndarray, w: float) -> float:
    mixed = blend(p_model, q_fair, w)
    return float(-np.mean(np.sum(outcomes * np.log(mixed), axis=1)))


def log_growth_at(
    p_model: np.ndarray,
    q_fair: np.ndarray,
    outcomes: np.ndarray,
    odds: np.ndarray,
    w: float,
    *,
    kelly_fraction: float = 0.25,
    cap: float = 0.02,
) -> float:
    """Mean realised log bankroll growth from staking on the blended probability.

    Stakes are sized by fractional Kelly on `p_used`, capped, and settled at the
    obtainable price. Selections with no positive Kelly fraction are skipped, so a
    blend that finds no bets scores zero rather than being penalised.
    """
    mixed = blend(p_model, q_fair, w)
    odds = np.asarray(odds, dtype=float)

    edge = mixed * odds - 1.0
    kelly = np.divide(
        edge, odds - 1.0, out=np.zeros_like(edge), where=(odds > 1.0)
    )
    stake = np.clip(kelly * kelly_fraction, 0.0, cap)
    stake[edge <= 0] = 0.0

    # One row can hold several positive-edge selections; total exposure per match
    # is capped so a single fixture cannot compound beyond the bankroll.
    total_stake = stake.sum(axis=1, keepdims=True)
    scale = np.where(total_stake > cap, cap / np.maximum(total_stake, EPSILON), 1.0)
    stake = stake * scale

    returns = np.sum(stake * (odds - 1.0) * outcomes, axis=1) - np.sum(
        stake * (1.0 - outcomes), axis=1
    )
    return float(np.mean(np.log(np.clip(1.0 + returns, EPSILON, None))))


def fit_w(
    p_model: np.ndarray,
    q_fair: np.ndarray,
    outcomes: np.ndarray,
    *,
    criterion: str = "log_loss",
    odds: np.ndarray | None = None,
    grid: np.ndarray = DEFAULT_GRID,
    bootstrap: int = 500,
    seed: int = 0,
    **growth_kwargs,
) -> ShrinkageFit:
    """Fit `w` on one criterion, with a bootstrap interval.

    The interval is over `w` itself: resample matches, refit, and report the
    spread. A point estimate without it invites reading 0.15 as edge when the data
    cannot distinguish it from zero.
    """
    p_model = np.asarray(p_model, dtype=float)
    q_fair = np.asarray(q_fair, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    if criterion not in ("log_loss", "log_growth"):
        raise ValueError(f"unknown criterion {criterion!r}")
    if criterion == "log_growth" and odds is None:
        raise ValueError("log_growth needs the obtainable odds")
    if len(p_model) == 0:
        raise ValueError("cannot fit w without matches")

    def score_grid(index: np.ndarray) -> np.ndarray:
        if criterion == "log_loss":
            return np.array([
                log_loss_at(p_model[index], q_fair[index], outcomes[index], w) for w in grid
            ])
        return np.array([
            log_growth_at(
                p_model[index], q_fair[index], outcomes[index],
                np.asarray(odds)[index], w, **growth_kwargs,
            )
            for w in grid
        ])

    everything = np.arange(len(p_model))
    scores = score_grid(everything)
    best = int(np.argmin(scores)) if criterion == "log_loss" else int(np.argmax(scores))

    rng = np.random.default_rng(seed)
    samples = np.empty(bootstrap)
    for b in range(bootstrap):
        pick = rng.integers(0, len(p_model), len(p_model))
        resampled = score_grid(pick)
        chosen = np.argmin(resampled) if criterion == "log_loss" else np.argmax(resampled)
        samples[b] = grid[chosen]

    return ShrinkageFit(
        w=float(grid[best]),
        criterion=criterion,
        score=float(scores[best]),
        ci_low=float(np.percentile(samples, 2.5)),
        ci_high=float(np.percentile(samples, 97.5)),
        n_matches=len(p_model),
        market_only_score=float(scores[0]),
        model_only_score=float(scores[-1]),
    )
