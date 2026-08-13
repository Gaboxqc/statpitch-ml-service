"""Prediction sets with a coverage guarantee (Roadmap §6.2).

A probability vector says "72% home". It does not say how much to trust that,
and MODEL_CARD §3 is blunt about why that matters here: the model's ECE is good,
but good calibration *on average* is compatible with being badly wrong on a
subset — a fixture where both clubs are unrated, or a competition with five
seasons of history.

Split conformal prediction turns the probabilities into a **set** with a marginal
coverage guarantee: over the calibration distribution, the true outcome falls in
the set at least `1 − alpha` of the time. "Home or draw, 80%" is a claim a reader
can act on and a claim that can be checked, which "72%" is not.

Adaptive prediction sets, not fixed-width
=========================================

The score is the cumulative probability of outcomes ranked most-likely-first, up
to and including the one that actually happened. A confident, correct prediction
scores low; a confident, wrong one scores near 1. Calibrating the `1 − alpha`
quantile of that score and then including outcomes until the cumulative
probability reaches it produces sets that are **small where the model is sharp
and large where it is not** — which is the entire point. A fixed-width rule would
return a set of the same size for a title-decider and a mismatch.

What the guarantee is, and what it is not
=========================================

Coverage is **marginal**: it holds averaged over fixtures, not for every subgroup.
A set covering 80% overall can cover 60% of cup ties and 85% of league matches.
That limitation is inherent to split conformal without conditioning, it is not a
defect of this implementation, and `coverage_by` exists so the subgroups can be
reported rather than assumed away.

The guarantee also transfers only as far as exchangeability does. Calibrating on
one season and serving the next assumes the two are exchangeable; football is not
stationary, so the empirical coverage on a later season is the number worth
quoting, and `evaluate` returns it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Default miscoverage rate. 0.20 gives 80% sets, which on a three-outcome market
#: are usually one or two outcomes — informative. At 0.05 almost every set is all
#: three, which is honest and useless.
DEFAULT_ALPHA = 0.20


class ConformalError(ValueError):
    pass


def scores(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cumulative probability down to the true outcome, ranked most likely first.

    Low when the model ranked the truth highly, near 1 when it was confidently
    wrong. This is the quantity whose quantile becomes the threshold.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probabilities.ndim != 2:
        raise ConformalError("probabilities must be (n_samples, n_classes)")
    if len(probabilities) != len(labels):
        raise ConformalError("probabilities and labels must have the same length")

    order = np.argsort(-probabilities, axis=1)
    ranked = np.take_along_axis(probabilities, order, axis=1)
    cumulative = np.cumsum(ranked, axis=1)
    # Where the true label sits in the ranking, per row.
    position = np.argmax(order == labels[:, None], axis=1)
    return cumulative[np.arange(len(labels)), position]


@dataclass(frozen=True, slots=True)
class Calibration:
    """A fitted threshold and what it was fitted on."""

    alpha: float
    threshold: float
    n_calibration: int

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "n_calibration": self.n_calibration,
        }


def calibrate(
    probabilities: np.ndarray, labels: np.ndarray, alpha: float = DEFAULT_ALPHA
) -> Calibration:
    """Fit the threshold on a held-out calibration set.

    The quantile uses the finite-sample correction `ceil((n+1)(1−alpha))/n`. It is
    not cosmetic: without it the guarantee is asymptotic, and on a few hundred
    calibration points the shortfall is real. With it, coverage is guaranteed at
    the stated level for any sample size.
    """
    if not 0.0 < alpha < 1.0:
        raise ConformalError(f"alpha must be in (0, 1), got {alpha}")
    values = scores(probabilities, labels)
    n = len(values)
    if n == 0:
        raise ConformalError("cannot calibrate on an empty set")

    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        # Too few points to certify this alpha; the honest threshold is 1.0,
        # which returns every outcome rather than a set that looks tighter than
        # the data can support.
        log.warning(
            "conformal: %d calibration points cannot certify alpha=%.2f; "
            "thresholding at 1.0, so every set contains every outcome", n, alpha,
        )
        return Calibration(alpha=alpha, threshold=1.0, n_calibration=n)

    threshold = float(np.sort(values)[rank - 1])
    return Calibration(alpha=alpha, threshold=threshold, n_calibration=n)


def prediction_sets(
    probabilities: np.ndarray, calibration: Calibration
) -> list[list[int]]:
    """Class indices per row, most likely first, until the threshold is reached."""
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(-probabilities, axis=1)
    ranked = np.take_along_axis(probabilities, order, axis=1)
    cumulative = np.cumsum(ranked, axis=1)

    out: list[list[int]] = []
    for row in range(len(probabilities)):
        reached = np.searchsorted(cumulative[row], calibration.threshold)
        # Always at least one outcome: an empty set is not a prediction.
        size = min(int(reached) + 1, probabilities.shape[1])
        out.append([int(c) for c in order[row, :size]])
    return out


def evaluate(
    probabilities: np.ndarray, labels: np.ndarray, calibration: Calibration
) -> dict[str, float]:
    """Empirical coverage and mean set size on data the threshold did not see."""
    sets = prediction_sets(probabilities, calibration)
    labels = np.asarray(labels, dtype=int)
    covered = np.array([label in s for s, label in zip(sets, labels, strict=True)])
    sizes = np.array([len(s) for s in sets], dtype=float)
    return {
        "coverage": float(covered.mean()),
        "target": 1.0 - calibration.alpha,
        "mean_set_size": float(sizes.mean()),
        "n": int(len(labels)),
    }


def coverage_by(
    probabilities: np.ndarray,
    labels: np.ndarray,
    calibration: Calibration,
    groups: pd.Series,
) -> pd.DataFrame:
    """Coverage per subgroup — the check the marginal guarantee does not make.

    Reported rather than assumed away: split conformal promises coverage averaged
    over fixtures, and a set covering 80% overall can cover 60% of one competition
    while over-covering another.
    """
    sets = prediction_sets(probabilities, calibration)
    labels = np.asarray(labels, dtype=int)
    frame = pd.DataFrame(
        {
            "group": np.asarray(groups),
            "covered": [label in s for s, label in zip(sets, labels, strict=True)],
            "size": [len(s) for s in sets],
        }
    )
    return (
        frame.groupby("group")
        .agg(coverage=("covered", "mean"), mean_set_size=("size", "mean"),
             n=("covered", "size"))
        .sort_values("n", ascending=False)
    )
