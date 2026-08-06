"""Probability calibration and reliability measurement (Design §5.1, FR-16b).

A model can rank fixtures well and still state probabilities that are wrong in
level — saying 60% for events that happen 52% of the time. Accuracy hides that
completely; log-loss and every downstream edge calculation do not.

It matters here more than in most projects. The Decision Layer computes
`edge_prob = p_model − q_fair` and stakes on the difference. A model running two
points hot across the board manufactures a two-point edge on every selection it
touches, which looks exactly like skill and is not.

Two pieces:

* `reliability` / `expected_calibration_error` — measurement, per FR-16b, which
  the `c_calib` grading sub-score later consumes.
* `MulticlassCalibrator` — a per-class isotonic fit with renormalisation, trained
  on **out-of-fold** predictions.

The out-of-fold requirement is not a detail. Fitting a calibrator on predictions
the model has already seen in training teaches it to correct an overfit that does
not exist on new data, and leaves the deployed probabilities worse than the raw
ones while appearing better in the report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

log = logging.getLogger(__name__)

#: Default bin count for reliability curves. Ten gives the per-decile breakdown
#: FR-16b asks for.
DEFAULT_BINS = 10

#: Probabilities are clipped away from 0 and 1 so log-loss stays finite and a
#: single confident miss cannot dominate the metric.
EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    mean_predicted: float
    observed_rate: float
    count: int

    @property
    def gap(self) -> float:
        return self.observed_rate - self.mean_predicted


def reliability(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int = DEFAULT_BINS
) -> list[ReliabilityBin]:
    """Reliability curve over flattened selection-level probabilities.

    A positive `gap` means the model is under-confident in that band (events
    happen more often than stated); negative means over-confident.
    """
    flat_p = np.asarray(probabilities, dtype=float).ravel()
    flat_y = np.asarray(outcomes, dtype=float).ravel()
    if flat_p.shape != flat_y.shape:
        raise ValueError("probabilities and outcomes must have the same shape")

    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(flat_p, edges[1:-1]), 0, bins - 1)

    out = []
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        out.append(
            ReliabilityBin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                mean_predicted=float(flat_p[mask].mean()),
                observed_rate=float(flat_y[mask].mean()),
                count=int(mask.sum()),
            )
        )
    return out


def expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int = DEFAULT_BINS
) -> float:
    """Count-weighted mean absolute gap between stated and observed frequency."""
    curve = reliability(probabilities, outcomes, bins)
    total = sum(b.count for b in curve)
    if total == 0:
        return 0.0
    return sum(abs(b.gap) * b.count for b in curve) / total


def one_hot(labels: np.ndarray, n_classes: int = 3) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    out = np.zeros((len(labels), n_classes))
    out[np.arange(len(labels)), labels] = 1.0
    return out


@dataclass
class MulticlassCalibrator:
    """Per-class isotonic regression, renormalised to a proper distribution.

    Each class's isotonic map is monotone, so on its own it only rescales — it
    cannot reorder fixtures. **The renormalisation breaks that guarantee**, and
    it is worth being precise about: the divisor is the row's own sum, so two
    fixtures with an identical home probability but different draw/away splits are
    scaled differently and can swap places. Ranking is therefore *nearly*
    preserved, not exactly. Anything that depends on strict rank order must not
    assume otherwise.
    """

    n_classes: int = 3
    models: list[IsotonicRegression] | None = None

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> MulticlassCalibrator:
        probabilities = np.asarray(probabilities, dtype=float)
        targets = one_hot(labels, self.n_classes)

        self.models = []
        for k in range(self.n_classes):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(probabilities[:, k], targets[:, k])
            self.models.append(iso)
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.models is None:
            raise ValueError("calibrator is not fitted")
        probabilities = np.asarray(probabilities, dtype=float)

        calibrated = np.column_stack(
            [self.models[k].transform(probabilities[:, k]) for k in range(self.n_classes)]
        )
        calibrated = np.clip(calibrated, EPSILON, 1.0)
        # Per-class isotonic has no reason to produce rows summing to one, so the
        # renormalisation is what turns three independent fits back into a
        # distribution.
        return calibrated / calibrated.sum(axis=1, keepdims=True)


def out_of_fold_probabilities(
    fit_predict, features, labels: np.ndarray, *, folds: int = 5, seed: int = 0
) -> np.ndarray:
    """Cross-fitted predictions, so the calibrator never sees in-sample output.

    `fit_predict(train_features, train_labels, test_features)` returns the
    probabilities for the held-out fold.

    Fitting a calibrator on in-sample predictions teaches it to correct an overfit
    that will not be present at inference, which makes the deployed probabilities
    worse while the calibration report looks better.
    """
    labels = np.asarray(labels)
    out = np.zeros((len(labels), 3))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for train_index, test_index in splitter.split(np.zeros(len(labels)), labels):
        out[test_index] = fit_predict(
            features.iloc[train_index], labels[train_index], features.iloc[test_index]
        )
    return out


def summarise(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = DEFAULT_BINS) -> str:
    lines = [f"{'band':>12} {'stated':>8} {'observed':>9} {'gap':>8} {'n':>7}"]
    for b in reliability(probabilities, outcomes, bins):
        lines.append(
            f"{b.lower:.2f}-{b.upper:.2f}".rjust(12)
            + f" {b.mean_predicted:8.4f} {b.observed_rate:9.4f} {b.gap:+8.4f} {b.count:7d}"
        )
    lines.append(f"ECE = {expected_calibration_error(probabilities, outcomes, bins):.5f}")
    return "\n".join(lines)
