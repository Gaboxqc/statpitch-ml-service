"""Calibration and reliability tests (Design §5.1, FR-16b)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statpitch.models import calibration as cal


def _confident(n=4000, seed=0, inflation=1.4):
    """Probabilities that are systematically over-confident by a known factor."""
    rng = np.random.default_rng(seed)
    truth = rng.dirichlet([5.0, 3.0, 3.0], size=n)
    stated = truth ** inflation
    stated = stated / stated.sum(axis=1, keepdims=True)

    picks = np.array([rng.choice(3, p=row) for row in truth])
    labels = picks
    return stated, labels, truth


# --- reliability --------------------------------------------------------------

def test_perfectly_calibrated_probabilities_have_near_zero_error():
    rng = np.random.default_rng(0)
    truth = rng.dirichlet([5.0, 3.0, 3.0], size=6000)
    picks = np.array([rng.choice(3, p=row) for row in truth])
    assert cal.expected_calibration_error(truth, cal.one_hot(picks)) < 0.02


def test_over_confident_probabilities_score_a_larger_error():
    stated, labels, truth = _confident()
    honest = cal.expected_calibration_error(truth, cal.one_hot(labels))
    inflated = cal.expected_calibration_error(stated, cal.one_hot(labels))
    assert inflated > honest


def test_reliability_bins_cover_the_sample():
    stated, labels, _ = _confident(n=1000)
    curve = cal.reliability(stated, cal.one_hot(labels))
    assert sum(b.count for b in curve) == stated.size


def test_reliability_gap_sign_indicates_direction():
    """Positive gap means under-confident: events happen more often than stated."""
    probabilities = np.tile([0.2, 0.4, 0.4], (500, 1))
    outcomes = np.zeros_like(probabilities)
    outcomes[:, 0] = 1.0            # the 20% outcome always happens
    curve = cal.reliability(probabilities, outcomes)
    band = next(b for b in curve if b.lower <= 0.2 < b.upper)
    assert band.gap > 0


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        cal.reliability(np.zeros((10, 3)), np.zeros((10, 2)))


def test_summary_reports_every_band_and_the_error():
    stated, labels, _ = _confident(n=800)
    text = cal.summarise(stated, cal.one_hot(labels))
    assert "ECE" in text
    assert "observed" in text


# --- the calibrator -----------------------------------------------------------

def test_calibration_fixes_a_known_over_confidence():
    stated, labels, _ = _confident(n=6000, inflation=1.5)
    split = 3000
    calibrator = cal.MulticlassCalibrator().fit(stated[:split], labels[:split])
    fixed = calibrator.transform(stated[split:])

    before = cal.expected_calibration_error(stated[split:], cal.one_hot(labels[split:]))
    after = cal.expected_calibration_error(fixed, cal.one_hot(labels[split:]))
    assert after < before


def test_calibrated_rows_sum_to_one():
    stated, labels, _ = _confident(n=2000)
    calibrator = cal.MulticlassCalibrator().fit(stated, labels)
    out = calibrator.transform(stated)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_each_isotonic_map_is_monotone_before_renormalisation():
    stated, labels, _ = _confident(n=3000)
    calibrator = cal.MulticlassCalibrator().fit(stated, labels)

    order = np.argsort(stated[:, 0])
    mapped = calibrator.models[0].transform(stated[order, 0])
    assert np.all(np.diff(mapped) >= -1e-9)


def test_renormalisation_can_reorder_and_that_is_documented():
    """A claim I originally got wrong, kept as a test so it stays right.

    The docstring said calibration preserves ranking exactly. It does not: the
    renormalisation divisor is the row's own sum, so two fixtures with identical
    home probability but different draw/away splits scale differently and can
    swap. The reordering is small, but code depending on strict rank order must
    not assume it away.
    """
    stated, labels, _ = _confident(n=3000)
    calibrator = cal.MulticlassCalibrator().fit(stated, labels)
    out = calibrator.transform(stated)

    order_before = np.argsort(stated[:, 0])
    after = out[order_before, 0]
    inversions = int((np.diff(after) < -1e-9).sum())
    assert inversions > 0, "renormalisation is expected to perturb the ordering"

    # Measured at rho ~0.989 on this fixture, not the ">0.99, essentially intact"
    # I first assumed. Most of the ordering survives, but the perturbation is
    # real and large enough to matter to anything ranking selections.
    from scipy.stats import spearmanr
    rho, _ = spearmanr(stated[:, 0], out[:, 0])
    assert 0.98 < rho < 1.0


def test_transform_before_fit_is_an_error():
    with pytest.raises(ValueError, match="not fitted"):
        cal.MulticlassCalibrator().transform(np.zeros((3, 3)))


def test_one_hot_encodes_labels():
    out = cal.one_hot(np.array([0, 1, 2, 0]))
    assert out.shape == (4, 3)
    assert np.array_equal(out[0], [1, 0, 0])
    assert np.array_equal(out[2], [0, 0, 1])


# --- out-of-fold --------------------------------------------------------------

def test_out_of_fold_predictions_cover_every_row():
    rng = np.random.default_rng(0)
    features = pd.DataFrame({"x": rng.normal(size=300)})
    labels = rng.integers(0, 3, 300)

    def fit_predict(xtr, ytr, xte):
        return np.tile([0.4, 0.3, 0.3], (len(xte), 1))

    out = cal.out_of_fold_probabilities(fit_predict, features, labels, folds=5)
    assert out.shape == (300, 3)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_out_of_fold_never_predicts_a_row_from_a_model_that_saw_it():
    """The guarantee that makes the calibrator honest.

    Fitting on in-sample predictions teaches the calibrator to correct an overfit
    that will not exist at inference, leaving deployed probabilities worse while
    the calibration report looks better.
    """
    rng = np.random.default_rng(1)
    features = pd.DataFrame({"row": np.arange(200)})
    labels = rng.integers(0, 3, 200)
    seen: list[set[int]] = []

    def fit_predict(xtr, ytr, xte):
        train_rows = set(xtr["row"].tolist())
        test_rows = set(xte["row"].tolist())
        seen.append(train_rows & test_rows)
        return np.tile([1 / 3, 1 / 3, 1 / 3], (len(xte), 1))

    cal.out_of_fold_probabilities(fit_predict, features, labels, folds=4)
    assert all(not overlap for overlap in seen)
