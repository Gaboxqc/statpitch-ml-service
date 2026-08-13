"""Conformal prediction sets (Roadmap §6.2).

The guarantee is the product. A set that does not cover at its stated rate is
worse than no set: it converts "72% home" into "home or draw, 80%", which reads
as more trustworthy while being less true.

So the tests are mostly about coverage holding under conditions that would break
a careless implementation — small calibration sets, a miscalibrated model, and an
alpha the data cannot support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statpitch.models import conformal


def _draw(n: int, seed: int = 0, skew: float = 1.0):
    """Probabilities and outcomes actually drawn from those probabilities."""
    rng = np.random.default_rng(seed)
    probabilities = rng.dirichlet([3 * skew, 2, 2], size=n)
    labels = np.array([rng.choice(3, p=p) for p in probabilities])
    return probabilities, labels


# --- the score ----------------------------------------------------------------

def test_a_confident_correct_prediction_scores_low():
    probabilities = np.array([[0.9, 0.05, 0.05]])
    assert conformal.scores(probabilities, np.array([0]))[0] == pytest.approx(0.9)


def test_a_confident_wrong_prediction_scores_high():
    """The truth ranked last means everything else had to be included first."""
    probabilities = np.array([[0.9, 0.05, 0.05]])
    assert conformal.scores(probabilities, np.array([2]))[0] == pytest.approx(1.0)


def test_the_score_is_cumulative_down_to_the_truth():
    probabilities = np.array([[0.5, 0.3, 0.2]])
    assert conformal.scores(probabilities, np.array([1]))[0] == pytest.approx(0.8)


def test_mismatched_lengths_are_refused():
    with pytest.raises(conformal.ConformalError, match="same length"):
        conformal.scores(np.zeros((3, 3)), np.array([0]))


# --- coverage -----------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.4])
def test_coverage_holds_on_unseen_data(alpha):
    """The guarantee, checked on data the threshold never saw."""
    calibration_p, calibration_y = _draw(4000, seed=1)
    test_p, test_y = _draw(4000, seed=2)

    fitted = conformal.calibrate(calibration_p, calibration_y, alpha=alpha)
    result = conformal.evaluate(test_p, test_y, fitted)
    # Marginal coverage is guaranteed in expectation; allow sampling noise around
    # the target rather than demanding it exactly.
    assert result["coverage"] >= (1 - alpha) - 0.03


def test_coverage_holds_even_when_the_model_is_miscalibrated():
    """Conformal makes no assumption that the probabilities are any good.

    Here the model is deliberately overconfident — probabilities sharpened away
    from the distribution the outcomes were drawn from. Coverage must survive it;
    the sets simply get wider.
    """
    calibration_p, calibration_y = _draw(4000, seed=3)
    test_p, test_y = _draw(4000, seed=4)

    def sharpen(p):
        sharp = p ** 3
        return sharp / sharp.sum(axis=1, keepdims=True)

    fitted = conformal.calibrate(sharpen(calibration_p), calibration_y, alpha=0.2)
    result = conformal.evaluate(sharpen(test_p), test_y, fitted)
    assert result["coverage"] >= 0.77


def test_a_sharper_model_gives_smaller_sets():
    """Adaptivity is the reason for this scheme over a fixed-width one."""
    vague_p, vague_y = _draw(3000, seed=5, skew=1.0)
    sharp_p, sharp_y = _draw(3000, seed=6, skew=12.0)

    vague = conformal.evaluate(
        vague_p, vague_y, conformal.calibrate(vague_p, vague_y, alpha=0.2)
    )
    sharp = conformal.evaluate(
        sharp_p, sharp_y, conformal.calibrate(sharp_p, sharp_y, alpha=0.2)
    )
    assert sharp["mean_set_size"] < vague["mean_set_size"]


# --- sets ---------------------------------------------------------------------

def test_a_set_is_never_empty():
    """An empty set is not a prediction, however low the threshold falls."""
    probabilities, labels = _draw(200, seed=7)
    fitted = conformal.calibrate(probabilities, labels, alpha=0.99)
    assert all(len(s) >= 1 for s in conformal.prediction_sets(probabilities, fitted))


def test_a_set_is_ordered_most_likely_first():
    probabilities = np.array([[0.2, 0.5, 0.3]])
    fitted = conformal.Calibration(alpha=0.2, threshold=1.0, n_calibration=100)
    assert conformal.prediction_sets(probabilities, fitted)[0] == [1, 2, 0]


def test_a_set_never_exceeds_the_number_of_outcomes():
    probabilities = np.array([[0.34, 0.33, 0.33]])
    fitted = conformal.Calibration(alpha=0.01, threshold=1.0, n_calibration=10)
    assert len(conformal.prediction_sets(probabilities, fitted)[0]) == 3


# --- the finite-sample correction ---------------------------------------------

def test_too_few_points_for_the_requested_alpha_widen_to_everything():
    """Honest rather than tight: 5 points cannot certify 99% coverage.

    Returning a narrow threshold here would produce sets that look sharper than
    the calibration data can support, which is the failure this whole module
    exists to avoid.
    """
    probabilities, labels = _draw(5, seed=8)
    fitted = conformal.calibrate(probabilities, labels, alpha=0.01)
    assert fitted.threshold == 1.0
    assert all(len(s) == 3 for s in conformal.prediction_sets(probabilities, fitted))


def test_an_empty_calibration_set_is_refused():
    with pytest.raises(conformal.ConformalError, match="empty"):
        conformal.calibrate(np.zeros((0, 3)), np.array([], dtype=int))


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_an_impossible_alpha_is_refused(alpha):
    probabilities, labels = _draw(100, seed=9)
    with pytest.raises(conformal.ConformalError, match="alpha"):
        conformal.calibrate(probabilities, labels, alpha=alpha)


# --- subgroup reporting -------------------------------------------------------

def test_coverage_by_reports_each_group_separately():
    """The marginal guarantee says nothing per subgroup, so it is measured."""
    probabilities, labels = _draw(2000, seed=10)
    fitted = conformal.calibrate(probabilities, labels, alpha=0.2)
    groups = pd.Series(["a"] * 1000 + ["b"] * 1000)
    table = conformal.coverage_by(probabilities, labels, fitted, groups)
    assert set(table.index) == {"a", "b"}
    assert (table["n"] == 1000).all()
