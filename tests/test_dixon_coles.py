"""Dixon-Coles tests (Design §5.1, §6.1).

This matrix is the single source of truth for ~60 markets, so an error here
propagates to all of them at once. The invariants are therefore checked exactly:
it sums to one, it reduces to independent Poisson at rho=0, and the correction
moves the four low-score cells in the direction Dixon-Coles was built for.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from statpitch.models import dixon_coles as dc


@pytest.fixture
def matrix():
    return dc.score_matrix(1.5, 1.1, rho=-0.05)


# --- invariants ---------------------------------------------------------------

def test_matrix_sums_to_one(matrix):
    """Phase 3 acceptance criterion, asserted rather than eyeballed."""
    assert float(matrix.matrix.sum()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("rho_fraction", [-0.9, -0.5, 0.0, 0.5, 0.9])
@pytest.mark.parametrize(("lh", "la"), [(0.5, 0.4), (1.5, 1.1), (3.0, 2.5)])
def test_matrix_sums_to_one_across_the_parameter_space(rho_fraction, lh, la):
    """rho is expressed as a fraction of the valid range, which depends on rates."""
    low, high = dc.rho_bounds(lh, la)
    rho = high * rho_fraction if rho_fraction > 0 else -low * rho_fraction
    assert float(dc.score_matrix(lh, la, rho).matrix.sum()) == pytest.approx(1.0, abs=1e-12)


def test_valid_rho_range_tightens_as_goal_rates_rise():
    """The bug a fixed +/-0.2 band hides.

    At 1.5 v 1.2 a rho of 0.15 is fine; at 3.0 v 2.5 it drives the 0-0 cell
    negative, because the ceiling is 1/(lh*la).
    """
    _, calm = dc.rho_bounds(1.5, 1.2)
    _, wild = dc.rho_bounds(3.0, 2.5)
    assert wild < calm
    dc.score_matrix(1.5, 1.2, rho=0.15)          # fine
    with pytest.raises(dc.DixonColesError, match="negative"):
        dc.score_matrix(3.0, 2.5, rho=0.15)      # not fine


def test_fitting_stays_inside_the_valid_range_for_high_scoring_data():
    rng = np.random.default_rng(4)
    n = 5_000
    lh, la = 3.0, 2.5
    home_goals = rng.poisson(lh, n)
    away_goals = rng.poisson(la, n)
    fitted = dc.fit_rho(home_goals, away_goals, np.full(n, lh), np.full(n, la))
    low, high = dc.rho_bounds(lh, la)
    assert low <= fitted <= high
    dc.score_matrix(lh, la, fitted)   # must not raise


def test_all_cells_are_non_negative(matrix):
    assert np.all(matrix.matrix >= 0.0)


def test_one_x_two_sums_to_one(matrix):
    assert sum(matrix.one_x_two()) == pytest.approx(1.0, abs=1e-12)


def test_over_and_under_a_half_line_are_complementary(matrix):
    """No push is possible on a half line, so the two must exhaust the space."""
    assert matrix.over(2.5) + matrix.under(2.5) == pytest.approx(1.0, abs=1e-12)


def test_a_whole_line_leaves_a_push(matrix):
    """On a whole line, over + under falls short by exactly P(total == line)."""
    gap = 1.0 - (matrix.over(2.0) + matrix.under(2.0))
    exact_two = sum(
        matrix.correct_score(i, 2 - i) for i in range(3)
    )
    assert gap == pytest.approx(exact_two, abs=1e-12)
    assert gap > 0


# --- reduction to independent Poisson -----------------------------------------

def test_rho_zero_reduces_to_independent_poisson():
    """Equal up to renormalisation, which truncation makes unavoidable."""
    lh, la = 1.4, 1.2
    m = dc.score_matrix(lh, la, rho=0.0)
    kept = sum(
        poisson.pmf(i, lh) * poisson.pmf(j, la)
        for i in range(dc.DEFAULT_MAX_GOALS + 1)
        for j in range(dc.DEFAULT_MAX_GOALS + 1)
    )
    for i in range(4):
        for j in range(4):
            expected = poisson.pmf(i, lh) * poisson.pmf(j, la) / kept
            assert m.correct_score(i, j) == pytest.approx(expected, abs=1e-12)


def test_tau_is_identity_at_rho_zero():
    goals = np.arange(4)
    home, away = np.meshgrid(goals, goals, indexing="ij")
    assert np.allclose(dc.tau(home, away, 1.5, 1.2, 0.0), 1.0)


def test_tau_touches_only_the_four_low_score_cells():
    goals = np.arange(5)
    home, away = np.meshgrid(goals, goals, indexing="ij")
    correction = dc.tau(home, away, 1.5, 1.2, -0.1)

    corrected = {(0, 0), (0, 1), (1, 0), (1, 1)}
    for i in range(5):
        for j in range(5):
            if (i, j) in corrected:
                assert correction[i, j] != 1.0
            else:
                assert correction[i, j] == 1.0


# --- the correction's direction -----------------------------------------------

def test_negative_rho_lifts_the_draws_and_lowers_the_one_nil_scores():
    """The empirical pattern Dixon-Coles exists to fix.

    Independent Poisson under-predicts 0-0 and 1-1 and over-predicts 1-0 and 0-1.
    A negative rho corrects exactly that.
    """
    lift = dc.low_score_lift(-0.1, 1.5, 1.2)
    assert lift["0-0"] > 0
    assert lift["1-1"] > 0
    assert lift["1-0"] < 0
    assert lift["0-1"] < 0


def test_positive_rho_moves_the_cells_the_other_way():
    lift = dc.low_score_lift(0.1, 1.5, 1.2)
    assert lift["0-0"] < 0
    assert lift["1-1"] < 0


def test_the_correction_is_local_to_low_scores():
    plain = dc.score_matrix(1.5, 1.2, rho=0.0)
    bent = dc.score_matrix(1.5, 1.2, rho=-0.1)
    # A 3-2 scoreline is untouched except by renormalisation, so the relative
    # change there is far smaller than at 0-0.
    high = abs(bent.correct_score(3, 2) / plain.correct_score(3, 2) - 1)
    low = abs(bent.correct_score(0, 0) / plain.correct_score(0, 0) - 1)
    assert low > high * 5


# --- market derivations -------------------------------------------------------

def test_a_stronger_home_side_wins_more_often():
    strong = dc.score_matrix(2.2, 0.9)
    weak = dc.score_matrix(0.9, 2.2)
    assert strong.home_win() > strong.away_win()
    assert weak.away_win() > weak.home_win()


def test_symmetric_rates_give_symmetric_outcomes():
    m = dc.score_matrix(1.3, 1.3, rho=-0.05)
    assert m.home_win() == pytest.approx(m.away_win(), abs=1e-12)


def test_higher_rates_raise_the_over(matrix):
    low = dc.score_matrix(0.8, 0.7)
    high = dc.score_matrix(2.2, 1.9)
    assert high.over(2.5) > low.over(2.5)


def test_over_is_monotonically_decreasing_in_the_line(matrix):
    lines = [0.5, 1.5, 2.5, 3.5, 4.5]
    overs = [matrix.over(line) for line in lines]
    assert overs == sorted(overs, reverse=True)


def test_both_teams_to_score_excludes_any_clean_sheet(matrix):
    manual = 1.0 - (matrix.matrix[0, :].sum() + matrix.matrix[:, 0].sum()
                    - matrix.matrix[0, 0])
    assert matrix.both_teams_to_score() == pytest.approx(manual, abs=1e-12)


def test_top_scores_are_ordered_and_sum_below_one(matrix):
    top = matrix.top_scores(10)
    probabilities = [p for _, _, p in top]
    assert probabilities == sorted(probabilities, reverse=True)
    assert 0 < sum(probabilities) < 1.0


def test_expected_goals_track_the_input_rates():
    m = dc.score_matrix(1.6, 1.1, rho=0.0)
    home, away = m.expected_goals()
    assert home == pytest.approx(1.6, abs=0.01)
    assert away == pytest.approx(1.1, abs=0.01)


# --- fitting rho --------------------------------------------------------------

def test_fit_recovers_a_known_rho():
    """Scores are simulated from a known rho; the fit must find it."""
    rng = np.random.default_rng(0)
    true_rho = -0.08
    lh, la = 1.5, 1.2
    m = dc.score_matrix(lh, la, true_rho)

    flat = m.matrix.ravel()
    picks = rng.choice(flat.size, size=40_000, p=flat)
    home_goals, away_goals = np.unravel_index(picks, m.matrix.shape)

    fitted = dc.fit_rho(
        home_goals, away_goals,
        np.full(len(home_goals), lh), np.full(len(away_goals), la),
    )
    assert fitted == pytest.approx(true_rho, abs=0.03)


def test_fit_returns_near_zero_for_independent_data():
    rng = np.random.default_rng(1)
    lh, la = 1.4, 1.2
    home_goals = rng.poisson(lh, 40_000)
    away_goals = rng.poisson(la, 40_000)
    fitted = dc.fit_rho(
        home_goals, away_goals,
        np.full(40_000, lh), np.full(40_000, la),
    )
    assert fitted == pytest.approx(0.0, abs=0.03)


def test_fit_rejects_empty_input():
    with pytest.raises(dc.DixonColesError, match="without matches"):
        dc.fit_rho(np.array([]), np.array([]), np.array([]), np.array([]))


def test_log_likelihood_rejects_a_rho_that_makes_a_cell_negative():
    home = np.array([1])
    away = np.array([1])
    rates = np.array([1.5])
    assert dc.log_likelihood(home, away, rates, rates, 2.0) == -np.inf


# --- input validation ---------------------------------------------------------

@pytest.mark.parametrize(("lh", "la"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0)])
def test_non_positive_rates_are_rejected(lh, la):
    with pytest.raises(dc.DixonColesError, match="must be positive"):
        dc.score_matrix(lh, la)


def test_non_finite_rates_are_rejected():
    with pytest.raises(dc.DixonColesError, match="finite"):
        dc.score_matrix(float("inf"), 1.0)


def test_an_extreme_rho_is_rejected_rather_than_silently_producing_negatives():
    """A negative cell would be a probability below zero propagating to 60 markets."""
    with pytest.raises(dc.DixonColesError, match="negative"):
        dc.score_matrix(1.5, 1.2, rho=1.5)


@pytest.mark.parametrize(("lh", "la", "ceiling"), [(1.5, 1.2, 1e-6), (2.5, 2.0, 1e-4)])
def test_truncation_discards_a_negligible_tail(lh, la, ceiling):
    """Measured, not asserted from memory.

    A first version of this claimed <1e-6 at every realistic rate; at a heavy
    2.5 v 2.0 the true figure is 7e-5. Renormalisation redistributes it
    proportionally, so it is harmless — but the number is stated honestly.
    """
    kept = sum(
        poisson.pmf(i, lh) * poisson.pmf(j, la)
        for i in range(dc.DEFAULT_MAX_GOALS + 1)
        for j in range(dc.DEFAULT_MAX_GOALS + 1)
    )
    assert 1.0 - kept < ceiling
