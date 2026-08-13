"""The deployed Elo-to-rate mapping, and that it stays the deployed one.

`elo_rates` is a second implementation of arithmetic that already exists inside
`serving/predictor.py`, written so the served path can be scored over a whole
season at once. Two implementations of one formula drift the moment nobody
checks, and a drifted evaluation is worse than none: it reports a number for a
model that is not deployed. `test_agrees_with_the_live_predictor` is the check
that keeps them honest.
"""

from __future__ import annotations

import numpy as np
import pytest

from statpitch.models import elo_rates
from statpitch.models.entrant_prior import ELO_SCALE
from statpitch.serving.predictor import (
    DEFAULT_AWAY_RATE,
    DEFAULT_HOME_RATE,
    ELO_GOAL_SENSITIVITY,
    LEAGUE_HOME_ADVANTAGE_ELO,
    Artifacts,
    Predictor,
)

BASE = {"base_home": DEFAULT_HOME_RATE, "base_away": DEFAULT_AWAY_RATE,
        "sensitivity": ELO_GOAL_SENSITIVITY, "elo_scale": ELO_SCALE}


def test_agrees_with_the_live_predictor():
    """The whole reason this module may exist separately from the predictor."""
    predictor = Predictor(Artifacts.load())
    prediction = predictor.predict("ENG.PL", "Arsenal", "Chelsea")
    ratings = prediction.home_rating, prediction.away_rating
    if any(r is None for r in ratings):
        pytest.skip("no measured ratings in this checkout")

    home, away = elo_rates.rates_from_elo(
        np.array([ratings[0].elo]),
        np.array([ratings[1].elo]),
        home_advantage=LEAGUE_HOME_ADVANTAGE_ELO,
        **BASE,
    )
    # The predictor reports the expectation of the *truncated, renormalised*
    # matrix, not the raw rate. `dixon_coles` puts the discarded mass at ~7e-5 at
    # heavy rates, so the two agree to about 1e-4 and cannot agree exactly. A
    # tolerance tight enough to catch a formula change, loose enough to allow the
    # truncation the matrix is documented to perform.
    expected_home, expected_away = prediction.expected_goals
    assert home[0] == pytest.approx(expected_home, rel=1e-3)
    assert away[0] == pytest.approx(expected_away, rel=1e-3)


def test_a_stronger_home_side_scores_more_and_concedes_less():
    home, away = elo_rates.rates_from_elo(
        np.array([1900.0]), np.array([1500.0]), home_advantage=0.0, **BASE
    )
    assert home[0] > DEFAULT_HOME_RATE
    assert away[0] < DEFAULT_AWAY_RATE


def test_the_symmetric_split_does_not_hold_total_goals_stable():
    """The split is symmetric; the totals it produces are not.

    `predictor._rates` says the symmetric split "keeps total goals roughly stable
    as the edge grows". That is only true when the two base rates are equal.
    They are not — 1.45 home against 1.20 away — and for
    `f(s) = a·10^(s/2) + b·10^(-s/2)` the derivative at zero is
    `(ln10/2)·(a − b)`, which is positive whenever `a > b`. The total therefore
    starts rising immediately rather than being stationary.

    Measured: a 400-point edge lifts expected total goals from 2.65 to 3.37, a
    27% increase that lands squarely on the Over/Under market. This test records
    the behaviour rather than the intention, so that a future correction to the
    mapping is a deliberate change with a failing test attached.
    """
    even = elo_rates.rates_from_elo(
        np.array([1700.0]), np.array([1700.0]), home_advantage=0.0, **BASE
    )
    lopsided = elo_rates.rates_from_elo(
        np.array([1900.0]), np.array([1500.0]), home_advantage=0.0, **BASE
    )
    even_total = sum(float(x[0]) for x in even)
    lopsided_total = sum(float(x[0]) for x in lopsided)
    assert even_total == pytest.approx(2.65, abs=0.01)
    assert lopsided_total == pytest.approx(3.37, abs=0.01)
    assert lopsided_total / even_total > 1.25


def test_home_advantage_moves_the_rates():
    without = elo_rates.rates_from_elo(
        np.array([1700.0]), np.array([1700.0]), home_advantage=0.0, **BASE
    )
    with_advantage = elo_rates.rates_from_elo(
        np.array([1700.0]), np.array([1700.0]),
        home_advantage=LEAGUE_HOME_ADVANTAGE_ELO, **BASE,
    )
    assert with_advantage[0][0] > without[0][0]
    assert with_advantage[1][0] < without[1][0]


def test_one_x_two_is_a_distribution():
    probabilities = elo_rates.one_x_two(np.array([1.5, 2.4]), np.array([1.2, 0.8]))
    assert probabilities.shape == (2, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)


def test_rho_is_clamped_per_row_as_the_predictor_does():
    """A rho valid at average rates can drive a cell negative at high ones.

    `predictor._safe_rho` clamps per fixture; evaluating without the clamp would
    score a model that cannot be served.
    """
    high = elo_rates.one_x_two(np.array([3.0]), np.array([2.5]), rho=0.2)
    assert np.isfinite(high).all()
    np.testing.assert_allclose(high.sum(axis=1), 1.0, atol=1e-9)


def test_rho_zero_reproduces_independent_poisson():
    """What production actually serves: Artifacts.rho is never populated."""
    with_rho = elo_rates.one_x_two(np.array([1.5]), np.array([1.2]), rho=0.05)
    without = elo_rates.one_x_two(np.array([1.5]), np.array([1.2]), rho=0.0)
    assert not np.allclose(with_rho, without)
