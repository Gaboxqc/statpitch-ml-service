"""Knockout resolution tests (FR-7, FR-8, Design §5.3).

The invariants: advancement probabilities exhaust the space, a stronger side
advances more often, and the two-leg orientation is right — getting that wrong
silently reverses every tie.
"""

from __future__ import annotations

import pytest

from statpitch.models import dixon_coles as dc
from statpitch.models import knockout as ko


@pytest.fixture
def even():
    return dc.score_matrix(1.3, 1.3, rho=-0.05)


@pytest.fixture
def home_favoured():
    return dc.score_matrix(2.0, 0.9, rho=-0.05)


# --- single leg ---------------------------------------------------------------

def test_someone_always_advances(even, home_favoured):
    for matrix in (even, home_favoured):
        r = ko.resolve_single_leg(matrix)
        assert r.home_advances + r.away_advances == pytest.approx(1.0, abs=1e-9)


def test_an_even_tie_is_even(even):
    r = ko.resolve_single_leg(even)
    assert r.home_advances == pytest.approx(0.5, abs=1e-9)


def test_the_stronger_side_advances_more(home_favoured):
    assert ko.resolve_single_leg(home_favoured).home_advances > 0.5


def test_a_draw_never_ends_the_tie(even):
    """The whole reason FR-8 exists: a draw is not a final result."""
    r = ko.resolve_single_leg(even)
    decided = r.home_in_regulation + r.away_in_regulation
    assert decided < 1.0
    assert r.reaches_extra_time == pytest.approx(1.0 - decided, abs=1e-9)


def test_extra_time_and_penalties_are_reported_separately(even):
    r = ko.resolve_single_leg(even)
    assert r.reaches_extra_time > r.reaches_penalties > 0


def test_roughly_the_right_share_reaches_penalties(even):
    """Measured: 39.5% of extra times went to a shootout."""
    r = ko.resolve_single_leg(even)
    share = r.reaches_penalties / r.reaches_extra_time
    assert 0.25 < share < 0.55


def test_skipping_extra_time_sends_draws_straight_to_penalties(even):
    r = ko.resolve_single_leg(even, extra_time=False)
    assert r.home_in_extra_time == 0.0
    assert r.reaches_penalties == pytest.approx(even.draw(), abs=1e-9)


def test_the_summary_dict_is_consistent(home_favoured):
    d = ko.resolve_single_leg(home_favoured).as_dict()
    assert d["home_advances"] + d["away_advances"] == pytest.approx(1.0, abs=1e-9)
    assert d["decided_in_regulation"] + d["reaches_extra_time"] == pytest.approx(
        1.0, abs=1e-9
    )


# --- extra time ---------------------------------------------------------------

def test_extra_time_scores_fewer_goals_than_regulation(even):
    et = ko.extra_time_matrix(even.lambda_home, even.lambda_away, even.rho)
    assert sum(et.expected_goals()) < sum(even.expected_goals())


def test_extra_time_is_more_open_than_a_pro_rata_extrapolation(even):
    """The conventional picture of cagey extra time does not hold.

    Measured: 1.101 goals in thirty minutes against 0.790 expected from the
    matches that actually reach it.
    """
    et = ko.extra_time_matrix(even.lambda_home, even.lambda_away, even.rho)
    pro_rata = sum(even.expected_goals()) * ko.EXTRA_TIME_FRACTION
    assert sum(et.expected_goals()) > pro_rata
    assert ko.EXTRA_TIME_RATE_MULTIPLIER > 1.0


def test_the_multiplier_is_configurable(even):
    quiet = ko.extra_time_matrix(1.3, 1.3, multiplier=1.0)
    lively = ko.extra_time_matrix(1.3, 1.3, multiplier=1.5)
    assert sum(lively.expected_goals()) > sum(quiet.expected_goals())


def test_a_stronger_side_wins_extra_time_more_often(home_favoured):
    """Extra time is more football, so strength carries into it.

    Measured across the Elo range: 23.4% / 56.7% / 80.0%.
    """
    r = ko.resolve_single_leg(home_favoured)
    assert r.home_in_extra_time > r.away_in_extra_time


# --- shootouts ----------------------------------------------------------------

def test_a_shootout_is_a_coin_flip_by_default():
    """Measured at 0.556 over 222 shootouts, p=0.315 against even.

    An effect a binomial test cannot separate from a coin flip is not encoded as
    one.
    """
    assert ko.shootout_probability(0.0) == pytest.approx(0.5)
    assert ko.SHOOTOUT_HOME_ADVANTAGE == 0.5


def test_strength_barely_moves_a_shootout():
    """The gradient is 52.4% to 63.6% against 23.4% to 80.0% in extra time."""
    strong = ko.shootout_probability(1.0)
    weak = ko.shootout_probability(-1.0)
    assert strong > 0.5 > weak
    assert strong - weak < 0.25          # far flatter than extra time


def test_shootout_probability_stays_a_probability():
    for edge in (-5.0, -1.0, 0.0, 1.0, 5.0):
        assert 0.0 < ko.shootout_probability(edge) < 1.0


def test_a_shootout_discards_most_of_the_strength_signal(home_favoured):
    """Reported uncertainty should grow as a tie goes deeper, not shrink."""
    r = ko.resolve_single_leg(home_favoured)
    regulation_edge = r.home_in_regulation / (
        r.home_in_regulation + r.away_in_regulation
    )
    shootout_edge = r.home_on_penalties / r.reaches_penalties
    assert regulation_edge > shootout_edge


# --- two legs (FR-7) ----------------------------------------------------------

def test_a_two_leg_tie_resolves_completely(even):
    r = ko.resolve_two_leg(even, even)
    assert r.home_advances + r.away_advances == pytest.approx(1.0, abs=1e-6)


def test_an_even_two_leg_tie_is_even(even):
    r = ko.resolve_two_leg(even, even)
    assert r.home_advances == pytest.approx(0.5, abs=1e-6)


def test_the_stronger_side_advances_over_two_legs(home_favoured, even):
    """Leg two is oriented from ITS home side, which is the tie's away side.

    Getting this flip wrong silently reverses every tie, which is why it is
    asserted rather than assumed.
    """
    # Tie's home side is strong at home; away side is ordinary at home in leg 2.
    r = ko.resolve_two_leg(home_favoured, even)
    assert r.home_advances > 0.5


def test_orientation_is_not_symmetric(home_favoured, even):
    forward = ko.resolve_two_leg(home_favoured, even).home_advances
    reversed_ = ko.resolve_two_leg(even, home_favoured).home_advances
    assert forward > reversed_


def test_conditioning_on_a_played_first_leg_changes_the_tie(even):
    neutral = ko.resolve_two_leg(even, even).home_advances
    won_first = ko.resolve_two_leg(even, even, first_leg_score=(3, 0)).home_advances
    lost_first = ko.resolve_two_leg(even, even, first_leg_score=(0, 3)).home_advances
    assert won_first > neutral > lost_first


def test_a_big_first_leg_lead_nearly_settles_it(even):
    r = ko.resolve_two_leg(even, even, first_leg_score=(4, 0))
    assert r.home_advances > 0.9


def test_a_level_first_leg_leaves_it_open(even):
    r = ko.resolve_two_leg(even, even, first_leg_score=(1, 1))
    assert 0.4 < r.home_advances < 0.6


def test_a_level_aggregate_goes_to_extra_time_then_penalties(even):
    r = ko.resolve_two_leg(even, even, first_leg_score=(1, 1))
    assert r.reaches_extra_time > 0
    assert r.reaches_penalties > 0


def test_away_goals_are_not_applied(even):
    """UEFA abolished the rule from 2021-2022.

    A 1-1 first leg and a 0-0 second leg is level on aggregate and must reach
    extra time, not hand the tie to the side with away goals.
    """
    r = ko.resolve_two_leg(even, even, first_leg_score=(1, 1))
    assert r.reaches_extra_time > 0
