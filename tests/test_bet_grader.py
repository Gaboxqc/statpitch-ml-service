"""Bet grading tests (FR-25, FR-33, Design §6.4).

The central property is the one the design calls its most important rule:
confidence in an edge is NOT monotonic. A bigger apparent edge must, past a
point, be graded as less trustworthy rather than more.
"""

from __future__ import annotations

import pytest

from statpitch.decision import bet_grader as bg
from statpitch.decision import market_engine as me
from statpitch.decision import value as v
from statpitch.models import dixon_coles as dc


@pytest.fixture(scope="module")
def selection():
    book = me.MarketBook.from_matrix(dc.score_matrix(1.6, 1.15, rho=-0.05))
    return book.get("1x2_home")


def _assess(selection, edge, odds=2.2, q_fair=0.45):
    return v.assess(selection, q_fair=q_fair, o_avail=odds, p_model=q_fair + edge)


def _good_context(**kwargs):
    base = {
        "p_std": 0.01, "book_margin": 0.03, "n_books": 25,
        "calibration_error": 0.005, "historical_clv": 0.01,
    }
    return bg.GradingContext(**{**base, **kwargs})


# --- the non-monotonic edge term ----------------------------------------------

def test_confidence_peaks_at_a_moderate_edge():
    """Design §6.4's most important rule, stated as a test."""
    at_peak = bg.edge_confidence(bg.DEFAULT_E_PEAK)
    assert at_peak == pytest.approx(1.0)
    assert bg.edge_confidence(0.01) < at_peak
    assert bg.edge_confidence(0.09) < at_peak


def test_a_bigger_edge_is_eventually_trusted_less():
    """The property that separates this from every naive value system.

    Measured in this project: selecting the largest apparent edge across markets
    returned -2.12% while committing to one market in advance returned +0.13%.
    Maximum-edge selection finds the model's own largest errors.
    """
    assert bg.edge_confidence(0.10) < bg.edge_confidence(0.04)
    assert bg.edge_confidence(0.11) < bg.edge_confidence(0.08)


def test_an_edge_above_the_ceiling_scores_zero():
    assert bg.edge_confidence(bg.DEFAULT_E_CEILING + 0.001) == 0.0
    assert bg.edge_confidence(0.30) == 0.0


def test_a_negative_edge_scores_zero():
    assert bg.edge_confidence(-0.02) == 0.0


def test_confidence_is_symmetric_around_the_peak():
    below = bg.edge_confidence(bg.DEFAULT_E_PEAK - 0.02)
    above = bg.edge_confidence(bg.DEFAULT_E_PEAK + 0.02)
    assert below == pytest.approx(above)


# --- the ceiling forces F and routes to review --------------------------------

def test_a_huge_edge_is_graded_f_not_a(selection):
    """A twenty-point edge is a diagnostic, not an opportunity."""
    bet = bg.grade(_assess(selection, 0.25), _good_context())
    assert bet.grade is bg.Grade.F
    assert bet.model_likely_blind
    assert "model likely blind" in bet.reasons[0]


def test_a_moderate_edge_outgrades_a_huge_one(selection):
    moderate = bg.grade(_assess(selection, 0.04), _good_context())
    huge = bg.grade(_assess(selection, 0.25), _good_context())
    assert moderate.grade is not bg.Grade.F
    assert huge.grade is bg.Grade.F


def test_blind_bets_reach_the_review_queue(selection):
    assessments = [_assess(selection, 0.04), _assess(selection, 0.30)]
    graded, queue = bg.grade_book(assessments, _good_context())
    assert len(queue) == 1
    assert queue.entries[0].model_likely_blind


def test_ordinary_bets_do_not_enter_the_review_queue(selection):
    _, queue = bg.grade_book([_assess(selection, 0.04)], _good_context())
    assert len(queue) == 0


# --- guardrails (FR-33) -------------------------------------------------------

@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"odds_coverage": False}, "no free odds coverage"),
        ({"lineup_confirmed": False, "key_player_doubtful": True}, "key player doubtful"),
        ({"dead_rubber": True}, "dead rubber"),
        ({"hours_to_other_competition_fixture": 48.0}, "another competition"),
        ({"p_std": 0.20}, "dispersion"),
        ({"book_margin": 0.15}, "book margin"),
    ],
)
def test_each_guardrail_forces_f_with_a_reason(selection, kwargs, fragment):
    bet = bg.grade(_assess(selection, 0.04), _good_context(**kwargs))
    assert bet.grade is bg.Grade.F
    assert any(fragment in r for r in bet.reasons), bet.reasons


def test_a_price_above_the_ceiling_is_suppressed(selection):
    bet = bg.grade(_assess(selection, 0.04, odds=12.0), _good_context())
    assert bet.grade is bg.Grade.F
    assert any("odds ceiling" in r for r in bet.reasons)


def test_guardrails_outrank_an_otherwise_perfect_bet(selection):
    """A guardrail is a structural statement, not a score to be outvoted."""
    clean = bg.grade(_assess(selection, 0.04), _good_context())
    assert clean.grade in (bg.Grade.A, bg.Grade.B)

    blocked = bg.grade(_assess(selection, 0.04), _good_context(dead_rubber=True))
    assert blocked.grade is bg.Grade.F
    assert blocked.composite == 0.0


def test_multiple_guardrails_all_get_reported(selection):
    bet = bg.grade(
        _assess(selection, 0.04),
        _good_context(dead_rubber=True, p_std=0.2, odds_coverage=False),
    )
    assert len(bet.reasons) >= 3


def test_a_cup_fixture_without_odds_coverage_is_never_staked(selection):
    """Requirements §9, enforced in code rather than in prose."""
    bet = bg.grade(_assess(selection, 0.04), _good_context(odds_coverage=False))
    assert not bet.is_stakeable


# --- sub-scores ---------------------------------------------------------------

def test_robustness_falls_as_dispersion_rises():
    assert bg.robustness(0.0) > bg.robustness(0.02) > bg.robustness(0.05)


def test_missing_dispersion_is_neutral_not_reassuring():
    """An unmeasured quantity must not score as a measured good one."""
    assert bg.robustness(None) == 0.5
    assert bg.robustness(None) < bg.robustness(0.0)


def test_market_quality_prefers_thin_margins_and_many_books():
    tight = bg.market_quality(0.02, 30)
    wide = bg.market_quality(0.09, 30)
    shallow = bg.market_quality(0.02, 2)
    assert tight > wide
    assert tight > shallow


def test_calibration_confidence_falls_as_error_rises():
    assert bg.calibration_confidence(0.0) > bg.calibration_confidence(0.02)
    assert bg.calibration_confidence(0.10) == 0.0


def test_support_is_neutral_with_no_history():
    """No evidence either way is neutral, not damning."""
    assert bg.support(None) == 0.5
    assert bg.support(0.0) == 0.5


def test_support_rewards_positive_historical_clv():
    """The sub-score carrying the only signal measured as significant here."""
    assert bg.support(0.02) > bg.support(0.0) > bg.support(-0.02)


def test_composite_is_a_weighted_average():
    sub = bg.SubScores(1.0, 1.0, 1.0, 1.0, 1.0)
    assert sub.composite(bg.DEFAULT_WEIGHTS) == pytest.approx(1.0)
    sub_zero = bg.SubScores(0.0, 0.0, 0.0, 0.0, 0.0)
    assert sub_zero.composite(bg.DEFAULT_WEIGHTS) == pytest.approx(0.0)


def test_zero_weights_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        bg.SubScores(1, 1, 1, 1, 1).composite({"c_edge": 0.0})


# --- grades and staking -------------------------------------------------------

def test_a_strong_bet_grades_well(selection):
    bet = bg.grade(_assess(selection, 0.04), _good_context())
    assert bet.grade in (bg.Grade.A, bg.Grade.B)
    assert bet.is_stakeable


def test_a_weak_bet_grades_poorly(selection):
    weak = bg.GradingContext(
        p_std=0.055, book_margin=0.075, n_books=2,
        calibration_error=0.045, historical_clv=-0.02,
    )
    bet = bg.grade(_assess(selection, 0.001, odds=2.6), weak)
    assert bet.grade in (bg.Grade.D, bg.Grade.F)


def test_non_positive_ev_is_never_graded_above_f(selection):
    bet = bg.grade(_assess(selection, 0.04, odds=1.05), _good_context())
    assert bet.grade is bg.Grade.F
    assert "expected value" in bet.reasons[0]


def test_grades_d_and_f_carry_no_stake():
    for letter in ("D", "F"):
        bet = bg.GradedBet("x", bg.Grade(letter), 0.4, None)
        assert bet.stake_multiplier() == 0.0
        assert not bet.is_stakeable


def test_stake_multipliers_decrease_with_grade():
    multipliers = [
        bg.GradedBet("x", bg.Grade(letter), 0.9, None).stake_multiplier()
        for letter in ("A", "B", "C")
    ]
    assert multipliers == sorted(multipliers, reverse=True)
    assert multipliers[0] == 1.0


def test_grade_boundaries_follow_the_cutoffs(selection):
    cutoffs = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3}
    perfect = bg.GradingContext(
        p_std=0.0, book_margin=0.0, n_books=40,
        calibration_error=0.0, historical_clv=0.05,
    )
    bet = bg.grade(_assess(selection, bg.DEFAULT_E_PEAK), perfect, cutoffs=cutoffs)
    assert bet.grade is bg.Grade.A
    assert bet.composite >= 0.9


def test_grade_book_returns_one_result_per_assessment(selection):
    assessments = [_assess(selection, e) for e in (0.01, 0.04, 0.30)]
    graded, _ = bg.grade_book(assessments, _good_context())
    assert len(graded) == 3
