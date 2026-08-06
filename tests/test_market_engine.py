"""All-markets engine tests (FR-23, Design §6.1).

Two properties carry the weight. Every payoff distribution must sum to one, and
selections derived from the same matrix must agree with each other — an Asian
Handicap at -0.5 is the same event as a 1X2 home win, so they cannot disagree.
"""

from __future__ import annotations

import pytest

from statpitch.decision import market_engine as me
from statpitch.models import dixon_coles as dc


@pytest.fixture(scope="module")
def matrix():
    return dc.score_matrix(1.6, 1.15, rho=-0.05)


@pytest.fixture(scope="module")
def book(matrix):
    return me.MarketBook.from_matrix(matrix)


# --- structure ----------------------------------------------------------------

def test_a_fixture_produces_the_expected_breadth_of_selections(book):
    assert len(book) >= 50


def test_every_family_is_represented(book):
    for family in me.MarketFamily:
        assert book.by_family(family), f"{family} produced no selections"


def test_selection_keys_are_unique(book):
    keys = [s.key for s in book.selections]
    assert len(keys) == len(set(keys))


# --- the payoff invariant -----------------------------------------------------

def test_every_payoff_distribution_sums_to_one(book):
    for s in book.selections:
        assert s.payoff.total == pytest.approx(1.0, abs=1e-9), s.key


def test_every_payoff_component_is_a_probability(book):
    for s in book.selections:
        for value in (s.payoff.win, s.payoff.half_win, s.payoff.push,
                      s.payoff.half_loss, s.payoff.loss):
            assert 0.0 <= value <= 1.0, s.key


# --- cross-market consistency -------------------------------------------------

def test_handicap_at_half_a_goal_equals_the_straight_home_win(book, matrix):
    """The same event summed two ways must give the same number.

    No selection has its own model — they are all sums over one grid — so a
    disagreement here means the derivation is wrong, not that two models differ.
    """
    ah = book.get("ah_home_-0.5")
    assert ah.payoff.win == pytest.approx(matrix.home_win(), abs=1e-9)
    assert ah.payoff.win == pytest.approx(book.get("1x2_home").payoff.win, abs=1e-9)


def test_double_chance_equals_the_sum_of_its_parts(book):
    home = book.get("1x2_home").payoff.win
    draw = book.get("1x2_draw").payoff.win
    assert book.get("dc_home_draw").payoff.win == pytest.approx(home + draw, abs=1e-9)


def test_one_x_two_sums_to_one(book):
    total = sum(book.get(k).payoff.win for k in ("1x2_home", "1x2_draw", "1x2_away"))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_draw_no_bet_pushes_on_the_draw(book):
    draw = book.get("1x2_draw").payoff.win
    dnb = book.get("dnb_home").payoff
    assert dnb.push == pytest.approx(draw, abs=1e-9)
    assert dnb.win == pytest.approx(book.get("1x2_home").payoff.win, abs=1e-9)


def test_over_and_under_on_a_half_line_are_complementary(book):
    over = book.get("over_2.5").payoff
    under = book.get("under_2.5").payoff
    assert over.win + under.win == pytest.approx(1.0, abs=1e-9)
    assert over.push == 0.0


def test_a_whole_totals_line_pushes(book):
    """Over/under 2.0 refunds when the match ends with exactly two goals."""
    book_with_whole = me.MarketBook.from_matrix(
        dc.score_matrix(1.6, 1.15, rho=-0.05), totals_lines=(2.0,)
    )
    over = book_with_whole.get("over_2.0").payoff
    assert over.push > 0.0
    assert over.win + over.push + over.loss == pytest.approx(1.0, abs=1e-9)


def test_btts_pair_is_complementary(book):
    yes = book.get("btts_yes").payoff.win
    no = book.get("btts_no").payoff.win
    assert yes + no == pytest.approx(1.0, abs=1e-9)


def test_away_handicap_mirrors_the_home_side(book):
    """A home -1.0 bet and an away +1.0 bet settle on the same match, opposed."""
    home = book.get("ah_home_-1.0").payoff
    away = book.get("ah_away_1.0").payoff
    assert home.win == pytest.approx(away.loss, abs=1e-9)
    assert home.loss == pytest.approx(away.win, abs=1e-9)
    assert home.push == pytest.approx(away.push, abs=1e-9)


# --- quarter lines ------------------------------------------------------------

def test_a_quarter_line_can_half_lose(book):
    """-0.25 is half on the draw-no-bet line and half giving half a goal.

    A drawn match refunds one half and loses the other, which a two-outcome
    formula has nowhere to represent.
    """
    payoff = book.get("ah_home_-0.25").payoff
    assert payoff.half_loss > 0.0
    assert payoff.half_win == 0.0
    assert payoff.push == 0.0


def test_a_quarter_line_can_half_win(book):
    payoff = book.get("ah_home_0.25").payoff
    assert payoff.half_win > 0.0
    assert payoff.half_loss == 0.0


def test_quarter_line_half_outcomes_match_the_neighbouring_pushes(book, matrix):
    """The half outcomes are exactly the neighbours' push regions."""
    payoff = book.get("ah_home_-0.25").payoff
    _, zero_push, _ = matrix.asian_handicap(0.0)
    assert payoff.half_loss == pytest.approx(zero_push, abs=1e-9)

    payoff = book.get("ah_home_-0.75").payoff
    _, minus_one_push, _ = matrix.asian_handicap(-1.0)
    assert payoff.half_win == pytest.approx(minus_one_push, abs=1e-9)


def test_quarter_line_win_probability_matches_the_averaged_matrix(book, matrix):
    """The engine and the matrix must agree on the expected fraction won.

    The matrix averages the neighbours, which is right for a probability. The
    engine keeps them apart, which is right for a payoff. They have to reconcile.

    A half-LOSS contributes nothing to the fraction won — half the stake is
    refunded and half is lost — so it does not enter this identity. Only a
    half-win does, at half weight. Adding the half-loss term was my first
    attempt and it inflated the figure by 13 points.
    """
    for line in (-0.25, -0.75, 0.25, 0.75):
        payoff = book.get(f"ah_home_{line}").payoff
        expected, _, _ = matrix.asian_handicap(line)
        assert payoff.win + payoff.half_win / 2.0 == pytest.approx(expected, abs=1e-9), line


# --- expected return ----------------------------------------------------------

def test_expected_return_at_fair_odds_is_zero(book):
    """Priced at exactly 1/p, a straight selection must break even."""
    selection = book.get("1x2_home")
    fair = 1.0 / selection.payoff.win
    assert selection.payoff.expected_return(fair) == pytest.approx(0.0, abs=1e-9)


def test_a_push_selection_returns_nothing_on_the_push(book):
    dnb = book.get("dnb_home").payoff
    outcomes = dict((round(p, 12), r) for p, r in dnb.outcomes(2.0))
    assert outcomes[round(dnb.push, 12)] == 0.0


def test_half_win_pays_half_the_price(book):
    payoff = me.Payoff(half_win=1.0)
    assert payoff.expected_return(3.0) == pytest.approx(1.0)


def test_half_loss_costs_half_the_stake():
    payoff = me.Payoff(half_loss=1.0)
    assert payoff.expected_return(3.0) == pytest.approx(-0.5)


def test_outcomes_cover_the_whole_distribution(book):
    for s in book.selections:
        assert sum(p for p, _ in s.payoff.outcomes(2.0)) == pytest.approx(1.0, abs=1e-9)


# --- staking exclusions -------------------------------------------------------

def test_correct_score_is_never_stakeable(book):
    scores = book.by_family(me.MarketFamily.CORRECT_SCORE)
    assert scores
    for s in scores:
        assert not s.stakeable


def test_correct_score_is_excluded_from_the_stakeable_set(book):
    families = {s.family for s in book.stakeable()}
    assert me.MarketFamily.CORRECT_SCORE not in families


def test_everything_else_is_stakeable(book):
    for s in book.selections:
        if s.family is not me.MarketFamily.CORRECT_SCORE:
            assert s.stakeable, s.key


def test_correct_scores_are_ordered_by_probability(book):
    scores = book.by_family(me.MarketFamily.CORRECT_SCORE)
    probabilities = [s.payoff.win for s in scores]
    assert probabilities == sorted(probabilities, reverse=True)


# --- sanity against the fixture ------------------------------------------------

def test_a_stronger_home_side_shifts_the_book():
    strong = me.MarketBook.from_matrix(dc.score_matrix(2.3, 0.8))
    weak = me.MarketBook.from_matrix(dc.score_matrix(0.8, 2.3))
    assert strong.get("1x2_home").payoff.win > weak.get("1x2_home").payoff.win
    assert strong.get("ah_home_-1.0").payoff.win > weak.get("ah_home_-1.0").payoff.win


def test_higher_scoring_fixture_raises_the_over(book):
    quiet = me.MarketBook.from_matrix(dc.score_matrix(0.8, 0.7))
    busy = me.MarketBook.from_matrix(dc.score_matrix(2.2, 1.9))
    assert busy.get("over_2.5").payoff.win > quiet.get("over_2.5").payoff.win
