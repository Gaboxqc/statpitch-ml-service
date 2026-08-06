"""Value tests (FR-16a, Design §6.3).

The properties that matter: the two market numbers never merge, the edge
decomposition is exact, and pushes are priced as refunds rather than losses.
"""

from __future__ import annotations

import pytest

from statpitch.decision import market_engine as me
from statpitch.decision import value as v
from statpitch.models import dixon_coles as dc


@pytest.fixture(scope="module")
def book():
    return me.MarketBook.from_matrix(dc.score_matrix(1.6, 1.15, rho=-0.05))


@pytest.fixture
def home(book):
    return book.get("1x2_home")


# --- the decomposition --------------------------------------------------------

def test_price_edge_and_model_edge_sum_to_total_ev(home):
    """The identity that lets a bet be attributed to its actual source."""
    a = v.assess(home, q_fair=0.45, o_avail=2.40, p_model=0.50)
    assert a.price_edge + a.model_edge == pytest.approx(a.expected_value, abs=1e-12)


def test_a_model_that_agrees_with_consensus_contributes_no_model_edge(home):
    a = v.assess(home, q_fair=0.46, o_avail=2.40, p_model=0.46)
    assert a.model_edge == pytest.approx(0.0, abs=1e-12)
    assert a.expected_value == pytest.approx(a.price_edge, abs=1e-12)


def test_price_edge_exists_with_no_model_skill(home):
    """Taking the best quote while believing exactly what consensus believes.

    This is the component with evidence behind it in this project; the model
    component is the one the fitted w says is worth nothing.
    """
    a = v.assess(home, q_fair=0.50, o_avail=2.20, p_model=0.50)
    assert a.price_edge > 0
    assert a.model_edge == pytest.approx(0.0, abs=1e-12)


def test_price_edge_is_negative_when_the_price_is_below_fair(home):
    a = v.assess(home, q_fair=0.50, o_avail=1.80, p_model=0.50)
    assert a.price_edge < 0


def test_driven_by_price_flags_the_source(home):
    priced = v.assess(home, q_fair=0.50, o_avail=2.30, p_model=0.50)
    modelled = v.assess(home, q_fair=0.40, o_avail=2.00, p_model=0.55)
    assert priced.driven_by_price
    assert not modelled.driven_by_price


# --- fair versus available ----------------------------------------------------

def test_fair_odds_come_from_the_consensus_not_the_available_price(home):
    a = v.assess(home, q_fair=0.40, o_avail=3.00, p_model=0.40)
    assert a.fair_odds == pytest.approx(2.5)
    assert a.o_avail == 3.00


def test_price_advantage_measures_the_gap_between_them(home):
    a = v.assess(home, q_fair=0.50, o_avail=2.20, p_model=0.50)
    assert a.price_advantage == pytest.approx(0.10)


def test_there_is_no_way_to_derive_fair_from_the_available_price(home):
    """FR-16a is enforced by the interface, not by a comment.

    De-vigging Max would make every selection look underpriced, because the
    maximum of N noisy quotes sits above consensus by construction. The two are
    separate required arguments and nothing computes one from the other.
    """
    a = v.assess(home, q_fair=0.46, o_avail=2.40, p_model=0.46)
    assert a.q_fair != pytest.approx(1.0 / a.o_avail)


def test_edge_prob_is_in_probability_points(home):
    a = v.assess(home, q_fair=0.44, o_avail=2.30, p_model=0.49)
    assert a.edge_prob == pytest.approx(0.05)


# --- expected value over payoff distributions ---------------------------------

def test_a_straight_selection_breaks_even_at_fair_odds(home):
    a = v.assess(home, q_fair=0.50, o_avail=2.00, p_model=0.50)
    assert a.expected_value == pytest.approx(0.0, abs=1e-12)


def test_a_push_is_a_refund_not_a_loss(book):
    """Draw No Bet must be worth more than the same probability without a refund.

    Pricing a push as a loss understates every DNB, whole-line total and
    whole-line handicap in the book.
    """
    dnb = book.get("dnb_home")
    straight = book.get("1x2_home")
    p = dnb.probability
    with_push = v.assess(dnb, q_fair=p, o_avail=2.0, p_model=p).expected_value
    without = v.assess(straight, q_fair=p, o_avail=2.0, p_model=p).expected_value
    assert with_push > without


def test_a_quarter_line_is_priced_over_its_full_distribution(book):
    """Half-loss must cost half a stake, not a whole one."""
    quarter = book.get("ah_home_-0.25")
    p = quarter.probability
    a = v.assess(quarter, q_fair=p, o_avail=2.0, p_model=p)
    naive = p * 2.0 - 1.0
    assert a.expected_value != pytest.approx(naive, abs=1e-6)
    assert a.expected_value > naive


def test_ev_rises_with_the_available_price(home):
    prices = [1.9, 2.0, 2.1, 2.2]
    evs = [v.assess(home, 0.5, o, 0.5).expected_value for o in prices]
    assert evs == sorted(evs)


def test_ev_rises_with_model_confidence(home):
    evs = [v.assess(home, 0.45, 2.2, p).expected_value for p in (0.40, 0.45, 0.50)]
    assert evs == sorted(evs)


# --- rescaling ----------------------------------------------------------------

def test_rescaling_preserves_the_push(book):
    """A push is a settlement rule, not an opinion about who wins."""
    dnb = book.get("dnb_home").payoff
    rescaled = v._rescale(dnb, 0.7)
    assert rescaled.push == pytest.approx(dnb.push, abs=1e-12)
    assert rescaled.total == pytest.approx(1.0, abs=1e-12)


def test_rescaling_hits_the_requested_probability(book):
    payoff = book.get("ah_home_-0.75").payoff
    rescaled = v._rescale(payoff, 0.62)
    assert rescaled.win + rescaled.half_win == pytest.approx(0.62, abs=1e-9)


def test_rescaling_keeps_the_distribution_valid(book):
    for selection in book.stakeable():
        for probability in (0.05, 0.5, 0.95):
            r = v._rescale(selection.payoff, probability)
            assert r.total == pytest.approx(1.0, abs=1e-9), selection.key
            assert min(r.win, r.half_win, r.push, r.half_loss, r.loss) >= -1e-12


# --- validation ---------------------------------------------------------------

@pytest.mark.parametrize("bad", [1.0, 0.5, -1.0])
def test_odds_at_or_below_one_are_rejected(home, bad):
    with pytest.raises(v.ValueError_, match="must exceed"):
        v.assess(home, q_fair=0.5, o_avail=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_a_fair_probability_outside_the_unit_interval_is_rejected(home, bad):
    with pytest.raises(v.ValueError_, match="q_fair"):
        v.assess(home, q_fair=bad, o_avail=2.0)


def test_an_impossible_model_probability_is_rejected(home):
    with pytest.raises(v.ValueError_, match="p_model"):
        v.assess(home, q_fair=0.5, o_avail=2.0, p_model=1.4)


# --- book-level ---------------------------------------------------------------

def test_assess_book_skips_selections_without_a_price(book):
    fair = {"1x2_home": 0.45, "1x2_draw": 0.27}
    available = {"1x2_home": 2.3}          # no price for the draw
    out = v.assess_book(book.selections, fair, available)
    assert [a.key for a in out] == ["1x2_home"]


def test_assess_book_never_invents_a_missing_price(book):
    """A market that was not quoted is not a free bet.

    Defaulting a missing price is how a backtest quietly starts trading markets
    that were never available.
    """
    out = v.assess_book(book.selections, {"1x2_home": 0.45}, {})
    assert out == []


def test_assess_book_excludes_non_stakeable_families(book):
    keys = [s.key for s in book.by_family(me.MarketFamily.CORRECT_SCORE)]
    fair = dict.fromkeys(keys, 0.08)
    available = dict.fromkeys(keys, 12.0)
    assert v.assess_book(book.selections, fair, available) == []


def test_assess_book_uses_the_selection_probability_by_default(book):
    home = book.get("1x2_home")
    out = v.assess_book(book.selections, {"1x2_home": 0.4}, {"1x2_home": 2.5})
    assert out[0].p_model == pytest.approx(home.probability)


def test_assess_book_accepts_an_override_probability(book):
    out = v.assess_book(
        book.selections, {"1x2_home": 0.4}, {"1x2_home": 2.5}, {"1x2_home": 0.55}
    )
    assert out[0].p_model == pytest.approx(0.55)


def test_summary_lists_selections_by_expected_value(book):
    out = v.assess_book(
        book.selections,
        {"1x2_home": 0.45, "1x2_away": 0.30},
        {"1x2_home": 2.10, "1x2_away": 4.00},
    )
    text = v.summarise(out)
    assert "price_ed" in text and "model_ed" in text
