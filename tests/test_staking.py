"""Staking tests (FR-27, FR-24, Design §6.5).

The properties that matter: Kelly is solved over the full payoff distribution so
pushes and quarter lines are priced correctly, caps actually bind, a correlated
slate is sized down relative to sequential Kelly, and nothing stakes from an
unfitted config.
"""

from __future__ import annotations

import numpy as np
import pytest

from statpitch.decision import bet_grader as bg
from statpitch.decision import market_engine as me
from statpitch.decision import staking as st
from statpitch.decision import value as v
from statpitch.models import dixon_coles as dc


@pytest.fixture(scope="module")
def book():
    return me.MarketBook.from_matrix(dc.score_matrix(1.6, 1.15, rho=-0.05))


def _graded(letter="A"):
    return bg.GradedBet("k", bg.Grade(letter), 0.85, None)


def _assessment(book, key="1x2_home", q=0.45, odds=2.4, p=None):
    return v.assess(book.get(key), q_fair=q, o_avail=odds, p_model=p if p else q + 0.04)


# --- log growth and Kelly -----------------------------------------------------

def test_growth_is_zero_at_zero_stake(book):
    payoff = book.get("1x2_home").payoff
    assert st.log_growth(payoff, 2.4, 0.0) == 0.0


def test_growth_peaks_at_the_kelly_fraction(book):
    payoff = v._rescale(book.get("1x2_home").payoff, 0.50)
    kelly = st.kelly_fraction(payoff, 2.4)
    at_kelly = st.log_growth(payoff, 2.4, kelly)
    assert at_kelly > st.log_growth(payoff, 2.4, kelly * 0.5)
    assert at_kelly > st.log_growth(payoff, 2.4, min(kelly * 1.5, 0.99))


def test_no_positive_stake_when_there_is_no_edge(book):
    payoff = v._rescale(book.get("1x2_home").payoff, 0.40)
    assert st.kelly_fraction(payoff, 2.0) == 0.0      # fair would be 2.5


def test_kelly_matches_the_closed_form_for_a_simple_bet():
    """A two-outcome bet has a closed form: (p*o - 1) / (o - 1)."""
    payoff = me.Payoff(win=0.55, loss=0.45)
    expected = (0.55 * 2.2 - 1) / (2.2 - 1)
    assert st.kelly_fraction(payoff, 2.2) == pytest.approx(expected, abs=1e-4)


def test_a_push_raises_the_optimal_stake():
    """A refund is not a loss, so the same win probability supports more stake.

    A two-outcome formula cannot express this and would under-stake every Draw No
    Bet and whole-line total in the book.
    """
    straight = me.Payoff(win=0.50, loss=0.50)
    with_push = me.Payoff(win=0.50, push=0.20, loss=0.30)
    assert st.kelly_fraction(with_push, 2.2) > st.kelly_fraction(straight, 2.2)


def test_a_half_loss_costs_less_than_a_full_one():
    full = me.Payoff(win=0.50, loss=0.50)
    half = me.Payoff(win=0.50, half_loss=0.50)
    assert st.kelly_fraction(half, 2.2) > st.kelly_fraction(full, 2.2)


def test_growth_is_negative_infinity_if_a_stake_could_wipe_out():
    payoff = me.Payoff(win=0.5, loss=0.5)
    assert st.log_growth(payoff, 2.0, 1.0) == -np.inf


# --- shrinkage ----------------------------------------------------------------

def test_shrinkage_at_zero_returns_the_market():
    """The case that matters here: w fitted at zero collapses to the market."""
    assert st.shrink(0.60, 0.45, 0.0) == pytest.approx(0.45)


def test_shrinkage_at_one_returns_the_model():
    assert st.shrink(0.60, 0.45, 1.0) == pytest.approx(0.60)


def test_shrinkage_interpolates():
    assert st.shrink(0.60, 0.40, 0.5) == pytest.approx(0.50)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_shrinkage_rejects_a_weight_outside_the_unit_interval(bad):
    with pytest.raises(st.StakingError, match=r"\[0, 1\]"):
        st.shrink(0.5, 0.5, bad)


# --- single-bet staking -------------------------------------------------------

def test_a_graded_bet_gets_a_positive_stake(book):
    stake = st.stake_for(_assessment(book), _graded("A"), w=1.0)
    assert stake.fraction > 0
    assert stake.is_placed


def test_grade_scales_the_stake(book):
    a = st.stake_for(_assessment(book), _graded("A"), w=1.0).fraction
    b = st.stake_for(_assessment(book), _graded("B"), w=1.0).fraction
    c = st.stake_for(_assessment(book), _graded("C"), w=1.0).fraction
    assert a > b > c > 0


def test_grades_d_and_f_stake_nothing(book):
    for letter in ("D", "F"):
        assert st.stake_for(_assessment(book), _graded(letter), w=1.0).fraction == 0.0


def test_the_per_bet_cap_binds(book):
    stake = st.stake_for(
        _assessment(book, q=0.30, odds=6.0, p=0.60), _graded("A"),
        w=1.0, cap_per_bet=0.01,
    )
    assert stake.fraction <= 0.01
    assert stake.kelly > 0.01     # uncapped Kelly would have been larger


def test_lambda_scales_the_stake(book):
    """Measured with the per-bet cap lifted, or the cap binds first and hides it.

    Worth noticing rather than working around: at realistic edges the 2% cap
    binds well before lambda does, so in practice the cap is the operative risk
    control and lambda only matters on marginal bets.
    """
    quarter = st.stake_for(
        _assessment(book), _graded("A"), w=1.0, kelly_lambda=0.25, cap_per_bet=1.0
    )
    full = st.stake_for(
        _assessment(book), _graded("A"), w=1.0, kelly_lambda=1.0, cap_per_bet=1.0
    )
    assert full.fraction > quarter.fraction


def test_the_per_bet_cap_binds_before_lambda_at_realistic_edges(book):
    quarter = st.stake_for(_assessment(book), _graded("A"), w=1.0, kelly_lambda=0.25)
    full = st.stake_for(_assessment(book), _graded("A"), w=1.0, kelly_lambda=1.0)
    assert quarter.fraction == full.fraction == pytest.approx(0.02)


def test_a_price_above_the_ceiling_stakes_nothing(book):
    stake = st.stake_for(
        _assessment(book, q=0.10, odds=12.0, p=0.15), _graded("A"),
        w=1.0, odds_ceiling=8.0,
    )
    assert stake.fraction == 0.0


def test_shrinkage_to_the_market_removes_the_stake(book):
    """With w=0 the model's disagreement is discarded, so nothing is bet."""
    assessment = _assessment(book, q=0.45, odds=2.15, p=0.55)
    with_model = st.stake_for(assessment, _graded("A"), w=1.0)
    without = st.stake_for(assessment, _graded("A"), w=0.0)
    assert with_model.fraction > 0
    assert without.fraction == 0.0


def test_a_tiny_stake_is_dropped(book):
    stake = st.stake_for(
        _assessment(book, q=0.4999, odds=2.0, p=0.5001), _graded("C"),
        w=1.0, kelly_lambda=0.01,
    )
    assert stake.fraction == 0.0


# --- ranking ------------------------------------------------------------------

def test_ranking_orders_by_growth():
    stakes = [
        st.Stake("low", 0.01, 0.04, 0.001, 0.5, 2.0, "B"),
        st.Stake("high", 0.02, 0.08, 0.004, 0.5, 2.0, "A"),
    ]
    assert [s.key for s in st.rank_by_growth(stakes)] == ["high", "low"]


def test_best_bet_ignores_unplaced_stakes():
    stakes = [
        st.Stake("skipped", 0.0, 0.0, 0.009, 0.5, 2.0, "F"),
        st.Stake("placed", 0.01, 0.04, 0.001, 0.5, 2.0, "B"),
    ]
    assert st.best_bet(stakes).key == "placed"


def test_best_bet_returns_nothing_when_no_stake_qualifies():
    assert st.best_bet([st.Stake("x", 0.0, 0.0, 0.0, 0.5, 2.0, "F")]) is None


# --- caps ---------------------------------------------------------------------

def test_matchday_cap_scales_the_slate_proportionally():
    stakes = [st.Stake(f"b{i}", 0.02, 0.08, 0.001, 0.5, 2.0, "A") for i in range(10)]
    capped = st.apply_matchday_cap(stakes, cap=0.10)
    assert sum(s.fraction for s in capped) == pytest.approx(0.10)
    # Proportional: every bet keeps its share of the slate.
    assert len({round(s.fraction, 10) for s in capped}) == 1


def test_matchday_cap_leaves_a_small_slate_alone():
    stakes = [st.Stake("a", 0.02, 0.08, 0.001, 0.5, 2.0, "A")]
    assert st.apply_matchday_cap(stakes, cap=0.10)[0].fraction == 0.02


# --- correlated slates --------------------------------------------------------

def _slate_bet(key, fixture, p=0.55, odds=2.1, max_fraction=0.02):
    return st.SlateBet(key, odds, me.Payoff(win=p, loss=1 - p), p, fixture, max_fraction)


def test_simulated_returns_are_correlated_within_a_fixture():
    """Two bets on one fixture must settle from the same draw."""
    bets = [_slate_bet("a", "F1"), _slate_bet("b", "F1")]
    returns = st.simulate_returns(bets, draws=2000)
    assert np.corrcoef(returns[:, 0], returns[:, 1])[0, 1] > 0.9


def test_simulated_returns_are_independent_across_fixtures():
    bets = [_slate_bet("a", "F1"), _slate_bet("b", "F2")]
    returns = st.simulate_returns(bets, draws=4000)
    assert abs(np.corrcoef(returns[:, 0], returns[:, 1])[0, 1]) < 0.1


def test_a_correlated_slate_is_staked_less_than_sequential_kelly():
    """Design §6.5 step 4, and the reason the joint solve exists.

    Four bets that all win or all lose together are one bet with four names.
    Sequential Kelly sizes each as though it were alone and quadruples the real
    exposure.
    """
    # Every cap lifted, including SlateBet.max_fraction, so the correlation
    # structure is the only thing binding. With production caps both slates
    # simply hit the matchday limit and the difference is invisible — worth
    # knowing in itself: the caps do most of the risk work, not the joint solve.
    correlated = [_slate_bet(f"c{i}", "SAME", max_fraction=0.5) for i in range(4)]
    independent = [_slate_bet(f"i{i}", f"F{i}", max_fraction=0.5) for i in range(4)]

    kwargs = {"cap_per_bet": 0.5, "cap_matchday": 2.0}
    total_correlated = sum(st.allocate_slate(correlated, **kwargs).values())
    total_independent = sum(st.allocate_slate(independent, **kwargs).values())
    assert total_correlated < total_independent


def test_production_caps_hide_the_correlation_adjustment():
    """Both slates reach the matchday cap, so the caps dominate in practice."""
    correlated = [_slate_bet(f"c{i}", "SAME") for i in range(4)]
    independent = [_slate_bet(f"i{i}", f"F{i}") for i in range(4)]
    assert sum(st.allocate_slate(correlated).values()) == pytest.approx(
        sum(st.allocate_slate(independent).values()), abs=1e-6
    )


def test_slate_allocation_respects_the_matchday_cap():
    bets = [_slate_bet(f"b{i}", f"F{i}", p=0.62) for i in range(12)]
    allocation = st.allocate_slate(bets, cap_matchday=0.08)
    assert sum(allocation.values()) <= 0.08 + 1e-6


def test_slate_allocation_respects_the_per_bet_cap():
    bets = [_slate_bet(f"b{i}", f"F{i}", p=0.70) for i in range(3)]
    allocation = st.allocate_slate(bets, cap_per_bet=0.01)
    assert all(f <= 0.01 + 1e-9 for f in allocation.values())


def test_an_empty_slate_allocates_nothing():
    assert st.allocate_slate([]) == {}


def test_a_slate_with_no_edge_stakes_nothing():
    bets = [st.SlateBet("x", 1.8, me.Payoff(win=0.5, loss=0.5), 0.5, "F1")]
    assert sum(st.allocate_slate(bets).values()) == 0.0


# --- lambda frontier ----------------------------------------------------------

def test_frontier_covers_every_lambda():
    rng = np.random.default_rng(0)
    returns = rng.choice([1.1, -1.0], size=(400, 3), p=[0.52, 0.48])
    points = st.lambda_frontier(returns, np.full(3, 0.02))
    assert [p.kelly_lambda for p in points] == [0.10, 0.25, 0.50, 1.00]


def test_bigger_lambda_means_deeper_drawdown():
    rng = np.random.default_rng(1)
    returns = rng.choice([1.1, -1.0], size=(600, 3), p=[0.52, 0.48])
    points = st.lambda_frontier(returns, np.full(3, 0.02))
    drawdowns = [p.max_drawdown for p in points]
    assert drawdowns == sorted(drawdowns, reverse=True)


# --- the gate -----------------------------------------------------------------

def test_the_engine_refuses_to_stake_from_placeholder_config():
    """A stake sized from unfitted parameters looks exactly like a real one."""
    from statpitch import decision_config
    from statpitch.decision_config import DecisionConfigError

    with pytest.raises(DecisionConfigError, match="size stakes"):
        st.StakingEngine(decision_config.config())


def test_the_engine_works_once_the_config_is_fitted(tmp_path, book):
    import json

    from statpitch import decision_config

    raw = json.loads(json.dumps(decision_config.config().raw))
    raw["status"] = "fitted"
    raw["market_shrinkage"] = {"w": 0.25, "w_fitted": True}
    path = tmp_path / "decision_config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    engine = st.StakingEngine(decision_config.load(path))
    stake = engine.stake(_assessment(book), _graded("A"))
    assert stake.fraction >= 0
