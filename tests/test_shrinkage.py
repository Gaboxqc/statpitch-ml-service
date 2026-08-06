"""Market-shrinkage tests (Design §6.5, FR-27).

Checked against data with a known answer: when the "model" is by construction
pure noise, w must fit near zero; when it is the true probability and the market
is noisy, w must fit near one.
"""

from __future__ import annotations

import numpy as np
import pytest

from statpitch.decision import shrinkage as sh


def _simulate(n=3000, seed=0, model_quality=0.5, market_quality=0.9):
    """Truth, plus a model and a market that each observe it with *independent* noise.

    The noise has to be independent per source. Degrading both toward uniform by
    the same amount makes them numerically identical, every w then scores the
    same, and the fit returns 0 for want of anything to choose between — which
    looks like a real "no information" verdict and is not.

    `quality` 1.0 reproduces the truth exactly; 0.0 is a source unrelated to it.
    """
    rng = np.random.default_rng(seed)
    truth = rng.dirichlet([5.0, 3.0, 3.0], size=n)

    def observe(quality: float) -> np.ndarray:
        if quality <= 0.0:
            # Unrelated to the truth, so it carries no information at all.
            return rng.dirichlet([5.0, 3.0, 3.0], size=n)
        sigma = (1.0 - quality) * 1.5
        noisy = truth * np.exp(sigma * rng.normal(size=truth.shape))
        return noisy / noisy.sum(axis=1, keepdims=True)

    p_model = observe(model_quality)
    q_fair = observe(market_quality)

    picks = np.array([rng.choice(3, p=row) for row in truth])
    outcomes = np.zeros_like(truth)
    outcomes[np.arange(n), picks] = 1.0
    return p_model, q_fair, outcomes, truth


# --- blending -----------------------------------------------------------------

def test_blend_at_zero_is_the_market():
    p, q, _, _ = _simulate(n=50)
    assert np.allclose(sh.blend(p, q, 0.0), q / q.sum(axis=1, keepdims=True))


def test_blend_at_one_is_the_model():
    p, q, _, _ = _simulate(n=50)
    assert np.allclose(sh.blend(p, q, 1.0), p / p.sum(axis=1, keepdims=True))


def test_blend_always_sums_to_one():
    p, q, _, _ = _simulate(n=50)
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert np.allclose(sh.blend(p, q, w).sum(axis=1), 1.0)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_blend_rejects_a_weight_outside_the_unit_interval(bad):
    p, q, _, _ = _simulate(n=10)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sh.blend(p, q, bad)


# --- the truth serum ----------------------------------------------------------

def test_a_worthless_model_fits_w_near_zero():
    """The finding Requirements §9 calls valid and publishable."""
    p, q, y, _ = _simulate(model_quality=0.0, market_quality=0.9, seed=1)
    fit = sh.fit_w(p, q, y, bootstrap=60)
    assert fit.w < 0.15
    assert "adds almost nothing" in fit.verdict() or not fit.interval_excludes_zero


def test_a_model_that_is_the_truth_fits_w_near_one():
    p, q, y, truth = _simulate(model_quality=1.0, market_quality=0.3, seed=2)
    fit = sh.fit_w(truth, q, y, bootstrap=60)
    assert fit.w > 0.8


def test_an_equally_good_model_lands_in_between():
    p, q, y, _ = _simulate(model_quality=0.8, market_quality=0.8, seed=3)
    fit = sh.fit_w(p, q, y, bootstrap=60)
    assert 0.15 < fit.w < 0.85


def test_blending_beats_either_source_alone_when_both_add_information():
    p, q, y, _ = _simulate(model_quality=0.7, market_quality=0.7, seed=4)
    fit = sh.fit_w(p, q, y, bootstrap=40)
    assert fit.score <= fit.market_only_score
    assert fit.score <= fit.model_only_score


def test_market_and_model_only_scores_are_reported():
    p, q, y, _ = _simulate(seed=5)
    fit = sh.fit_w(p, q, y, bootstrap=20)
    assert fit.market_only_score == pytest.approx(sh.log_loss_at(p, q, y, 0.0))
    assert fit.model_only_score == pytest.approx(sh.log_loss_at(p, q, y, 1.0))


# --- honesty about uncertainty ------------------------------------------------

def test_every_fit_carries_an_interval():
    p, q, y, _ = _simulate(n=800, seed=6)
    fit = sh.fit_w(p, q, y, bootstrap=80)
    assert fit.ci_low <= fit.w <= fit.ci_high


def test_a_noisy_fit_reports_an_interval_spanning_zero():
    """A small sample cannot distinguish a small w from none at all."""
    p, q, y, _ = _simulate(n=120, model_quality=0.05, market_quality=0.9, seed=7)
    fit = sh.fit_w(p, q, y, bootstrap=120)
    assert not fit.interval_excludes_zero
    assert "includes zero" in fit.verdict()


def test_verdict_flags_a_suspiciously_high_w():
    p, q, y, truth = _simulate(model_quality=1.0, market_quality=0.2, seed=8)
    fit = sh.fit_w(truth, q, y, bootstrap=40)
    if fit.interval_excludes_zero:
        assert "audit for leakage" in fit.verdict()


def test_fitting_without_matches_is_an_error():
    empty = np.zeros((0, 3))
    with pytest.raises(ValueError, match="without matches"):
        sh.fit_w(empty, empty, empty, bootstrap=5)


def test_unknown_criterion_is_rejected():
    p, q, y, _ = _simulate(n=50)
    with pytest.raises(ValueError, match="unknown criterion"):
        sh.fit_w(p, q, y, criterion="vibes", bootstrap=5)


def test_log_growth_requires_odds():
    p, q, y, _ = _simulate(n=50)
    with pytest.raises(ValueError, match="needs the obtainable odds"):
        sh.fit_w(p, q, y, criterion="log_growth", bootstrap=5)


# --- log growth ---------------------------------------------------------------

def _odds_from(q_fair, margin=0.05):
    """Prices implied by the market's own probabilities, plus a margin."""
    return 1.0 / (q_fair * (1.0 + margin))


def test_log_growth_is_zero_when_no_bet_has_positive_edge():
    """A blend that finds nothing scores zero rather than being penalised.

    Constructed explicitly rather than by pricing a simulated market below fair:
    the blend is not the market, so a handful of selections keep positive edge
    there and the growth is merely small, not zero.
    """
    p = np.array([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3]])
    q = np.array([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3]])
    y = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # Every price sits below its fair value, so no blend can find an edge.
    odds = 1.0 / (p * 1.5)
    for w in (0.0, 0.5, 1.0):
        assert sh.log_growth_at(p, q, y, odds, w) == pytest.approx(0.0, abs=1e-12)


def test_log_growth_prefers_the_better_information_source():
    p, q, y, truth = _simulate(n=4000, model_quality=1.0, market_quality=0.6, seed=10)
    odds = _odds_from(q)
    at_model = sh.log_growth_at(truth, q, y, odds, 1.0)
    at_market = sh.log_growth_at(truth, q, y, odds, 0.0)
    assert at_model > at_market


def test_log_growth_criterion_fits_and_reports():
    p, q, y, truth = _simulate(n=2500, model_quality=1.0, market_quality=0.6, seed=11)
    fit = sh.fit_w(truth, q, y, criterion="log_growth", odds=_odds_from(q), bootstrap=30)
    assert fit.criterion == "log_growth"
    assert 0.0 <= fit.w <= 1.0
    assert fit.beats_market_alone


def test_stake_is_capped_per_match():
    """Total exposure on one fixture cannot exceed the cap.

    Without this a single match holding several positive-edge selections could
    compound past the bankroll and make the growth figure meaningless.
    """
    p = np.array([[0.9, 0.9, 0.9]])
    q = np.array([[1 / 3, 1 / 3, 1 / 3]])
    y = np.array([[1.0, 0.0, 0.0]])
    odds = np.array([[5.0, 5.0, 5.0]])
    growth = sh.log_growth_at(p, q, y, odds, 1.0, cap=0.02)
    # Best case: 2% staked, one winner at 5.0 -> bounded well under log(1.09).
    assert growth < np.log(1.09)
