"""De-vig tests (FR-28, Design §6.2).

The properties that matter are mathematical, so they are checked exactly rather
than against fixtures: probabilities sum to one, ordering is preserved, and the
methods differ from proportional in the specific direction the
favourite-longshot bias implies.
"""

from __future__ import annotations

import numpy as np
import pytest

from statpitch.decision import devig as dv

# A typical 1X2 book: strong favourite, mid draw, longshot away.
FAVOURITE_LONGSHOT = [1.40, 4.80, 8.50]
# A near-balanced two-way market, as Asian Handicap and O/U tend to be.
TWO_WAY = [1.95, 1.98]
# A flat three-way with equal prices.
SYMMETRIC = [3.15, 3.15, 3.15]


@pytest.mark.parametrize("method", dv.METHODS)
@pytest.mark.parametrize("odds", [FAVOURITE_LONGSHOT, TWO_WAY, SYMMETRIC])
def test_probabilities_sum_to_one(method, odds):
    result = dv.devig(odds, method)
    assert float(np.sum(result.probabilities)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("method", dv.METHODS)
@pytest.mark.parametrize("odds", [FAVOURITE_LONGSHOT, TWO_WAY, SYMMETRIC])
def test_probabilities_are_valid(method, odds):
    p = dv.devig(odds, method).probabilities
    assert np.all(p > 0.0)
    assert np.all(p < 1.0)


@pytest.mark.parametrize("method", dv.METHODS)
def test_ordering_is_preserved(method):
    """De-vigging redistributes margin; it never reorders outcomes."""
    p = dv.devig(FAVOURITE_LONGSHOT, method).probabilities
    assert p[0] > p[1] > p[2]


@pytest.mark.parametrize("method", dv.METHODS)
def test_every_method_reduces_the_raw_implied_total(method):
    raw = dv.implied(FAVOURITE_LONGSHOT)
    p = dv.devig(FAVOURITE_LONGSHOT, method).probabilities
    assert float(np.sum(raw)) > 1.0
    assert float(np.sum(p)) < float(np.sum(raw))


@pytest.mark.parametrize("method", dv.METHODS)
def test_symmetric_market_gives_equal_probabilities(method):
    p = dv.devig(SYMMETRIC, method).probabilities
    assert p == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-9)


# --- the favourite-longshot direction ----------------------------------------

@pytest.mark.parametrize("method", ["power", "shin"])
def test_power_and_shin_take_more_margin_off_the_longshot(method):
    """The reason the method choice is load-bearing.

    Books load margin onto longshots. Proportional de-vig assumes an even spread
    and so leaves the longshot's fair probability too high — which is how a model
    manufactures phantom value on draws and away underdogs, exactly where losses
    accumulate.
    """
    flat = dv.proportional(FAVOURITE_LONGSHOT).probabilities
    bent = dv.devig(FAVOURITE_LONGSHOT, method).probabilities

    assert bent[-1] < flat[-1]   # longshot assigned less probability
    assert bent[0] > flat[0]     # favourite assigned more


def test_proportional_keeps_the_ratio_of_any_two_prices():
    """Its defining property, and precisely the assumption under suspicion."""
    p = dv.proportional(FAVOURITE_LONGSHOT).probabilities
    raw = dv.implied(FAVOURITE_LONGSHOT)
    assert p[0] / p[2] == pytest.approx(raw[0] / raw[2])


def test_shin_and_power_do_not_keep_that_ratio():
    raw = dv.implied(FAVOURITE_LONGSHOT)
    for method in ("power", "shin"):
        p = dv.devig(FAVOURITE_LONGSHOT, method).probabilities
        assert p[0] / p[2] != pytest.approx(raw[0] / raw[2], rel=1e-3)


def test_methods_agree_when_the_market_is_symmetric():
    """With no favourite there is no favourite-longshot bias to correct."""
    results = [dv.devig(SYMMETRIC, m).probabilities for m in dv.METHODS]
    for other in results[1:]:
        assert other == pytest.approx(results[0], abs=1e-9)


# --- fitted parameters --------------------------------------------------------

def test_power_exponent_exceeds_one_for_a_real_book():
    result = dv.power(FAVOURITE_LONGSHOT)
    assert result.parameter > 1.0


def test_shin_insider_share_is_a_valid_proportion():
    result = dv.shin(FAVOURITE_LONGSHOT)
    assert 0.0 <= result.parameter < 1.0


def test_overround_and_margin_are_reported():
    result = dv.devig(FAVOURITE_LONGSHOT, "shin")
    assert result.overround > 1.0
    assert result.margin == pytest.approx(result.overround - 1.0)
    assert result.overround == pytest.approx(dv.overround(FAVOURITE_LONGSHOT))


def test_margin_matches_a_hand_computed_book():
    # 1/1.40 + 1/4.80 + 1/8.50 = 0.7143 + 0.2083 + 0.1176 = 1.0403
    assert dv.overround(FAVOURITE_LONGSHOT) == pytest.approx(1.0403, abs=1e-4)


# --- degenerate and hostile input ---------------------------------------------

def test_a_market_with_no_margin_is_left_alone():
    fair = [2.0, 2.0]
    for method in dv.METHODS:
        p = dv.devig(fair, method).probabilities
        assert p == pytest.approx([0.5, 0.5], abs=1e-9)


def test_an_arbitrage_book_still_normalises():
    """Overround below one means stale or crossed quotes, not a margin."""
    arb = [2.10, 2.10]
    for method in dv.METHODS:
        p = dv.devig(arb, method).probabilities
        assert float(np.sum(p)) == pytest.approx(1.0)


def test_a_very_wide_book_still_solves():
    wide = [1.20, 12.0, 30.0]   # ~15% margin
    for method in dv.METHODS:
        p = dv.devig(wide, method).probabilities
        assert float(np.sum(p)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("bad", [[1.0, 2.0], [0.5, 3.0], [-2.0, 2.0]])
def test_odds_at_or_below_one_are_rejected(bad):
    with pytest.raises(dv.DevigError, match="must exceed 1.0"):
        dv.devig(bad, "shin")


def test_non_finite_odds_are_rejected():
    with pytest.raises(dv.DevigError, match="non-finite"):
        dv.devig([2.0, float("nan")], "shin")


def test_a_single_selection_is_rejected():
    with pytest.raises(dv.DevigError, match="at least two selections"):
        dv.devig([2.0], "shin")


def test_unknown_method_is_rejected():
    with pytest.raises(dv.DevigError, match="unknown de-vig method"):
        dv.devig(TWO_WAY, "vibes")  # type: ignore[arg-type]


# --- vectorised path ----------------------------------------------------------

def test_devig_many_matches_the_scalar_path():
    matrix = np.array([FAVOURITE_LONGSHOT, SYMMETRIC, [2.5, 3.4, 3.0]])
    for method in dv.METHODS:
        bulk = dv.devig_many(matrix, method)
        for row, got in zip(matrix, bulk, strict=True):
            assert got == pytest.approx(dv.devig(row, method).probabilities)


def test_devig_many_rejects_a_flat_array():
    with pytest.raises(dv.DevigError, match="two-dimensional"):
        dv.devig_many(np.array([2.0, 3.0]), "shin")


def test_devig_many_rows_each_sum_to_one():
    matrix = np.array([FAVOURITE_LONGSHOT, SYMMETRIC])
    out = dv.devig_many(matrix, "shin")
    assert np.allclose(out.sum(axis=1), 1.0)
