"""The market-as-offset experiment (Roadmap §5.4).

The experiment's conclusion is recorded in MODEL_CARD §4; what needs a test is the
plumbing that could make it wrong without looking wrong.

The offset is the whole design. Handed the market as ordinary features, a tree
ensemble cannot reproduce it — the identity map on three continuous inputs is
what piecewise-constant models approximate worst — and the first version of this
experiment measured that wrapper at ~0.024 log-loss, an order of magnitude more
than the effect being looked for. As a `base_margin` the market passes through
untouched, and a model with nothing to add scores exactly the market. That
property is what makes the comparison attributable, so it is asserted here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb


def _load():
    spec = importlib.util.spec_from_file_location(
        "market_as_feature", Path("scripts/market_as_feature.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mf = _load()


def test_the_margin_is_the_log_of_the_market():
    frame = pd.DataFrame(
        {"market_home": [0.5], "market_draw": [0.3], "market_away": [0.2]}
    )
    np.testing.assert_allclose(
        mf._margin(frame), np.log([[0.5, 0.3, 0.2]]), rtol=1e-12
    )


def test_the_margin_survives_a_zero_probability():
    """A de-vig can return a vanishing probability; log(0) would poison the fold."""
    frame = pd.DataFrame(
        {"market_home": [0.0], "market_draw": [0.5], "market_away": [0.5]}
    )
    assert np.isfinite(mf._margin(frame)).all()


def test_a_model_with_no_trees_reproduces_the_market_exactly():
    """The property the whole comparison rests on.

    With the market as `base_margin` and zero boosting rounds, the prediction must
    be the market itself. If the offset did not pass through untouched, every
    difference this experiment reports would be measuring the wrapper instead of
    the features.
    """
    rng = np.random.default_rng(0)
    n = 200
    market = rng.dirichlet([4, 3, 3], size=n)
    frame = pd.DataFrame(market, columns=mf.MARKET_COLUMNS)
    frame["x"] = rng.normal(size=n)
    frame["result"] = rng.choice(["H", "D", "A"], size=n)

    matrix = xgb.DMatrix(frame[["x"]], label=mf._labels(frame))
    matrix.set_base_margin(mf._margin(frame))
    booster = xgb.train(
        {"objective": "multi:softprob", "num_class": 3}, matrix, num_boost_round=0
    )
    np.testing.assert_allclose(booster.predict(matrix), market, atol=1e-6)


def test_labels_follow_the_project_wide_class_order():
    """`training.CLASSES` is H, D, A; a silent reordering would invert the test."""
    frame = pd.DataFrame({"result": ["H", "D", "A"]})
    np.testing.assert_array_equal(mf._labels(frame), [0, 1, 2])


def test_only_odds_covered_competitions_are_in_scope():
    """Requirements §9: cups have no free odds source, so no closing line."""
    assert "ENG.FA_CUP" not in mf.LEAGUES
    assert "UEFA.UCL" not in mf.LEAGUES
    assert len(mf.LEAGUES) == 8


def test_market_probabilities_are_a_proper_distribution():
    devigged = mf.market_probabilities()
    if devigged.empty:
        pytest.skip("no closing odds in this checkout")
    totals = devigged[mf.MARKET_COLUMNS].sum(axis=1)
    np.testing.assert_allclose(totals.to_numpy(), 1.0, atol=1e-6)


def test_market_probabilities_are_deduplicated_by_match():
    devigged = mf.market_probabilities()
    if devigged.empty:
        pytest.skip("no closing odds in this checkout")
    assert devigged["match_id"].is_unique
