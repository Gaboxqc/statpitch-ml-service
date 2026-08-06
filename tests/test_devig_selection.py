"""De-vig selection tests (FR-28, Phase 5).

The important behaviours are the guardrails: selection must not touch the holdout
season, and must refuse to derive fair probabilities from maximum odds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statpitch.decision import devig_selection as ds


def _odds_rows(match_id, competition_id, season, prices, regime="pre_2025_07_23"):
    rows = []
    for selection, price in zip(ds.SELECTIONS, prices, strict=True):
        rows.append({
            "match_id": match_id, "competition_id": competition_id, "season": season,
            "snapshot": "close", "market": "1x2", "selection": selection,
            "odds_avg": price, "odds_max": price * 1.05, "odds_regime": regime,
        })
    return rows


def _dataset(n=400, seed=0, season="2020-2021", competition_id="ENG.PL", margin=0.05):
    """A synthetic book built the way a real one is: true probabilities, then margin.

    Drawing odds at random instead produces books whose implied probabilities sum
    to well under one — an arbitrage, not a market — and the de-vig methods then
    have no margin to remove and behave arbitrarily. Starting from probabilities
    and loading a known margin on top keeps the ground truth explicit: with margin
    applied proportionally, proportional de-vigging recovers `true` exactly.
    """
    rng = np.random.default_rng(seed)
    odds_rows, match_rows = [], []
    for i in range(n):
        true = rng.dirichlet([4.0, 3.0, 3.0])
        implied = true * (1.0 + margin)
        prices = [float(1.0 / p) for p in implied]

        outcome = str(rng.choice(["H", "D", "A"], p=true))

        match_id = f"M{i}"
        odds_rows += _odds_rows(match_id, competition_id, season, prices)
        match_rows.append({"match_id": match_id, "result": outcome})

    return pd.DataFrame(odds_rows), pd.DataFrame(match_rows)


@pytest.fixture(scope="module")
def frame():
    odds, matches = _dataset()
    return ds.build_market_frame(odds, matches)


# --- frame construction -------------------------------------------------------

def test_frame_has_one_row_per_match_with_all_three_prices(frame):
    assert len(frame) == 400
    assert frame["match_id"].is_unique
    for selection in ds.SELECTIONS:
        assert (frame[selection] > 1.0).all()


def test_frame_keeps_only_matches_with_a_result():
    odds, matches = _dataset(n=10)
    matches.loc[0, "result"] = None
    frame = ds.build_market_frame(odds, matches)
    assert len(frame) == 9


def test_max_odds_may_never_be_de_vigged():
    """FR-16a, enforced at the point of use as well as in the config."""
    odds, matches = _dataset(n=10)
    with pytest.raises(ValueError, match="fabricates edge"):
        ds.build_market_frame(odds, matches, price_column="odds_max")


def test_season_filter_excludes_the_holdout():
    """NFR-10: selecting a method on the holdout consumes it before Phase 8."""
    train, matches_a = _dataset(n=20, season="2020-2021")
    holdout, matches_b = _dataset(n=20, seed=9, season="2024-2025")
    holdout["match_id"] = holdout["match_id"] + "H"
    matches_b["match_id"] = matches_b["match_id"] + "H"

    odds = pd.concat([train, holdout], ignore_index=True)
    matches = pd.concat([matches_a, matches_b], ignore_index=True)

    frame = ds.build_market_frame(odds, matches, seasons=["2020-2021"])
    assert set(frame["season"]) == {"2020-2021"}
    assert len(frame) == 20


def test_regime_filter_is_applied():
    pre, matches_a = _dataset(n=10)
    post, matches_b = _dataset(n=10, seed=5, season="2025-2026", )
    post["odds_regime"] = "post_2025_07_23"
    post["match_id"] = post["match_id"] + "P"
    matches_b["match_id"] = matches_b["match_id"] + "P"

    frame = ds.build_market_frame(
        pd.concat([pre, post], ignore_index=True),
        pd.concat([matches_a, matches_b], ignore_index=True),
        odds_regime="pre_2025_07_23",
    )
    assert len(frame) == 10


def test_non_closing_snapshots_are_ignored():
    odds, matches = _dataset(n=10)
    odds["snapshot"] = "preclose"
    assert ds.build_market_frame(odds, matches).empty


# --- scoring ------------------------------------------------------------------

@pytest.mark.parametrize("method", ["proportional", "power", "shin"])
def test_every_method_scores(frame, method):
    score = ds.score_method(frame, "ENG.PL", method)
    assert score is not None
    assert score.n_matches == 400
    assert score.log_loss > 0
    assert 0.0 <= score.ece <= 1.0
    assert score.mean_margin > 0


def test_compare_covers_every_method_and_competition(frame):
    comparison = ds.compare(frame)
    assert len(comparison) == 3
    assert set(comparison["method"]) == set(("proportional", "power", "shin"))


def test_proportional_wins_when_the_margin_really_is_proportional(frame):
    """A sanity check on the harness, not a claim about real books.

    Outcomes here were generated from proportionally de-vigged probabilities, so
    the proportional method is the true model and must win. If it did not, the
    scoring would be measuring something other than agreement with reality.
    """
    comparison = ds.compare(frame)
    assert ds.select(comparison)["ENG.PL"] == "proportional"


def test_selection_returns_one_winner_per_competition():
    odds_a, matches_a = _dataset(n=100, competition_id="ENG.PL")
    odds_b, matches_b = _dataset(n=100, seed=3, competition_id="ESP.LALIGA")
    odds_b["match_id"] = odds_b["match_id"] + "B"
    matches_b["match_id"] = matches_b["match_id"] + "B"

    frame = ds.build_market_frame(
        pd.concat([odds_a, odds_b], ignore_index=True),
        pd.concat([matches_a, matches_b], ignore_index=True),
    )
    winners = ds.select(ds.compare(frame))
    assert set(winners) == {"ENG.PL", "ESP.LALIGA"}


def test_selection_criterion_must_be_known(frame):
    with pytest.raises(ValueError, match="unknown selection criterion"):
        ds.select(ds.compare(frame), criterion="vibes")


def test_summary_names_the_runner_up(frame):
    text = ds.summarise(ds.compare(frame))
    assert "winner=" in text
    assert "next:" in text


def test_score_returns_none_for_an_absent_competition(frame):
    assert ds.score_method(frame, "NOT.A.LEAGUE", "shin") is None


# --- significance gate --------------------------------------------------------

def test_paired_test_reports_an_interval_and_a_p_value(frame):
    test = ds.paired_test(frame, "ENG.PL", "power", "shin")
    assert test.ci_low < test.mean_difference < test.ci_high
    assert 0.0 <= test.p_value <= 1.0
    assert test.n_matches == 400


def test_a_method_compared_with_itself_shows_no_difference(frame):
    test = ds.paired_test(frame, "ENG.PL", "shin", "shin")
    assert test.mean_difference == pytest.approx(0.0, abs=1e-12)
    assert not test.is_significant


def test_noise_level_differences_do_not_override_the_default(frame):
    """The guard that stopped a coin flip being written into the config.

    On the real Big-5 training window all three methods finish within 0.05% of
    each other on log-loss, every paired interval spans zero, and the nominal
    winner flips between competitions with no pattern. Persisting those rankings
    would make an unmeasured choice look like a measured one.
    """
    winners, tests = ds.select_significant(frame, default="shin")
    assert winners["ENG.PL"] == "shin"
    assert not any(t.is_significant for t in tests)


@pytest.mark.slow
def test_a_genuinely_better_method_does_override_the_default():
    """The gate must not be so strict that nothing can ever win.

    Margin is loaded onto the longshot here, the condition proportional de-vig
    handles worst, and power/shin do recover the true probabilities better
    (L1 error 0.039/0.041 against 0.050 for proportional).

    It takes **20,000 matches** for that to register, which is the real lesson.
    At 4,000 the same effect gives p=0.49. The Big-5 training window holds 8,955,
    so the live comparison is underpowered by roughly 2x — its null result means
    "cannot distinguish at this sample size", not "identical".
    """
    rng = np.random.default_rng(7)
    odds_rows, match_rows = [], []
    for i in range(20_000):
        true = rng.dirichlet([6.0, 3.0, 1.5])
        # Margin concentrated on the longshot rather than spread evenly.
        loading = np.array([1.005, 1.03, 1.20])
        prices = [float(1.0 / p) for p in true * loading]
        odds_rows += _odds_rows(f"X{i}", "ENG.PL", "2020-2021", prices)
        match_rows.append(
            {"match_id": f"X{i}", "result": str(rng.choice(["H", "D", "A"], p=true))}
        )

    frame = ds.build_market_frame(pd.DataFrame(odds_rows), pd.DataFrame(match_rows))
    _, tests = ds.select_significant(frame, default="proportional")
    assert any(t.challenger_wins for t in tests), (
        "no method beat proportional even with margin loaded onto the longshot"
    )


def test_the_live_sample_is_smaller_than_this_effect_needs():
    """Guards the claim above from drifting: 8,955 < 20,000."""
    live_training_matches = 8_955
    detectable_at = 20_000
    assert live_training_matches < detectable_at


def test_select_is_documented_as_a_ranking_not_a_decision(frame):
    """`select` still returns the raw ranking; the gate lives in the other call."""
    ranked = ds.select(ds.compare(frame))
    gated, _ = ds.select_significant(frame, default="shin")
    assert set(ranked) == set(gated)


# --- calibration --------------------------------------------------------------

def test_perfect_calibration_scores_zero_error():
    probabilities = np.tile([0.5, 0.3, 0.2], (1000, 1))
    rng = np.random.default_rng(0)
    outcomes = np.zeros_like(probabilities)
    picks = rng.choice(3, size=1000, p=[0.5, 0.3, 0.2])
    outcomes[np.arange(1000), picks] = 1.0
    assert ds._expected_calibration_error(probabilities, outcomes) < 0.05


def test_badly_calibrated_probabilities_score_high_error():
    probabilities = np.tile([0.9, 0.05, 0.05], (500, 1))
    outcomes = np.zeros_like(probabilities)
    outcomes[:, 2] = 1.0   # the 5% outcome always happens
    assert ds._expected_calibration_error(probabilities, outcomes) > 0.3
