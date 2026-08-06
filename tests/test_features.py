"""Feature construction tests (Design §4, NFR-10).

The central property is that no feature can see its own match or any later one.
That is asserted directly — including by a test that would catch the specific
failure a missing `shift(1)` produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statpitch.features import build as fb


def _match(match_id, date, home, away, hg, ag, competition_id="ENG.PL", season="2024-2025"):
    return {
        "match_id": match_id, "competition_id": competition_id, "season": season,
        "date": pd.Timestamp(date), "home_team": home, "away_team": away,
        "home_goals": hg, "away_goals": ag,
    }


@pytest.fixture
def simple_log():
    """Arsenal win, draw, lose — so form is unambiguous at each step."""
    return pd.DataFrame([
        _match("m1", "2024-08-01", "Arsenal", "Chelsea", 3, 0),
        _match("m2", "2024-08-08", "Arsenal", "Everton", 1, 1),
        _match("m3", "2024-08-15", "Arsenal", "Spurs", 0, 2),
        _match("m4", "2024-08-22", "Arsenal", "Fulham", 2, 1),
    ])


# --- the leakage guarantee ----------------------------------------------------

def test_the_first_match_has_no_history(simple_log):
    features = fb.build_features(simple_log)
    first = features[features.match_id == "m1"].iloc[0]
    # Missing is NaN rather than None: pandas coerces it in a numeric column, and
    # NaN is what XGBoost reads as "no value" natively.
    assert pd.isna(first["home_form_5"])
    assert pd.isna(first["home_rest_days"])
    assert first["home_matches_played"] == 0


def test_a_match_never_contributes_to_its_own_features(simple_log):
    """The failure a missing shift(1) produces.

    Arsenal won 3-0 in m1. If that result leaked into m1's own features, its
    form would already read 3.0 points per game before a ball was kicked.
    """
    features = fb.build_features(simple_log)
    assert pd.isna(features[features.match_id == "m1"].iloc[0]["home_form_5"])
    # By m2 the win is visible, and only the win.
    assert features[features.match_id == "m2"].iloc[0]["home_form_5"] == 3.0


def test_features_use_only_earlier_matches(simple_log):
    features = fb.build_features(simple_log).set_index("match_id")
    # m3 sees a win and a draw: (3 + 1) / 2 = 2.0
    assert features.loc["m3", "home_form_5"] == pytest.approx(2.0)
    # m4 sees win, draw, loss: (3 + 1 + 0) / 3 = 1.333...
    assert features.loc["m4", "home_form_5"] == pytest.approx(4 / 3)


def test_reordering_the_input_does_not_change_the_output(simple_log):
    """Construction sorts internally, so input order is irrelevant."""
    shuffled = simple_log.sample(frac=1.0, random_state=3)
    a = fb.build_features(simple_log).sort_values("match_id").reset_index(drop=True)
    b = fb.build_features(shuffled).sort_values("match_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_truncating_the_future_leaves_earlier_features_unchanged(simple_log):
    """The strongest statement of the guarantee.

    Features for the first two matches must be identical whether or not the later
    matches exist at all. If any later result reached backwards, they would differ.
    """
    full = fb.build_features(simple_log)
    partial = fb.build_features(simple_log[simple_log.date <= "2024-08-08"])

    columns = [c for c in full.columns if c != "match_id"]
    early_full = full[full.match_id.isin(["m1", "m2"])][columns].reset_index(drop=True)
    early_partial = partial[columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(early_full, early_partial)


def test_outcomes_are_attached_separately_not_built_in(simple_log):
    features = fb.build_features(simple_log)
    assert "result" not in features.columns
    assert "home_goals" not in features.columns

    with_targets = fb.attach_outcomes(features, simple_log)
    assert set(with_targets["result"]) <= {"H", "D", "A"}


def test_targets_are_excluded_from_the_model_columns(simple_log):
    frame = fb.attach_outcomes(fb.build_features(simple_log), simple_log)
    columns = fb.feature_columns(frame)
    for leak in ("result", "home_goals", "away_goals", "total_goals", "match_id", "date"):
        assert leak not in columns
    assert "home_form_5" in columns


# --- form ---------------------------------------------------------------------

def test_form_is_points_per_game(simple_log):
    features = fb.build_features(simple_log).set_index("match_id")
    assert features.loc["m2", "home_form_5"] == 3.0     # one win
    assert features.loc["m3", "home_form_5"] == 2.0     # win + draw


def test_form_windows_differ_once_history_exceeds_the_shorter_one():
    rows = [
        _match(f"m{i}", f"2024-08-{i + 1:02d}", "Arsenal", f"Opp{i}", 3, 0)
        for i in range(6)
    ]
    rows.append(_match("m6", "2024-08-20", "Arsenal", "Late", 0, 3))
    features = fb.build_features(pd.DataFrame(rows)).set_index("match_id")
    # Six straight wins then the loss is not yet visible to m6 itself.
    assert features.loc["m6", "home_form_5"] == 3.0
    assert features.loc["m6", "home_form_10"] == 3.0


def test_goals_for_and_against_are_tracked_from_each_clubs_perspective(simple_log):
    features = fb.build_features(simple_log).set_index("match_id")
    # Arsenal scored 3 in m1, so by m2 their goals-for average is 3.
    assert features.loc["m2", "home_goals_for_5"] == 3.0
    assert features.loc["m2", "home_goals_against_5"] == 0.0


def test_an_away_defeat_is_recorded_as_such():
    log = pd.DataFrame([
        _match("m1", "2024-08-01", "Arsenal", "Chelsea", 3, 0),
        _match("m2", "2024-08-08", "Chelsea", "Everton", 1, 0),
    ])
    features = fb.build_features(log).set_index("match_id")
    # Chelsea lost away in m1, so they arrive at m2 on zero points.
    assert features.loc["m2", "home_form_5"] == 0.0
    assert features.loc["m2", "home_goals_against_5"] == 3.0


# --- rest and congestion ------------------------------------------------------

def test_rest_days_measure_the_gap_since_the_last_match(simple_log):
    features = fb.build_features(simple_log).set_index("match_id")
    assert features.loc["m2", "home_rest_days"] == 7.0


def test_rest_days_are_capped():
    log = pd.DataFrame([
        _match("m1", "2024-05-01", "Arsenal", "Chelsea", 1, 0),
        _match("m2", "2024-08-20", "Arsenal", "Everton", 1, 0),
    ])
    features = fb.build_features(log).set_index("match_id")
    assert features.loc["m2", "home_rest_days"] == fb.MAX_REST_DAYS


def test_congestion_counts_matches_in_the_last_fortnight():
    dates = ["2024-08-01", "2024-08-04", "2024-08-08", "2024-08-11", "2024-08-14"]
    log = pd.DataFrame([
        _match(f"m{i}", d, "Arsenal", f"Opp{i}", 1, 0) for i, d in enumerate(dates)
    ])
    features = fb.build_features(log).set_index("match_id")
    assert features.loc["m4", "home_matches_14d"] == 4


def test_congestion_ignores_matches_outside_the_window():
    log = pd.DataFrame([
        _match("m0", "2024-06-01", "Arsenal", "Old", 1, 0),
        _match("m1", "2024-08-01", "Arsenal", "New", 1, 0),
    ])
    features = fb.build_features(log).set_index("match_id")
    assert features.loc["m1", "home_matches_14d"] == 0


def test_congestion_and_form_cross_competitions():
    """FR-17: a Wednesday European tie makes a club tired on Saturday.

    Computing this per competition would report the club as fully rested.
    """
    log = pd.DataFrame([
        _match("c1", "2024-08-05", "Arsenal", "Euro Club", 2, 0, competition_id="UEFA.UCL"),
        _match("l1", "2024-08-08", "Arsenal", "Chelsea", 1, 0, competition_id="ENG.PL"),
    ])
    features = fb.build_features(log).set_index("match_id")
    assert features.loc["l1", "home_matches_14d"] == 1
    assert features.loc["l1", "home_rest_days"] == 3.0
    assert features.loc["l1", "home_form_5"] == 3.0   # the cup win counts


# --- head to head -------------------------------------------------------------

def test_head_to_head_counts_only_prior_meetings():
    log = pd.DataFrame([
        _match("m1", "2024-08-01", "Arsenal", "Chelsea", 3, 0),
        _match("m2", "2024-12-01", "Arsenal", "Chelsea", 1, 1),
    ])
    features = fb.build_features(log).set_index("match_id")
    assert features.loc["m1", "h2h_matches"] == 0
    assert features.loc["m2", "h2h_matches"] == 1
    assert features.loc["m2", "h2h_home_ppg"] == 3.0


def test_head_to_head_is_symmetric_in_the_pair_but_not_in_the_points():
    """Reversing the fixture must find the same tie, scored the other way."""
    log = pd.DataFrame([
        _match("m1", "2024-08-01", "Arsenal", "Chelsea", 3, 0),
        _match("m2", "2024-12-01", "Chelsea", "Arsenal", 0, 0),
    ])
    features = fb.build_features(log).set_index("match_id")
    assert features.loc["m2", "h2h_matches"] == 1
    # Chelsea are home in m2 and have no prior points *as home side* in this pair.
    assert pd.isna(features.loc["m2", "h2h_home_ppg"])


# --- Elo ----------------------------------------------------------------------

def test_elo_is_taken_from_the_lookup(simple_log):
    lookup = {
        ("Arsenal", pd.Timestamp("2024-08-01")): 1900.0,
        ("Chelsea", pd.Timestamp("2024-08-01")): 1800.0,
    }
    features = fb.build_features(simple_log, elo_lookup=lookup).set_index("match_id")
    assert features.loc["m1", "home_elo"] == 1900.0
    assert features.loc["m1", "elo_diff"] == 100.0


def test_missing_elo_yields_null_not_zero(simple_log):
    features = fb.build_features(simple_log, elo_lookup={}).set_index("match_id")
    assert pd.isna(features.loc["m1", "home_elo"])
    assert pd.isna(features.loc["m1", "elo_diff"])


# --- differentials and hygiene ------------------------------------------------

def test_differentials_are_emitted(simple_log):
    features = fb.build_features(simple_log)
    for column in ("form_diff_5", "form_diff_10", "rest_diff", "congestion_diff"):
        assert column in features.columns


def test_burn_in_rows_are_droppable(simple_log):
    features = fb.build_features(simple_log)
    trimmed = fb.drop_burn_in(features, min_matches=2)
    assert len(trimmed) < len(features)
    assert (trimmed["home_matches_played"] >= 2).all()


def test_merge_match_log_combines_and_sorts():
    league = pd.DataFrame([_match("l1", "2024-08-10", "Arsenal", "Chelsea", 1, 0)])
    cups = pd.DataFrame([
        _match("c1", "2024-08-05", "Arsenal", "Minnows", 4, 0, competition_id="ENG.FA_CUP")
    ])
    merged = fb.merge_match_log(league, cups)
    assert list(merged["match_id"]) == ["c1", "l1"]
    assert merged["date"].is_monotonic_increasing


def test_merge_match_log_drops_unplayed_and_duplicate_matches():
    league = pd.DataFrame([
        _match("l1", "2024-08-10", "Arsenal", "Chelsea", 1, 0),
        _match("l1", "2024-08-10", "Arsenal", "Chelsea", 1, 0),
        _match("l2", "2024-08-11", "Everton", "Fulham", None, None),
    ])
    merged = fb.merge_match_log(league)
    assert list(merged["match_id"]) == ["l1"]


def test_empty_input_returns_empty_output():
    assert fb.build_features(pd.DataFrame()).empty


def test_every_match_produces_exactly_one_row(simple_log):
    features = fb.build_features(simple_log)
    assert len(features) == len(simple_log)
    assert features["match_id"].is_unique


def _busy_log(n=200, seed=0):
    rng = np.random.default_rng(seed)
    clubs = [f"Club {i}" for i in range(10)]
    rows = []
    day = pd.Timestamp("2024-08-01")
    for i in range(n):
        home, away = rng.choice(clubs, size=2, replace=False)
        rows.append(
            _match(f"m{i}", day + pd.Timedelta(days=i // 5 * 3), home, away,
                   int(rng.integers(0, 4)), int(rng.integers(0, 4)))
        )
    return pd.DataFrame(rows)


def test_no_feature_column_is_entirely_null_on_real_shaped_input():
    log = _busy_log()
    rng = np.random.default_rng(1)
    xg = {mid: (float(rng.uniform(0.2, 3.0)), float(rng.uniform(0.2, 3.0)))
          for mid in log["match_id"]}
    features = fb.drop_burn_in(fb.build_features(log, xg_lookup=xg))
    for column in fb.feature_columns(features):
        assert features[column].notna().any(), f"{column} is entirely null"


# --- expected goals -----------------------------------------------------------

def test_xg_columns_are_null_when_no_xg_is_supplied():
    """Absent xG must read as "not measured", never as zero.

    Understat covers the Big 5 from 2014/15 only, so most of the archive has no
    xG at all. A zero would mean "created no chances" and drag every rolling
    average down for the seasons that predate coverage.
    """
    features = fb.build_features(_busy_log(n=60))
    for column in ("home_xg_for_5", "away_xg_against_10", "xg_diff_5"):
        assert features[column].isna().all()
    assert (features["home_xg_matches"] == 0).all()


def test_rolling_xg_uses_only_earlier_matches():
    log = pd.DataFrame([
        _match("m1", "2024-08-01", "Arsenal", "Chelsea", 1, 0),
        _match("m2", "2024-08-08", "Arsenal", "Everton", 0, 0),
    ])
    xg = {"m1": (2.5, 0.4), "m2": (1.1, 1.0)}
    features = fb.build_features(log, xg_lookup=xg).set_index("match_id")
    assert pd.isna(features.loc["m1", "home_xg_for_5"])   # its own xG is invisible
    assert features.loc["m2", "home_xg_for_5"] == pytest.approx(2.5)
    assert features.loc["m2", "home_xg_against_5"] == pytest.approx(0.4)


def test_xg_is_recorded_from_each_clubs_perspective():
    log = pd.DataFrame([
        _match("m1", "2024-08-01", "Arsenal", "Chelsea", 1, 0),
        _match("m2", "2024-08-08", "Chelsea", "Everton", 0, 0),
    ])
    xg = {"m1": (2.5, 0.4), "m2": (1.0, 1.0)}
    features = fb.build_features(log, xg_lookup=xg).set_index("match_id")
    # Chelsea were away in m1: they created 0.4 and faced 2.5.
    assert features.loc["m2", "home_xg_for_5"] == pytest.approx(0.4)
    assert features.loc["m2", "home_xg_against_5"] == pytest.approx(2.5)


def test_overperformance_is_goals_minus_xg():
    """The signal rolling goals cannot see: finishing above or below chances."""
    log = pd.DataFrame([
        _match("m1", "2024-08-01", "Arsenal", "Chelsea", 3, 0),
        _match("m2", "2024-08-08", "Arsenal", "Everton", 1, 0),
    ])
    xg = {"m1": (1.0, 0.5), "m2": (1.0, 1.0)}
    features = fb.build_features(log, xg_lookup=xg).set_index("match_id")
    # Arsenal scored 3 from 1.0 xG, so they arrive at m2 two goals to the good.
    assert features.loc["m2", "home_xg_overperformance"] == pytest.approx(2.0)


def test_xg_window_is_not_shortened_by_matches_without_xg():
    """A club can have long form history and only a few measured matches.

    Tracking xG in the same deque as goals would silently truncate the xG window
    to the last N matches overall rather than the last N measured ones.
    """
    rows = [_match(f"m{i}", f"2024-08-{i + 1:02d}", "Arsenal", f"Opp{i}", 1, 0)
            for i in range(8)]
    # Only the first two matches have xG.
    xg = {"m0": (2.0, 0.5), "m1": (2.0, 0.5)}
    features = fb.build_features(pd.DataFrame(rows), xg_lookup=xg).set_index("match_id")
    last = features.loc["m7"]
    assert last["home_xg_matches"] == 2
    assert last["home_xg_for_5"] == pytest.approx(2.0)   # both measured matches
    assert last["home_matches_played"] == 7              # full form history
