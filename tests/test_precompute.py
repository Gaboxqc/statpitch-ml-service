"""Precomputed predictions (Roadmap §8, closing §2's measured gap).

The API derives goal rates from Elo and pays +0.0064 log-loss for it
(MODEL_CARD §3), because the fitted model needs rolling-form features that do not
exist for an arbitrary fixture. They do exist for a *known* one: rolling form
depends only on matches already played, so a scheduled fixture can be given a
real feature row offline and its rates read back at request time.

The risk that arrangement introduces is leakage in the other direction. A
scheduled fixture now travels through the same chronological pass as a played
one, and if it updated club state it would inject an invented match — with null
goals — into every later fixture's features. That is what most of this file is
about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from statpitch.data import club_elo as ce
from statpitch.features import build as fb
from statpitch.serving.app import app, predictor

PLAYED = pd.DataFrame(
    {
        "match_id": ["m1", "m2", "m3"],
        "competition_id": ["ENG.PL"] * 3,
        "season": ["2025-2026"] * 3,
        "date": pd.to_datetime(["2026-01-03", "2026-01-10", "2026-01-17"]),
        "home_team": ["Arsenal", "Chelsea", "Arsenal"],
        "away_team": ["Chelsea", "Arsenal", "Chelsea"],
        "home_goals": [2, 1, 3],
        "away_goals": [0, 1, 1],
    }
)

FIXTURES = pd.DataFrame(
    {
        "fixture_id": ["f1"],
        "competition_id": ["ENG.PL"],
        "season": ["2026-2027"],
        "date": pd.to_datetime(["2026-08-21"]),
        "home_team": ["Arsenal"],
        "away_team": ["Chelsea"],
    }
)


# --- the match log ------------------------------------------------------------

def test_fixtures_are_appended_with_null_goals():
    merged = fb.merge_match_log(PLAYED, fixtures=FIXTURES)
    assert len(merged) == 4
    scheduled = merged[merged["match_id"] == "f1"].iloc[0]
    assert pd.isna(scheduled["home_goals"])


def test_the_played_filter_still_applies_without_fixtures():
    """A result-less row must not reach the training log by any other route."""
    with_null = pd.concat(
        [PLAYED, PLAYED.tail(1).assign(match_id="m4", home_goals=np.nan)]
    )
    assert len(fb.merge_match_log(with_null)) == 3


# --- leakage ------------------------------------------------------------------

def test_a_scheduled_fixture_gets_a_feature_row():
    features = fb.build_features(fb.merge_match_log(PLAYED, fixtures=FIXTURES))
    assert "f1" in set(features["match_id"])


def test_a_scheduled_fixture_sees_the_played_history():
    """Otherwise the row would be all-null and the exercise pointless."""
    features = fb.build_features(
        fb.merge_match_log(PLAYED, fixtures=FIXTURES)
    ).set_index("match_id")
    assert features.loc["f1", "home_matches_played"] == 3


def test_a_scheduled_fixture_contributes_nothing_to_club_state():
    """The leakage guarantee, in the direction precompute introduces.

    Two scheduled fixtures in a row: the second must see exactly the history the
    first saw, because nothing happened in between.
    """
    two = pd.DataFrame(
        {
            "fixture_id": ["f1", "f2"],
            "competition_id": ["ENG.PL"] * 2,
            "season": ["2026-2027"] * 2,
            "date": pd.to_datetime(["2026-08-21", "2026-08-28"]),
            "home_team": ["Arsenal", "Arsenal"],
            "away_team": ["Chelsea", "Chelsea"],
        }
    )
    features = fb.build_features(
        fb.merge_match_log(PLAYED, fixtures=two)
    ).set_index("match_id")
    assert (
        features.loc["f2", "home_matches_played"]
        == features.loc["f1", "home_matches_played"]
    )
    assert features.loc["f2", "h2h_matches"] == features.loc["f1", "h2h_matches"]


def test_played_rows_are_unchanged_by_appending_fixtures():
    """Adding a future fixture must not alter a single historical feature."""
    without = fb.build_features(fb.merge_match_log(PLAYED)).set_index("match_id")
    with_fixtures = fb.build_features(
        fb.merge_match_log(PLAYED, fixtures=FIXTURES)
    ).set_index("match_id")
    pd.testing.assert_frame_equal(without, with_fixtures.loc[without.index])


# --- the Elo lookup -----------------------------------------------------------

ELO_TABLE = pd.DataFrame(
    {
        "clubelo_name": ["Arsenal", "Arsenal", "Arsenal"],
        "source_name": [None, None, None],
        "elo": [1800.0, 1850.0, 1900.0],
        "valid_from": pd.to_datetime(["2026-01-01", "2026-06-01", "2026-08-21"]),
    }
)


def test_lookup_takes_the_rating_in_force_before_the_date():
    lookup = ce.build_lookup(ELO_TABLE, [("Arsenal", pd.Timestamp("2026-07-01"))])
    assert lookup[("Arsenal", pd.Timestamp("2026-07-01"))] == 1850.0


def test_lookup_is_strict_so_a_match_day_rating_cannot_leak():
    """Club Elo's interval covering a match already reflects that match (NFR-10)."""
    lookup = ce.build_lookup(ELO_TABLE, [("Arsenal", pd.Timestamp("2026-08-21"))])
    assert lookup[("Arsenal", pd.Timestamp("2026-08-21"))] == 1850.0


def test_lookup_keys_on_clubelo_name_not_source_name():
    """`source_name` is null for every club that entered as a cup entrant.

    Keying on it silently loses 187 of 428 clubs, which shows up as fixtures
    predicted without a rating rather than as an error.
    """
    assert ce.build_lookup(ELO_TABLE, [("Arsenal", pd.Timestamp("2026-07-01"))])


def test_lookup_omits_a_club_with_no_earlier_rating():
    lookup = ce.build_lookup(ELO_TABLE, [("Arsenal", pd.Timestamp("2025-01-01"))])
    assert lookup == {}


# --- the rates override -------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_supplied_rates_replace_the_elo_mapping(client):
    engine = predictor()
    default = engine.predict("ENG.PL", "Arsenal", "Chelsea")
    overridden = engine.predict("ENG.PL", "Arsenal", "Chelsea", rates=(2.5, 0.4, 0.0))
    assert overridden.expected_goals[0] > default.expected_goals[0]
    assert overridden.expected_goals[0] == pytest.approx(2.5, rel=1e-3)


def test_supplied_rho_reaches_the_matrix(client):
    """Serving applies no rho of its own, so a precomputed one must travel."""
    engine = predictor()
    without = engine.predict("ENG.PL", "Arsenal", "Chelsea", rates=(1.5, 1.2, 0.0))
    with_rho = engine.predict("ENG.PL", "Arsenal", "Chelsea", rates=(1.5, 1.2, 0.1))
    assert without.one_x_two != with_rho.one_x_two


# --- what the API reports -----------------------------------------------------

@pytest.fixture(scope="module")
def precomputed(client):
    if not predictor().artifacts.predicted_rates:
        pytest.skip("no precomputed predictions in this checkout")
    return True


def test_precomputed_fixtures_say_so(client, precomputed):
    body = client.get("/fixtures/upcoming?limit=5&include_predictions=true").json()
    for fixture in body["fixtures"]:
        assert fixture["prediction_source"] == "fitted_goal_model"
        assert fixture["prediction_model_version"].startswith("goals-")


def test_a_fixture_without_a_precomputed_rate_falls_back_and_says_so(
    client, precomputed
):
    """The fallback is the Elo path, and a consumer must be able to see that."""
    artifacts = predictor().artifacts
    saved = artifacts.predicted_rates
    artifacts.predicted_rates = {}
    try:
        body = client.get("/fixtures/upcoming?limit=3&include_predictions=true").json()
        for fixture in body["fixtures"]:
            assert fixture["prediction_source"] == "elo-poisson"
            assert fixture["prediction_model_version"].startswith("elo-poisson")
    finally:
        artifacts.predicted_rates = saved


# --- explanations (FR-32) -----------------------------------------------------

@pytest.fixture(scope="module")
def explained(client, precomputed):
    if not predictor().artifacts.explanations:
        pytest.skip("no explanations in this checkout")
    return True


def test_precomputed_fixtures_carry_an_explanation(client, explained):
    body = client.get("/fixtures/upcoming?limit=3&include_predictions=true").json()
    for fixture in body["fixtures"]:
        explanation = fixture["explanation"]
        assert set(explanation) >= {"home", "away", "units"}
        assert explanation["home"], "no contributions for the home rate"


def test_contributions_name_their_feature_and_its_value(client, explained):
    fixture = client.get(
        "/fixtures/upcoming?limit=1&include_predictions=true"
    ).json()["fixtures"][0]
    for contribution in fixture["explanation"]["home"]:
        assert contribution["feature"]
        assert "contribution" in contribution
        # The log link is why this is carried: +0.31 is x1.36 on the rate, not
        # +0.31 goals, and a frontend that renders it as goals would be wrong by
        # however far the fixture sits from its competition's baseline.
        assert contribution["multiplier"] > 0


def test_the_units_are_stated_rather_than_assumed(client, explained):
    fixture = client.get(
        "/fixtures/upcoming?limit=1&include_predictions=true"
    ).json()["fixtures"][0]
    assert "log goal-rate" in fixture["explanation"]["units"]


def test_no_explanation_is_attached_to_an_elo_fallback_prediction(client, explained):
    """An explanation of the fitted rates beside an Elo number describes a
    prediction nobody made."""
    artifacts = predictor().artifacts
    saved = artifacts.predicted_rates
    artifacts.predicted_rates = {}
    try:
        body = client.get("/fixtures/upcoming?limit=3&include_predictions=true").json()
        for fixture in body["fixtures"]:
            assert fixture["prediction_source"] == "elo-poisson"
            assert "explanation" not in fixture
    finally:
        artifacts.predicted_rates = saved
