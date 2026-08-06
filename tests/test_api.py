"""API tests (Design §7, NFR-2, NFR-11, NFR-13).

The contracts under test are the ones that break silently: v1 routes keeping
their shape, cup fixtures returning a stated reason rather than a bare null, and
staking staying disabled while the config is unfitted.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from statpitch.serving.app import DISCLAIMER, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --- v1 contract (NFR-13) -----------------------------------------------------

V1_ROUTES = [
    "/", "/health", "/competitions", "/teams/ENG.PL",
    "/predict/ENG.PL/Arsenal/Chelsea", "/value-bets/today",
    "/backtest/ENG.PL", "/today",
]


@pytest.mark.parametrize("route", V1_ROUTES)
def test_every_v1_route_still_answers(client, route):
    """No v1 path may be renamed or removed (Design §7.1)."""
    assert client.get(route).status_code == 200


def test_value_bets_keeps_its_v1_shape(client):
    """Richer graded output lives at /card/today so this contract is untouched."""
    body = client.get("/value-bets/today").json()
    assert set(body) >= {"date", "value_bets"}
    assert isinstance(body["value_bets"], list)


def test_prediction_response_carries_the_v1_fields(client):
    body = client.get("/predict/ENG.PL/Arsenal/Chelsea").json()
    for field in ("competition_id", "home_team", "away_team", "probabilities",
                  "expected_goals", "correct_scores"):
        assert field in body


def test_new_keys_are_additive_only(client):
    """A client ignoring unknown keys sees no difference."""
    body = client.get("/predict/ENG.PL/Arsenal/Chelsea").json()
    assert "bet_recommendation" in body          # new key, not a renamed one
    assert body["probabilities"]["home"] > 0     # existing key unchanged


# --- predictions --------------------------------------------------------------

def test_probabilities_sum_to_one(client):
    p = client.get("/predict/ENG.PL/Arsenal/Chelsea").json()["probabilities"]
    assert p["home"] + p["draw"] + p["away"] == pytest.approx(1.0, abs=1e-6)


def test_a_league_fixture_has_no_tie_resolution(client):
    """A league match can be drawn, so there is nothing to resolve."""
    body = client.get("/predict/ENG.PL/Arsenal/Chelsea").json()
    assert body["format"] == "round_robin"
    assert "tie" not in body


def test_a_cup_fixture_resolves_extra_time_and_penalties(client):
    """FR-8: a draw is not a final result in a knockout."""
    body = client.get("/predict/ENG.FA_CUP/Arsenal/Chelsea?stage=round_of_16").json()
    assert body["format"] == "single_leg_knockout"
    assert "tie" in body
    tie = body["tie"]
    assert tie["home_advances"] + tie["away_advances"] == pytest.approx(1.0, abs=1e-6)
    assert tie["reaches_penalties"] > 0


def test_a_two_legged_stage_returns_an_aggregate(client):
    """Copa del Rey semi-finals are two-legged (FR-7)."""
    body = client.get(
        "/predict/ESP.COPA_DEL_REY/Barcelona/Real%20Madrid?stage=semi_final"
    ).json()
    assert body["format"] == "two_leg_knockout"
    assert body["tie"]["home_advances"] + body["tie"]["away_advances"] == pytest.approx(
        1.0, abs=1e-6
    )


def test_format_is_resolved_per_stage_not_per_competition(client):
    """The branch Design §5.3 depends on, and it fails silently if wrong."""
    quarter = client.get(
        "/predict/ESP.COPA_DEL_REY/Barcelona/Sevilla?stage=quarter_final"
    ).json()
    semi = client.get(
        "/predict/ESP.COPA_DEL_REY/Barcelona/Sevilla?stage=semi_final"
    ).json()
    assert quarter["format"] == "single_leg_knockout"
    assert semi["format"] == "two_leg_knockout"


def test_a_neutral_final_removes_home_advantage(client):
    at_home = client.get("/predict/ENG.FA_CUP/Arsenal/Chelsea?stage=round_of_16").json()
    at_wembley = client.get("/predict/ENG.FA_CUP/Arsenal/Chelsea?stage=final").json()
    assert at_wembley["neutral_venue"]
    assert not at_home["neutral_venue"]
    assert at_wembley["probabilities"]["home"] < at_home["probabilities"]["home"]


def test_the_tie_endpoint_conditions_on_a_played_first_leg(client):
    base = "/predict/tie/UEFA.UCL/Arsenal/Chelsea"
    level = client.get(base).json()["tie"]["home_advances"]
    ahead = client.get(
        f"{base}?first_leg_home_goals=3&first_leg_away_goals=0"
    ).json()["tie"]["home_advances"]
    behind = client.get(
        f"{base}?first_leg_home_goals=0&first_leg_away_goals=3"
    ).json()["tie"]["home_advances"]
    assert ahead > level > behind


def test_post_predict_accepts_the_same_fixture(client):
    body = client.post("/predict", json={
        "competition_id": "ENG.PL", "home_team": "Arsenal", "away_team": "Chelsea",
    }).json()
    assert body["probabilities"]["home"] > 0


def test_an_unknown_competition_returns_404_with_guidance(client):
    response = client.get("/predict/NOT.A.LEAGUE/A/B")
    assert response.status_code == 404
    assert "/competitions" in response.json()["detail"]


# --- the odds-coverage gate (Requirements §9) ---------------------------------

def test_a_cup_fixture_states_why_no_bet_is_offered(client):
    """§9 enforced per request, not left to documentation."""
    body = client.get("/predict/ENG.FA_CUP/Arsenal/Chelsea").json()
    assert body["bet_recommendation"] is None
    assert "No free odds source" in body["bet_recommendation_reason"]
    assert body["odds_coverage"] is False


def test_the_field_is_present_rather_than_omitted(client):
    """A missing key is indistinguishable from a client-side bug."""
    body = client.get("/predict/UEFA.UCL/Arsenal/Chelsea").json()
    assert "bet_recommendation" in body
    assert "bet_recommendation_reason" in body


def test_a_league_fixture_reports_the_unfitted_config_instead(client):
    body = client.get("/predict/ENG.PL/Arsenal/Chelsea").json()
    assert body["bet_recommendation"] is None
    assert "unfitted" in body["bet_recommendation_reason"]


def test_backtest_is_unavailable_for_cups_with_a_reason(client):
    body = client.get("/backtest/ENG.FA_CUP").json()
    assert body["available"] is False
    assert "data-availability limit" in body["reason"]


def test_backtest_reports_the_league_gap_honestly(client):
    """NFR-3: report where the model does not beat the market."""
    body = client.get("/backtest/ENG.PL").json()
    assert body["model"]["log_loss"] > body["market"]["log_loss"]
    assert body["gap"] > 0


# --- NFR-11, advisory only ----------------------------------------------------

@pytest.mark.parametrize(
    "route", ["/", "/card/today", "/clv/report", "/ledger", "/edge-map"]
)
def test_stake_bearing_responses_carry_the_disclaimer(client, route):
    assert client.get(route).json()["disclaimer"] == DISCLAIMER


def test_the_disclaimer_says_no_wagers_are_placed(client):
    assert "does not place wagers" in DISCLAIMER


# --- staking stays disabled while unfitted ------------------------------------

def test_the_card_refuses_while_the_config_is_a_placeholder(client):
    body = client.get("/card/today").json()
    assert body["bets"] == []
    assert body["total_exposure"] == 0.0
    assert "unfitted" in body["reason"]


def test_health_reports_staking_as_disabled(client):
    assert client.get("/health").json()["staking_enabled"] is False


def test_best_bet_returns_nothing_and_explains_why(client):
    """Both reasons are measured, not asserted."""
    body = client.get("/best-bet/ENG.PL/Arsenal/Chelsea").json()
    assert body["best_bet"] is None
    assert "0.000" in body["reason"]
    assert "-2.12%" in body["reason"]


# --- markets (FR-23) ----------------------------------------------------------

def test_markets_returns_the_full_book(client):
    body = client.get("/markets/ENG.PL/Arsenal/Chelsea").json()
    assert body["count"] >= 50
    assert len(body["selections"]) == body["count"]


def test_every_selection_carries_a_payoff_distribution(client):
    for s in client.get("/markets/ENG.PL/Arsenal/Chelsea").json()["selections"]:
        payoff = s["payoff"]
        total = sum(payoff[k] for k in
                    ("win", "half_win", "push", "half_loss", "loss"))
        assert total == pytest.approx(1.0, abs=1e-4), s["key"]


def test_correct_score_is_marked_non_stakeable(client):
    selections = client.get("/markets/ENG.PL/Arsenal/Chelsea").json()["selections"]
    scores = [s for s in selections if s["family"] == "correct_score"]
    assert scores and all(not s["stakeable"] for s in scores)


# --- simulation (FR-20) -------------------------------------------------------

def test_bracket_simulation_produces_one_champion(client):
    body = client.get(
        "/simulate/ENG.FA_CUP?teams=Arsenal,Chelsea,Everton,Fulham&runs=500"
    ).json()
    assert sum(t["win"] for t in body["teams"]) == pytest.approx(1.0, abs=1e-6)


def test_a_field_that_is_not_a_power_of_two_is_rejected(client):
    response = client.get("/simulate/ENG.FA_CUP?teams=Arsenal,Chelsea,Everton&runs=100")
    assert response.status_code == 400


def test_continental_competitions_use_a_fixed_bracket(client):
    body = client.get(
        "/simulate/UEFA.UCL?teams=Arsenal,Chelsea,Everton,Fulham&runs=300"
    ).json()
    assert body["draw_type"] == "fixed"


def test_domestic_cups_redraw_at_random(client):
    body = client.get(
        "/simulate/ENG.FA_CUP?teams=Arsenal,Chelsea,Everton,Fulham&runs=300"
    ).json()
    assert body["draw_type"] == "random"


# --- CLV and ledger -----------------------------------------------------------

def test_clv_report_serves_an_empty_ledger_honestly(client):
    body = client.get("/clv/report").json()
    assert body["label"] == "Friday-to-close CLV"
    assert body["n"] == 0
    assert body["verdict"] == "no settled bets"


def test_ledger_paginates(client):
    body = client.get("/ledger?limit=5&offset=0").json()
    assert body["limit"] == 5
    assert isinstance(body["entries"], list)


def test_edge_map_reports_the_measured_findings(client):
    findings = client.get("/edge-map").json()["findings"]
    assert findings["market_shrinkage_w"] == 0.0
    assert findings["measured"]["sharp_reference_clv"]["t"] > 3


def test_bankroll_simulation_refuses_an_empty_track_record(client):
    body = client.get("/bankroll/simulate?lambda=0.25").json()
    assert body["paths"] == 0
    assert "simulation of nothing" in body["reason"]


# --- docs ---------------------------------------------------------------------

def test_openapi_docs_are_served(client):
    assert client.get("/openapi.json").status_code == 200
