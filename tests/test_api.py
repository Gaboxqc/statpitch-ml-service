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


def test_a_league_fixture_reports_the_config_state_in_force(client):
    """`/predict` never recommends a bet for a single fixture on its own — that
    is `/bets/today`'s job, under the selection rule. What it must do is say
    which state the config is in rather than going quiet."""
    body = client.get("/predict/ENG.PL/Arsenal/Chelsea").json()
    assert "bet_recommendation" in body
    assert body["bet_recommendation_reason"]


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

def test_the_card_refuses_only_while_the_config_is_a_placeholder(client):
    """And carries bets otherwise. Pinning "always refuses" would have made
    enabling staking look like a contract break rather than the intended change."""
    from statpitch import decision_config

    body = client.get("/card/today").json()
    if decision_config.config().is_placeholder:
        assert body["bets"] == []
        assert body["total_exposure"] == 0.0
        assert "unfitted" in body["reason"]
    else:
        assert isinstance(body["bets"], list)
        assert body["total_exposure"] >= 0.0


def test_health_reports_the_staking_state_that_is_actually_in_force(client):
    """It used to assert False, which was the committed config's state rather
    than a property of the code. What must hold is that /health agrees with the
    config actually loaded — a health check that disagrees is worse than none."""
    from statpitch import decision_config

    reported = client.get("/health").json()["staking_enabled"]
    assert reported is (not decision_config.config().is_placeholder)


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


# --- the card routes now read a computed artifact (Plan §4 Phase B) -----------

def test_card_today_reports_that_it_computed_something(client):
    """An empty slate must be distinguishable from unwritten code.

    Before Phase B this route returned a hardcoded `[]` whatever the state of the
    world, so "nothing qualified" and "nobody wired it up" were the same JSON.
    """
    body = client.get("/card/today").json()
    assert "assessed" in body
    assert "grades" in body
    if "refusal" in body:
        assert body["refusal"]["reason_code"] in {
            "DECISION_CONFIG_UNFITTED", "NO_QUALIFYING_SELECTION", "NO_CARD_SOURCE",
        }
    elif body["bets"]:
        pass
    else:
        # Staking on, nothing priced today. Not a refusal — no gate is closed —
        # but it must still name the cause rather than going quiet.
        assert body["binding_constraint"], "empty slate with no reason given"
        assert body["empty_because"]["cause"]


def test_card_today_keeps_its_v1_keys(client):
    """NFR-13: `bets`, `total_exposure` and `reason` are already stored downstream."""
    body = client.get("/card/today").json()
    for key in ("date", "bets", "total_exposure", "reason", "disclaimer"):
        assert key in body
    assert isinstance(body["bets"], list)


def test_value_bets_today_keeps_its_v1_keys(client):
    body = client.get("/value-bets/today").json()
    for key in ("date", "value_bets", "note", "disclaimer"):
        assert key in body
    # `reason` is the card route's name for it; this route must not grow one in
    # place of `note`, which a v1 consumer reads.
    assert "reason" not in body


def test_the_two_slate_routes_agree_on_what_is_recommended(client):
    card = client.get("/card/today").json()
    value = client.get("/value-bets/today").json()
    assert card["bets"] == value["value_bets"]
    assert card["assessed"] == value["assessed"]


def test_a_staking_refusal_carries_the_measurement_behind_it(client):
    """The project's rule: a refusal cites the number that caused it."""
    body = client.get("/card/today").json()
    if body.get("refusal", {}).get("reason_code") == "DECISION_CONFIG_UNFITTED":
        measurement = body["refusal"]["measurement"]
        assert measurement["w"] == 0.0
        assert measurement["n_validation_matches"] == 5306
        assert "selections_assessed" in measurement


# --- the assessed card is visible even when nothing is recommended ------------

def test_assessments_return_the_analysis_behind_an_empty_slate(client):
    """An empty slate is the right recommendation; it is not a reason to hide
    the work. "126 assessed, none qualified" is only checkable if the 126 are
    available."""
    body = client.get("/card/assessments?limit=5").json()
    if body.get("total", 0) == 0:
        pytest.skip("no card built in this checkout")
    row = body["assessments"][0]
    for field in ("odds", "consensus_odds", "fair_odds", "q_fair", "p_model",
                  "price_edge", "model_edge", "grade", "stake_fraction"):
        assert field in row, field


def test_assessments_keep_the_two_market_numbers_apart(client):
    """FR-16a: fair probability comes from the consensus, the bet from the best
    quote, and a consumer must be able to see both."""
    body = client.get("/card/assessments?limit=50").json()
    if body.get("total", 0) == 0:
        pytest.skip("no card built in this checkout")
    for row in body["assessments"]:
        if row["consensus_odds"] and row["odds"]:
            assert row["odds"] >= row["consensus_odds"] - 1e-9


def test_assessments_say_they_are_not_recommendations(client):
    body = client.get("/card/assessments?limit=1").json()
    assert "not recommendations" in body["note"]
    assert body["disclaimer"]


def test_assessments_can_be_filtered_by_grade(client):
    body = client.get("/card/assessments?graded=F&limit=5").json()
    if body.get("total", 0) == 0:
        pytest.skip("no card built in this checkout")
    assert all(r["grade"] == "F" for r in body["assessments"])


def test_an_unparseable_date_is_a_400_not_a_silent_empty_list(client):
    assert client.get("/card/assessments?date=not-a-date").status_code == 400


def test_an_empty_slate_says_which_kind_of_empty_it_is(client):
    """Four different states used to collapse into `assessed: 0`.

    Never built, built and all played, no football today, or fixtures today with
    no price — the last is the common one and the least obvious, because the
    price feed publishes one matchday block at a time rather than a rolling week.
    """
    body = client.get("/card/today").json()
    if body["bets"]:
        pytest.skip("something is staked in this checkout")
    cause = body["empty_because"]["cause"]
    assert cause in {
        "no_card_artifact", "card_is_empty", "assessed_but_nothing_qualified",
        "priced_fixtures_all_played", "no_fixtures_today",
        "fixtures_today_carry_no_price",
    }


# --- the upcoming slate, not just today ---------------------------------------

def _fake_card(client, dates):
    """Point the loaded artifacts at a card covering chosen dates."""
    import pandas as pd

    from statpitch.serving.app import predictor

    rows = []
    for i, day in enumerate(dates):
        rows.append({
            "fixture_id": f"ENG.PL|2026-2027|A{i} FC|B{i} FC",
            "competition_id": "ENG.PL", "date": pd.Timestamp(day),
            "kickoff_utc": pd.Timestamp(day), "home_team": f"A{i} FC",
            "away_team": f"B{i} FC", "selection_key": "1x2_home",
            "market_family": "1x2", "line": None, "description": "Home win",
            "p_model": 0.5, "q_fair": 0.45, "p_used": 0.45,
            "odds_avg": 2.1, "fair_odds": 2.22, "odds_max": 2.2,
            "edge_prob": 0.0, "expected_value": -0.01, "price_edge": -0.01,
            "model_edge": 0.0, "grade": "F", "composite": 0.3, "reasons": "",
            "stake_fraction": 0.0, "book_margin": 0.05, "max_book_sum": 1.02,
            "n_books": 7, "capture_id": "T1", "w": 0.0,
            "config_version": "test", "config_status": "placeholder",
            "model_version": "test", "generated_at": "2026-08-25T00:00:00+00:00",
        })
    artifacts = predictor().artifacts
    saved = artifacts.card
    artifacts.card = pd.DataFrame(rows)
    return saved


def test_upcoming_shows_a_slate_that_today_would_hide(client):
    """The Thursday-before-Saturday case.

    The feed publishes a matchday block days before it is played, so filtering to
    the current date returns nothing while a full assessed slate sits in the
    card. Accurate, and indistinguishable from a broken service.
    """
    from datetime import UTC, datetime, timedelta

    from statpitch.serving.app import predictor

    soon = (datetime.now(UTC).date() + timedelta(days=2)).isoformat()
    saved = _fake_card(client, [soon, soon])
    try:
        upcoming = client.get("/card/upcoming").json()
        today = client.get("/card/today").json()
        assert upcoming["assessed"] == 2
        assert today["assessed"] == 0, "the fixtures are not today, by construction"
        assert upcoming["dates_covered"] == [soon]
    finally:
        predictor().artifacts.card = saved


def test_upcoming_excludes_what_has_already_been_played(client):
    from datetime import UTC, datetime, timedelta

    from statpitch.serving.app import predictor

    past = (datetime.now(UTC).date() - timedelta(days=3)).isoformat()
    saved = _fake_card(client, [past])
    try:
        assert client.get("/card/upcoming").json()["assessed"] == 0
    finally:
        predictor().artifacts.card = saved


def test_upcoming_carries_the_analysis_not_only_the_bets(client):
    from datetime import UTC, datetime, timedelta

    from statpitch.serving.app import predictor

    soon = (datetime.now(UTC).date() + timedelta(days=1)).isoformat()
    saved = _fake_card(client, [soon])
    try:
        body = client.get("/card/upcoming").json()
        assert body["bets"] == []
        assert len(body["assessments"]) == 1
        assert body["assessments"][0]["odds"] == 2.2
        assert body["assessments"][0]["consensus_odds"] == 2.1
    finally:
        predictor().artifacts.card = saved


# --- the reason must blame the thing that is actually binding -----------------

def test_an_empty_card_does_not_blame_the_staking_gate(client):
    """With nothing assessed there is nothing to size, fitted or not.

    Telling a reader "staking is disabled" sends them to look at the decision
    config, which is not what is stopping them.
    """
    body = client.get("/card/today").json()
    if body["assessed"]:
        pytest.skip("something is assessed for today in this checkout")
    assert body["binding_constraint"] != "decision_config_unfitted"
    assert body["reason"].startswith(("Nothing to assess", "No fixtures priced"))


def test_the_refusal_code_is_unchanged_by_the_clearer_prose(client):
    """NFR-13: a consumer branches on the code, which has meant this since v1.

    Only asserted when the route is refusing — it carries bets once a selection
    rule is live.
    """
    from statpitch import decision_config

    if not decision_config.config().is_placeholder:
        pytest.skip("the config gate is not what is refusing")
    body = client.get("/card/today").json()
    if "refusal" not in body:
        pytest.skip("a selection rule is live, so /card/today is answering")
    assert body["refusal"]["reason_code"] == "DECISION_CONFIG_UNFITTED"


# --- the daily bet, and the rule behind it ------------------------------------

def test_bets_today_reports_the_rule_it_selected_under(client):
    """A recommendation without its basis is not checkable."""
    body = client.get("/bets/today").json()
    rule = body["selection_rule"]
    assert rule["reference"], "a rule with no reference cannot select"
    assert rule["market_families"], "an unrestricted rule is the -2.12% failure"
    assert "evidence" in rule


def test_the_rule_is_restricted_to_the_family_it_was_measured_on(client):
    """MODEL_CARD §4: max-edge ACROSS markets measured -2.12% ROI against +0.13%
    for committing to one market. A daily pick ranked over all 86 selections
    would be exactly that failure."""
    body = client.get("/bets/today").json()
    assert body["selection_rule"]["market_families"] == ["1x2"]
    for bet in body["bets"]:
        assert bet["market_family"] == "1x2"


def test_every_rule_recommendation_cleared_the_rule(client):
    """Scoped to the rule tier. A confidence pick is surfaced BECAUSE nothing
    cleared the rule, so asserting that it did would contradict its own reason."""
    body = client.get("/bets/today").json()
    for bet in body["bets"]:
        assert bet["stake_fraction"] > 0
        assert bet["selection_basis"] in {"rule", "confidence"}
        if bet["selection_basis"] == "rule":
            assert bet["rule_qualified"] is True
            assert bet["rule_edge"] is not None


def test_the_two_tiers_never_appear_on_the_same_day(client):
    """The fallback fires only when the rule leaves the day empty, so a day
    carrying both would mean it ran when it should not have."""
    body = client.get("/bets/today").json()
    bases = {b["selection_basis"] for b in body["bets"]}
    assert bases != {"rule", "confidence"}


def test_a_confidence_pick_says_what_it_is(client):
    """It is the most LIKELY outcome, not a mispriced one. A consumer rendering
    it as a value bet has been misled by the payload."""
    body = client.get("/bets/today").json()
    if not body.get("by_basis", {}).get("confidence"):
        pytest.skip("no confidence pick today")
    assert "confidence_caveat" in body
    assert "-2.12%" in body["confidence_caveat"]
    for bet in body["bets"]:
        if bet["selection_basis"] == "confidence":
            assert bet["rule_qualified"] is False


def test_a_rule_recommendation_shows_the_sharp_price_it_beat(client):
    """Only the rule tier makes that claim — a confidence pick beat nothing."""
    body = client.get("/bets/today").json()
    for bet in body["bets"]:
        if bet["selection_basis"] != "rule":
            continue
        assert bet["reference_odds"] is not None
        assert bet["odds"] >= bet["reference_odds"]


def test_every_pick_carries_a_price_of_one_kind_or_the_other(client):
    """Market where a book quotes it, model-implied where none does. A pick with
    neither would be a recommendation with no number attached."""
    body = client.get("/bets/today").json()
    for bet in body["bets"]:
        assert bet["pricing"] in {"market", "model"}
        assert bet["odds"] or bet["model_odds"], bet["fixture_id"]


def test_an_experimental_rule_says_so_on_every_response_carrying_a_bet(client):
    """The rule has five seasons; the panel it runs on has none.

    A consumer storing these rows must store that with them.
    """
    body = client.get("/bets/today").json()
    if body["selection_rule"]["status"] == "fitted":
        pytest.skip("rule promoted to fitted in this checkout")
    assert "caveat" in body
    assert body["refusal"]["reason_code"] == "SELECTION_RULE_EXPERIMENTAL"


def test_no_qualifying_price_yields_no_bet_rather_than_the_least_bad(client):
    """A day with nothing clearing the threshold is a day with no bet."""
    body = client.get("/bets/today").json()
    if body["count"] == 0:
        assert body["bets"] == []
        assert "reason" in body


def test_bets_are_ranked_by_the_rule_edge(client):
    body = client.get("/bets/today").json()
    edges = [b["rule_edge"] for b in body["bets"]]
    assert edges == sorted(edges, reverse=True)


# --- matchday odds ------------------------------------------------------------

def test_matchday_groups_prices_by_fixture(client):
    body = client.get("/odds/matchday").json()
    for fixture in body["fixtures"]:
        assert fixture["home_team"] and fixture["away_team"]
        assert fixture["markets_priced"]
        for family, selections in fixture["markets"].items():
            assert selections, family


def test_matchday_carries_every_market_that_was_captured(client):
    """1X2 daily everywhere; totals and handicaps only where a fixture plays."""
    body = client.get("/odds/matchday").json()
    if not body["fixtures"]:
        pytest.skip("no priced fixtures today in this checkout")
    assert any("1x2" in f["markets_priced"] for f in body["fixtures"])


def test_matchday_rejects_an_unparseable_date(client):
    assert client.get("/odds/matchday?date=nonsense").status_code == 400


def test_matchday_can_be_filtered_by_competition(client):
    body = client.get("/odds/matchday?competition_id=ENG.PL").json()
    for fixture in body["fixtures"]:
        assert fixture["competition_id"] == "ENG.PL"
