"""The downstream integration contract (Roadmap §8).

StatPitch owns no database. A separate API consumes it and persists the results,
which makes the response shape the product rather than an implementation detail.
These tests cover the three ways that contract breaks without anyone noticing
until a season of rows has already been stored.

**A response_model that silently truncates.** Documenting a route by attaching a
`response_model` normally *filters* the payload to the declared fields. Every key
a model forgets to declare disappears — a field removal, forbidden by NFR-13,
delivered as a side effect of adding documentation. `contract.OpenModel` allows
extras precisely to prevent this, and that behaviour is asserted here rather than
assumed from a pydantic setting that a future refactor could drop.

**Provenance going missing.** A stored prediction with no `model_version` cannot
be interpreted once the weekly retrain lands: two rows under the same fixture key
may come from different models and nothing distinguishes them.

**Refusals degrading to prose.** This project refuses rather than answering badly,
and every refusal cites its measurement. Stored downstream, an English sentence is
a blob; the structured form is what lets a consumer group by cause and chart the
number the sentence quotes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from statpitch.serving import contract
from statpitch.serving.app import app

PROVENANCE_KEYS = {
    "model_version", "config_version", "schema_version", "generated_at",
}

#: Every route a downstream consumer is expected to store rows from.
CONSUMED_ROUTES = [
    "/", "/health", "/competitions", "/teams/ENG.PL",
    "/predict/ENG.PL/Arsenal/Chelsea", "/markets/ENG.PL/Arsenal/Chelsea",
    "/value-bets/today", "/best-bet/ENG.PL/Arsenal/Chelsea", "/card/today",
    "/clv/report", "/ledger", "/edge-map", "/bankroll/simulate",
    "/backtest/ENG.PL", "/today",
]

#: Routes that decline to answer, and the code each must report.
#: Routes that refuse unconditionally, whatever the config says.
REFUSING_ROUTES = [
    ("/best-bet/ENG.PL/Arsenal/Chelsea", contract.ReasonCode.MAX_EDGE_SELECTION_HARMFUL),
    ("/bankroll/simulate", contract.ReasonCode.EMPTY_LEDGER),
]

#: Routes that refuse only while the decision config is a placeholder. Once a
#: selection rule is live they carry bets instead, so the code they emit is
#: asserted only when a refusal is actually present. The pairing still matters:
#: both codes are true under a placeholder and each route has emitted its own
#: since v1, so sharing a helper must not unify them.
CONDITIONALLY_REFUSING_ROUTES = [
    ("/card/today", contract.ReasonCode.DECISION_CONFIG_UNFITTED),
    ("/value-bets/today", contract.ReasonCode.SHRINKAGE_WEIGHT_ZERO),
]
# `/today` is deliberately absent: once a fixture artifact is built it answers
# rather than refusing. Its refusal path — artifact missing, which must not be
# confused with a day that has no football — is covered in test_fixtures.py.


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --- provenance ---------------------------------------------------------------

@pytest.mark.parametrize("route", CONSUMED_ROUTES)
def test_every_consumed_route_carries_provenance(client, route):
    body = client.get(route).json()
    assert set(body) >= PROVENANCE_KEYS, f"{route} is missing {PROVENANCE_KEYS - set(body)}"


def test_model_version_names_the_served_path_not_the_package(client):
    """The served path is the Elo mapping, not the fitted goal model.

    MODEL_CARD §3 measures a matrix driven by fitted XGBoost rates; the deployed
    predictor derives goal rates from Elo. Reporting the package version here
    would let a consumer believe it had stored the evaluated model's output.
    Roadmap §2 closes that gap, and this string is how the change becomes visible.
    """
    version = client.get("/health").json()["model_version"]
    assert version.startswith(contract.SERVED_MODEL)


def test_post_predict_carries_provenance_too(client):
    body = client.post(
        "/predict",
        json={
            "competition_id": "ENG.PL",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
        },
    ).json()
    assert set(body) >= PROVENANCE_KEYS


# --- the response_model must not truncate (NFR-13) ----------------------------

def test_response_model_does_not_drop_undeclared_fields(client):
    """`PredictionResponse` declares a core; the rest of the payload must survive.

    These four keys are deliberately undeclared, so they only appear if extras
    pass through. If a refactor tightens the model to `extra="forbid"` or
    `"ignore"`, this is the test that fails instead of a consumer's ETL.
    """
    body = client.get("/predict/ENG.PL/Arsenal/Chelsea").json()
    for undeclared in ("over_under", "btts", "correct_scores", "ratings"):
        assert undeclared in body, f"response_model truncated {undeclared!r}"


def test_declared_core_fields_are_present_and_typed(client):
    body = client.get("/predict/ENG.PL/Arsenal/Chelsea").json()
    assert isinstance(body["fully_rated"], bool)
    assert isinstance(body["odds_coverage"], bool)
    assert set(body["probabilities"]) == {"home", "draw", "away"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


def test_openapi_documents_the_typed_routes(client):
    """A consumer codegens its client from this document."""
    schema = client.get("/openapi.json").json()
    health = schema["paths"]["/health"]["get"]["responses"]["200"]
    assert "application/json" in health["content"]
    assert "$ref" in str(health["content"]["application/json"]["schema"])


# --- structured refusals ------------------------------------------------------

@pytest.mark.parametrize("route,code", REFUSING_ROUTES)
def test_refusals_are_machine_readable(client, route, code):
    body = client.get(route).json()
    assert "refusal" in body, f"{route} refuses in prose only"
    assert body["refusal"]["reason_code"] == str(code)
    assert body["refusal"]["available"] is False


@pytest.mark.parametrize("route,code", CONDITIONALLY_REFUSING_ROUTES)
def test_a_conditional_refusal_keeps_its_own_code(client, route, code):
    """Asserted when a refusal is present, not that one always is.

    These routes refused unconditionally while the config was a placeholder.
    They now carry bets when a selection rule is live, and pinning "always
    refuses" would have made enabling staking look like a contract break.
    """
    from statpitch import decision_config

    body = client.get(route).json()
    if "refusal" not in body:
        assert body.get("bets") or body.get("value_bets"), (
            f"{route} neither refuses nor recommends"
        )
        return

    emitted = body["refusal"]["reason_code"]
    assert body["refusal"]["available"] is False

    if decision_config.config().is_placeholder:
        # The config gate is what is stopping it, and each route has emitted its
        # OWN code for that since v1. Sharing a helper must not unify them.
        assert emitted == str(code)
    else:
        # Staking is on, so an empty slate means nothing cleared the cutoff —
        # a different cause with its own code, and pinning the config-gate code
        # here would have made "assessed 42, none qualified" look like a break.
        assert emitted in {str(code), "NO_QUALIFYING_SELECTION"}


@pytest.mark.parametrize("route,_code", REFUSING_ROUTES)
def test_refusal_prose_is_preserved_alongside_the_structure(client, route, _code):
    """NFR-13: the structure is additive. The sentence consumers already read stays."""
    body = client.get(route).json()
    prose = body.get("reason") or body.get("note")
    assert prose, f"{route} lost its human-readable reason"
    assert body["refusal"]["reason"] == prose


def test_best_bet_refusal_carries_both_findings(client):
    """Two independent measurements close this route; a consumer needs both.

    w=0 says the model adds nothing over the market. Max-edge selection measuring
    -2.12% is a separate finding that would keep the route closed even if w moved.
    """
    measurement = client.get("/best-bet/ENG.PL/Arsenal/Chelsea").json()["refusal"][
        "measurement"
    ]
    assert measurement["w"] == 0.0
    assert measurement["n_validation_matches"] == 5306
    assert measurement["best_bet_per_match_roi"] == -0.0212


def test_cup_fixture_refuses_bet_recommendation_with_a_code(client):
    """Requirements §9: no free odds source covers cups, and the API says so."""
    body = client.get("/predict/ENG.FA_CUP/Arsenal/Chelsea").json()
    assert body["bet_recommendation"] is None
    assert (
        body["bet_recommendation_refusal"]["reason_code"]
        == str(contract.ReasonCode.NO_ODDS_COVERAGE)
    )
    # The v1 prose field is untouched.
    assert body["bet_recommendation_reason"]


def test_today_answers_in_the_v1_shape_whether_or_not_it_refuses(client):
    """Both paths keep the keys a v1 consumer reads (NFR-13).

    Whether `/today` lists fixtures or refuses for want of an artifact, `date`,
    `fixtures` and `note` are present. Which of the two happened is carried by
    the presence of `refusal`, not by the shape changing underneath the client.
    """
    body = client.get("/today").json()
    assert {"date", "fixtures", "note"} <= set(body)
    assert isinstance(body["fixtures"], list)


def test_reason_codes_are_unique(client):
    """They are stored downstream, so a duplicate value would merge two causes."""
    values = [c.value for c in contract.ReasonCode]
    assert len(values) == len(set(values))


# --- readiness (Roadmap §9.2) -------------------------------------------------

def test_health_reports_readiness_and_version(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["artifacts_loaded"] is True
    # Whatever the config says — a health check that disagrees with the loaded
    # config is worse than none. It asserted False when that was the committed
    # state rather than a property of the code.
    from statpitch import decision_config

    assert body["staking_enabled"] is (not decision_config.config().is_placeholder)


def test_health_reports_not_ready_instead_of_raising(client, monkeypatch):
    """A sync job must tell "still starting" from "broken".

    Raising a 500 here makes both look identical to a client that only sees a
    failed request, so it either abandons a healthy instance or retries a dead
    one forever.
    """
    from statpitch.serving import app as app_module

    def boom():
        raise RuntimeError("artifacts not on disk")

    monkeypatch.setattr(app_module, "predictor", boom)
    body = client.get("/health").json()
    assert body["ready"] is False
    assert body["status"] == "starting"
    assert "artifacts not on disk" in body["error"]


def test_the_two_slate_routes_keep_their_own_refusal_codes(client):
    """Only meaningful while both are refusing; skipped once they carry bets."""
    """Sharing a helper must not unify two codes a consumer branches on.

    Both are true while the config is a placeholder — `w` is 0.000 *and* the
    config is unfitted — and each route has emitted its own since v1. Which one
    a route returns is as much a part of the contract as its path.
    """
    from statpitch import decision_config

    if not decision_config.config().is_placeholder:
        pytest.skip("the config gate is not what is refusing; codes differ by cause")
    card_body = client.get("/card/today").json()
    value_body = client.get("/value-bets/today").json()
    if "refusal" not in card_body or "refusal" not in value_body:
        pytest.skip("a selection rule is live, so the slate routes are answering")
    assert card_body["refusal"]["reason_code"] == "DECISION_CONFIG_UNFITTED"
    assert value_body["refusal"]["reason_code"] == "SHRINKAGE_WEIGHT_ZERO"


def test_the_bet_recommendation_fields_survive_every_config_state(client):
    """NFR-13. `bet_recommendation_reason` was emitted on every response while
    the config was a placeholder, and dropped out the moment staking was
    enabled — a consumer reading it would have started seeing KeyError on that
    exact day. All three keys are now present in every state."""
    for route in (
        "/predict/ENG.PL/Arsenal/Chelsea",       # covered, staking on
        "/predict/ENG.FA_CUP/Arsenal/Chelsea",   # no odds coverage
    ):
        body = client.get(route).json()
        assert "bet_recommendation" in body
        assert body.get("bet_recommendation_reason"), route
        assert body.get("bet_recommendation_refusal"), route
