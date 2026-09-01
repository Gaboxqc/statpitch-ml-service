"""FastAPI application (Design §7, NFR-2, NFR-11, NFR-13).

Artifacts load once at startup and every request is arithmetic over them — no
training, no file reads, no network calls on the request path (NFR-2).

Three contracts this module exists to keep
==========================================

**NFR-13, backward compatibility.** Every v1 route keeps its exact path and
parameter order, and no existing response field is renamed, removed or retyped.
New capability lives at new routes; `disclaimer` and `bet_recommendation` are
added as new keys, which a client that ignores unknown keys will never see. The
provenance and `refusal` keys added for the downstream consumer follow the same
rule — additive, alongside the prose they describe rather than in place of it.
See `statpitch.serving.contract` for why response models here allow extra fields:
a filtering `response_model` would remove keys as a side effect of documenting
them, which is the one thing NFR-13 forbids.

**NFR-11, advisory only.** Any response carrying a stake recommendation carries a
disclaimer saying so. Nothing here places a bet or talks to a bookmaker.

**Requirements §9, honest scoping.** A competition with `odds_coverage=false`
returns `bet_recommendation: null` **with a stated reason** rather than omitting
the field. There is no free odds source covering cups, and the API says so per
request rather than leaving it to documentation.

Two things are deliberately unavailable rather than faked. The staking routes
refuse while `decision_config` is a placeholder, because a stake sized from
unfitted parameters looks exactly like a real one. And the CLV report serves an
empty ledger honestly rather than inventing a track record.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from statpitch import __version__, decision_config, paths, taxonomy
from statpitch.decision import clv_tracker as clv
from statpitch.decision.market_engine import MarketFamily
from statpitch.models import bracket as bk
from statpitch.serving import contract
from statpitch.serving.contract import ReasonCode, provenance, refusal
from statpitch.serving.predictor import Artifacts, Predictor

log = logging.getLogger(__name__)

#: NFR-11. Attached to every response carrying a stake recommendation.
DISCLAIMER = (
    "Simulation and analysis only. StatPitch does not place wagers, hold funds, "
    "or integrate with any bookmaker. Stake fractions are advisory output."
)

#: Requirements §9. Returned wherever the Decision Layer is gated off.
NO_ODDS_REASON = (
    "No free odds source covers this competition, so no market benchmark exists. "
    "This is a data-availability limit, not a modelling choice."
)

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load artifacts once, before the first request (NFR-2).

    A lifespan handler rather than the deprecated `on_event`, so a prediction is
    never the thing that pays for loading the model.
    """
    predictor()
    log.info("statpitch: artifacts loaded, ready to serve")
    yield
    _state.clear()


app = FastAPI(
    title="StatPitch v2",
    version=__version__,
    lifespan=lifespan,
    description=(
        "Calibrated club-football prediction and decision layer. "
        "Advisory only — see NFR-11."
    ),
)


def predictor() -> Predictor:
    if "predictor" not in _state:
        _state["predictor"] = Predictor(Artifacts.load())
    return _state["predictor"]


def ledger() -> clv.BetLedger:
    if "ledger" not in _state:
        _state["ledger"] = clv.BetLedger(paths.bet_ledger_file())
    return _state["ledger"]


# --- schemas ------------------------------------------------------------------

class PredictRequest(BaseModel):
    competition_id: str
    home_team: str
    away_team: str
    stage: str | None = None
    season: str | None = None
    neutral: bool | None = None
    first_leg_home_goals: int | None = Field(default=None, ge=0)
    first_leg_away_goals: int | None = Field(default=None, ge=0)
    #: Round each club ENTERED the competition. Only consulted for a club with no
    #: measured rating, and never defaulted from `stage` — a round-1 entrant that
    #: wins three ties is still a round-1 calibre club in round 4.
    home_entry_stage: str | None = None
    away_entry_stage: str | None = None


# Response models declare the fields a downstream consumer may rely on and store.
# They are deliberately partial: `contract.OpenModel` allows extras through, so
# these document and guarantee a core without truncating the fuller payload the
# routes already return. See `contract.OpenModel` for why filtering would be a
# NFR-13 violation delivered as a side effect of adding documentation.

class HealthResponse(contract.OpenModel):
    status: str
    ready: bool
    model_version: str
    config_version: str
    schema_version: int
    generated_at: str
    artifacts_loaded: bool
    staking_enabled: bool


class PredictionResponse(contract.OpenModel):
    competition_id: str
    home_team: str
    away_team: str
    format: str
    probabilities: dict[str, float]
    expected_goals: dict[str, float]
    odds_coverage: bool
    #: False when either club's rating is a prior rather than a measured Elo. Part
    #: of the contract because it once was not: 187 of 428 clubs silently fell
    #: through to a flat 1400 and the probabilities could not express it.
    fully_rated: bool
    model_version: str
    config_version: str
    schema_version: int
    generated_at: str


# --- helpers ------------------------------------------------------------------

def _competition_or_404(competition_id: str):
    try:
        return taxonomy.get(competition_id)
    except taxonomy.TaxonomyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown competition '{competition_id}'. "
                f"See /competitions for the {len(taxonomy.registry())} in scope."
            ),
        ) from None


def _bet_recommendation(competition) -> dict[str, Any]:
    """The odds-coverage gate, applied per request (Requirements §9).

    `bet_recommendation_reason` keeps its exact prose; `bet_recommendation_refusal`
    carries the same fact as data, so a consumer storing the response can group and
    alert on the cause instead of only rendering the sentence.
    """
    if not competition.odds_coverage:
        # Which half is missing changes what would fix it, so the refusal says.
        # A competition that gains a price but no closing-odds history is still
        # unrecommendable, and a gate reading one flag could not tell that apart
        # from full coverage.
        missing = []
        if not competition.live_odds_coverage:
            missing.append("no price can be obtained for an upcoming fixture")
        if not competition.benchmark_coverage:
            missing.append("no historical closing odds exist to validate against")
        reason = f"{NO_ODDS_REASON} Specifically: {'; and '.join(missing)}."
        return {
            "bet_recommendation": None,
            "bet_recommendation_reason": reason,
            "bet_recommendation_refusal": contract.refusal_object(
                ReasonCode.NO_ODDS_COVERAGE,
                reason,
                competition_id=competition.competition_id,
                odds_coverage=False,
                live_odds_coverage=competition.live_odds_coverage,
                benchmark_coverage=competition.benchmark_coverage,
            ),
        }

    config = decision_config.config()
    if config.is_placeholder:
        reason = (
            f"decision_config '{config.config_version}' is unfitted "
            f"(status={config.status}). Staking is disabled until the market "
            "shrinkage weight w is fitted — a stake sized from placeholder "
            "parameters is indistinguishable from a real one."
        )
        return {
            "bet_recommendation": None,
            "bet_recommendation_reason": reason,
            "bet_recommendation_refusal": contract.refusal_object(
                ReasonCode.DECISION_CONFIG_UNFITTED,
                reason,
                config_version=config.config_version,
                status=config.status,
                w_fitted=config.w_fitted,
            ),
            "disclaimer": DISCLAIMER,
        }
    # Staking is enabled, and this route still recommends nothing — deliberately.
    # A bet is chosen by the selection rule across the whole slate, with a
    # per-day cap, so a single fixture asked about in isolation has no
    # recommendation to give: MODEL_CARD §4 measured picking per-fixture at
    # -2.12% ROI. `/bets/today` is where selections come from.
    #
    # `bet_recommendation_reason` is emitted here rather than omitted. It was
    # present on every response while the config was a placeholder, and dropping
    # a field once the config changed would be exactly the removal NFR-13
    # forbids — a consumer reading it would start seeing KeyError on the day
    # staking was enabled.
    rule = config.selection_rule
    reason = (
        "Per-fixture bet recommendations are not issued. Selections are made "
        f"across the slate by the '{rule.reference}' rule (status={rule.status}, "
        f"markets={','.join(rule.market_families) or 'all'}, "
        f"max {rule.max_per_day} per day) — see /bets/today. Ranking a single "
        "fixture's markets in isolation measured -2.12% ROI against +0.13% for "
        "committing to one market (MODEL_CARD §4)."
    )
    return {
        "bet_recommendation": None,
        "bet_recommendation_reason": reason,
        "bet_recommendation_refusal": contract.refusal_object(
            ReasonCode.MAX_EDGE_SELECTION_HARMFUL,
            reason,
            selection_rule_status=rule.status,
            see="/bets/today",
            **contract.MAX_EDGE_MEASUREMENT,
        ),
        "disclaimer": DISCLAIMER,
    }


# --- v1 routes (NFR-13: paths and fields must not change) ---------------------

@app.get("/")
def root() -> dict:
    return {
        "name": "StatPitch",
        "version": __version__,
        "competitions": len(taxonomy.registry()),
        "docs": "/docs",
        "disclaimer": DISCLAIMER,
        **provenance(),
    }


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    """Liveness, readiness and version, in one call.

    The consuming API's sync job polls this before a batch, and the free plan's
    ~15-minute spin-down means it is routinely the call that pays the cold start.
    Two things follow.

    It must distinguish "still starting" from "broken". A failure to load
    artifacts is reported as `ready: false` with the error, not raised as a 500 —
    a job that cannot tell a boot in progress from a broken deploy will either
    give up on a healthy instance or retry a dead one forever.

    And it carries `model_version`, because a retrain (Roadmap §11) changes what
    the same fixture key returns. Polling this is how a downstream store learns to
    segregate its rows without diffing predictions to find out.
    """
    config = decision_config.config()
    base = {
        "decision_config": config.config_version,
        "staking_enabled": not config.is_placeholder,
        **provenance(),
    }
    try:
        artifacts = predictor().artifacts
    except Exception as exc:  # noqa: BLE001 — reported, not swallowed
        log.exception("statpitch: artifacts unavailable")
        return {
            **base,
            "status": "starting",
            "ready": False,
            "artifacts_loaded": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        **base,
        "status": "ok",
        "ready": bool(artifacts.elo),
        "artifacts_loaded": bool(artifacts.elo),
        "clubs_rated": len(artifacts.elo),
        "club_name_aliases": len(artifacts.aliases),
        "entrant_prior_buckets": len(artifacts.entrant_prior),
    }


@app.get("/competitions")
def competitions() -> dict:
    return {
        "competitions": [
            {
                "competition_id": c.competition_id,
                "name": c.name,
                "country": c.country,
                "type": c.competition_type,
                "format": c.format,
                "tier": c.tier,
                "odds_coverage": c.odds_coverage,
                "live_odds_coverage": c.live_odds_coverage,
                "benchmark_coverage": c.benchmark_coverage,
            }
            for c in taxonomy.registry()
        ],
        **provenance(),
    }


@app.get("/teams/{competition_id}")
def teams(competition_id: str) -> dict:
    _competition_or_404(competition_id)
    ratings = predictor().artifacts.elo
    ranked = sorted(ratings.items(), key=lambda kv: -kv[1])
    return {
        "competition_id": competition_id,
        "teams": [{"team": t, "elo": round(e, 1)} for t, e in ranked],
        **provenance(),
    }


@app.get("/predict/{competition_id}/{home}/{away}", response_model=PredictionResponse)
def predict(
    competition_id: str,
    home: str,
    away: str,
    stage: str | None = None,
    season: str | None = None,
    home_entry_stage: str | None = Query(
        None, description="Round the home club ENTERED, not the round played"
    ),
    away_entry_stage: str | None = Query(
        None, description="Round the away club ENTERED, not the round played"
    ),
) -> dict:
    competition = _competition_or_404(competition_id)
    prediction = predictor().predict(
        competition_id, home, away, stage=stage, season=season,
        home_entry_stage=home_entry_stage, away_entry_stage=away_entry_stage,
    )
    return {**prediction.as_dict(), **_bet_recommendation(competition), **provenance()}


@app.post("/predict", response_model=PredictionResponse)
def predict_post(request: PredictRequest) -> dict:
    competition = _competition_or_404(request.competition_id)
    first_leg = None
    if (
        request.first_leg_home_goals is not None
        and request.first_leg_away_goals is not None
    ):
        first_leg = (request.first_leg_home_goals, request.first_leg_away_goals)

    prediction = predictor().predict(
        request.competition_id, request.home_team, request.away_team,
        stage=request.stage, season=request.season, neutral=request.neutral,
        first_leg_score=first_leg,
        home_entry_stage=request.home_entry_stage,
        away_entry_stage=request.away_entry_stage,
    )
    return {**prediction.as_dict(), **_bet_recommendation(competition), **provenance()}


@app.get("/predict/tie/{competition_id}/{team_a}/{team_b}")
def predict_tie(
    competition_id: str,
    team_a: str,
    team_b: str,
    season: str | None = None,
    first_leg_home_goals: int | None = None,
    first_leg_away_goals: int | None = None,
) -> dict:
    """Two-legged aggregate qualification (FR-7)."""
    competition = _competition_or_404(competition_id)
    first_leg = None
    if first_leg_home_goals is not None and first_leg_away_goals is not None:
        first_leg = (first_leg_home_goals, first_leg_away_goals)

    prediction = predictor().predict_tie(
        competition_id, team_a, team_b, season=season, first_leg_score=first_leg
    )
    if prediction.tie is None:
        raise HTTPException(
            status_code=400,
            detail=f"{competition_id} does not play two-legged ties at this stage",
        )
    return {**prediction.as_dict(), **_bet_recommendation(competition), **provenance()}


@app.get("/value-bets/today")
def value_bets_today() -> dict:
    """v1 shape preserved (Design §7.1), now reading the computed card.

    `value_bets` lists what is actually recommended — selections that survived
    grading and were sized — not everything with positive expected value. Under
    `w`=0 those are not the same set and the difference matters: EV is
    `price_edge` alone, and Phase A measured that the best-of-N price it comes
    from is a high-water mark rather than a simultaneously available quote.
    `positive_expected_value` carries the wider count for anyone who wants it.

    `note` keeps its name; the route predates `/card/today` and a consumer reads
    it (NFR-13).
    """
    body = _card_response(
        key="value_bets", placeholder_code=ReasonCode.SHRINKAGE_WEIGHT_ZERO
    )
    body["note"] = body.pop("reason")
    return body


@app.get("/simulate/{competition_id}")
def simulate(
    competition_id: str,
    teams: str = Query(..., description="Comma-separated clubs, a power of two"),
    runs: int = Query(10_000, ge=100, le=50_000),
) -> dict:
    """Bracket simulation from the current stage to the final (FR-20)."""
    competition = _competition_or_404(competition_id)
    field = [t.strip() for t in teams.split(",") if t.strip()]

    two_leg = competition.competition_type == "continental_cup"
    draw = (
        bk.DrawType.FIXED
        if competition.competition_type == "continental_cup"
        else bk.DrawType.RANDOM
    )
    try:
        bracket = bk.Bracket(
            teams=field,
            rounds=bk.knockout_rounds(len(field), two_leg_until_final=two_leg),
            draw_type=draw,
        )
    except bk.BracketError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    engine = predictor()

    def provide(home: str, away: str, neutral: bool):
        return engine.predict(
            competition_id, home, away, neutral=neutral
        ).matrix

    result = bk.simulate(bracket, provide, runs=runs)
    return {
        "competition_id": competition_id,
        "draw_type": str(draw),
        "runs": result.runs,
        "rounds": result.rounds,
        "teams": [
            {"team": t, "win": round(p, 4), "reach": {
                r: round(result.reach[t][r], 4) for r in result.rounds
            }}
            for t, p in result.ranked()
        ],
        **provenance(),
    }


# --- Decision Layer routes (new in v2) ----------------------------------------

@app.get("/markets/{competition_id}/{home}/{away}")
def markets(competition_id: str, home: str, away: str,
            stage: str | None = None) -> dict:
    """Every priced selection derived from the score matrix (FR-23)."""
    competition = _competition_or_404(competition_id)
    prediction = predictor().predict(competition_id, home, away, stage=stage)
    book = prediction.markets()

    return {
        "competition_id": competition_id,
        "home_team": home,
        "away_team": away,
        "selections": [
            {
                "key": s.key,
                "family": str(s.family),
                "line": s.line,
                "probability": round(s.probability, 6),
                "fair_odds": round(1.0 / s.probability, 4) if s.probability > 0 else None,
                "stakeable": s.stakeable,
                "payoff": {
                    "win": round(s.payoff.win, 6),
                    "half_win": round(s.payoff.half_win, 6),
                    "push": round(s.payoff.push, 6),
                    "half_loss": round(s.payoff.half_loss, 6),
                    "loss": round(s.payoff.loss, 6),
                },
            }
            for s in book.selections
        ],
        "count": len(book),
        **_bet_recommendation(competition),
        **provenance(),
    }


@app.get("/best-bet/{competition_id}/{home}/{away}")
def best_bet(competition_id: str, home: str, away: str) -> dict:
    """Highest-ranked selection for a fixture (FR-24).

    Returns no bet, and says why. The fitted w is zero, so the model adds nothing
    over the market; and ranking a fixture's markets to pick a single best bet
    measured WORSE than committing to one market in advance (-2.12% against
    +0.13%), because maximum-edge selection finds the model's own largest errors.
    """
    competition = _competition_or_404(competition_id)
    reason = (
        "No selection is recommended. The fitted market-shrinkage weight w is "
        "0.000 over 5,306 validation matches, so the model demonstrates no "
        "information beyond the closing line. Separately, best-bet-per-match "
        "selection was measured at -2.12% ROI against +0.13% for committing to "
        "a single market, because ranking on model-versus-market disagreement "
        "selects the model's largest errors."
    )
    return {
        "competition_id": competition_id,
        "home_team": home,
        "away_team": away,
        "best_bet": None,
        "reason": reason,
        # Two independent findings refuse this route, and a consumer that stores
        # only one of them loses the reason the endpoint would stay closed even if
        # the other were overturned.
        **refusal(
            ReasonCode.MAX_EDGE_SELECTION_HARMFUL,
            reason,
            **contract.W_MEASUREMENT,
            **contract.MAX_EDGE_MEASUREMENT,
        ),
        **_bet_recommendation(competition),
        **provenance(),
    }


#: Returned when the card artifact is absent entirely. Distinct from a computed
#: card in which nothing qualified — the first is missing work, the second is a
#: finding, and before Plan §4 Phase B this route could not tell them apart
#: because it returned a hardcoded empty list either way.
NO_CARD_REASON = (
    "No card artifact is loaded. The matchday card is built offline by "
    "scripts/build_card.py and read at startup, because deriving 86 selections "
    "per fixture and solving a joint Kelly allocation is far outside NFR-2's "
    "latency budget."
)


def _card_today() -> tuple[pd.DataFrame | None, str]:
    """Today's card rows, and the ISO date they were selected for."""
    artifacts = predictor().artifacts
    today = datetime.now(UTC).date()
    if artifacts.card is None:
        return None, today.isoformat()
    card = artifacts.card
    if card.empty:
        return card, today.isoformat()
    return card[card["date"].dt.date == today], today.isoformat()


def _why_the_card_is_empty(card: pd.DataFrame | None) -> dict[str, Any]:
    """Which of several different things "nothing today" actually means.

    `assessed: 0` conflated four states, and they want different responses:
    the card was never built, it was built and every fixture on it has since
    been played, there is genuinely no football today, or today's fixtures exist
    but carry no price.

    The last one is the common case and the least obvious. football-data.co.uk's
    `fixtures.csv` is **not** a rolling week-ahead feed — it publishes one
    matchday block and then holds it, already-played, until the next block goes
    up. So on a midweek day the price feed can be entirely stale while the
    fixture list is perfectly current.
    """
    today = datetime.now(UTC).date()
    detail: dict[str, Any] = {"card_dates": [], "fixtures_today": 0}

    fixtures = predictor().artifacts.fixtures
    if fixtures is not None and not fixtures.empty:
        detail["fixtures_today"] = int((fixtures["date"].dt.date == today).sum())

    if card is None:
        detail["cause"] = "no_card_artifact"
        return detail
    if card.empty:
        detail["cause"] = "card_is_empty"
        return detail

    dates = sorted({d.date() for d in card["date"]})
    detail["card_dates"] = [d.isoformat() for d in dates]
    detail["card_covers_today"] = today in dates

    if today in dates:
        detail["cause"] = "assessed_but_nothing_qualified"
    elif dates and max(dates) < today:
        detail["cause"] = "priced_fixtures_all_played"
        detail["note"] = (
            "Every fixture the card was built from has been played. The price "
            "feed publishes one matchday block at a time rather than a rolling "
            "week, so between blocks there is nothing upcoming to price."
        )
    elif detail["fixtures_today"] == 0:
        detail["cause"] = "no_fixtures_today"
    else:
        detail["cause"] = "fixtures_today_carry_no_price"
        detail["note"] = (
            f"{detail['fixtures_today']} fixture(s) today, none of them priced. "
            "The price feed has not published this matchday block yet."
        )
    return detail


def _card_row(record: dict) -> dict:
    """One graded selection, in the shape a downstream store can key on."""
    q_fair = record.get("q_fair")
    return {
        "fixture_id": record.get("fixture_id"),
        "competition_id": record.get("competition_id"),
        "home_team": record.get("home_team"),
        "away_team": record.get("away_team"),
        "selection": record.get("selection_key"),
        "market_family": record.get("market_family"),
        "line": record.get("line"),
        "description": record.get("description"),
        # The two market numbers, kept apart exactly as FR-16a requires: fair
        # probability is de-vigged from the consensus, the price is the best quote.
        "q_fair": q_fair,
        "fair_odds": round(1.0 / q_fair, 4) if q_fair else None,
        "consensus_odds": record.get("odds_avg"),
        "odds": record.get("odds_max"),
        "p_model": record.get("p_model"),
        "p_used": record.get("p_used"),
        # Decomposed, and never summed into one "edge" — the whole point of
        # FR-16a is that a consumer can see which half is doing the work.
        "edge_prob": record.get("edge_prob"),
        "expected_value": record.get("expected_value"),
        "price_edge": record.get("price_edge"),
        "model_edge": record.get("model_edge"),
        # The rule's own quantity: the best quote against the SHARP book's fair
        # value, which is what MODEL_CARD 5's finding is defined on. Distinct
        # from `price_edge`, which is measured against the consensus and which
        # Phase C showed to be mean reversion.
        "reference_odds": record.get("reference_odds"),
        "rule_edge": record.get("rule_edge"),
        "rule_qualified": bool(record.get("rule_qualified")),
        "grade": record.get("grade"),
        "composite": record.get("composite"),
        "stake_fraction": record.get("stake_fraction"),
        "reasons": [r for r in str(record.get("reasons") or "").split("; ") if r],
    }


def _card_response(*, key: str, placeholder_code: ReasonCode) -> dict:
    """Shared body for the two slate routes.

    `key` is the list's name, which differs between them and must not change:
    `/value-bets/today` predates `/card/today` and a consumer already reads
    `value_bets` (NFR-13).

    `placeholder_code` differs for the same reason. Both codes are true while the
    config is a placeholder — `w` really is 0.000 *and* the config really is
    unfitted — and each route has been emitting its own since v1. A consumer
    branches on that code, so which one a given route returns is as much a part
    of the contract as the path is, and sharing this helper must not quietly
    unify them.
    """
    config = decision_config.config()
    rows, date = _card_today()

    if rows is None:
        return {
            "date": date, key: [], "total_exposure": 0.0,
            "reason": NO_CARD_REASON,
            **refusal(
                ReasonCode.NO_CARD_SOURCE, NO_CARD_REASON,
                artifact="data/processed/card.parquet", loaded=False,
            ),
            "assessed": 0,
            "disclaimer": DISCLAIMER, **provenance(),
        }

    records = rows.to_dict("records")
    staked = [r for r in records if float(r.get("stake_fraction") or 0.0) > 0.0]
    grades: dict[str, int] = {}
    for record in records:
        grade = str(record.get("grade") or "?")
        grades[grade] = grades.get(grade, 0) + 1
    positive_ev = sum(1 for r in records if float(r.get("expected_value") or 0.0) > 0)

    if config.is_placeholder and not records:
        # Naming the config gate here would be true and misleading. With nothing
        # assessed there is nothing to size, fitted or not, so the proximate
        # cause is the absence of priced fixtures — and a reader told "staking is
        # disabled" goes and looks at the config, which is not the problem.
        detail = _why_the_card_is_empty(predictor().artifacts.card)
        reason = (
            f"Nothing to assess: {detail.get('note') or detail['cause']}. "
            f"Staking is also disabled (decision_config "
            f"'{config.config_version}' is unfitted), so nothing would be sized "
            "even if there were."
        )
        binding = detail["cause"]
        # The reason_code stays DECISION_CONFIG_UNFITTED: the config really is
        # unfitted, a consumer branches on that code, and it has meant this since
        # v1. The prose and `binding_constraint` carry the proximate cause, which
        # is additive rather than a re-typing of the existing field (NFR-13).
        structured = refusal(
            placeholder_code, reason,
            config_version=config.config_version,
            status=config.status,
            w_fitted=config.w_fitted,
            selections_assessed=0,
            binding_constraint=binding,
            **contract.W_MEASUREMENT,
            **contract.SELECTION_RULE_MEASUREMENT,
        )
    elif config.is_placeholder:
        # Now the config gate IS what is binding: selections exist and graded,
        # and StakingEngine refuses to size from unfitted parameters.
        reason = (
            f"Staking is disabled: decision_config '{config.config_version}' is "
            f"unfitted (status={config.status}). {len(records)} selection(s) were "
            "assessed and graded; none can be sized until the config is fitted."
        )
        binding = "decision_config_unfitted"
        structured = refusal(
            placeholder_code, reason,
            config_version=config.config_version,
            status=config.status,
            w_fitted=config.w_fitted,
            selections_assessed=len(records),
            # Two independent gates, both shut. `w`=0 removes the model's
            # contribution; the selection study removes the price rule's, because
            # the only reference with a multi-season result is one the live feed
            # does not publish. A consumer needs both to chart the cause.
            **contract.W_MEASUREMENT,
            **contract.SELECTION_RULE_MEASUREMENT,
        )
    elif records and not staked:
        binding = "nothing_reached_the_cutoff"
        reason = (
            f"{len(records)} selection(s) were assessed and none reached the "
            f"staking cutoff. Grades: "
            f"{', '.join(f'{g}={n}' for g, n in sorted(grades.items()))}."
        )
        structured = refusal(
            ReasonCode.NO_QUALIFYING_SELECTION, reason,
            selections_assessed=len(records),
            positive_expected_value=positive_ev,
            grades=grades,
        )
    elif not records:
        # Staking is on and there is simply nothing to stake: today's fixtures
        # carry no price. Not a refusal — no gate is closed — but the cause has
        # to be named, or `binding_constraint: null` beside an empty slate reads
        # as "no reason given".
        #
        # It happens for real. On 2026-09-01 the only fixture was Hamburg
        # Eimsbütteler BC, a fifth-tier side, against Borussia Dortmund, and no
        # bookmaker in the panel quoted it.
        detail = _why_the_card_is_empty(predictor().artifacts.card)
        binding = detail["cause"]
        reason = f"No fixtures priced for today: {detail.get('note') or binding}."
        structured = {}
    else:
        binding = None
        reason = f"{len(staked)} selection(s) recommended."
        structured = {}

    body_extra: dict[str, Any] = {"binding_constraint": binding}
    if not staked:
        body_extra["empty_because"] = _why_the_card_is_empty(
            predictor().artifacts.card
        )

    return {
        "date": date,
        key: [_card_row(r) for r in staked],
        "total_exposure": round(
            sum(float(r.get("stake_fraction") or 0.0) for r in staked), 6
        ),
        "reason": reason,
        **structured,
        # Additive, and the reason this route is now worth calling even when it
        # recommends nothing: a consumer can see the card was computed, how the
        # grades fell out, and how much of the apparent value was price rather
        # than model.
        "assessed": len(records),
        "positive_expected_value": positive_ev,
        "grades": grades,
        "card_generated_at": predictor().artifacts.card_generated_at,
        **body_extra,
        "disclaimer": DISCLAIMER,
        **provenance(),
    }


@app.get("/card/today")
def card_today() -> dict:
    """Graded matchday slate with correlation-aware sizing (FR-27).

    Reads the precomputed card rather than returning a fixed empty list. The
    slate may still be empty — with `w`=0.000 and an unfitted config it will be —
    but the response now distinguishes *computed and nothing qualified* from
    *never built*, and carries the assessment behind either.
    """
    return _card_response(
        key="bets", placeholder_code=ReasonCode.DECISION_CONFIG_UNFITTED
    )


@app.get("/bets/today")
def bets_today() -> dict:
    """Today's recommendations, and why each one was selected.

    A selection reaches this route only by clearing the rule in
    `decision_config.selection_rule` — the best available quote beating a sharp
    book's de-vigged fair value — and then surviving grading and sizing.

    Two things about that rule are worth stating on the route that serves it.

    **It is measured, not chosen.** MODEL_CARD 5: +0.51% Friday-to-close CLV at
    t=+7.53 clustered, over five pre-break seasons and 7,790 matches. Until The
    Odds API was wired in there was no live source publishing the reference book
    it is defined on, so it could be measured backwards and never run forwards.

    **It is restricted to 1X2 on purpose.** MODEL_CARD 4 measured picking the
    largest apparent edge ACROSS markets at -2.12% ROI against +0.13% for
    committing to one market in advance, because maximum-edge selection finds
    the model's own largest errors. Ranking a daily pick over all 86 selections
    would be exactly that.

    Empty is a valid answer. If nothing clears the threshold today, this returns
    nothing rather than promoting the least-bad selection — a day with no
    qualifying price is a day with no bet.
    """
    config = decision_config.config()
    rows, date = _card_today()
    rule = config.selection_rule

    if rows is None:
        return {
            "date": date, "bets": [], "count": 0, "total_exposure": 0.0,
            "reason": NO_CARD_REASON,
            **refusal(
                ReasonCode.NO_CARD_SOURCE, NO_CARD_REASON,
                artifact="data/processed/card.parquet", loaded=False,
            ),
            "disclaimer": DISCLAIMER, **provenance(),
        }

    records = rows.to_dict("records")
    staked = sorted(
        (r for r in records if float(r.get("stake_fraction") or 0.0) > 0.0),
        key=lambda r: -float(r.get("rule_edge") or 0.0),
    )
    qualified = [r for r in records if r.get("rule_qualified")]

    body = {
        "date": date,
        "bets": [_card_row(r) for r in staked],
        "count": len(staked),
        "total_exposure": round(
            sum(float(r.get("stake_fraction") or 0.0) for r in staked), 6
        ),
        "assessed": len(records),
        "qualified_by_rule": len(qualified),
        "selection_rule": {
            "status": rule.status,
            "reference": rule.reference,
            "threshold": rule.threshold,
            "market_families": list(rule.market_families),
            "max_per_day": rule.max_per_day,
            "evidence": rule.evidence,
        },
        "config_status": config.status,
        "disclaimer": DISCLAIMER,
    }

    if rule.status != "fitted":
        # Loud, on every response that carries a bet. `experimental` means the
        # rule has multi-season evidence and the price panel it now runs on does
        # not, and a consumer storing these rows needs that stored with them.
        body["caveat"] = (
            f"selection_rule.status={rule.status}. The rule has five seasons of "
            "measured CLV; the 25-book panel it now runs on has none, so its "
            "calibration is inherited rather than re-measured. Treat these as a "
            "live test of a measured rule on a new panel, not a validated edge."
        )
        body["refusal"] = contract.refusal_object(
            ReasonCode.SELECTION_RULE_EXPERIMENTAL,
            body["caveat"],
            **contract.SELECTION_RULE_MEASUREMENT,
        )

    if not staked:
        body["reason"] = (
            f"No selection cleared the rule today. {len(records)} assessed, "
            f"{len(qualified)} qualified, none sized."
            if records else
            f"Nothing to assess: {_why_the_card_is_empty(rows)['cause']}."
        )
        body["empty_because"] = _why_the_card_is_empty(
            predictor().artifacts.card
        )
    else:
        body["reason"] = f"{len(staked)} selection(s) recommended."

    body.update(provenance())
    return body


@app.get("/odds/matchday")
def odds_matchday(
    date: str | None = Query(None, description="ISO date. Defaults to today."),
    competition_id: str | None = None,
) -> dict:
    """Every priced market for a day's fixtures, grouped by fixture.

    `/card/assessments` returns a flat list of selections; this is the same
    prices arranged the way a matchday is actually read — one entry per fixture,
    each carrying its 1X2, totals and handicap quotes together.

    Which markets are present is a function of what was captured. The daily sweep
    asks for 1X2 across every competition, and adds totals and handicaps only for
    competitions playing that day, because the price API bills per market per
    competition and asking for all three everywhere every day does not fit inside
    the free allowance.
    """
    artifacts = predictor().artifacts
    when = datetime.now(UTC).date()
    if date is not None:
        try:
            when = pd.Timestamp(date).date()
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"unparseable date '{date}'"
            ) from None

    if artifacts.card is None:
        return {
            "date": when.isoformat(), "fixtures": [], "count": 0,
            "reason": NO_CARD_REASON,
            **refusal(
                ReasonCode.NO_CARD_SOURCE, NO_CARD_REASON,
                artifact="data/processed/card.parquet", loaded=False,
            ),
            "disclaimer": DISCLAIMER, **provenance(),
        }

    frame = artifacts.card
    frame = frame[frame["date"].dt.date == when] if not frame.empty else frame
    if competition_id is not None:
        _competition_or_404(competition_id)
        frame = frame[frame["competition_id"] == competition_id]

    fixtures = []
    for fixture_id, group in frame.groupby("fixture_id", sort=False):
        first = group.iloc[0]
        markets: dict[str, list[dict]] = {}
        for record in group.to_dict("records"):
            markets.setdefault(str(record.get("market_family")), []).append(
                _card_row(record)
            )
        fixtures.append({
            "fixture_id": fixture_id,
            "competition_id": first["competition_id"],
            "home_team": first["home_team"],
            "away_team": first["away_team"],
            "kickoff_utc": (
                None if pd.isna(first.get("kickoff_utc"))
                else str(first["kickoff_utc"])
            ),
            "markets": markets,
            "markets_priced": sorted(markets),
            "recommended": [
                _card_row(r) for r in group.to_dict("records")
                if float(r.get("stake_fraction") or 0.0) > 0.0
            ],
        })

    return {
        "date": when.isoformat(),
        "fixtures": fixtures,
        "count": len(fixtures),
        "selections": int(len(frame)),
        "note": (
            "1X2 is captured for every competition daily; totals and handicaps "
            "only for competitions playing that day, because the price API bills "
            "per market per competition."
        ),
        "card_generated_at": artifacts.card_generated_at,
        "disclaimer": DISCLAIMER,
        **provenance(),
    }


@app.get("/card/upcoming")
def card_upcoming(
    days: int = Query(
        14, ge=1, le=60,
        description="How far ahead to include. The price feed rarely reaches "
                    "beyond the next matchday block, so a longer window is "
                    "harmless rather than useful.",
    ),
) -> dict:
    """The whole priced slate ahead, not only the fixtures kicking off today.

    `/card/today` filters to the current date, which is right for a "what is on
    now" view and wrong for almost every other purpose. The price feed publishes
    a matchday block a few days before it is played, so on the Thursday before a
    Saturday round `/card/today` returns nothing while a full assessed slate is
    sitting in the card. Answering "nothing" there is technically accurate and
    reads exactly like a broken service.

    Same shape as `/card/today`, same refusal, same disclaimer — a wider window.
    """
    artifacts = predictor().artifacts
    today = datetime.now(UTC).date()
    horizon = today + timedelta(days=days)

    if artifacts.card is None:
        return {
            "from": today.isoformat(), "to": horizon.isoformat(),
            "bets": [], "assessments": [], "total_exposure": 0.0,
            "reason": NO_CARD_REASON,
            **refusal(
                ReasonCode.NO_CARD_SOURCE, NO_CARD_REASON,
                artifact="data/processed/card.parquet", loaded=False,
            ),
            "assessed": 0, "disclaimer": DISCLAIMER, **provenance(),
        }

    card = artifacts.card
    window = card[
        (card["date"].dt.date >= today) & (card["date"].dt.date <= horizon)
    ] if not card.empty else card

    records = window.to_dict("records")
    staked = [r for r in records if float(r.get("stake_fraction") or 0.0) > 0.0]
    grades: dict[str, int] = {}
    for record in records:
        grades[str(record.get("grade") or "?")] = (
            grades.get(str(record.get("grade") or "?"), 0) + 1
        )

    return {
        "from": today.isoformat(),
        "to": horizon.isoformat(),
        "bets": [_card_row(r) for r in staked],
        # The assessed slate travels with it. A caller asking what is coming up
        # wants the analysis, and making them issue a second request to
        # /card/assessments to get it is the same omission this route exists to
        # correct, one level down.
        "assessments": [_card_row(r) for r in records],
        "assessed": len(records),
        "total_exposure": round(
            sum(float(r.get("stake_fraction") or 0.0) for r in staked), 6
        ),
        "grades": grades,
        "dates_covered": sorted({str(d.date()) for d in window["date"]})
        if not window.empty else [],
        "reason": (
            f"{len(staked)} selection(s) recommended." if staked
            else f"{len(records)} selection(s) assessed, none recommended."
        ),
        "note": (
            "`assessments` are priced and graded, not recommendations. "
            "stake_fraction says which is which."
        ),
        "card_generated_at": artifacts.card_generated_at,
        "disclaimer": DISCLAIMER,
        **provenance(),
    }


@app.get("/card/assessments")
def card_assessments(
    date: str | None = Query(
        None, description="ISO date. Defaults to every date the card covers."
    ),
    competition_id: str | None = None,
    graded: str | None = Query(
        None, description="Filter to one grade, e.g. `A`. Default returns all."
    ),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
) -> dict:
    """Every assessed selection, priced and graded — recommendation or not.

    This exists because the analysis was invisible. `/card/today` returns only
    selections that were *staked*, so with staking gated off it returned an empty
    list and a count, and a consumer could not see the odds, the de-vigged fair
    probability, the edge decomposition or the grade — all of which are computed
    for every selection and written to `card.parquet` on every run.

    An empty slate is the correct recommendation. It is not a reason to hide the
    work behind it, and "we assessed 126 selections and none qualified" is only
    checkable if the 126 are available.

    **Nothing here is a recommendation.** `stake_fraction` is on every row and is
    the field that says so; under the current configuration it is 0.0 everywhere.
    The prices are real, the fair probabilities are real, and the grades are real
    — the reason not to bet them is in `/card/today`'s refusal, with the
    measurement behind it.
    """
    artifacts = predictor().artifacts
    if artifacts.card is None:
        return {
            "assessments": [], "count": 0, "total": 0,
            "reason": NO_CARD_REASON,
            **refusal(
                ReasonCode.NO_CARD_SOURCE, NO_CARD_REASON,
                artifact="data/processed/card.parquet", loaded=False,
            ),
            "disclaimer": DISCLAIMER, **provenance(),
        }

    frame = artifacts.card
    if date is not None:
        try:
            wanted = pd.Timestamp(date).date()
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"unparseable date '{date}'"
            ) from None
        frame = frame[frame["date"].dt.date == wanted]
    if competition_id is not None:
        _competition_or_404(competition_id)
        frame = frame[frame["competition_id"] == competition_id]
    if graded is not None:
        frame = frame[frame["grade"].str.upper() == graded.upper()]

    total = int(len(frame))
    window = frame.sort_values(
        ["date", "competition_id", "fixture_id", "selection_key"]
    ).iloc[offset : offset + limit]
    rows = [_card_row(record) for record in window.to_dict("records")]

    return {
        "assessments": rows,
        "count": len(rows),
        "total": total,
        "dates_covered": sorted({str(d.date()) for d in frame["date"]}),
        "grades": {
            str(g): int(n) for g, n in frame["grade"].value_counts().items()
        },
        # Said in the payload rather than only in the docs, because this route
        # returns priced selections and a reader could otherwise mistake the list
        # for a slate.
        "note": (
            "Assessed selections, not recommendations. stake_fraction is 0.0 on "
            "every row while staking is gated; see /card/today for the refusal "
            "and the measurement behind it."
        ),
        "card_generated_at": artifacts.card_generated_at,
        "disclaimer": DISCLAIMER,
        **provenance(),
    }


@app.get("/clv/report")
def clv_report() -> dict:
    """Live CLV track record from the ledger (FR-26) — the headline number."""
    entries = ledger().entries
    report = clv.report(entries)
    return {
        "label": report.label,
        "n": report.n,
        "mean_clv": round(report.mean_clv, 6),
        "clv_se": round(report.clv_se, 6),
        "clv_t": round(report.clv_t, 4),
        "positive_rate": round(report.positive_rate, 4),
        "mean_roi": round(report.mean_roi, 6),
        "roi_t": round(report.roi_t, 4),
        "verdict": report.verdict(),
        "by_competition": {
            k: {"n": v.n, "mean_clv": round(v.mean_clv, 6)}
            for k, v in clv.report_by(entries, "competition_id").items()
        },
        "disclaimer": DISCLAIMER,
        **provenance(),
    }


@app.get("/ledger")
def ledger_route(
    limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)
) -> dict:
    """Paginated bet ledger, including suppressed bets and their reasons (FR-29)."""
    entries = ledger().entries
    window = entries[offset : offset + limit]
    return {
        "total": len(entries),
        "offset": offset,
        "limit": limit,
        "entries": [
            {
                "ts_flagged": e.ts_flagged,
                "fixture_id": e.fixture_id,
                "selection": e.selection,
                "grade": e.grade,
                "odds_taken": e.odds_taken,
                "price_source": e.price_source,
                "stake_fraction": e.stake_fraction,
                "clv_pct": e.clv_pct,
                "result": e.result,
                "config_version": e.config_version,
            }
            for e in window
        ],
        "disclaimer": DISCLAIMER,
        **provenance(),
    }


@app.get("/edge-map")
def edge_map() -> dict:
    """Where measured edge lives, by competition and market (FR-31)."""
    return {
        "label": clv.CLV_LABEL,
        "cells": [],
        "findings": {
            "market_shrinkage_w": 0.0,
            "w_confidence_interval": [0.0, 0.100],
            "model_vs_market_log_loss_gap": 0.0147,
            "measured": {
                "1x2": {"roi": -0.0449, "t": -1.99, "n": 7482},
                "over_under_2.5": {"roi": -0.0162, "t": -1.36, "n": 9037},
                "asian_handicap": {"roi": 0.0046, "t": 0.48, "n": 9143},
                "sharp_reference_clv": {"clv": 0.0051, "t": 3.47, "n": 4929},
            },
            "summary": (
                "The model shows no edge over the closing line on any market. The "
                "only statistically significant signal is closing-line value from "
                "using a sharp book as reference, which requires a live sharp price "
                "feed and is therefore outside the zero-cost constraint."
            ),
        },
        "disclaimer": DISCLAIMER,
        **provenance(),
    }


@app.get("/bankroll/simulate")
def bankroll_simulate(
    kelly_lambda: float = Query(0.25, alias="lambda", gt=0.0, le=1.0)
) -> dict:
    """Drawdown distribution and risk of ruin (FR-30)."""
    config = decision_config.config()
    settled = ledger().settled()
    if config.is_placeholder or not settled:
        reason = (
            "No settled bets in the ledger to resample. A bankroll simulation "
            "over an empty track record would be a simulation of nothing."
        )
        return {
            "lambda": kelly_lambda,
            "paths": 0,
            "reason": reason,
            **refusal(
                ReasonCode.EMPTY_LEDGER
                if not settled
                else ReasonCode.DECISION_CONFIG_UNFITTED,
                reason,
                settled_bets=len(settled),
                config_version=config.config_version,
                staking_enabled=not config.is_placeholder,
            ),
            "disclaimer": DISCLAIMER,
            **provenance(),
        }
    return {
        "lambda": kelly_lambda, "paths": 0, "disclaimer": DISCLAIMER, **provenance(),
    }


@app.get("/backtest/{competition_id}")
def backtest(competition_id: str) -> dict:
    """Brier / log-loss / RPS / ECE against closing odds (FR-13, FR-14, FR-16b)."""
    competition = _competition_or_404(competition_id)
    if not competition.odds_coverage:
        return {
            "competition_id": competition_id,
            "available": False,
            "reason": NO_ODDS_REASON,
            **refusal(
                ReasonCode.NO_ODDS_COVERAGE,
                NO_ODDS_REASON,
                competition_id=competition_id,
                odds_coverage=False,
            ),
            **provenance(),
        }
    return {
        "competition_id": competition_id,
        "available": True,
        "window": "2019-2020 to 2023-2024",
        "holdout_excluded": "2024-2025",
        "model": {"log_loss": 0.9845, "ece": 0.00317},
        "market": {"log_loss": 0.9698, "ece": 0.01012},
        "gap": 0.0147,
        "note": (
            "Model trails the de-vigged closing consensus. Reported as measured "
            "(NFR-3), including where it does not beat the market."
        ),
        **provenance(),
    }


#: Returned when the fixture artifact is absent entirely. Distinct from an empty
#: list, which means the source is present and there is genuinely nothing on.
NO_FIXTURES_REASON = (
    "No fixture artifact is loaded. Upcoming fixtures are built offline by "
    "scripts/build_fixtures.py and read at startup, because NFR-2 forbids a "
    "network call on a request path."
)


def _fixture_refusal() -> dict[str, Any]:
    return refusal(
        ReasonCode.NO_FIXTURE_SOURCE,
        NO_FIXTURES_REASON,
        artifact="data/processed/fixtures.parquet",
        loaded=False,
    )


def _fixture_rows(frame, *, with_predictions: bool) -> list[dict]:
    """Serialise fixture rows, optionally attaching a prediction to each.

    A prediction that raises is reported on the fixture rather than propagated: a
    single unrecognised club name must not empty an entire matchday. The failure
    is visible per fixture, which is the only place it can be acted on.
    """
    engine = predictor()
    rows: list[dict] = []
    for record in frame.to_dict("records"):
        date = record.get("date")
        row = {
            "fixture_id": record.get("fixture_id"),
            "competition_id": record.get("competition_id"),
            "season": record.get("season"),
            "stage": record.get("stage"),
            "format": record.get("format"),
            "date": None if date is None or pd.isna(date) else str(date.date()),
            "kickoff": record.get("kickoff") or None,
            # False when the schedule published no kickoff time, which is how a
            # provisional matchday date announces itself. See §7 of docs/API.md.
            "date_confirmed": bool(record.get("date_confirmed", False)),
            "home_team": record.get("home_team"),
            "away_team": record.get("away_team"),
            "neutral_venue": bool(record.get("neutral_venue", False)),
            "odds_coverage": bool(record.get("odds_coverage", False)),
        }
        if with_predictions:
            # Precomputed rates come from the fitted goal model; the Elo mapping
            # is the fallback for a fixture that was never precomputed. They
            # differ by +0.0064 log-loss (MODEL_CARD §3), so which one answered
            # is part of the answer rather than an implementation detail.
            artifacts = engine.artifacts
            rates = artifacts.predicted_rates.get(str(record.get("fixture_id")))
            try:
                prediction = engine.predict(
                    str(record["competition_id"]),
                    str(record["home_team"]),
                    str(record["away_team"]),
                    stage=record.get("stage"),
                    season=record.get("season"),
                    rates=rates,
                )
            except Exception as exc:  # noqa: BLE001 — surfaced, not swallowed
                log.warning(
                    "fixture %s: prediction failed: %s", record.get("fixture_id"), exc
                )
                row["prediction"] = None
                row["prediction_error"] = f"{type(exc).__name__}: {exc}"
            else:
                row["prediction"] = prediction.as_dict()
                row["prediction_source"] = (
                    "fitted_goal_model" if rates is not None else contract.SERVED_MODEL
                )
                row["prediction_model_version"] = (
                    artifacts.predictions_model_version
                    if rates is not None
                    else contract.model_version()
                )
                # FR-32. Only attached where the prediction came from the model
                # that produced them: an explanation of the fitted rates next to
                # an Elo-fallback number would describe a prediction nobody made.
                explanation = artifacts.explanations.get(str(record.get("fixture_id")))
                if explanation is not None and rates is not None:
                    row["explanation"] = {
                        "units": (
                            "Contributions are additive in log goal-rate and "
                            "multiplicative on goals: +0.31 multiplies the rate "
                            "by e^0.31. The base is the competition's own goal "
                            "environment, so these describe how far this fixture "
                            "departs from it."
                        ),
                        **explanation,
                    }
        rows.append(row)
    return rows


@app.get("/fixtures/upcoming")
def fixtures_upcoming(
    from_date: str | None = Query(
        None, alias="from",
        description="ISO date, inclusive. Defaults to today — past fixtures are "
                    "excluded unless you ask for them.",
    ),
    to_date: str | None = Query(None, alias="to", description="ISO date, inclusive"),
    competition_id: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    include_predictions: bool = Query(
        False, description="Attach a full prediction to each fixture"
    ),
) -> dict:
    """A whole round in one round-trip (Roadmap §8.4).

    The consuming API syncs from here on a schedule, so this is built to be
    called once per matchday rather than once per fixture: hundreds of requests
    into a free instance is the failure mode this endpoint exists to prevent.

    `generated_at_source` is when the fixture artifact was built, which is not
    when this response was generated. A fixture list is a claim about the future
    and kickoff times move, so its age belongs in the payload rather than being
    inferred from the fact that a response arrived.
    """
    artifacts = predictor().artifacts
    if artifacts.fixtures is None:
        return {
            "fixtures": [], "count": 0, "total": 0,
            "note": NO_FIXTURES_REASON,
            **_fixture_refusal(),
            **provenance(),
        }

    frame = artifacts.fixtures
    if competition_id is not None:
        _competition_or_404(competition_id)
        frame = frame[frame["competition_id"] == competition_id]

    # Default to today rather than to the start of the artifact. The list is
    # filtered to the future when it is *built*, so without this the window of
    # already-played fixtures grows every day the artifact ages, and a consumer
    # syncing "upcoming" keeps re-ingesting last week.
    lower = pd.Timestamp(from_date) if from_date else pd.Timestamp(
        datetime.now(UTC).date()
    )
    frame = frame[frame["date"] >= lower]
    if to_date is not None:
        frame = frame[frame["date"] <= pd.Timestamp(to_date)]

    total = len(frame)
    window = frame.sort_values(["date", "competition_id"]).iloc[offset : offset + limit]
    return {
        "fixtures": _fixture_rows(window, with_predictions=include_predictions),
        "count": len(window),
        "total": total,
        "offset": offset,
        "limit": limit,
        "from": str(lower.date()),
        "generated_at_source": artifacts.fixtures_generated_at,
        **provenance(),
    }


@app.get("/today")
def today() -> dict:
    """Today's fixtures across all in-scope competitions, with predictions.

    v1 shape preserved (NFR-13): `date`, `fixtures`, `note`. What changed is that
    `fixtures` can now be non-empty.

    An empty list here means there is genuinely no football today. The absence of
    the artifact is a *refusal* instead, carrying NO_FIXTURE_SOURCE — a consumer
    that cannot tell those apart will record a quiet Tuesday and a broken deploy
    as the same thing.
    """
    artifacts = predictor().artifacts
    date = datetime.now(UTC).date()
    if artifacts.fixtures is None:
        return {
            "date": None, "fixtures": [], "note": NO_FIXTURES_REASON,
            **_fixture_refusal(), **provenance(),
        }

    frame = artifacts.fixtures
    todays = frame[frame["date"] == pd.Timestamp(date)]
    return {
        "date": str(date),
        "fixtures": _fixture_rows(todays, with_predictions=True),
        "note": (
            f"{len(todays)} fixture(s) in scope today."
            if len(todays)
            else "No fixtures scheduled today in the twelve competitions in scope."
        ),
        "generated_at_source": artifacts.fixtures_generated_at,
        **provenance(),
    }


def market_families() -> list[str]:
    return [str(f) for f in MarketFamily]
