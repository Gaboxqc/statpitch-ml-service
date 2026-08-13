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
from typing import Any

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
        return {
            "bet_recommendation": None,
            "bet_recommendation_reason": NO_ODDS_REASON,
            "bet_recommendation_refusal": contract.refusal_object(
                ReasonCode.NO_ODDS_COVERAGE,
                NO_ODDS_REASON,
                competition_id=competition.competition_id,
                odds_coverage=False,
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
    return {"bet_recommendation": None, "disclaimer": DISCLAIMER}


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
    """v1 shape preserved exactly (Design §7.1).

    Richer graded output lives at /card/today so this route's contract is
    untouched and the existing frontend keeps working.
    """
    config = decision_config.config()
    if config.is_placeholder:
        note = (
            "No value bets are flagged. The fitted market-shrinkage weight w is "
            "0.000, meaning the model adds nothing over the closing line, and the "
            "decision config is unfitted."
        )
        structured = refusal(
            ReasonCode.SHRINKAGE_WEIGHT_ZERO,
            note,
            config_version=config.config_version,
            **contract.W_MEASUREMENT,
        )
    else:
        note = "No value bets flagged for today."
        structured = {}

    return {
        "date": None,
        "value_bets": [],
        "note": note,
        **structured,
        "disclaimer": DISCLAIMER,
        **provenance(),
    }


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


@app.get("/card/today")
def card_today() -> dict:
    """Graded matchday slate with correlation-aware sizing (FR-27)."""
    config = decision_config.config()
    if config.is_placeholder:
        reason = (
            f"Staking is disabled: decision_config '{config.config_version}' is "
            f"unfitted (status={config.status})."
        )
        structured = refusal(
            ReasonCode.DECISION_CONFIG_UNFITTED,
            reason,
            config_version=config.config_version,
            status=config.status,
            w_fitted=config.w_fitted,
        )
    else:
        reason = "No qualifying bets on today's slate."
        structured = {}

    return {
        "date": None,
        "bets": [],
        "total_exposure": 0.0,
        "reason": reason,
        **structured,
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


@app.get("/today")
def today() -> dict:
    """Today's fixtures across all in-scope competitions, with predictions.

    Still a stub. The refusal is structured so a consumer's sync job can tell
    "no fixture source wired" apart from "no fixtures today" — an empty list
    means the latter, and only the latter. Roadmap §7 closes this.
    """
    note = (
        "Live fixture listing requires a fixture source, which is not yet wired. "
        "The API-Football feed is quota-budgeted to 100 requests/day (NFR-9) and "
        "not exercised here."
    )
    return {
        "date": None,
        "fixtures": [],
        "note": note,
        **refusal(ReasonCode.NO_FIXTURE_SOURCE, note, fixture_source=None),
        **provenance(),
    }


def market_families() -> list[str]:
    return [str(f) for f in MarketFamily]
