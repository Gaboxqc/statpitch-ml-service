# StatPitch API

Reference for every endpoint: what to send, what comes back, and what the fields
mean. Written for the API that consumes StatPitch and stores its output.

**Base URL** — your Render service, e.g. `https://statpitch-api.onrender.com`.
**Interactive schema** — `/docs` (Swagger UI) and `/openapi.json` for codegen.

---

## Contents

1. [Before you start](#1-before-you-start)
2. [Fields on every response](#2-fields-on-every-response)
3. [Refusals](#3-refusals)
4. [Reading a fixture](#4-reading-a-fixture)
5. [Prediction endpoints](#5-prediction-endpoints)
6. [Reference endpoints](#6-reference-endpoints)
7. [Decision Layer endpoints](#7-decision-layer-endpoints)
8. [Evidence endpoints](#8-evidence-endpoints)
9. [Recommended sync pattern](#9-recommended-sync-pattern)
10. [Error responses](#10-error-responses)

---

## 1. Before you start

Four things that shape everything below.

**StatPitch stores nothing.** It is a stateless prediction service. It has no
database, no user accounts and no memory between requests. Your API is what
persists results and serves the frontend.

**It never returns a bet.** `/best-bet`, `/card/today` and `/value-bets/today`
deliberately return nothing, and say why. This is not a feature that is missing —
it is a measured result. The model does not beat the closing line, so staking is
disabled. See [Refusals](#3-refusals).

**The free instance sleeps.** After ~15 minutes idle, the next request takes tens
of seconds instead of ~3 ms. Size your client timeouts for 60 seconds, and sync on
a schedule rather than on a user's request. See
[Recommended sync pattern](#9-recommended-sync-pattern).

**Twelve competitions, five with odds.**

| competition_id | name | odds |
|---|---|---|
| `ENG.PL` | Premier League | ✅ |
| `ESP.LALIGA` | La Liga | ✅ |
| `GER.BUNDESLIGA` | Bundesliga | ✅ |
| `ITA.SERIEA` | Serie A | ✅ |
| `FRA.LIGUE1` | Ligue 1 | ✅ |
| `ENG.FA_CUP` | FA Cup | ❌ |
| `ESP.COPA_DEL_REY` | Copa del Rey | ❌ |
| `GER.DFB_POKAL` | DFB-Pokal | ❌ |
| `ITA.COPPA_ITALIA` | Coppa Italia | ❌ |
| `FRA.COUPE_DE_FRANCE` | Coupe de France | ❌ |
| `UEFA.UCL` | Champions League | ❌ |
| `UEFA.UEL` | Europa League | ❌ |

`odds_coverage: false` means no free odds source exists for that competition, so
it gets predictions but never a bet comparison. Always present on the response —
never inferred by your client from a competition list.

---

## 2. Fields on every response

Every endpoint returns these four. Store them next to whatever else you keep.

| field | type | meaning |
|---|---|---|
| `model_version` | string | which model produced the numbers |
| `config_version` | string | which decision config was in force |
| `schema_version` | int | response shape version; bumps only on a breaking change |
| `generated_at` | string | ISO-8601 UTC, when this response was built |

### Why `model_version` matters to you

The model is retrained on a schedule. Two predictions for the same fixture, stored
a month apart, can come from different models — and nothing else in the payload
would tell you.

- Store `model_version` on **every row**.
- Treat predictions as **immutable**: a new model writes new rows, it does not
  update old ones.
- Poll `/health` to learn the current version without diffing predictions.

Values look like `goals-20260813-bb07c99e` (the fitted model) or `elo-poisson-2.0.0.dev0`
(the fallback — see [Reading a fixture](#4-reading-a-fixture)).

---

## 3. Refusals

When StatPitch declines to answer, it says so in a structured way so you can store
the *reason*, not just a message.

```json
{
  "best_bet": null,
  "reason": "No selection is recommended. The fitted market-shrinkage weight w is 0.000 …",
  "refusal": {
    "available": false,
    "reason_code": "MAX_EDGE_SELECTION_HARMFUL",
    "reason": "…the same sentence…",
    "measurement": {
      "w": 0.0,
      "w_confidence_interval": [0.0, 0.1],
      "n_validation_matches": 5306,
      "best_bet_per_match_roi": -0.0212
    }
  }
}
```

`reason` (prose) and `refusal` (structured) always carry the same statement. The
prose is for display, the code is for grouping and alerting, `measurement` is the
evidence.

| reason_code | meaning |
|---|---|
| `NO_ODDS_COVERAGE` | no free odds source for this competition |
| `DECISION_CONFIG_UNFITTED` | staking parameters are placeholders; no stake can be sized |
| `SHRINKAGE_WEIGHT_ZERO` | the model adds nothing over the closing line |
| `MAX_EDGE_SELECTION_HARMFUL` | picking the biggest edge per match measured worse than not |
| `EMPTY_LEDGER` | nothing settled to resample |
| `NO_FIXTURE_SOURCE` | the fixture artifact is not loaded |

**Render the reason, not an empty state.** "No bets today" throws away the most
useful thing here. "No selection: the model shows no edge over the closing line
(w = 0.000 across 5,306 matches)" is the honest version, and it is already written
for you.

Two refusal keys exist because a response can refuse at two levels:

- `refusal` — the whole endpoint declined.
- `bet_recommendation_refusal` — the prediction is fine, only the *betting* part
  declined. Appears on `/predict` and `/markets`.

---

## 4. Reading a fixture

Fixtures returned by `/fixtures/upcoming` and `/today` carry two fields that tell
you how much to trust the prediction.

**`prediction_source`**

| value | meaning |
|---|---|
| `fitted_goal_model` | the trained model, using precomputed rolling-form features |
| `elo-poisson` | fallback: goal rates derived from Elo ratings alone |

The fallback is measurably weaker (+0.0064 log-loss). It appears for fixtures that
were not in the last precompute run — typically a newly added fixture. Worth
surfacing, or at least storing.

**`fully_rated`** (inside `prediction`)

`false` means at least one club had no measured Elo rating and fell back to a
prior. The prediction is still a well-formed number, and it is a much weaker claim.
This exists because it once did not: 187 clubs known only as cup entrants silently
defaulted to the same rating, and two fourth-tier sides came back as equals of each
other and of the club hosting them. Check it before showing a confident number.

`prediction.ratings.home.source` tells you which tier of evidence was used:
`club_elo`, `entrant_prior`, `pooled_prior` or `default`.

---

## 5. Prediction endpoints

### `GET /fixtures/upcoming`

The main endpoint. Returns a page of scheduled fixtures — designed to be called
once per matchday, not once per fixture.

**Query parameters**

| name | type | default | notes |
|---|---|---|---|
| `from` | ISO date | — | inclusive lower bound |
| `to` | ISO date | — | inclusive upper bound |
| `competition_id` | string | — | 404 if unknown |
| `limit` | int 1–1000 | `200` | |
| `offset` | int ≥ 0 | `0` | |
| `include_predictions` | bool | `false` | attaches full prediction + explanation |

```
GET /fixtures/upcoming?from=2026-08-21&to=2026-08-24&include_predictions=true&limit=100
```

**Response**

```json
{
  "fixtures": [ /* see below */ ],
  "count": 12,
  "total": 655,
  "offset": 0,
  "limit": 100,
  "generated_at_source": "2026-08-13T10:27:00+00:00"
}
```

`generated_at_source` is when the **fixture list was built**, which is not when the
response was generated. A fixture list is a claim about the future and kickoff
times move, so its age is part of the answer. Rebuilt weekly; if it is more than a
week or two old, treat kickoff times as provisional.

**Each fixture**

```json
{
  "fixture_id": "ESP.LALIGA|2026-2027|Club Atlético de Madrid|Málaga CF",
  "competition_id": "ESP.LALIGA",
  "season": "2026-2027",
  "stage": "matchday_1",
  "format": "round_robin",
  "date": "2026-08-16",
  "home_team": "Club Atlético de Madrid",
  "away_team": "Málaga CF",
  "neutral_venue": false,
  "odds_coverage": true,
  "prediction_source": "fitted_goal_model",
  "prediction_model_version": "goals-20260813-bb07c99e",
  "prediction": { /* the prediction object, §5.2 */ },
  "explanation": { /* §5.3 */ }
}
```

`fixture_id` **excludes the date on purpose**, so a postponed match keeps its
identity rather than appearing as a new fixture plus a vanished one. Use it as your
primary key. `date` is an attribute that can change.

`prediction`, `prediction_source`, `prediction_model_version` and `explanation`
appear only when `include_predictions=true`.

---

### `GET /today`

Today's fixtures, with predictions always attached. Same fixture shape as above.

```json
{
  "date": "2026-08-21",
  "fixtures": [ ... ],
  "note": "7 fixture(s) in scope today.",
  "generated_at_source": "2026-08-13T10:27:00+00:00"
}
```

**`fixtures: []` means no football today.** If the fixture artifact is missing you
get a `refusal` with `NO_FIXTURE_SOURCE` instead. Do not treat those as the same
thing — one is a quiet Tuesday, the other is a broken deploy.

---

### `GET /predict/{competition_id}/{home}/{away}`

One fixture, computed live. Use this for an ad-hoc lookup; use `/fixtures/upcoming`
for anything scheduled, since that path serves the stronger fitted model.

**Path**: `competition_id`, `home`, `away` (URL-encode club names).
**Query**: `stage`, `season`, `home_entry_stage`, `away_entry_stage` — all optional.

`*_entry_stage` is the round a club **entered** a cup, not the round being played.
It is only consulted for a club with no measured rating, and is deliberately not
inferred from `stage`: a round-1 entrant that wins three ties is still a round-1
calibre club in round 4.

**Response**

```json
{
  "competition_id": "ENG.PL",
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "format": "round_robin",
  "stage": null,
  "neutral_venue": false,
  "probabilities": { "home": 0.717, "draw": 0.174, "away": 0.109 },
  "expected_goals": { "home": 2.28, "away": 0.76 },
  "over_under": { "over_1.5": 0.82, "over_2.5": 0.60, "over_3.5": 0.37 },
  "btts": 0.48,
  "correct_scores": [ { "home": 2, "away": 0, "probability": 0.13 } ],
  "ratings": {
    "home": { "elo": 2063.76, "source": "club_elo" },
    "away": { "elo": 1841.20, "source": "club_elo" }
  },
  "fully_rated": true,
  "odds_coverage": true,
  "bet_recommendation": null,
  "bet_recommendation_reason": "decision_config '…' is unfitted …",
  "bet_recommendation_refusal": { "reason_code": "DECISION_CONFIG_UNFITTED", … },
  "disclaimer": "Simulation and analysis only. …"
}
```

`probabilities` sums to 1. `correct_scores` is the top 10 scorelines. Every market
here is derived from one score matrix, so they are mutually consistent by
construction.

---

### `POST /predict`

Same output as the GET, with fields the URL cannot express.

```json
{
  "competition_id": "UEFA.UCL",
  "home_team": "Arsenal",
  "away_team": "Real Madrid",
  "stage": "quarter_final",
  "season": "2026-2027",
  "neutral": false,
  "first_leg_home_goals": 2,
  "first_leg_away_goals": 1,
  "home_entry_stage": null,
  "away_entry_stage": null
}
```

Only `competition_id`, `home_team` and `away_team` are required.

---

### `GET /predict/tie/{competition_id}/{team_a}/{team_b}`

Two-legged aggregate qualification. Adds a `tie` object with qualification
probabilities. Returns **400** if the competition does not play two-legged ties at
that stage.

**Query**: `season`, `first_leg_home_goals`, `first_leg_away_goals`.

---

### `GET /markets/{competition_id}/{home}/{away}`

All **86 selections** derived from the same score matrix.

**Query**: `stage` (optional).

```json
{
  "selections": [
    {
      "key": "1x2_home",
      "family": "1x2",
      "line": null,
      "probability": 0.5231,
      "fair_odds": 1.9117,
      "stakeable": true,
      "payoff": { "win": 1.0, "half_win": 0.0, "push": 0.0, "half_loss": 0.0, "loss": -1.0 }
    }
  ],
  "count": 86
}
```

Families: `1x2`, `double_chance`, `draw_no_bet`, `totals`, `team_totals`,
`asian_handicap`, `btts`, `correct_score`.

`fair_odds` is `1 / probability` — a **no-vig** price, not one you can bet. A real
bookmaker price includes margin and will always be shorter. `payoff` matters for
Asian lines, where a result can push or half-win.

---

### `GET /simulate/{competition_id}`

Monte Carlo over a knockout bracket.

**Query**: `teams` (required, comma-separated, count must be a power of two),
`runs` (100–50,000, default 10,000).

```
GET /simulate/UEFA.UCL?teams=Arsenal,Real%20Madrid,Bayern,Inter&runs=10000
```

```json
{
  "draw_type": "fixed",
  "runs": 10000,
  "rounds": ["semi_final", "final"],
  "teams": [
    { "team": "Real Madrid", "win": 0.342, "reach": { "semi_final": 1.0, "final": 0.58 } }
  ]
}
```

Sorted by win probability.

---

### Explanations

When `include_predictions=true`, each fixture carries an `explanation` showing
what drove each side's goal rate.

```json
{
  "units": "Contributions are additive in log goal-rate and multiplicative on goals…",
  "home": [
    { "feature": "elo_diff", "feature_value": 259.05, "contribution": 0.287, "multiplier": 1.333 },
    { "feature": "home_rest_days", "feature_value": 30.0, "contribution": -0.030, "multiplier": 0.970 },
    { "feature": "other", "feature_value": null, "contribution": 0.041, "multiplier": 1.042 }
  ],
  "away": [ … ]
}
```

**Read `multiplier`, not `contribution`, unless you know what you are doing.**
The model works on a log link, so `contribution: +0.287` means the goal rate is
multiplied by **1.333** — a 33% increase. It does **not** mean "+0.287 goals".

- Ranked by absolute impact; negative contributions are kept, since a feature
  arguing *against* a rate is part of the explanation.
- `other` is the summed remainder of features outside the top 6, so the parts still
  reconstruct the whole. Do not drop it from a chart's total.
- The baseline is the competition's own goal environment, so this reads as "this
  fixture is 1.33× its league's normal rate".
- Only present when `prediction_source` is `fitted_goal_model`.

---

## 6. Reference endpoints

### `GET /health`

Poll before a sync batch.

```json
{
  "status": "ok",
  "ready": true,
  "artifacts_loaded": true,
  "staking_enabled": false,
  "clubs_rated": 456,
  "club_name_aliases": 477,
  "entrant_prior_buckets": 7,
  "decision_config": "dec-2026.08.0-placeholder",
  "model_version": "elo-poisson-2.0.0.dev0"
}
```

`status` is `ok` or `starting`. **A failure to load artifacts returns `ready: false`
with an `error` field, not a 500** — so your job can distinguish "still booting"
from "broken" and retry appropriately.

### `GET /`

Service name, version, competition count, disclaimer.

### `GET /competitions`

All twelve, each with `competition_id`, `name`, `country`, `type`, `format`,
`tier`, `odds_coverage`. Fetch once and cache.

### `GET /teams/{competition_id}`

Every rated club with its current Elo, sorted strongest first.

```json
{ "competition_id": "ENG.PL", "teams": [ { "team": "Arsenal", "elo": 2063.8 } ] }
```

Note: this returns the **whole rated population** — 456 clubs across every
competition — not just that competition's current members. `competition_id` is
validated (404 if unknown) but does not filter the list.

---

## 7. Decision Layer endpoints

All of these currently refuse. Each refusal cites its measurement, and that is the
content worth displaying.

### `GET /best-bet/{competition_id}/{home}/{away}`

Always `best_bet: null`, with `MAX_EDGE_SELECTION_HARMFUL`. Two independent
findings keep it closed: the model shows no edge over the closing line, **and**
picking the largest edge per match measured −2.12% ROI against +0.13% for
committing to a single market. Ranking on model-versus-market disagreement selects
the model's own largest errors.

### `GET /card/today`

`bets: []`, `total_exposure: 0.0`, `DECISION_CONFIG_UNFITTED`. Staking parameters
have never been fitted, and a stake sized from placeholders is indistinguishable
from a real one.

### `GET /value-bets/today`

`value_bets: []`, `SHRINKAGE_WEIGHT_ZERO`, with the full `w` measurement attached.

### `GET /bankroll/simulate`

**Query**: `lambda` (0–1, default 0.25) — the Kelly fraction.

Returns `EMPTY_LEDGER`: resampling an empty track record is a simulation of
nothing.

---

## 8. Evidence endpoints

These always return real data and are what a docs or methodology page should read
from.

### `GET /backtest/{competition_id}`

```json
{
  "available": true,
  "window": "2019-2020 to 2023-2024",
  "holdout_excluded": "2024-2025",
  "model": { "log_loss": 0.9845, "ece": 0.00317 },
  "market": { "log_loss": 0.9698, "ece": 0.01012 },
  "gap": 0.0147,
  "note": "Model trails the de-vigged closing consensus. Reported as measured…"
}
```

Returns `available: false` with `NO_ODDS_COVERAGE` for cups. Lower log-loss is
better, so a positive `gap` means the model trails the market.

### `GET /edge-map`

Measured returns by market, and the headline finding:

```json
{
  "findings": {
    "market_shrinkage_w": 0.0,
    "w_confidence_interval": [0.0, 0.1],
    "model_vs_market_log_loss_gap": 0.0147,
    "measured": {
      "1x2": { "roi": -0.0449, "t": -1.99, "n": 7482 },
      "over_under_2.5": { "roi": -0.0162, "t": -1.36, "n": 9037 },
      "asian_handicap": { "roi": 0.0046, "t": 0.48, "n": 9143 },
      "sharp_reference_clv": { "clv": 0.0051, "t": 3.47, "n": 4929 }
    }
  }
}
```

### `GET /clv/report`

Closing-line value from the bet ledger — the project's headline metric, because it
resolves an edge on roughly an eighth of the sample ROI needs.

`verdict` is the field to display: positive ROI with negative CLV is reported as
the **absence** of demonstrated edge, not as success.

### `GET /ledger`

**Query**: `limit` (1–1000, default 100), `offset` (default 0).

Paginated, append-only, including suppressed bets and why they were suppressed.
Currently empty.

---

## 9. Recommended sync pattern

The service sleeps after ~15 minutes idle. Never call it from a user's request path.

```
1. Scheduled job (nightly, plus a pre-matchday pass)
2. GET /health          — wait for ready:true, note model_version
3. GET /fixtures/upcoming?from=…&to=…&include_predictions=true&limit=200
   … page with offset until offset+count >= total
4. Upsert on (fixture_id, model_version)
5. Serve your frontend from your own database
```

**Client settings**

- Timeout **60 s**, not 5 — the first call after idle pays the cold start.
- Retry with backoff on timeout; the second call is usually milliseconds.
- One batch call per matchday, not one per fixture.

**When `model_version` changes**, a retrain has been promoted. Insert new rows;
do not overwrite the old ones, or your track record becomes uninterpretable.

---

## 10. Error responses

| status | when | body |
|---|---|---|
| `200` | success, **including every refusal** | see [Refusals](#3-refusals) |
| `400` | invalid request — e.g. a tie for a competition that has no two-legged ties | `{"detail": "…"}` |
| `404` | unknown `competition_id` | `{"detail": "unknown competition 'XX'. See /competitions for the 12 in scope."}` |
| `422` | parameter failed validation — e.g. `runs` out of range | FastAPI validation detail |

**A refusal is a 200, not an error.** "No bet recommended" is a successful,
well-formed answer to the question asked. Only malformed requests get 4xx.

---

## Compatibility

Existing fields are never renamed, removed or retyped. New capability arrives as
new keys or new routes, so **ignore unknown fields** rather than validating
strictly against a closed schema. `schema_version` bumps only on a breaking change;
it is `1` today.
