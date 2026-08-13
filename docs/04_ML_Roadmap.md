# StatPitch v3 — ML + Platform Roadmap

**Status:** proposed, not started.
**Goal:** a trained model that retrains on new results and is served from Render as
a stateless API, consumed by a separate downstream API that persists the results to
its own database and feeds a frontend — all at $0.

> Follows the convention of [`01_Requirements.md`](01_Requirements.md),
> [`02_Design.md`](02_Design.md), [`03_Tasks.md`](03_Tasks.md): every phase has an
> exit criterion that can be **measured**, not merely reviewed.

---

## 0. Is this a real ML model?

Partly today, fully after Phase 1.

**What is already real ML.** [`models/goals.py`](../src/statpitch/models/goals.py)
is two XGBoost `count:poisson` regressors with a per-competition `base_margin`
offset. It was trained on 61,321 feature rows × 74 columns and evaluated properly:
time-ordered splits, an untouched 2024/25 holdout, out-of-fold isotonic
calibration, and a leakage test that truncates the future and asserts earlier
feature rows come back byte-identical. Measured at 0.9852 log-loss.

**What is not real yet.** That model is not trained by any script, not saved to any
artifact, and not used by the deployed API. `GoalsModel.fit` is called only from
`tests/test_goals.py`. The live service derives goal rates from Elo through a
hard-coded constant. So the *research* is real ML; the *product* is currently a
parametric model wearing its results.

**Three more things are specced and unwired**, discovered while writing this plan.
All three are declared dependencies that nothing in `src/`, `scripts/` or `tests/`
imports:

| dependency | declared for | status |
|---|---|---|
| `xgboost>=2.1` | the goal model | fit only in tests, never persisted |
| `shap>=0.46` | **FR-32 per-bet explainability** | never imported |
| `optuna>=3.6` | hyperparameter search | never imported |
| `cloudscraper>=1.2.71` | **Transfermarkt (Phase 2)** | never imported |

This plan wires all four. After Phase 1 the answer is unambiguous: learned
parameters, fitted on data, versioned as artifacts, retrained on a schedule,
promoted only on measured improvement.

---

## 0.1 What this plan is fighting

[`MODEL_CARD.md`](MODEL_CARD.md) §1: the market-shrinkage weight `w` fits at
**0.000**, CI [0.000, 0.100], over 5,306 validation matches, on both log-loss and
log-growth. §4 logs eight attempts to overturn it; none did. §7: best-bet-per-match
measured **−2.12%** ranked by EV, **−3.27%** by log-growth.

Three consequences shape everything below.

**Headroom is bounded and small.** The CI upper bound says even a much stronger
model on this feature set competes for ≲10% of a blend. Phases that add
*information* rank above phases that add *capacity*. Every one carries a kill
criterion.

**The requested features — best bet per match, best bet of the day — are the two
with measured negative results.** They are built here in full, behind
`require_fitted()` and a new selector gate that cites its own measurement. Nothing
in this roadmap removes a guard. If a gate fails, the endpoint keeps refusing and
the refusal keeps quoting the number that caused it.

**CLV is the gate metric, not ROI.** Over the same selections, ROI on 38,763
settled bets could not resolve whether an edge existed (t = 1.19); CLV on 4,929
priced bets resolved it clearly (t = 3.47) — roughly an eighth of the sample. This
is what makes a one-season evaluation loop possible instead of an eight-season one.

---

# Part I — The model

## Phase 1 — Reproducible training

Nothing else works while training is a thing that happened once.

**1.1 `scripts/train.py`** — one command end to end: build features → fit the two
regressors with the competition offset → fit `rho` per competition → write a
versioned artifact.

**1.2 Artifact format and registry.** `models/goals-<version>.json` via xgboost's
own `save_model` — **not pickle**, which ties an artifact to the exact sklearn and
Python versions and breaks on any runtime upgrade. Alongside it
`models/registry.json` recording per artifact: training window, feature list *and
column order*, git SHA, source-data checksums, and every headline metric. An
artifact whose feature list disagrees with the live builder must fail loudly at
load, never silently mis-order columns.

**1.3 Walk-forward validation replaces the single split.** Expanding-window folds,
refit at each season boundary, metrics per fold. This turns `w` from a number into
a time series — the input to Phase 9's monitoring.

**1.4 The 2024/25 holdout stays untouched** (NFR-10). Add a test that fails if the
holdout season appears in any training or validation split.

**1.5 Multiple-comparisons discipline.** This plan proposes ~25 new features and
several selectors. Testing each at p < 0.05 against one validation window
manufactures about one false positive per twenty tests — exactly how a null result
gets overturned by accident. Before Phase 3, write the full ablation list, fix
Holm–Bonferroni over the pre-registered set, record it in the registry. A feature
clearing only an uncorrected threshold does not ship.

**Exit:** `python scripts/train.py` reproduces 0.9852 from a clean checkout and
registers an artifact.

## Phase 2 — Serve what was actually measured

The deployed path is not the evaluated path.
[`predictor.py`](../src/statpitch/serving/predictor.py) derives λ from Elo via
`ELO_GOAL_SENSITIVITY = 0.55`; the card measures a matrix driven by *fitted* rates.
The served numbers have no row in the evaluation table.

The 512 MB split stays — `requirements-serving.txt` excludes xgboost deliberately,
enforced by `tests/test_deployment.py`.

**2.1** Add `elo→λ` as a row in MODEL_CARD §3 on the same validation window.
**2.2** If it costs more than ~0.005 log-loss, export fitted rates: train offline,
write a λ lookup or distilled coefficient table to parquet, load at startup.
Serving still imports no xgboost.

**Exit:** every probability the API serves traces to a measured number.

## Phase 3 — Momentum and form

What exists in [`features/build.py`](../src/statpitch/features/build.py): rolling
form at 5/10, goals for/against, rolling xG and overperformance, venue-split form,
**scoring streak**, **clean-sheet streak**, goal volatility, rest days,
`matches_14d` congestion, h2h PPG over 10 meetings.

Note what is missing despite those streak columns: **there is no result-streak
feature at all.** `scoring_streak` counts matches with a goal; nothing counts
consecutive wins, losses or unbeaten matches.

Every feature below is computed inside the existing single chronological pass, with
club state updated **after** the row is emitted. That structure is what makes
NFR-10 hold — a feature added via `groupby/shift` outside the pass forfeits the
guarantee. Extend the truncation leakage test to each new column.

**3.1 Result streaks** (the direct ask). Consecutive wins, losses, unbeaten run,
winless run, matches since last win / last loss. Cheap, and genuinely absent.

**3.2 Opponent-adjusted form.** Current form is raw points and goals — three wins
over relegation sides reads identically to three over the top four. Weight each
match's contribution by the opponent's Elo at the time. Corrects a known bias
rather than adding a new signal; strongest item in the phase.

**3.3 Exponentially-weighted form.** Replace flat 5/10 windows with a fitted decay
half-life. A flat window says match 5 and match 1 count equally and match 6 not at
all; neither is true.

**3.4 Elo momentum.** Club Elo is 1,274,186 as-of-date intervals, so the derivative
is free: Δelo over the last 5 and 10 matches, and over 30/90 days. The level is the
strongest feature already; the trend is absent.

**3.5 Expected-points over/underperformance.** Convert rolling xG to xPts and
subtract actual points — separates "winning while outplayed" from real form.
Distinct from the existing `xg_overperformance`, which is goals-vs-xG.

**3.6 Fatigue and travel.** Minutes in the last 7/14/21 days rather than match
counts; away travel distance from free club coordinates; days since last match.

**3.7 Motivation and stakes.** Table position and points-from-target at match time
(derivable from the existing log), season stage, "big match within N days"
lookahead-trap flag.

**Gate:** each feature measured on the gap to the closing line and on `w`, under
Holm from 1.5. The prior from §4 is explicitly pessimistic — venue-split form and
streaks were tried, verdict "real effect, redundant with rolling xG". 3.1, 3.2 and
3.4 are the three with a genuine claim to new information rather than re-encoding.

## Phase 4 — New data, free only

Ranked by the property that decides their value: **backfillable or not.** Only
backfillable data can re-test `w`.

### Backfillable

**4.1 Transfermarkt squad market values.** §6 names squad values as a gap, and
`cloudscraper` is already a declared dependency for exactly this. Free, historical,
and — crucially — *not* shot-derived, so it escapes the argument that killed xG
("bookmakers use the same public shot data"). Encodes transfer activity and squad
depth that rolling xG picks up only with a lag. **Highest expected value here.**

**4.2 Manager changes.** Date-stamped, free, backfillable. A new-manager bounce is
a regime break that rolling form models as noise.

**4.3 Deeper cup history.** Coupe de France has **7 rows**; Coppa Italia 90, Copa
del Rey 119, DFB-Pokal 194. §6 calls joint training with a competition embedding
"load-bearing rather than elegant" — it is currently load-bearing on almost
nothing. Cannot move `w` (no cup odds) but improves the cup predictions actually
served.

**4.4 FBref / StatsBomb event detail** via `soccerdata` — possession, set-piece
share, PPDA. Honest expectation: **low.** Same public event-data family as xG,
which moved the gap by 0.0007. Last, or not at all.

### Forward-only

**4.5 API-Football lineups and injuries.** [`quota.py`](../src/statpitch/quota.py)
already implements the full budget — 100/day, hard stop at 90, fixture-keyed 24h
cache, one attempt per key, returns `None` rather than raising, fully tested.
**No caller was ever written.** Confirmed XI lands ~1h before kickoff with no
historical archive at the free tier, so this **cannot re-test `w`** — it feeds live
predictions only. Budget works for a match-day card: 90 calls against ~50 fixtures
in a five-league weekend round.

Its evaluation is a season away **and the clock starts only when collection
starts.** Every week it is not running is validation data that cannot be recovered.
Wire it early; expect nothing until 2027.

## Phase 5 — Model class improvements

Capacity, not information — so ranked below Phases 3 and 4, and each is a
hypothesis with a null.

**5.1 Optuna tuning.** The dependency is declared and unused. `DEFAULT_PARAMS` in
`goals.py` are hand-set. Tune under the walk-forward folds from 1.3, never a single
split, with the search budget and seed recorded in the registry.

**5.2 Monotonic constraints.** A higher `elo_diff` must never reduce the home goal
rate. XGBoost supports monotone constraints directly; they cost nothing, remove a
class of nonsense on sparse feature rows, and make the model defensible. This is
also a partial substitute for the `LAMBDA_BOUNDS` clip, which currently catches
such failures after the fact.

**5.3 Overdispersion.** `count:poisson` assumes variance equals mean. Football
scorelines are mildly overdispersed. Test a negative-binomial head against Poisson
on the same folds. Dixon-Coles already corrects the low-score cells; this is a
different correction and they may be partly redundant — measure, do not assume.

**5.4 Market-as-feature, not market-as-blend.** The most interesting untried idea.
Today `p_used = w·p_model + (1−w)·q_fair` is a *linear post-hoc blend*, and `w` = 0
says the blend is the market. A different formulation is to feed de-vigged market
probabilities in as **input features** and train the model to predict where the
market is wrong. This permits nonlinear interactions the blend cannot express, and
is the standard sharp-modelling formulation. The risk is obvious and must be
controlled for: the model can simply learn to copy the market and look excellent.
The test is whether the non-market features add anything *given* the market
features — a nested-model comparison, not a headline log-loss. Only applicable to
the five leagues that have odds.

**5.5 Joint / multi-task head.** Predict home goals, away goals and result
together, sharing a representation, so the score matrix and the 1X2 output cannot
disagree. §3 currently reports them as separate models that happen to agree.

**Explicitly out of scope:** deep learning. With 61k rows and 74 tabular features
against a gradient-boosted baseline, it is the least likely thing in this document
to help and the most expensive to run at $0.

## Phase 6 — Explainability and uncertainty

This is where the frontend gets something to show, and it satisfies a requirement
already on the books.

**6.1 SHAP per prediction (FR-32).** `shap>=0.46` is declared for "per-bet
explainability" and never imported. Compute SHAP values at prediction time in the
offline job (Phase 8), store the top contributing features per fixture in the
database, and serve them. This turns every prediction from a number into "Elo gap
+0.31, home rest advantage +0.08, away winless run +0.06" — the single highest
product-value item in this plan, and it is already specced.

**6.2 Conformal prediction intervals.** Calibrated coverage on the goal-rate
outputs, giving the frontend an honest "1.8 goals, 80% interval [0.9, 3.1]" instead
of a bare point estimate. Cheap, distribution-free, and it composes with the
existing calibration measurement.

**6.3 Surface rating provenance.** Already computed — `fully_rated` and the rating
source (measured Elo / fitted entry prior / pooled entrant / bare default) are in
every response because 187 of 428 clubs once silently fell through to a flat 1400.
The frontend must display it, not hide it behind a clean number.

---

# Part II — The platform

> **Architecture.** StatPitch is deployed on Render as a **stateless prediction
> service**. It owns no database. A separate, independently developed API consumes
> it and persists the results to its own DB, which feeds the frontend.
>
> This is a clean division and it keeps `STATPITCH_READ_ONLY=1` correct exactly as
> [`render.yaml`](../render.yaml) explains it — the ephemeral disk stops being a
> problem when nothing here needs to persist. StatPitch computes; the consumer
> remembers.
>
> The consequence is that **the API response contract is now the product.** A
> field renamed here is a migration in someone else's database. Everything in
> Part II follows from that.

## Phase 7 — The blocker: there is no fixture source

This has to come first, because the two endpoints a daily consumer would poll are
currently stubs.

```
GET /today            → {"date": null, "fixtures": [], "note": "…not exercised here."}
GET /value-bets/today → {"date": null, "value_bets": [], "note": "…"}
```

`/today` returns nothing because live fixture listing needs the API-Football feed
that was never wired (Phase 4.5). Without a list of upcoming fixtures there is
nothing for a consumer to poll, and the integration has no daily entry point.

Two free options, and they are not exclusive:

**7.1 openfootball schedules.** The repos already ingested for cup history carry
**full season fixture lists, including future fixtures**, and there is already a
working client in [`data/openfootball.py`](../src/statpitch/data/openfootball.py).
No quota, no key, backfillable, and it covers the leagues and cups already in
scope. **This is the cheaper path and should be tried first.** Risk: schedule
freshness — postponements and TV-driven date changes may lag.

**7.2 API-Football fixtures** as the accurate/live source, spending part of the
90-call daily budget `quota.py` already enforces. Better freshness, hard ceiling.

Recommended: openfootball as the base schedule, API-Football as a same-day
correction pass for kickoff times and postponements, which is a far cheaper use of
the quota than fetching every fixture.

**Exit:** `/today` returns a real fixture list with predictions attached.

## Phase 8 — The integration contract

The most important phase in this plan now, and the cheapest to get wrong.

**8.1 Typed responses.** Almost every endpoint is annotated `-> dict`; only
`PredictRequest` is a Pydantic model. That means FastAPI's generated OpenAPI schema
describes nearly nothing, the consumer cannot codegen a client, and a field can
change type without any test noticing. Add `response_model` to every route. This is
also the strongest possible enforcement of the NFR-13 contract the module docstring
already commits to — "no existing response field is renamed, removed or retyped" —
which is currently a promise kept by discipline alone.

**8.2 Provenance on every response.** `model_version`, `config_version`,
`generated_at`, and the schema version. The consumer's DB rows are only as
trustworthy as their traceability; a stored prediction with no model version cannot
be interpreted after a retrain. This is what makes Phase 9's retraining safe for a
downstream database rather than a source of silent history rewrites.

**8.3 Machine-readable refusals.** Today a refusal is prose in a `note` string:

```json
{"value_bets": [], "note": "No value bets are flagged. The fitted market-shrinkage weight w is 0.000, …"}
```

Stored in a consumer's DB, that is an unparseable blob. The refusal philosophy is
right and should be kept — it should just be structured so it survives the hop:

```json
{"available": false,
 "reason_code": "SHRINKAGE_WEIGHT_ZERO",
 "reason": "…human-readable, unchanged…",
 "measurement": {"w": 0.0, "ci": [0.0, 0.1], "n_matches": 5306}}
```

A consumer can then store the code, render the reason, and show the measurement.
The prose stays; it stops being the only machine-visible thing.

**8.4 Batch endpoints.** A matchday is ~50 fixtures across five leagues. A consumer
that must issue one request per fixture per market family will make hundreds of
calls into a free instance. Add `POST /predict/batch` and a
`GET /fixtures/upcoming?from=&to=&include=markets,explanations` that returns a
whole round in one round-trip.

**8.5 Cheap polling.** `updated_at` per fixture plus `ETag`/`If-None-Match`, so a
consumer syncing every 15 minutes transfers nothing when nothing changed.

**8.6 Determinism and idempotency.** For a given `(fixture, model_version)` the
response must be byte-identical on repeat calls, so re-polling cannot produce
conflicting rows downstream. Assert it in a test. Where a prediction legitimately
changes — new lineup data, updated Elo — that must move `model_version` or
`generated_at`, never mutate silently under a stable key.

**8.7 An API key.** A public unauthenticated service on a free instance is a
denial-of-service surface, and burning the instance hours affects the one consumer
that matters. A single static key checked by a dependency is enough.

## Phase 9 — Cold start: resolved by the architecture

Render's free plan spins down after ~15 minutes idle — warm path **2.5 ms**, first
request after idle **tens of seconds**, plus ~0.7 s artifact load
([`DEPLOYMENT.md`](DEPLOYMENT.md)).

**This is not a user-facing problem in this architecture.** The frontend is served
entirely from the consuming API's database; nothing on a user's request path
touches StatPitch. A cold start therefore lands on a scheduled sync job, which does
not care. No keep-alive ping is needed — which is fortunate, since `render.yaml`
rules one out on the grounds that it "would burn the monthly instance hours for no
user."

Two residual items only, both small, and both belong to the sync job rather than to
this service:

**9.1 Timeouts sized for a cold start, not a warm one.** The sync job needs a
~60-second timeout with retry-and-backoff. A client configured for the 2.5 ms warm
path will fail its first call after every idle period and look like an outage.

**9.2 A readiness signal worth polling.** `/health` should report artifact-load
state and `model_version`, so the sync job can distinguish "still starting" from
"broken" instead of guessing from a timeout. Cheap, and it also gives the consumer
the version-change signal that 11.3 depends on.

## Phase 10 — Model artifacts and the deployed image

Retraining changes what the deployed service must load, and the 512 MB ceiling does
not move.

**10.1 Artifacts published to GitHub Releases**, not committed to the repo — the
training job uploads, the deploy downloads at build time. Keeps the repo small and
decouples a retrain from a code push.

**10.2 The serving split stays enforced.** `requirements-serving.txt` excludes
xgboost, shap, optuna and the scrapers, and `tests/test_deployment.py` asserts that
importing `statpitch.serving.app` pulls in nothing outside it. Phase 2.2's exported
λ table and Phase 6.1's precomputed SHAP values are the mechanisms that keep it
that way: heavy libraries run in CI, the deployed image reads their output.

**10.3 SHAP values are precomputed, never computed on request.** Explaining a
prediction with the `shap` library at request time would drag the training stack
into the deployed image and blow both the memory ceiling and NFR-2. The training
job writes explanations alongside the model; serving reads them.

## Phase 11 — The learning loop

What "learns and improves" means concretely. Today the system trains once; the data
arrives weekly.

**11.1 Weekly retrain workflow** — `.github/workflows/retrain.yml`, alongside
`flag-card.yml` and `settle-ledger.yml`, on the same principles those already
establish: idempotent, honest about actual vs nominal run time, free of the API
budget. Fetch new results → rebuild features → retrain → evaluate → register.

**11.2 Promotion gate.** A new artifact replaces the live one only if it is not
worse than the incumbent across walk-forward folds by a margin exceeding fold
noise. Otherwise: registered, not promoted, reason recorded. Automatic retraining
without a promotion gate is a mechanism for silently shipping regressions.

**11.3 Retraining is a breaking event for a downstream database.** This is the
integration-specific hazard. When a new model is promoted, every stored prediction
in the consumer's DB was produced by a different model. Handle it explicitly:
`model_version` on every row (8.2), an endpoint the consumer can poll for the
current version, and a documented policy — predictions are immutable, a new model
writes new rows. Silently re-serving different numbers under the same fixture key
corrupts a track record that lives in someone else's system.

**11.4 Drift monitors.** Rolling log-loss vs the closing line, ECE, feature
distribution shift, exposed at `/monitoring` for the consumer to store. The
2025-07-23 Pinnacle regime break is the worked example: a benchmark can change
underneath a model without any code changing.

**11.5 `w` as a monitored series.** Re-fit every retrain, track with CI.
Requirements §9's truth serum becomes a continuous measurement. If it ever leaves
zero with a CI excluding zero, that is the signal this project exists to detect —
and it should be detected by a monitor, not by someone re-reading a model card.

**11.6 Close the outcome loop.** The append-only ledger and `clv_tracker` already
exist. Feed settled results into 11.2 so promotion is judged on realized CLV as
well as offline log-loss.

## Phase 12 — Best bet per match, and best bet of the day

Built last, behind a gate.

**12.1 Fit `decision_config`.** Shipped as `dec-2026.08.0-placeholder` with
`w_fitted` false; `StakingEngine.require_fitted()` refuses to size a stake from it
([`staking.py:356`](../src/statpitch/decision/staking.py)). Fitting it unlocks
`/card/today`. The guard is satisfied, not weakened.

**12.2 Redesign the selector around the recorded failure.** §4 explains *why*
best-bet lost: max-edge selection "reliably finds the model's largest errors, not
the market's", and put 54.5% of picks in 1X2, the worst-performing market. A better
ranker over the same raw-edge signal reproduces that. Rank instead by **graded
confidence** — `bet_grader` already encodes it, its confidence *falls* past ~4
points — and constrain market mix so the selector cannot concentrate in 1X2.

**12.3 Best of the day = top-N under the same gate**, with the correlation-aware
fractional Kelly in `staking.py` handling overlapping exposure. A day's card is not
N independent bets.

**12.4 The gate.** Ships only if selections beat the *unselected* baseline on
Friday-to-close CLV with t > 3 over a pre-registered window. The baseline matters:
§5 shows the whole book at −1.19% best-price and −0.09% average-price, so a rule
must beat its own no-rule baseline, not zero.

**If the gate fails, the endpoints keep refusing** and the refusal cites the new
measurement alongside the old. That is a successful completion of this phase.

---

## Credentials and secrets

**Current state: no API key is read anywhere in this codebase.** `.gitignore`
reserves `.env` with the comment "Secrets (API-Football key)", but nothing calls
`load_dotenv`, no `os.environ` lookup returns a key, and there is no `.env.example`.
The only environment variables in use are `STATPITCH_DATA`, `STATPITCH_MODELS`,
`STATPITCH_READ_ONLY` and `STATPITCH_QUOTA_HARD_STOP`. The credentials path is
built from scratch in Phase 8.

### What actually needs a key

| source | variable | needed for | tier |
|---|---|---|---|
| **API-Football** | `STATPITCH_API_FOOTBALL_KEY` | 4.5 lineups/injuries; 7.2 same-day fixture corrections | free, 100 req/day |
| **football-data.org** *(optional)* | `STATPITCH_FOOTBALL_DATA_ORG_KEY` | fallback fixture source if openfootball schedules prove too stale | free tier, key required |
| **StatPitch itself** | `STATPITCH_API_KEY` | 8.7 — protects the free instance from anonymous traffic | self-chosen value, not from a provider |

### What needs no key at all

football-data.co.uk (results and odds), Club Elo, Understat, openfootball
(including the Phase 7.1 schedules), Transfermarkt via `cloudscraper`, FBref via
`soccerdata`. **Every phase in the first three sequencing slots runs without a
single credential** — the key only becomes load-bearing at Phase 4.5.

No database URL is required: the downstream API owns persistence.

### Where each secret lives

Three places, and forgetting the second or third is the usual failure:

1. **Local `.env`** — development only, gitignored, never committed.
2. **Render environment variables** — dashboard or `render.yaml` with `sync: false`
   so the value is set in the dashboard rather than written into the repo.
3. **GitHub Actions secrets** — the scheduled jobs (retrain, fixture sync, lineup
   collection) run there, not on Render, so they need their own copy.

**8.8 Credential plumbing.** Add a committed `.env.example` listing every variable
with empty values, `load_dotenv()` at entry points only (never at import time in
library code), and a startup check that logs which optional sources are configured
— by name, never by value. A missing key must degrade to the documented fallback
path, exactly as `quota.py` already returns `None` on exhaustion rather than
raising, and never appear in a log line, an error message or an API response.

## Sequencing

Ordered so the consuming API has something real to integrate against as early as
possible, and so nothing downstream is built on a contract that will move.

| # | phase | why here |
|---|---|---|
| 1 | **7 — fixture source** | `/today` is a stub; without it there is no daily integration at all |
| 2 | **8 — integration contract** | typed responses, provenance, structured refusals, batch. Freeze it before the consumer builds on it |
| 3 | 1 — reproducible training | everything model-side depends on it |
| 4 | 4.5 — start the lineup collector | forward-only; the clock starts now |
| 5 | 9.2 — readiness signal on `/health` | tiny; the sync job and 11.3 both want it |
| 6 | 2 — serve what was measured | small, fixes a live integrity gap |
| 7 | 6.1 — SHAP explanations | highest product value, already specced as FR-32 |
| 8 | 10 — artifacts + image discipline | keeps the 512 MB split intact under retraining |
| 9 | 4.1 — Transfermarkt values | best expected value, and backfillable |
| 10 | 3.1 / 3.2 / 3.4 — streaks, opponent-adjusted form, Elo momentum | the genuinely new features |
| 11 | 11 — learning loop | needs 1 and something worth retraining |
| 12 | 5 — model class | capacity after information |
| 13 | 3.3–3.7, 4.2, 4.3, 6.2 | second tier |
| 14 | 12 — best bet, gated | needs all of the above |
| — | 4.4 — event detail | only if something above moves `w` |

**Why the contract comes before the model.** Phases 7 and 8 are cheap and they are
the ones another codebase will build against. Every week the consumer integrates
against untyped `-> dict` responses and prose refusals is a week of downstream code
written against a shape that this plan intends to change.

## The honest expectation

Phases 1, 2, 6–11 are engineering. They will land, and they turn this into a real
ML service: trained artifacts, versioned predictions, explanations, monitoring, and
a contract another system can safely persist. That work is not speculative.

Phases 3, 4, 5 and 12 are aimed at the *result*, and the recorded prior says the
market has already priced most of what free data can see. The plan is built so a
null result arrives as a **finding** with an ablation table behind it rather than as
an unexplained absence of improvement — and so the endpoints stay honest either
way, including in someone else's database.
