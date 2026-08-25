# StatPitch v2

Calibrated club-football prediction and decision layer for Europe's major leagues,
domestic cups and continental competitions — built on free data sources only.

Specs live in `docs/`: `01_Requirements.md`, `02_Design.md`, `03_Tasks.md`. Every
module traces back to an FR/NFR and a design section; the docstrings carry those
references so the code and the spec stay legible together.

> **The headline result is negative, and it is the point.** The market-shrinkage
> weight `w` fits at **0.000** on both criteria — this model does not beat the
> closing line. What survived measurement is closing line value on
> sharp-reference selections (+0.51%, t=7.53 clustered) — and that rule is
> defined on **Pinnacle**, which the live price feed does not publish. Every
> reference the feed does carry was tested and none clears the ≥2-season bar
> (`data/selection_rule_study.json`). Read
> **[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)** before anything else.
>
> Advisory only (NFR-11): no bookmaker integration, no wagers, no funds. Staking
> is disabled in code, not by convention.

## Layout

```
src/statpitch/
  paths.py             Canonical filesystem layout (Design §8)
  taxonomy.py          Competition taxonomy, season/stage-aware format resolution (§2)
  decision_config.py   Typed, versioned Decision Layer parameters (NFR-12)
  quota.py             API-Football 100/day request budget (NFR-9, Design §3.2)
  data/
    http.py            Polite, cached, rate-limited HTTP (NFR-5)
    football_data.py   football-data.co.uk results + full odds set (Design §3.1)
    football_data_live.py  Pre-match prices for upcoming fixtures, same publisher
    club_elo.py        As-of-date strength ratings, reported name reconciliation
    openfootball.py    Domestic cups + UCL/UEL, with ET and shootout columns kept apart
    openligadb.py      Keyless DFB-Pokal fixtures, with the round parsed
    odds_api.py        The six cups nothing free reaches; needs a key, /events is free
    understat.py       Shot-based xG; club map derived from fixture identity
  features/
    build.py           Single chronological pass — the leakage guarantee (NFR-10)
  models/
    dixon_coles.py     The score matrix: one source of truth for ~60 markets (§6.1)
    goals.py           Poisson rates with a per-competition base_margin offset
    calibration.py     Reliability curves and ECE (FR-16b)
    entrant_prior.py   Entry-round Elo prior for unrated cup clubs (FR-9)
    knockout.py        Extra time and shootouts, measured not assumed (FR-8, FR-7)
    bracket.py         Fixed brackets vs random redraws — not interchangeable (FR-20)
  decision/
    devig.py           Proportional, power, Shin (FR-28)
    devig_selection.py Empirical selection that refuses non-significant winners
    shrinkage.py       p_used = w·p_model + (1−w)·q_fair — the truth serum
    market_engine.py   86 selections, each with a full payoff distribution (FR-23)
    value.py           Edge decomposed into price and model, which never mix (FR-16a)
    bet_grader.py      Confidence that falls as the apparent edge grows (FR-25/33)
    staking.py         Kelly over payoff distributions, correlation-aware (FR-27)
    clv_tracker.py     Append-only ledger; refuses cross-source CLV (FR-26/29)
  serving/
    predictor.py       Format-aware inference, artifacts loaded once (§5.3, §7)
    app.py             FastAPI; refusals carry the measurement behind them
  ops/
    jobs.py            flag_card and settle_ledger, idempotent and clock-honest
data/
  competitions.json    12 competitions, incl. the odds_coverage gate
  decision_config.json Placeholder parameters — nothing here has seen data yet
  processed/           matches_clean, closing_odds, features, elo_ratings_all, …
  processed/live_odds.parquet  Append-only capture log of pre-match prices
  processed/card.parquet       Graded selections; empty slate, computed reason
docs/05_Gaps_and_Plan.md  The three blockers on bets, and the plan to clear them
docs/MODEL_CARD.md     What was measured, what failed, and what it means
docs/DEPLOYMENT.md     Scheduled jobs, Render blueprint, and the free-tier limits
render.yaml            Free-plan blueprint; read-only, serving deps only
.github/workflows/     CI, plus the two scheduled ledger jobs
tests/                 Offline; no test touches the network
```

## Setup

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt && .venv/Scripts/python.exe -m pip install -e . --no-deps
```

Run the suite:

```bash
.venv/Scripts/python.exe -m pytest
```

Ingest the archive (downloads are cached; re-runs do not re-hit the origin):

```bash
.venv/Scripts/python.exe -m statpitch.data.football_data
```

## Current state

Phases 0–9 complete, plus Plan Phases A–E. **1,202 tests**, all offline.

| Layer | Item | Status |
|---|---|---|
| 0 | Taxonomy for 12 competitions, format resolved by stage and season | done |
| 0 | `decision_config.json`, versioned, refuses to stake while placeholder | done |
| 0 | API-Football quota guard, before any real API call exists | done |
| 1 | football-data.co.uk, Club Elo, openfootball cups, Understat xG | done |
| 2 | Leakage-safe chronological feature build | done |
| 3 | Dixon-Coles matrix, Poisson goal model, calibration | done |
| 3 | Extra time + shootouts (FR-8), bracket simulator (FR-20) | done |
| 5 | **`w` checkpoint — fits at 0.000** | done |
| 5.5 | Market engine, value, grading, staking, CLV ledger | done |
| 7 | Format-aware predictor + FastAPI serving layer | done |
| 8 | Model card | done |
| 9 | Scheduled jobs, CI, Render blueprint | done |
| A | Live pre-match prices, keyed to the fixture list ([plan](docs/05_Gaps_and_Plan.md)) | done |
| B | Card built from prices + predictions; slate routes read it | done |
| C | Sharp-reference study — **no tradeable rule clears the 2-season bar** | done |
| D | Cup fixtures: keyless OpenLigaDB + Odds API client for the other six | done |
| E | Daily odds capture on its own schedule; model card corrected | done |

Ingested: **64,795 matches** — 59,079 league (1993/94–), 5,716 cup and
continental — **417,631 tidy odds rows**, **1,274,186 Club Elo rating intervals**
across 428 clubs, and Understat xG joined on **100.0%** of covered fixtures.
**61,321 feature rows**, 74 columns.

### What the evaluation says

| | log-loss | accuracy | ECE |
|---|---|---|---|
| Dixon-Coles + xG | 0.9845 | 0.5264 | **0.00317** |
| market (de-vigged close) | **0.9698** | **0.5439** | 0.01012 |

`w` = 0.000, CI [0.000, 0.100] on log-loss and [0.000, 0.215] on log-growth. The
blend is the market. The full account — what was tried against this, what the
odds-coverage gap means, and the eight corrections to the spec — is in the
[model card](docs/MODEL_CARD.md).

### Known limitation in the Elo table

Club Elo rates only the top two tiers, contradicting FR-9 and Requirements §7.1.
Clubs below that return an *empty CSV rather than a 404*, which is why the gap is
easy to miss. Deeper cup entrants are handled by a fitted entry-round prior
instead: FA Cup round-3 entrants come back at 1790 Elo against round-1's 1331, a
460-point tier separation recovered from results rather than assumed.

## Deployment

Two scheduled GitHub Actions jobs and one Render web service, all free-tier.
Full detail, including the failure modes they are built around, is in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

```bash
python -m statpitch.ops.jobs flag_card
```

Three things worth knowing before relying on any of it:

- **The jobs flag nothing today, and say so.** `w` is 0.000 and the decision
  config is unfitted, so there is nothing to stake. Both jobs still run and both
  emit a reason, because a job that silently produces an empty result is
  indistinguishable from one that is broken.
- **The free Render instance spins down after ~15 minutes.** The warm path is
  3.9 ms against NFR-2's ~200 ms budget; the first request after idle is tens of
  seconds. That is a plan limit, not a code path, and there is deliberately no
  keep-alive ping to hide it.
- **The deployed API is read-only.** A free instance's disk is ephemeral, so a
  ledger write would succeed and then vanish. `BetLedger` refuses writes under
  `STATPITCH_READ_ONLY`; the ledger is owned by the scheduled job that commits it.

## Three findings from Phase 1 that change the plan

These came out of checking the live files rather than trusting the spec's
description of them, and all three are now enforced in code and tests.

**1. Consensus closing odds start in 2019/20, not at the start of the archive.**
Requirements §7.3 describes the `C`-suffixed closing columns as if they run the
length of the data. They do not. There are three schema eras:

| era | seasons | consensus pre-close | consensus close | AH | kickoff time |
|---|---|---|---|---|---|
| legacy | …–2004/05 | none | none | none | no |
| betbrain | 2005/06–2018/19 | `BbAv*` / `BbMx*` | none | `BbAHh` | no |
| modern | 2019/20– | `Avg*` / `Max*` | `AvgC*` / `MaxC*` | `AHh` | yes |

CLV, de-vig selection and the whole Decision Layer need consensus *closing* odds,
so they are confined to 2019/20 onward. Combined with the ban on pooling across
the 23/07/2025 Pinnacle break, the clean backtest window is **2019/20–2024/25**:
10,707 matches over 6 seasons. That comfortably clears the "≥2 full seasons" bar
in Requirements §8.3, but it is a quarter of the archive, not all of it. The
earlier seasons remain fully usable for *training* — it is the market benchmark
that is window-limited.

**2. Pinnacle's own closing price reaches back to 2012/13**, which gives
calibration and RPS work a 13-season runway on a single-book benchmark. It is not
a consensus and is typed separately, but Requirements §7.3 only licenses Pinnacle
as a sharp benchmark *before* 2025/26 — exactly the window it covers.

**3. Nine files (2002/03–2004/05) have rows wider than their header.** Left
unhandled they cost ~3,200 matches, silently, because the parse failure is caught
per-file. The overflow is always trailing empty commas, so it is trimmed — but
only after checking the discarded fields are actually empty, and a row that would
lose real data is dropped loudly instead.

## Two conventions worth knowing before reading the code

**Fair probability and available price never share a column.** `odds_avg`
(consensus) is what fair probability is derived from; `odds_max` is the price
actually obtainable. Max-of-N is above consensus by construction, so de-vigging it
fabricates edge — FR-16a, and the separation is structural rather than a comment.

**Reconstructed panel prices are never written into the published consensus
columns.** Individual books span all 25 seasons while the published `Avg*`/`Max*`
only start in 2005/06, so `odds_panel_avg` / `odds_panel_max` carry a panel
reconstruction that extends the price series backwards. Measured against 2024/25,
the 7-book panel average runs **+0.038 above** the published ~30-book consensus —
a small but systematic gap, which is precisely why the two stay in separate
columns.
