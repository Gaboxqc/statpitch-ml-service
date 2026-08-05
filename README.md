# StatPitch v2

Calibrated club-football prediction and decision layer for Europe's major leagues,
domestic cups and continental competitions — built on free data sources only.

Specs live in `docs/`: `01_Requirements.md`, `02_Design.md`, `03_Tasks.md`. Every
module traces back to an FR/NFR and a design section; the docstrings carry those
references so the code and the spec stay legible together.

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
data/
  competitions.json    12 Phase-1 competitions, incl. the odds_coverage gate
  decision_config.json Placeholder parameters — nothing here has seen data yet
  processed/           matches_clean.parquet, closing_odds.parquet
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

Phase 0 complete, Phase 1 in progress.

| Item | Status |
|---|---|
| Taxonomy for all 12 competitions, format resolution by stage and season | done |
| `decision_config.json`, versioned, refuses to stake while placeholder | done |
| API-Football quota guard, before any real API call exists | done |
| football-data.co.uk ingestion, full odds column set, regime-tagged | done |
| Club Elo ingestion, as-of-date lookups, name reconciliation | done |
| openfootball cups + continental | next |

Ingested so far: **59,079 matches** across the 5 leagues, 1993/94 to date,
**417,631 tidy odds rows**, and **926,697 Club Elo rating intervals** covering all
245 clubs.

### Known limitation in the Elo table

It currently covers only clubs that have appeared in one of the five top leagues
since 1993/94. Domestic cups admit entrants from far down the pyramid, and those
clubs are not in it yet — they get added when the openfootball cup ingestion
reveals their names. Club Elo itself covers them (that is why FR-9 is buildable
at all); the roster is simply scoped to what has been needed so far.

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
