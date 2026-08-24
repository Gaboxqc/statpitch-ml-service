# StatPitch — Gap review and remediation plan

Written 2026-08-23 against `322181c`. Every claim below was checked against the
live files or the running code, not against the spec's description of them.

---

## 1. Why no bets are thrown

Three independent blockers, stacked. Odds are the first, and fixing only the
first changes nothing.

### Blocker 1 — there is no live price anywhere in the pipeline

`refresh_fixtures` runs `build_fixtures` → `collect_fixtures` → `precompute_predictions`.
That chain produces `fixtures.parquet` (703 rows) and `predictions.parquet`
(703 rows, `lambda_home/away`, `prob_home/draw/away`). Neither artifact carries a
price. `closing_odds.parquet` is the historical archive — football-data.co.uk
publishes it *after* the match.

So `value.assess_book(selections, fair, available)` has never been called with a
real `available` dict on an upcoming fixture. `MODEL_CARD` §6 states this
plainly: "No live odds… the single largest gap."

### Blocker 2 — `w` = 0.000 means model edge is zero *by construction*

`staking.shrink(p_model, q_fair, 0.0)` returns `q_fair` exactly. Feed that into
`value.assess` as `p_model` and:

- `edge_prob = p_model − q_fair = 0.0`
- `model_edge = expected_value − price_edge = 0.0`
- `expected_value = price_edge`

There is no arrangement of live odds under which a model-driven bet appears. The
only surviving quantity is `price_edge` — EV earned by taking the best quote
while believing exactly what the consensus believes.

### Blocker 3 — the grader zeroes out under that regime, and the card endpoint is a stub

`bet_grader.edge_confidence` returns `0.0` when `edge_prob <= 0`. With `w` = 0
that is every bet, so `c_edge` = 0 always. `c_edge` carries weight 0.30, so the
composite caps at **0.70** against a B cutoff of 0.65 and an A cutoff of 0.80.
Realistic sub-scores put every bet at C or D — and `stake_multiplier` is 0.25 for
C, **0.00 for D**.

Separately, `serving/app.py::card_today` and `value_bets_today` return a
hardcoded `"bets": []` / `"value_bets": []`. There is no code path that builds a
card, fitted config or not. `ops/jobs.py::flag_card` admits it in the fitted
branch: *"config is fitted but no fixture source is wired in"*.

### The consequence

**The honest product is a price-shopping / CLV engine, not a model-edge engine.**
That is not a downgrade — it is the only thing in the project that survived
measurement: Friday-to-close CLV on sharp-reference selections, **+0.51%,
t = 3.47, n = 4,929** (MODEL_CARD §5). The grader must therefore grade on
`price_advantage`, not on `edge_prob`.

---

## 2. Why cups are missing

Not a code bug. `SCHEDULE_SOURCES` maps all seven cup competitions correctly and
`build_fixtures` already asks for them. **openfootball has stopped publishing
them.** Verified 2026-08-23:

| path | result |
|---|---|
| `england/2026-27/` | only `1-premierleague.txt`, `2-championship.txt` |
| `england/2026-27/facup.txt` | 404 |
| `england/2025-26/facup.txt` | 404 |
| `deutschland/2026-27/cup.txt` | 404 (2025-26 still 200) |
| `espana/2026-27/cup.txt` | 404 |
| `italy/2026-27/cup.txt` | 404 |
| `europe/france/2026-27_frcup.txt` | 404 |
| `champions-league/` | no `2026-27` directory at all |
| `champions-league/2025-26/` | `cl.txt`, `clq.txt`, `confq.txt`, `elq.txt` — **no `el.txt`** |

`build_fixtures` logs the absence and treats it as normal ("a cup with no
published draw has no fixtures to publish"), which was true when written and is
now masking an upstream source that has gone dark. The cup modelling stack —
`entrant_prior`, `knockout`, `bracket`, stage-aware `taxonomy.resolve_format` —
is complete and has nothing to run on.

---

## 3. The two sources that fix both problems

### `football-data.co.uk/fixtures.csv` — free, keyless, schema-identical

Verified live: 200, 47 KB, ~180 fixtures across 20 divisions for the coming week.
Header is **the same modern-era schema `data/football_data.py` already parses**:

```
B365H/D/A  BFDH/D/A  BVH/D/A  BWH/D/A  PPH/D/A  SKBH/D/A
MaxH/D/A   AvgH/D/A                     <- consensus best / average
B365>2.5  Max>2.5  Avg>2.5  (and <2.5)
AHh  MaxAHH/AHA  AvgAHH/AHA
...C-suffixed closing columns, empty until played
```

Three properties that matter:

1. **Same source as the historical benchmark.** `clv_tracker` refuses
   cross-source CLV. Friday snapshot and closing price both come from
   football-data.co.uk, so the +0.51% result can be traded forward under the
   exact label it was measured with.
2. **`Avg*` and `Max*` in one row** — the FR-16a separation (fair from
   consensus, price from best-of-N) is available directly, no reconstruction.
3. **Confirmed kickoff times and referee, keylessly**, for the five leagues.
   This is a better date-correction source than football-data.org and needs no
   credential at all.

Covers leagues only. No cups.

### The Odds API — key required, covers all 12 competitions

Confirmed sport keys exist for every competition in the taxonomy:

```
soccer_epl  soccer_spain_la_liga  soccer_germany_bundesliga
soccer_italy_serie_a  soccer_france_ligue_one
soccer_fa_cup  soccer_spain_copa_del_rey  soccer_germany_dfb_pokal
soccer_italy_coppa_italia  soccer_france_coupe_de_france
soccer_uefa_champs_league  soccer_uefa_europa_league
```

Free tier is 500 credits/month. Critically, **`/v4/sports/{key}/events` costs 0
credits** and returns `id`, `home_team`, `away_team`, `commence_time` — so it is
a *free* current-season fixture source for the seven competitions openfootball
abandoned. Credits are only spent on `/v4/sports/{key}/odds`, at 1 per
competition per region-market call.

Its historical endpoints are paid-tier only, so cups gain live prices but remain
**unbacktestable**. That distinction has to survive into the API contract.

---

## 4. Plan

### Phase A — live odds for the five leagues ($0, keyless) — **done**

Delivered: `src/statpitch/data/football_data_live.py`,
`scripts/collect_live_odds.py`, `tests/test_football_data_live.py` (27 tests),
`paths.live_odds_file()`. First capture `20260823T1935Z` recorded 266 selection
rows over 38 fixtures in four competitions, **100% keyed** to `fixture_id`, and
confirmed 38 kickoff times of which 16 moved the fixture to a different day.

Four things worth carrying forward:

- **All 29 distinct live `selection_key` values match `market_engine.derive`
  exactly**, lines included, down to its `ah_away_-0.0` spelling for a level
  handicap. Phase B's join is a lookup, and a test asserts the contract rather
  than leaving Phase B to discover it.
- **`_BOOK_PREFIXES_1X2` was stale**, and it fed `c_market`. 2025/26 dropped WH
  and 1XB and added BFD/BMGM/BV/CL; the fixture feed adds PP/SKB. The archive
  counted 5 of 2025/26's 9 books and 3 of the feed's 7 — an undercount that
  penalised recent seasons for being recent. Six prefixes added.
- **Bookmaker-confirmed kickoffs come free with the prices.** That is the job
  `collect_fixtures.py` cannot do on API-Football's free plan at all, and it
  needs no credential.
- **The clock is Europe/London, not UTC.** Serie A's 20:45 CEST appears as
  19:45. Read as UTC it would mistime every near-kickoff capture by an hour in
  summer.

The original plan for this phase follows.

- `src/statpitch/data/football_data_live.py` — fetch `fixtures.csv`, reuse the
  existing era/column resolution in `football_data.py`, emit tidy rows.
- New artifact `data/processed/live_odds.parquet`, keyed
  `(fixture_id, market_key, book, odds, captured_at, snapshot_id)`.
  **Append-only.** Overwriting the Friday snapshot destroys the CLV baseline,
  which is the only measured signal the project has.
- Club-name reconciliation: `fixtures.csv` uses football-data names,
  `fixtures.parquet` uses openfootball names. Build
  `data/processed/fixture_odds_map.json` under the same rule the Elo and
  Transfermarkt resolvers already follow — unique candidate within ±3 days, or
  leave unmatched rather than guess.
- **Market key mapping table** — this is the actual integration work and it does
  not exist today. `market_engine` emits 86 selection keys; free sources quote
  about seven of them:

  | selection key | fair (`Avg*`) | price (`Max*`) |
  |---|---|---|
  | `1x2_home/draw/away` | `AvgH/D/A` | `MaxH/D/A` |
  | `ou_2.5_over/under` | `Avg>2.5` / `Avg<2.5` | `Max>2.5` / `Max<2.5` |
  | `ah_{AHh}_home/away` | `AvgAHH` / `AvgAHA` | `MaxAHH` / `MaxAHA` |

  The other ~79 stay unpriced; `assess_book` already skips selections with no
  quoted price, which is the correct behaviour and needs no change.
- Fold confirmed kickoff times from the same fetch into `date_confirmed`.

### Phase B — make the card compute

- `scripts/build_card.py`: fixtures × predictions × live_odds →
  `dixon_coles` matrix (with the per-competition `rho` already carried in
  `predictions.parquet`) → `market_engine.derive` → `devig` on the `Avg*`
  triplet (Shin) → `value.assess_book` with `p_model = shrink(p, q_fair, w)` →
  `bet_grader.grade_book` → `staking.allocate_slate` →
  `data/processed/card.parquet`.
- `serving/app.py`: `/card/today`, `/value-bets/today`, `/best-bet` read
  `card.parquet` instead of returning a literal `[]`. When the card is empty,
  keep the structured refusal — but have it name *which* gate emptied it, not a
  fixed reason string.
- `ops/jobs.py::flag_card` calls the same builder; its "no fixture source is
  wired in" branch is deleted.
- Guard: `test_deployment.py` must still pass — the card is built offline, the
  API only reads the parquet.

### Phase C — re-fit the decision config for a price-driven regime

> **Measured on the first live capture (2026-08-23, 38 fixtures), and it
> constrains this phase.** De-vigging the consensus `Avg*` triplet gives a mean
> overround of 6.45% and fair probabilities summing to 1.000000. Doing the same
> on `Max*` gives a mean of 1.019 — and **3 of 38 fixtures (8%) sum below 1.0**,
> as low as 0.892. A book sum under 1.0 is a riskless arbitrage, so those quotes
> demonstrably were not all live at the same moment. Everton–Crystal Palace
> priced a 37.6% "edge" on the home side; Newcastle–Liverpool priced a positive
> edge on *both* home and away at once.
>
> `Max` is a high-water mark over the quoting period, not a takeable price.
> Any `price_edge` computed from it is therefore biased upward, and the naive
> top-of-list selections are all artefacts. Phase C must either model the decay
> from `Max` to a takeable price or select on a single named book. This is the
> same failure FR-16a exists to prevent, arriving through a different door, and
> it is why the existing `e_ceiling = 0.12` automatic-F guardrail is load-bearing
> rather than decorative.

- Re-fit `grading.e_peak` / `sigma` / `e_ceiling` against **`price_advantage`**,
  not `edge_prob`, on the 2019/20–2024/25 window using the sharp-reference rule
  that produced +0.51%. Add a `c_price` sub-score or repoint `c_edge`; either
  way `c_edge = 0` on every bet must stop being the default.
- Decide the sharp reference. Pinnacle (`PS*`) is in the archive but **not in
  the fixture feed**, so the rule as measured cannot be run forward.
  `odds_bfe` (Betfair Exchange) is carried by Phase A as the candidate
  replacement and needs validating against the archive before it is used.
- Add a `selection_rule` block: reference book, edge threshold (2% per the
  measured rule), `fair_price_column: odds_avg`, `available_price_column: odds_max`.
- Set `status: "fitted"`, `w: 0.0`, `w_fitted: true`. **Do not change `w`.**
  A fitted config whose fitted value is zero is the honest state, and it lets
  `require_fitted()` pass without asserting an edge that was not measured.
- Re-run `scripts/select_devig_method.py` to populate
  `devig.method_per_competition` for the seven currently-missing competitions.

### Phase D — cups

- `src/statpitch/data/odds_api.py` — `/events` (0 credits) and `/odds`
  (1 credit), with a monthly budget guard modelled on `statpitch/quota.py`,
  reading `x-requests-remaining` from the response header rather than counting
  locally.
- `build_fixtures.py` merges openfootball (still the source for history and for
  whatever it does publish) with Odds API events. Dedupe on
  (competition, date ±3, club pair).
- Stage is not supplied by the Odds API. Emit `stage_detail=None`,
  `stage_confirmed=false`, and let `taxonomy.resolve_format` fall through to the
  competition default — same honesty rule as `date_confirmed`.
- **Split `odds_coverage` in `competitions.json` into two flags:**
  - `live_odds_coverage` — a price is obtainable now (all 12, once D lands)
  - `benchmark_coverage` — historical closing odds exist for validation
    (the 5 leagues only; nothing else, ever, at $0)

  The serving gate becomes: recommend only where **both** hold. Cups get
  predictions *and* prices *and* a stated refusal citing the absent benchmark —
  which is stronger than today's silence, and keeps the "refusals cite their
  measurement" rule intact.
- Verify `precompute_predictions` handles cup rows once they exist: unrated
  lower-tier entrants must fall through to `entrant_prior`, and
  `neutral_venue`/`format` must reach the matrix.

### Phase E — hardening

- Offline test fixtures for both new sources (repo rule: no test touches the
  network). A trimmed `fixtures.csv` and one `/events` + one `/odds` JSON.
- Two-snapshot schedule: a Friday capture and a T−2h capture, both from
  football-data.co.uk, so both ends of the CLV measurement share a source.
- `requirements-serving.txt` must not gain either scraper;
  `test_deployment.py` enforces it.
- Rewrite MODEL_CARD §6 "No live odds" once it is false. Keep and sharpen the
  cup limitation: prices without a backtest.

---

## 5. Smaller gaps worth closing

- **`collect_fixtures.py` carries dead weight.** API-Football's free plan is
  verified useless (seasons 2022–2024 only) and Phase A gives keyless confirmed
  kickoffs for the same five leagues. The quota machinery has no live caller
  left. Demote or delete.
- **Bundesliga was absent from this week's `fixtures.csv`.** Per-division
  absence is normal; the join must tolerate it without failing the run.
- **`date_confirmed` is mostly false today**, which degrades `/today` and would
  mistime any T−2h odds snapshot.
- **The card artifact inherits the ledger's persistence problem.** Render's disk
  is ephemeral and `STATPITCH_READ_ONLY` blocks writes, so `card.parquet` must be
  built and committed by the scheduled Action, exactly as the ledger is.

---

## 6. Sequencing

A → B → C are one thread and deliver the first real bet recommendation for the
five leagues. D is independent of them and can run in parallel; it delivers cup
*predictions* immediately and cup *prices* behind the new two-flag gate. E closes
behind whichever lands first.
