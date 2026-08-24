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

### Phase B — make the card compute — **done**

Delivered: `src/statpitch/decision/card.py`, `scripts/build_card.py`,
`tests/test_card.py` (18 tests), `paths.card_file()`, plus the serving and
`flag_card` rewiring. First build: **126 selections over 18 fixtures**.

The measurement that matters, and it corrects §1 of this document:

> Every selection grades **F**. Not C or D as estimated above — F.
>
> With `w`=0, `edge_prob` is 0, so `c_edge` is 0 and contributes nothing against
> its 0.30 weight. The remaining sub-scores are not the 1.0 the earlier estimate
> implicitly assumed: `c_robust`, `c_calib` and `c_support` all return their
> 0.5 "unknown" default because no ensemble dispersion, calibration history or
> realised CLV is wired in, and `c_market` reads **0.269** on a normal 6.5%
> consensus overround. The composite is **0.3038**, against a D cutoff of 0.35.
>
> So nothing can grade above F while `w`=0 — regardless of price edge, and
> regardless of what Phase C does to `e_peak`. Repointing `c_edge` at
> `price_advantage` is necessary but **not sufficient**: `c_market` and the three
> 0.5 defaults have to be addressed too, or the ceiling stays below the cutoff.

Of 126 selections, 104 were F'd for non-positive EV, and the 22 with positive EV
were all price-driven — `model_edge` is exactly 0.0 everywhere, as arithmetic
rather than as prose. Two fixtures were flagged with `max_book_sum` below 1.0.

One deviation from the plan below: **`/best-bet` still refuses.** The plan said
it should read the card, but that was written before re-reading MODEL_CARD §4 —
best-bet-per-match selection measured at −2.12% ROI against +0.13% for
committing to one market, a finding independent of `w`. Wiring it to the card
would contradict a measurement, so it keeps its `MAX_EDGE_SELECTION_HARMFUL`
refusal.

The original plan for this phase follows.


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

### Phase C — re-fit the decision config for a price-driven regime — **done, and the answer is no**

Delivered: `src/statpitch/decision/selection_study.py`,
`scripts/study_selection_rules.py`, `data/selection_rule_study.json`,
`odds_bfe` promoted into the archive, and a `selection_rule` block in
`decision_config.json`. **Staking is not enabled**, and that is the result
rather than unfinished work.

**The question.** MODEL_CARD §5's +0.51% CLV was measured on *Pinnacle*-referenced
selections. Phase A found Pinnacle is not published in the live fixture feed. So:
does any reference the feed *does* carry behave the same way?

**Scored `avg → avg`, never `best → best`.** Every rule selects using `odds_max`,
so scoring it on how `odds_max` then moves scores a variable on itself. Measured:
on selected rows the max-vs-consensus spread narrows from +11.64% to +10.23%
while the consensus moves **0.28% against** the bet. That is regression to the
mean wearing the costume of edge, and it is exactly what the tradeable
consensus-referenced rule turned out to be.

**Pre-break — 2019/20–2023/24, holdout excluded, 7,790 matches:**

| reference | in live feed | CLV | clustered t |
|---|---|---|---|
| none (whole book) | — | −0.09% | −2.58 |
| **Pinnacle** | **no** | **+0.51%** | **+7.53** |
| B365 | yes | −0.07% | −1.09 |
| consensus | yes | −0.23% | −3.35 |

Only Pinnacle works, and it is the one book that cannot be traded. Betfair
Exchange does not exist before 2024/25, and 2024/25 is the NFR-10 holdout.

**Post-break — 2025/26 onward, 1,781 matches, the regime the live feed is in:**

| reference | in live feed | CLV | clustered t |
|---|---|---|---|
| none (whole book) | — | −0.03% | −0.39 |
| **Betfair Exchange** | **yes** | **+2.51%** | **+7.86** |
| Pinnacle | no | +1.79% | +3.82 |
| B365 | yes | +1.63% | +4.69 |
| consensus | yes | +1.17% | +1.80 |

Promising, and **not sufficient**. Requirements line 250 sets the primary edge
criterion at CLV "over ≥2 full historical seasons"; this is one. And the same
consensus-referenced rule has *opposite signs* in the two regimes (−0.23% then
+1.17%), which is what a regime-specific artifact looks like — a single
post-break season cannot tell that apart from a real effect.

**So `status` stays `placeholder`.** The plan below proposed flipping it to
`fitted` so `require_fitted()` would pass. That was written before this
measurement. Flipping it now would size real stakes using a rule whose only
multi-season evidence belongs to a book the live feed does not publish.

**The grading changes in the plan are also withdrawn.** Repointing `c_edge` at
`price_advantage` would encode the consensus-referenced rule — the one just shown
to be mean reversion. `c_edge = 0` on every selection is the *correct* output
given `w`=0, not a bug to be tuned away.

**Unblock condition:** 2026/27 completes, giving a second post-break season for
Betfair Exchange. Re-run `scripts/study_selection_rules.py`; if the result holds,
the rule qualifies under Requirements line 250 and `selection_rule.status` can
move from `candidate` to `fitted`.

The original plan for this phase follows, superseded above.


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

### Phase D — cups — **partly done; 1 of 7, and the rest need a credential**

Delivered: `src/statpitch/data/openligadb.py` (keyless), merged into
`build_fixtures.py`, the `odds_coverage` split, and the FR-9 fill in
`precompute_predictions.py`. **Cup fixtures are in the artifact for the first
time** — 2 DFB-Pokal round-1 ties, with confirmed UTC kickoffs and a parsed
round.

**Root cause reconfirmed.** openfootball has not come back: every 2026-27 cup
path still 404s and `champions-league` still has no 2026-27 directory.

**What is now covered.** OpenLigaDB is keyless and gives the DFB-Pokal with
`matchDateTimeUTC` *and* `group.groupName` ("1. Runde"), which
`openfootball.normalise_stage` already parses because the same German labels
appear in the files it was written for. Stage is not cosmetic: it drives
`resolve_format` and `is_neutral_venue`, so an unknown stage would price a
two-legged tie as a single leg.

**What is not.** The FA Cup, Copa del Rey, Coppa Italia, Coupe de France and both
UEFA competitions have no keyless source. The Odds API covers all of them but
needs a key, and none is configured — no `.env` exists. `build_fixtures` now
distinguishes *undrawn* from *unsourced* in its logging, because conflating them
is how a dead upstream hides for a fortnight.

**The bug this phase exposed.** The first cup fixture ever to reach the offline
prediction path produced **Hamburg Eimsbütteler BC at 52.9% to beat Borussia
Dortmund** — a fifth-tier amateur side as favourite. Club Elo rates only the top
two tiers, so the club had no rating; left as a null, the fitted model did not
abstain, it predicted from the remaining features and invented a number.

Worse, the two routes disagreed: `/predict` applied the FR-9 entrant prior and
said 19.6%, while `/fixtures/upcoming` served the precomputed 52.9% for the same
fixture — and the bulk route is the one a consumer syncs from. `precompute` now
applies the pooled entrant level through
`entrant_prior.fill_missing_ratings`, and the fixture reads **9.1%**, in line
with the Osnabrück–Bayern tie at 11.3%. The rating tier travels with the row as
`home_rating_source` / `away_rating_source`.

**The coverage split shipped.** `live_odds_coverage` (a price can be had) and
`benchmark_coverage` (history exists to validate against) are recorded
separately, with `odds_coverage` kept as their conjunction and its exact meaning
(NFR-13). Today all three move together; they will diverge the moment a keyed
odds feed gives the cups prices their closing-odds history will never have. The
cup refusal now names which half is missing.

The original plan for this phase follows.


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
