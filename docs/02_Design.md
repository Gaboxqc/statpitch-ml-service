# StatPitch v2 — Design
## Spec-Driven Development · Artifact 2/3: Design (Diseño)

*Derived from `01_Requirements.md` v2.0. Every design decision below traces back to a functional or non-functional requirement (FR-x / NFR-x).*

**Spec revision 2.0** — adds Layer 5, the Decision Layer (§6), with full formula specifications. Sections marked 🆕 are new in this spec revision.

---

## 1. Architecture Overview

StatPitch v2 keeps v1's four-layer shape (Data → Features → Models → Serving) but adds a **Competition Taxonomy layer underneath everything**, because — unlike a single World Cup model — v2 has to reason about 12+ competitions with different formats (round-robin league, single-elimination cup, two-legged ties, Swiss-format continental league phase) on one shared foundation.

🆕 This spec revision inserts a **Decision Layer between Models and Serving**. The Models layer answers *what will happen*; the Decision Layer answers *what to do about it*. Keeping these separate matters architecturally: the model can be retrained, swapped, or recalibrated without touching staking logic, and staking parameters can be re-tuned without retraining anything.

```
Competition Taxonomy  →  Data  →  Features  →  Models  →  Decision  →  Serving
                                                            🆕
```

---

## 2. Competition Taxonomy & Unified Data Model

Every match in the system is tagged with:

| Field | Example values |
|---|---|
| `competition_id` | `ENG.PL`, `ESP.LALIGA`, `ESP.COPA_DEL_REY`, `UEFA.UCL` |
| `competition_type` | `league` \| `domestic_cup` \| `continental_cup` |
| `format` | `round_robin` \| `single_leg_knockout` \| `two_leg_knockout` \| `swiss_league_phase` |
| `tier` | integer, 1 = top flight (used for lower-division cup-entrant priors, FR-9) |
| `leg_number` | `null` \| `1` \| `2` (two-legged ties only) |
| `neutral_venue` | boolean (finals are usually neutral) |
| 🆕 `odds_coverage` | boolean — whether free closing-odds data exists for this competition. **Gates the entire Decision Layer.** True for the 5 leagues, false for all cups and continental competitions in the first release |

This is the key structural upgrade over v1, which only ever had one format (single group-then-knockout tournament). The v2 model layer reads `format` and branches its inference logic (§5.3) instead of needing a separate bespoke system per competition — this is what lets NFR-6 (scale to 20+ competitions without a redesign) hold.

🆕 The `odds_coverage` flag is the mechanism that keeps the honest scoping of Requirements §9 enforced in code rather than in prose: a fixture without odds coverage returns predictions but no bet recommendations, automatically, with a stated reason.

---

## 3. Layer 1 — Data

| Source | What it feeds | Ingestion method | Refresh cadence |
|---|---|---|---|
| football-data.co.uk | Results, match stats, pre-closing + closing odds (1X2, O/U 2.5, AH), market Max & Avg | Scheduled CSV download | Weekly during season |
| openfootball (GitHub, CC0) | League + domestic cup historical results, UCL/UEL fixtures & results | `git pull` of public repos | Weekly |
| understat.com | Shot-level xG, per-match xG for Big 5 + RFPL | Polite scraper, cached, rate-limited | Weekly |
| clubelo.com API | Elo rating per club per date, full pyramid | Free CSV API call | Daily (cheap, no key needed) |
| Transfermarkt | Squad market value, avg. age | `cloudscraper`, cached (as in StatPitch v1) | Each transfer window (Jan/Aug) + monthly light refresh |
| API-Football (free tier) | Live fixtures, injuries, confirmed lineups | REST calls, **hard-budgeted** — see §3.2 | Matchday-driven |

### 3.1 🆕 Odds data model

football-data.co.uk gives two price snapshots per fixture, which is the foundation of the CLV work (FR-26):

| Snapshot | Columns | Collected |
|---|---|---|
| Pre-closing | `B365H`, `MaxH`, `AvgH`, `B365>2.5`, `AHh`, `MaxAHH`, `AvgAHH`, … | Friday afternoon (weekend fixtures), Tuesday afternoon (midweek) |
| Closing | Same names with `C` inserted: `B365CH`, `MaxCH`, `AvgCH`, `MaxCAHH`, … | Just before kickoff |

Two derived series per selection, used throughout the Decision Layer:

```
q_fair(t)      = devig(AvgC*)   →  best consensus estimate of true probability   (FR-16a)
price_avail(t) = Max*           →  the price actually obtainable                  (FR-16a)
```

**Never use `Max*` to compute fair probability.** Max odds across many books are systematically biased — the maximum of N noisy prices is above consensus by construction, so de-vigging it produces fair probabilities that look like free money and are not. Fair probability comes from `Avg*`, price comes from `Max*`, and the design keeps these in separate fields to make the mistake structurally hard to commit.

**Pinnacle regime boundary.** Per Requirements §7.3, Pinnacle's feed became unreliable on 23/07/2025 and is excluded from Max/Avg from that date. The ingestion layer tags every row with `odds_regime ∈ {pre_2025_07_23, post_2025_07_23}`. Backtests must either restrict to one regime or model the break explicitly; silently pooling them is a correctness bug, not a stylistic choice.

### 3.2 🆕 API-Football request budget (NFR-9)

100 requests/day, hard. Budget:

| Call | Frequency | Daily cost (heavy Saturday) |
|---|---|---|
| Fixtures for the day, all competitions | 1× at 06:00 UTC | 1 |
| Injuries, batched per competition | 5× | 5 |
| Confirmed lineup, one call per fixture at T−45min | ≤50× | ≤50 |
| Reserve for retries / manual queries | — | ~44 |

Rules enforced in `live_fetcher.py`: no polling loops, one lineup attempt per fixture, a local cache keyed by fixture ID with no re-fetch inside 24h, and a hard daily counter that stops issuing calls at 90 and falls back to pre-match estimates (NFR-7). The fallback path is a quota-protection mechanism, not only a robustness one.

**Odds coverage note**: football-data.co.uk covers league divisions only. Domestic cups and UCL/UEL rely on results-only data from openfootball. No free odds source with cup coverage exists — the-odds-api was evaluated and rejected (Requirements §7.2). The closing-line benchmark and Decision Layer are therefore scoped to league matches in the first release via the `odds_coverage` flag.

---

## 4. Layer 2 — Features

Builds on StatPitch v1's 42 features, restructured into groups, with new club-football-specific groups added.

**Carried over from StatPitch v1 (adapted per-club instead of per-nation):**
- Elo ratings (now sourced from clubelo.com instead of computed from scratch — more reliable, covers the full pyramid)
- Rolling form & rolling xG (last 5 / last 10)
- Head-to-head history
- Rest days
- Squad market values (5 features)

**New — club-football-only groups:**

| Group | Features | Why it matters (ties to requirement) |
|---|---|---|
| Cross-competition Elo bridge | UCL/UEL-adjusted cross-league strength delta | FR-11: enables true cross-league comparison |
| Competition context | `competition_type`, `format`, `leg_number`, `tier`, cup-round importance weight | Ties to §2 taxonomy; drives model branching |
| Fixture congestion | Matches played in last 14 days *across all competitions*, days since last match | FR-17 |
| Confirmed lineup delta | (Confirmed XI strength) − (pre-match estimated XI strength), populated ~60 min pre-kick-off | FR-18 |
| Manager tenure | Days since appointment, "new manager bounce" flag (first 6 matches) | FR-19 |
| Cup-specific priors | Lower-tier Elo prior for sub-top-flight cup entrants, rotation-risk flag | FR-9, FR-10 |
| Two-leg state | Aggregate score so far, away-goals-rule flag (historical seasons only — abolished by UEFA in 2021), home/away leg order | FR-7 |
| 🆕 Line movement | `Δ = q_fair(closing) − q_fair(pre-closing)`, sign and magnitude, per selection | Feeds FR-34 edge decay; also a *feature* — line drift carries information the model may lack |
| 🆕 Motivation / dead rubber | Points-to-safety, points-to-title, mathematically-eliminated flag, matchweek index | FR-33 guardrails; markets price these unevenly late in the season |

### 4.1 🆕 A caution on the line-movement feature

Using line movement as a *model input* creates a subtle trap: the model partially learns to reproduce the market, which inflates apparent accuracy while destroying measured edge (you cannot beat a line you are copying). The design therefore trains **two model variants**:

- **`model_pure`** — no odds-derived features. Used for all edge, CLV, and value calculations.
- **`model_informed`** — includes line movement. Used for the standalone accuracy/Brier/RPS headline numbers only.

Reporting must state which variant produced which number. Conflating them is the single easiest way to produce a dishonest result in a project like this.

---

## 5. Layer 3 — Models

### 5.1 Shared ensemble (same core idea as StatPitch v1, extended)

- **XGBoost Classifier** for 1X2, trained across **all** competitions jointly with a `competition_id` embedding feature, so data-rich leagues (Premier League) improve predictions for thinner-data competitions (Coupe de France) via shared structure — this is the multi-task design that lets one model serve 12+ competitions instead of 12 separate models.
- **XGBoost-Poisson regressors** for home/away expected goals — but with a **per-competition goal-environment offset**, since e.g. Bundesliga's long-run average goals/match runs meaningfully higher than Serie A's or Ligue 1's. Without this correction, a pooled model systematically over- or under-predicts goals depending on which league a match belongs to.
- **Dixon-Coles score matrix** per match, tau-corrected, built from the above lambda values — unchanged approach from StatPitch v1, just parameterized per competition. 🆕 This matrix is now the **single source of truth for every market** (§6.1), so its accuracy matters far more than in v1.
- **Neural network (MLP)**, same 128-64-32 shape, same standardized-input approach as StatPitch v1.
- **Isotonic calibration + logistic-regression meta-learner**, same blending approach as StatPitch v1.
- **Time-decay sample weights**: `exp(-0.15 × years_ago)`, with the multiplier extended so continental knockout matches get the highest weight, domestic cup matches a medium weight, and league matches the baseline.
- 🆕 **Ensemble dispersion is retained, not discarded.** The spread of the individual member predictions around the blended output is persisted per fixture as `p_std`, and becomes the edge-robustness input to bet grading (§6.4). Most ensembles throw this away; here it is a product feature.

### 5.2 League-embedding detail

Rather than one-hot encoding 12 competitions (which fragments training data), the model learns a low-dimensional embedding per `competition_id`, alongside the numeric features. This is the mechanism that satisfies NFR-6 — adding a 13th competition later means adding one new embedding row, not retraining a bespoke pipeline.

### 5.3 Format-aware inference branching

At inference time, the served model branches on `format`:

- `round_robin` / `swiss_league_phase` → standard 1X2 + score-matrix flow (as StatPitch v1).
- `single_leg_knockout` → same score matrix, plus an extra-time/penalty sub-model: if regulation ends level, apply a symmetric coin-weighted-by-form penalty-shootout probability (penalties are close to a coin flip in the literature, lightly adjusted by penalty-taking history where available).
- `two_leg_knockout` → run the leg-1 score matrix and leg-2 score matrix as two draws, then **Monte Carlo combine** them (≥10,000 simulated tie outcomes) to produce an aggregate qualification probability (FR-7). Leg 2 draws condition on the actual leg-1 result once played.

### 5.4 Evaluation module (metrics only — staking moved to Layer 4)

An evaluation pipeline, not a predictive model:

1. For every backtested match with odds coverage, de-vig closing odds using the per-competition method selected in §6.2.
2. Compute **Brier score, log-loss, and RPS** for both StatPitch and the market, side by side (FR-14).
3. Produce calibration curves and ECE per competition × probability decile, for model and market (FR-16b).
4. Output `backtest_report.json` — reported honestly per NFR-3, including seasons and competitions where the model does *not* beat the market.

🆕 **RPS specification.** For ordered outcomes (Home, Draw, Away), with cumulative predicted probabilities and cumulative outcome indicators:

```
RPS = (1 / (r − 1)) · Σ_{i=1}^{r−1} ( Σ_{j=1}^{i} (p_j − e_j) )²        with r = 3
```

RPS is the standard metric in the football-forecasting literature precisely because Brier score treats "predicted home, got draw" and "predicted home, got away win" as equally wrong, which for an ordinal outcome they are not.

🆕 **Validation protocol (NFR-10).** Purged walk-forward only: train on seasons ≤ *n*, validate on *n+1*, with a purge gap so no rolling feature spans the boundary. One season is designated an untouched holdout and is not looked at until the final Phase 8 report. Every reported ROI carries a bootstrap CI over per-bet returns and a t-statistic.

---

## 6. 🆕 Layer 4 — Decision Layer

Where Layer 3 stops at *what will happen*, Layer 4 answers *what to do*. Five modules, in strict dependency order — each depends on the one before it, which is also the recommended build order.

```
market_engine.py  →  devig.py  →  value.py  →  bet_grader.py  →  staking.py  →  clv_tracker.py
```

### 6.1 `market_engine.py` — score matrix → all markets (FR-23)

Input: the Dixon-Coles matrix `P[i][j]` = probability of exactly `i` home goals and `j` away goals (truncated at 10-10, renormalised). Output: ~50–60 selections, each a simple summation over matrix cells.

| Market | Derivation |
|---|---|
| 1X2 | `Σ P[i>j]`, `Σ P[i=j]`, `Σ P[i<j]` |
| Double Chance | Pairwise sums of the above |
| Draw No Bet | `P(home) / (1 − P(draw))` |
| Over/Under line *L* | `Σ P[i+j > L]` |
| Team totals | Row/column marginals |
| BTTS | `Σ P[i≥1 and j≥1]` |
| Asian Handicap *h* | `Σ P[i + h > j]` (win), `= j` (push), `< j` (loss) |
| Correct score | `P[i][j]` directly |

**Quarter lines** (e.g. −0.25, +0.75) split the notional stake across the two adjacent half-lines, producing win / half-win / push / half-loss / loss outcomes. The engine must return the full outcome distribution per selection, not just a win probability, because the staking module needs the payoff distribution to compute log-growth correctly.

**Correct score is generated but tagged `stakeable = false`** (Requirements §3.2) so it can be displayed under FR-4 while being structurally excluded from every staking path.

### 6.2 `devig.py` — three methods, empirically selected (FR-28)

With raw implied probabilities `p_i = 1 / o_i` and `S = Σ p_i` (the overround):

| Method | Formula |
|---|---|
| Proportional | `q_i = p_i / S` |
| Power | solve for `k` such that `Σ (p_i)^k = 1`, then `q_i = (p_i)^k` |
| Shin | solve for `z` such that `Σ q_i = 1`, where `q_i = [ √(z² + 4(1−z)·p_i²/S) − z ] / (2(1−z))` |

Both power and Shin are one-dimensional root-finds — `scipy.optimize.brentq` over a bounded interval, a few lines each.

**Selection procedure**: for each competition, de-vig all historical closing odds with each method, then compare each method's fair probabilities against realized outcomes by log-loss and calibration error. Persist the winner per competition in `decision_config.json`.

**Why this is load-bearing and not a detail.** Bookmakers do not distribute margin uniformly; they load it onto longshots. Proportional de-vig therefore *overstates* the fair probability of longshots, which means the model will appear to find value on draws and away underdogs where none exists. Getting this wrong doesn't produce a slightly worse system — it produces a system whose value flags point at exactly the bets that lose money.

### 6.3 `value.py` — edge and expected value (FR-16a)

Strict separation of the two market numbers:

```
q_fair       = devig(AvgC*)[selection]          # consensus true probability
o_avail      = Max*[selection]                  # best obtainable price
p_model      = model_pure prediction            # never model_informed (§4.1)

edge_prob    = p_model − q_fair                 # in probability points
EV           = p_model · o_avail − 1            # at the obtainable price
```

Two distinct edge sources fall out of this and are reported separately, because they behave differently and decay differently:

- **Model edge** — `p_model` disagrees with consensus.
- **Price edge** — `o_avail` is better than consensus implies, i.e. one book is off the market, independent of the model.

Price edge is often the more reliable of the two and is available even when the model is merely accurate rather than superior.

### 6.4 `bet_grader.py` — A–F classification (FR-25)

Composite confidence score in [0, 1] from five sub-scores:

| Sub-score | Definition |
|---|---|
| `c_edge` | Non-monotonic in edge size — see below |
| `c_robust` | Decreasing in `p_std` (ensemble dispersion, §5.1); optionally grade on the bootstrap lower bound of `edge_prob` |
| `c_market` | Decreasing in book margin, increasing in number of quoting books and line stability |
| `c_calib` | Historical ECE for this competition × probability decile, inverted |
| `c_support` | Realized CLV of historically similar bets (same market family, price bucket, edge bucket) |

**The non-monotonic edge term.** This is the design's most important single rule:

```
c_edge = exp( − ( (edge_prob − e_peak) / σ )² )        e_peak ≈ 0.04, σ ≈ 0.05
c_edge = 0  and  grade = F                              if edge_prob > e_ceiling (≈ 0.12)
```

In a market this efficient, real edges are 2–5 probability points. An apparent 20-point edge is overwhelmingly more likely to mean the model is blind to something — a key injury not yet in the feature set, a dead rubber, confirmed rotation before a midweek European tie — than that the market has mispriced by 20 points. Naive systems bet hardest exactly there and lose fastest. StatPitch instead grades these **F ("model likely blind")**, never stakes them, and routes them to a review queue where they become the most valuable model-diagnostic signal the system produces.

Grade cutoffs and their staking multipliers (all in `decision_config.json`, NFR-12):

| Grade | Composite | Action |
|---|---|---|
| A | ≥ 0.80 | Full fractional-Kelly stake (1.0 × λ) |
| B | 0.65–0.80 | Half stake (0.5 × λ) |
| C | 0.50–0.65 | Quarter stake (0.25 × λ), logged only |
| D | 0.35–0.50 | No bet — monitor |
| F | < 0.35, or edge > ceiling, or a guardrail fired | No bet — investigate |

**Guardrails (FR-33)** are evaluated before grading and force an F with a logged reason: unconfirmed lineup with a key player doubtful, dead rubber, fixture within 72h in another competition, `p_std` above threshold, book margin above threshold, price above the odds ceiling, `odds_coverage = false`.

### 6.5 `staking.py` — risk-managed Kelly (FR-27)

**Step 1 — shrink toward the market.** The model's probability is an estimate with real error; full Kelly on an estimated probability is a bankruptcy machine.

```
p_used = w · p_model + (1 − w) · q_fair
```

`w` is **fitted, not assumed** — chosen on validation data to maximise realized log-growth. It is the project's most informative single number: it quantifies how much information the model adds over the market. If it fits near zero, the honest conclusion is that the model adds nothing, and Requirements §8.4 makes reporting it mandatory.

**Step 2 — Kelly fraction and log-growth.**

```
f*  = ( p_used · o − 1 ) / ( o − 1 )
g   = p_used · ln(1 + f*·(o−1)) + (1 − p_used) · ln(1 − f*)
```

For quarter lines and other push-capable selections, `g` is computed over the full payoff distribution from §6.1 rather than the two-outcome form above.

**`g` is the ranking key for FR-24's "best bet".** Ranking by EV instead would always crown the longshot: a nominal 20% edge on a 15.0 correct score beats a 3% edge on AH −0.5 at 1.95 on EV, but the first is model error and the second is the realizable bet. Log-growth is the correct objective for a compounding bankroll and penalises variance automatically — it is not a heuristic tie-breaker but the mathematically right answer to "which of these is the best bet."

**Step 3 — fraction, caps, exclusions.**

```
stake = grade_multiplier · λ · f*
stake = min(stake, cap_per_bet)                        # default 2% of bankroll
stake = 0  if  o > odds_ceiling  or  f* ≤ 0  or  grade ∈ {D, F}
```

`λ` defaults to 0.25. The backtest reports the full frontier across `λ ∈ {0.10, 0.25, 0.50, 1.00}` — quarter Kelly captures most of the long-run growth at a small fraction of the drawdown, and publishing that curve is more informative than asserting a single value.

**Step 4 — simultaneous allocation.** A Saturday slate runs 10+ bets concurrently, many correlated (same fixture, same team, overlapping markets — "Home win" and "Over 2.5" are positively correlated for a strong favourite). Sequential single-bet Kelly over-stakes badly here.

The system already has the machinery to do this properly: simulate the joint outcome of the slate from the per-fixture score matrices (fixtures independent, selections within a fixture jointly determined by the matrix), then solve

```
maximise  E[ ln( 1 + Σ_k f_k · r_k ) ]     subject to  Σ f_k ≤ cap_matchday,  f_k ≥ 0
```

numerically over the simulated outcome paths (`scipy.optimize.minimize`, SLSQP). This is the same Monte Carlo apparatus built for FR-7 and FR-20, reused — no new infrastructure.

### 6.6 `clv_tracker.py` — the ledger and the verdict (FR-26, FR-29)

Every graded recommendation is appended to `bet_ledger.jsonl` at flag time:

```json
{
  "ts_flagged": "2026-08-08T14:02:11Z", "fixture_id": "ENG.PL-2026-08-09-ARS-CHE",
  "selection": "AH_HOME_-0.5", "book": "Max", "odds_taken": 1.98,
  "p_model": 0.548, "q_fair": 0.521, "edge_prob": 0.027, "p_std": 0.019,
  "grade": "B", "stake_fraction": 0.0091, "kelly_lambda": 0.25, "w": 0.34,
  "config_version": "dec-2026.08.1",
  "odds_closing": null, "result": null, "clv_pct": null
}
```

A post-match GitHub Actions job settles each row, filling `odds_closing`, `result`, and `clv_pct`.

```
CLV%     = (odds_taken / odds_closing) − 1
CLV_prob = q_fair_closing − q_fair_at_flag
```

**CLV is the headline metric, not ROI** (FR-26). Over a few hundred bets, ROI is dominated by variance — a positive figure is routinely produced by luck alone. CLV converges far faster and is the standard evidence of genuine edge. The reporting rule is explicit: **positive ROI with negative CLV is reported as absence of demonstrated edge, not as success.**

Because football-data.co.uk's base snapshot is Friday afternoon rather than a true opening line, this must always be labelled **"Friday-to-close CLV"**. It understates what an early bettor could capture. It is a valid edge signal reported with an accurate name.

### 6.7 `bankroll_sim.py` — risk of ruin (FR-30)

Resample the settled ledger (≥10,000 paths, block bootstrap to preserve matchday correlation) and report per `λ`: median ROI, 5th/95th percentiles, max-drawdown distribution, probability of ruin at −50%, and time-to-recovery. The fan chart of bankroll paths is the single most communicative output the project produces — it turns "positive expected value" into a visible picture of how bad a bad season can look.

### 6.8 `edge_map.py` — where the edge actually lives (FR-31)

Backtested mean CLV, ROI, bet count, and t-statistic, cross-tabulated by competition × market family × price bucket, rendered as a heatmap with cells below a minimum sample size greyed out rather than shown as noise.

Expected reading, stated in advance so the result is interpreted honestly rather than rationalised after the fact: **1X2 in the Big 5 will likely show no edge** — those markets are brutally efficient. Asian Handicap and totals are the plausible hunting grounds, on margin grounds alone. The cup hypothesis from FR-31 cannot be tested in v1 because cup odds are unavailable free.

---

## 7. Layer 5 — Serving

**FastAPI backend**, same pattern as StatPitch v1: load all `.pkl` artifacts once at startup, serve in-memory.

### API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info & version |
| GET | `/health` | Status check |
| GET | `/competitions` | All in-scope competitions, current stage, `odds_coverage` flag |
| GET | `/teams/{competition_id}` | Teams ranked by Elo within a competition |
| GET | `/predict/{competition_id}/{home}/{away}` | Full single-match prediction |
| POST | `/predict` | Same, with optional date, injuries, confirmed lineup override |
| GET | `/predict/tie/{competition_id}/{team_a}/{team_b}` | Two-legged aggregate tie prediction (FR-7) |
| GET | `/today` | All fixtures today across all in-scope competitions, with predictions |
| GET | `/simulate/{competition_id}` | Full bracket/table simulation for a cup or continental competition |
| GET | `/value-bets/today` | Fixtures where model probability diverges from market odds (FR-16) |
| GET | `/backtest/{competition_id}` | Brier / log-loss / RPS / ECE vs. closing odds (FR-13–14, FR-16b) |
| 🆕 GET | `/markets/{competition_id}/{home}/{away}` | All ~60 priced selections for a fixture, with edge, EV, log-growth and grade (FR-23) |
| 🆕 GET | `/best-bet/{competition_id}/{home}/{away}` | Single highest-log-growth selection + grade + SHAP reasoning (FR-24, FR-32) |
| 🆕 GET | `/card/today` | Full graded matchday slate, correlation-aware simultaneous Kelly sizing (FR-27) |
| 🆕 GET | `/clv/report` | Live CLV track record from the ledger — the project's headline number (FR-26) |
| 🆕 GET | `/edge-map` | Efficiency heatmap by competition × market × price bucket (FR-31) |
| 🆕 GET | `/bankroll/simulate?lambda=0.25` | Drawdown distribution and risk of ruin (FR-30) |
| 🆕 GET | `/ledger` | Paginated bet ledger, including suppressed bets with their guardrail reasons |
| GET | `/docs` | Auto-generated interactive API docs |

### 🆕 7.1 API backward-compatibility contract

**The existing frontend must keep working against v2 without modification.** This is a binding constraint on the build, not a courtesy:

1. **No v1 endpoint is renamed, removed, or given a different path.** Every route in the v1 design is present above with identical spelling and parameter order.
2. **No existing response field is renamed, removed, or has its type changed.** v2 is **additive only** — new fields may appear in existing responses, but a frontend that ignores unknown keys sees no difference.
3. **New functionality lives at new routes.** All Decision Layer capability is reachable via the 🆕 endpoints; none of it alters the shape of a v1 response.
4. **Deprecation, if ever needed, is versioned** — a `/v2/...` prefix alongside the unprefixed original, never an in-place change to a live route.

Two fields *are* added to existing responses, both safe under rule 2 because they are new keys rather than changes to existing ones:

| Field | Where | Behaviour for an unmodified frontend |
|---|---|---|
| `disclaimer` | Any response carrying a stake recommendation (NFR-11) | Ignored — a string the old client doesn't read |
| `bet_recommendation` | Prediction responses; `null` with a reason string where `odds_coverage = false` | Ignored — the old client never asked for it |

`/value-bets/today` retains its v1 response shape exactly. Richer, graded output is served from the new `/card/today` route instead, so the existing frontend's value-bet view is untouched and can be upgraded later by pointing it at the new route when convenient.

**Acceptance check for Phase 7**: point the existing frontend at the v2 deployment with zero code changes and confirm every screen renders as before.

### Deployment

- **Render.com free tier**, connected to GitHub for auto-redeploy on push.
- **GitHub Actions scheduled workflows** (free, 2,000 min/month on public repos) for scheduled scraping and refresh, 🆕 plus two new jobs: `settle_ledger` (post-match, fills closing odds and results) and `flag_card` (pre-matchday, writes the day's graded card to the ledger). Ledger settlement is what makes the CLV track record accumulate automatically.
- **Google Colab + Google Drive** for training, identical pattern to StatPitch v1.

---

## 8. Data storage schema

```
data/
  competitions.json          Taxonomy table (§2), incl. odds_coverage flag
  team_stats.json            Current stats per club (Elo, form, xG, squad value, memberships)
  elo_ratings.json           Cross-competition Elo history, sourced from clubelo.com
  h2h_stats.json             Head-to-head lookup, all club pairs across competitions played
  cup_bracket_state.json     Current round, remaining ties, aggregate scores for active cups
  model_config.json          Feature columns, per-competition Dixon-Coles rho, model version
  closing_odds.json          De-vigged closing odds by match, with odds_regime tag
  backtest_report.json       Brier / log-loss / RPS / ECE vs. market (§5.4 output)
🆕 decision_config.json      λ, w, grade cutoffs, caps, odds ceiling, per-competition de-vig
                             method, config_version — every Decision Layer parameter (NFR-12)
🆕 bet_ledger.jsonl          Append-only log of every graded recommendation (FR-29)
🆕 clv_report.json           Aggregated CLV: mean, SE, positive rate, by competition & market
🆕 edge_map.json             Competition × market × price-bucket edge table (FR-31)
🆕 bankroll_sim.json         Drawdown / risk-of-ruin distributions per λ (FR-30)
🆕 calibration.json          Per-competition, per-decile calibration curves and ECE (FR-16b)
```

---

## 9. Architecture diagram

```
Training pipeline (Google Colab)
  ├── Notebook 01  Data collection        football-data.co.uk + openfootball → matches_clean.csv
  ├── Notebook 02  Elo integration        clubelo.com API → elo_ratings.json
  ├── Notebook 03  Feature engineering    rolling form/xG, H2H, congestion, cup priors
  ├── Notebook 04  Base model training    XGBoost + Poisson per competition → .pkl files
  ├── Notebook 05  League embedding       joint multi-competition training → shared .pkl
  ├── Notebook 06  Model improvements     Optuna, Dixon-Coles per-league rho, stacking, NN
  ├── Notebook 07  xG enrichment          understat.com scrape → xg_features.csv
  ├── Notebook 08  Squad values           Transfermarkt scrape → squad_values.json
  ├── Notebook 09  Two-leg tie engine     Monte Carlo aggregate-tie simulator
  ├── Notebook 10  Cup/bracket simulator  Monte Carlo domestic cup + UCL/UEL simulation
  ├── Notebook 11  Market evaluation      Brier / log-loss / RPS / ECE vs. closing odds
  ├──🆕 Notebook 12  De-vig comparison     3 methods × competitions → decision_config.json
  ├──🆕 Notebook 13  All-markets engine    Score matrix → ~60 selections, validated vs. book prices
  ├──🆕 Notebook 14  Staking & grading     Fit w, sweep λ, tune grade cutoffs, Kelly backtest
  └──🆕 Notebook 15  CLV & risk analysis   CLV backtest, edge map, bankroll sim, risk of ruin

Production stack (Render.com, free tier)
  ├── main.py               FastAPI app — HTTP routing
  ├── predictor.py          Model inference — .pkl loading, format-aware branching (§5.3)
  ├── tie_engine.py         Two-legged aggregate simulation
  ├── live_fetcher.py       Fixtures, injuries, lineups — quota-budgeted (§3.2)
  ├──🆕 market_engine.py    Score matrix → all markets (§6.1)
  ├──🆕 devig.py            Proportional / power / Shin (§6.2)
  ├──🆕 value.py            Edge & EV, fair-vs-available separation (§6.3)
  ├──🆕 bet_grader.py       A–F grading + guardrails (§6.4)
  ├──🆕 staking.py          Fractional & simultaneous Kelly (§6.5)
  ├──🆕 clv_tracker.py      Ledger append & settlement (§6.6)
  ├──🆕 bankroll_sim.py     Risk of ruin (§6.7)
  ├──🆕 edge_map.py         Efficiency heatmap (§6.8)
  ├── models/               Trained .pkl files
  └── data/                 See §8 schema

Automation (GitHub Actions, free)
  ├── scheduled scraping/refresh workflows → git push → Render auto-redeploy
  ├──🆕 flag_card     pre-matchday   → append graded card to bet_ledger.jsonl
  └──🆕 settle_ledger post-match     → fill closing odds, results, CLV
```

---

## 10. Technology stack

| Layer | Technology |
|---|---|
| Training | Python, Google Colab, Google Drive |
| ML models | XGBoost, scikit-learn, scipy, MLPClassifier |
| Data | pandas, numpy |
| Statistics | Poisson distribution, Dixon-Coles correction, Elo rating system |
| Optimisation | Optuna |
| Calibration | Isotonic regression (`CalibratedClassifierCV`) |
| Simulation | Monte Carlo (two-leg ties, cup brackets, 🆕 bankroll paths, 🆕 joint slate outcomes) |
| 🆕 Root-finding | `scipy.optimize.brentq` (power & Shin de-vig) |
| 🆕 Constrained optimisation | `scipy.optimize.minimize` SLSQP (simultaneous Kelly) |
| 🆕 Explainability | SHAP (per-bet reasoning, FR-32) |
| API | FastAPI, Uvicorn |
| Deployment | Render.com (free tier), GitHub |
| Automation | GitHub Actions (scheduled workflows) |
| Data scraping | `cloudscraper`, BeautifulSoup, `requests` |
| Odds & results | football-data.co.uk, openfootball (CC0) |
| xG data | understat.com |
| Elo ratings | clubelo.com API |
| Squad values | Transfermarkt (scraped) |
| Live fixtures | API-Football (free tier, 100 req/day) |

---

## 11. Design decisions explicitly deferred to Tasks

Implementation details rather than architectural choices: exact Optuna trial counts per competition, the exact GitHub Actions cron schedule, 🆕 the initial numeric values for `e_peak`, `σ`, `e_ceiling`, and the grade cutoffs (§6.4 gives defaults to be tuned in Notebook 14), 🆕 the score-matrix truncation point, and 🆕 the minimum sample size for an edge-map cell to be displayed rather than greyed out. See `03_Tasks.md`.

## 12. 🆕 Design decisions made and their rationale

| Decision | Alternative rejected | Why |
|---|---|---|
| Decision Layer separate from Models | Staking logic inside `predictor.py` | Lets the model be retrained without touching staking, and staking re-tuned without retraining |
| Rank by log-growth | Rank by EV | EV always crowns the longshot; log-growth is the correct compounding objective |
| Non-monotonic edge confidence | Confidence increasing in edge | Large apparent edges in an efficient market signal model blindness, not opportunity |
| Fair from `Avg`, price from `Max` | De-vig `Max` for both | Max-of-N is upward-biased by construction; de-vigging it fabricates edge |
| CLV as headline metric | ROI as headline metric | ROI over a few hundred bets is dominated by variance; CLV converges far faster |
| `w` fitted, not assumed | Fixed blend weight | `w` is the honest measure of whether the model adds anything over the market |
| Two model variants (pure / informed) | One model with odds features | Odds features inflate accuracy while destroying measurable edge; separating them prevents a self-deceiving result |
| `odds_coverage` as a taxonomy field | Prose note about cup scoping | Enforces the honest limitation in code rather than in documentation |
