# StatPitch v2 — Tasks
## Spec-Driven Development · Artifact 3/3: Tasks (Tareas)

*Derived from `02_Design.md` v2.0. Each task references the requirement(s) and design section it implements. Tasks are grouped into phases; within a phase, order matters less, but phases should generally proceed in sequence since later phases depend on earlier data/artifacts.*

**Spec revision 2.0** — adds Phase 5.5 (the Decision Layer), tightens Phase 5 with validation-protocol tasks, and adds an early "truth serum" checkpoint. Items marked 🆕 are new in this spec revision.

---

## Phase 0 — Setup

- [ ] Create GitHub repo, Google Drive project folder, and Colab notebook skeleton (Design §9 pipeline structure).
- [ ] Set up `competitions.json` taxonomy file by hand for the 12 in-scope competitions (Design §2). *This is the foundation every later task depends on — get the `format` field right for each competition (round_robin / single_leg_knockout / two_leg_knockout / swiss_league_phase).*
- [ ] 🆕 Add the `odds_coverage` boolean to every taxonomy row — `true` for the 5 leagues, `false` for all cups and continental competitions. This flag gates the entire Decision Layer and enforces the honest scoping of Requirements §9 in code rather than in prose.
- [ ] Register free access: API-Football free tier (direct or via RapidAPI). No signup needed for football-data.co.uk, openfootball, understat, clubelo, Transfermarkt.
- [ ] 🆕 Write the API-Football request budget into `live_fetcher.py` from day one (Design §3.2, NFR-9): daily counter, hard stop at 90 calls, fixture-keyed cache with no re-fetch inside 24h, no polling loops. Retrofitting a quota guard after building a polling-based fetcher means rewriting it.
- [ ] Set up Render.com free-tier service, connected to GitHub, confirm auto-redeploy works end-to-end with a placeholder "hello world" FastAPI app.
- [ ] Set up a GitHub Actions workflow file (empty steps for now) on a weekly cron schedule, to confirm the free scheduled-automation pattern works before wiring real scraping into it (Design §7, NFR-1).
- [ ] 🆕 Create `decision_config.json` with placeholder values and a `config_version` string (NFR-12). Every Decision Layer parameter lives here, never in code, so any backtest result can be reproduced from its parameter set.

**Acceptance**: empty API is live on Render, taxonomy file covers all 12 competitions with correct `format` and `odds_coverage`, scheduled workflow runs and completes, quota guard is in place before any real API call is made.

---

## Phase 1 — Data collection (Requirements §7, Design §3)

- [ ] Notebook 01a: download and clean football-data.co.uk CSVs for the 5 in-scope leagues, back to 1993/94 → `matches_clean.csv`.
- [ ] Notebook 01b: clone/pull openfootball repos for the 5 domestic cups + Champions League + Europa League → merge into the same match schema.
- [ ] Reconcile team-name mismatches between football-data.co.uk and openfootball (a known pain point — build a name-mapping table once, reuse everywhere downstream).
- [ ] Tag every match with `competition_id`, `competition_type`, `format`, `leg_number` using the Phase 0 taxonomy file.
- [ ] Notebook 02: pull full clubelo.com Elo history via the free CSV API for every club that has appeared in an in-scope competition, including lower-division clubs that entered via domestic cups (FR-9).
- [ ] 🆕 **Ingest the full odds column set, not just 1X2** (Design §3.1): pre-closing and closing (`C`-suffixed) columns for 1X2, Over/Under 2.5, and Asian Handicap, including both `Max*` (available price) and `Avg*` (consensus). The AH and O/U columns are what make the low-margin market focus possible; skipping them here blocks Phase 5.5 entirely.
- [ ] 🆕 Tag every odds row with `odds_regime ∈ {pre_2025_07_23, post_2025_07_23}` (Requirements §7.3). Pinnacle's feed became unreliable on 23/07/2025 and was dropped from Max/Avg; pooling the two regimes without adjustment is a correctness bug, not a stylistic one.
- [ ] 🆕 Build the two-snapshot price series per selection (Friday/Tuesday pre-closing → closing). This is the free line-movement data that the whole CLV programme runs on.

**Acceptance**: one unified `matches_clean.csv` covering all 12 competitions with consistent team IDs and correct format tags; Elo history for every club including lower-division cup entrants; 🆕 a `closing_odds.json` carrying both price snapshots for 1X2, O/U 2.5 and AH across the 5 leagues, regime-tagged.

---

## Phase 2 — Feature engineering (Design §4)

- [ ] Rolling form & rolling xG (last 5/10), ported from StatPitch v1 logic, computed **per club across all competitions combined** (a club's "form" should reflect all its matches, not just one competition's).
- [ ] Head-to-head lookup table, all club pairs that have met in any in-scope competition.
- [ ] Rest days / fixture congestion: matches played in the last 14 days *across all competitions* (FR-17) — requires the merged multi-competition match log from Phase 1.
- [ ] Cross-competition Elo bridge: for each league pair, compute the average Elo delta implied by historical UCL/UEL results between clubs of those two leagues (FR-11). Document the method clearly — this is a key differentiator, so it should be explainable, not a black box.
- [ ] Cup-specific features: `tier`-based lower-division prior, two-leg aggregate-state features, cup-round importance weight (FR-9, FR-7).
- [ ] Manager tenure / new-manager-bounce flag — requires a manager-change dataset; source from Wikipedia club season pages or a free manager-history dataset, cache locally.
- [ ] Notebook 07: scrape understat.com xG data for Big 5 leagues, rate-limited and cached, merge into rolling-xG features (cups won't have xG coverage from this source, and that's expected).
- [ ] Notebook 08: scrape Transfermarkt squad values/ages for all in-scope clubs, cached (reuse StatPitch v1's `cloudscraper` approach).
- [ ] 🆕 Motivation / dead-rubber features (Design §4): points-to-safety, points-to-title, mathematically-eliminated flag, matchweek index. Feeds the FR-33 guardrails.
- [ ] 🆕 Line-movement features: `q_fair(closing) − q_fair(pre-closing)`, sign and magnitude, per selection.
- [ ] 🆕 **Split the feature set into two variants** (Design §4.1) and keep them separate for the rest of the project:
  - `model_pure` — **no odds-derived features**. Used for all edge, value and CLV calculations.
  - `model_informed` — includes line movement. Used for standalone accuracy/Brier/RPS numbers only.

  *Why this matters*: a model fed line movement partially learns to reproduce the market. Accuracy goes up, measurable edge goes to zero — you cannot beat a line you are copying. Every reported number must state which variant produced it. Conflating them is the easiest way to produce a result that looks good and means nothing.

**Acceptance**: full `features_dataset.csv` with all feature groups populated for every match in Phase 1's dataset, documented handling for missing xG (cup matches) and missing lineup data, 🆕 and two cleanly separated feature variants with no odds leakage into `model_pure`.

---

## Phase 3 — Base model training (Design §5.1–5.2)

- [ ] Train per-competition-goal-environment XGBoost-Poisson regressors for home/away expected goals.
- [ ] Train the joint multi-competition XGBoost 1X2 classifier with `competition_id` embedding.
- [ ] Train the MLP on the same feature set, standardized inputs.
- [ ] Fit isotonic calibration on a held-out validation window.
- [ ] Train the logistic-regression meta-learner on out-of-fold predictions from all three models.
- [ ] Confirm time-decay + competition-type sample weighting is applied correctly (Design §5.1: continental > domestic cup > league weighting).
- [ ] 🆕 **Persist ensemble dispersion (`p_std`) per fixture** — the spread of individual member predictions around the blended output (Design §5.1). Most pipelines discard this; here it is the edge-robustness input to bet grading in Phase 5.5, so it must be saved alongside the point prediction from the start.
- [ ] 🆕 Verify the Dixon-Coles score matrix sums to ~1.0 after truncation and renormalisation, and sanity-check its implied 1X2 and O/U 2.5 probabilities against the classifier's direct output. **The matrix becomes the single source of truth for ~60 markets in Phase 5.5** — an error here propagates everywhere, so validate it now rather than discovering it through bad bets later.

**Acceptance**: a single trained ensemble producing reasonable 1X2 + goal predictions for a held-out season across all 5 leagues, before cup-specific logic is layered on; 🆕 `p_std` persisted; 🆕 score matrix validated against direct classifier output.

---

## Phase 4 — Cup & knockout logic (Design §5.3)

- [ ] Build the extra-time/penalty sub-model for single-leg knockout matches that end level.
- [ ] Build the two-leg Monte Carlo tie engine (`tie_engine.py`): simulate leg 1 and leg 2 score matrices together, ≥10,000 runs, output aggregate qualification probability. Validate against a handful of known historical ties as a sanity check before trusting it broadly.
- [ ] Wire `format`-based branching into the serving-layer inference path so the right sub-model is used automatically per match (Design §5.3).
- [ ] Build the cup/bracket Monte Carlo simulator (domestic cups + UCL/UEL from current stage to final), generalized to knockout brackets that update as real results come in.
- [ ] 🆕 Structure the Monte Carlo code so it can be reused for the bankroll simulation (Phase 5.5) and the simultaneous-Kelly joint-outcome solve. Same apparatus, different payoff function — building it generically here avoids writing a third simulator later.

**Acceptance**: a two-legged tie between two known clubs returns an aggregate probability that changes sensibly once leg 1's real result is supplied; a full domestic cup bracket simulation completes and produces per-team title-probability estimates.

---

## Phase 5 — Market evaluation (Design §5.4)

*Metrics only. Staking moved to Phase 5.5.*

- [ ] 🆕 **Notebook 12 — de-vig comparison first** (FR-28, Design §6.2). Implement proportional, power, and Shin. Compare each method's fair probabilities against realized outcomes by log-loss and calibration error, per competition. Persist the winner per competition in `decision_config.json`.

  *Do this before anything else in this phase.* Every downstream number depends on it, and proportional de-vig systematically overstates longshot probabilities — it will manufacture phantom value on draws and away underdogs, which is precisely where the losses would accumulate.

- [ ] Merge closing odds into the backtest dataset using the selected per-competition method.
- [ ] Compute **Brier score, log-loss and RPS** for StatPitch vs. market-implied probability, per competition, per season (FR-14). Implement RPS per the Design §5.4 formula — it is the correct metric for an ordinal outcome and the standard in the football-forecasting literature.
- [ ] 🆕 Produce calibration curves and ECE per competition × probability decile, for both model and market (FR-16b) → `calibration.json`. This feeds the `c_calib` grading input in Phase 5.5.
- [ ] 🆕 **Implement the validation protocol properly** (NFR-10) before reporting any number: purged walk-forward splits with a gap so no rolling feature spans the train/validate boundary, and one season designated untouched holdout, not examined until Phase 8.
- [ ] Generate `backtest_report.json` and a human-readable summary — **report results honestly**, including any competition or season where StatPitch does not beat the market (NFR-3).

### 🆕 Checkpoint — fit `w` early and decide whether to continue

- [ ] Before building Phase 5.5, fit the market-shrinkage weight `w` (Design §6.5) on validation data: `p_used = w·p_model + (1−w)·q_fair`, choosing `w` to maximise realized log-growth.

**This is the project's truth serum.** `w` measures how much information the model adds over the market. Read the result before proceeding:

| Fitted `w` | Interpretation | Action |
|---|---|---|
| ≈ 0 | Model adds nothing over the market | **Report this honestly and prominently.** Build the Decision Layer as a demonstration of correct methodology, sized as a smaller effort, and make "we measured it and there was no edge" the headline finding — that is a genuinely valuable and uncommon result |
| 0.1–0.3 | Modest but real information | Proceed with Phase 5.5 as specified, with heavy shrinkage |
| > 0.4 | Substantial edge | Proceed, **and audit for leakage first** — a high `w` against closing lines is more likely a bug than a discovery |

Measuring this at Phase 5 rather than Phase 8 is the point: it tells you how much to invest in Phase 5.5 before you invest it.

**Acceptance**: de-vig method selected and documented per competition; Brier / log-loss / RPS / ECE reported for ≥2 full historical seasons of league matches under a leakage-free protocol; `w` fitted and its value recorded.

---

## Phase 5.5 — 🆕 Decision layer (Design §6)

*Build in strict dependency order — each module consumes the previous one's output.*

### 5.5a — All-markets engine (`market_engine.py`, FR-23)

- [ ] Derive every selection in Requirements §3.2 from the Dixon-Coles matrix: 1X2, Double Chance, DNB, O/U all lines, team totals, BTTS, Asian Handicap all lines, correct score.
- [ ] Implement quarter-line handling: split the notional stake across two adjacent half-lines, returning the full outcome distribution (win / half-win / push / half-loss / loss), not just a win probability. The staking module needs the payoff distribution to compute log-growth correctly.
- [ ] Tag correct score `stakeable = false` — displayed under FR-4, structurally excluded from every staking path.
- [ ] **Validate against real book prices**: de-vig actual closing 1X2, O/U 2.5 and AH prices and compare with the engine's derived probabilities. Systematic divergence on a market family means the score matrix is wrong for that family — catch it here, not in the backtest.

### 5.5b — Value calculation (`value.py`, FR-16a)

- [ ] Implement the strict fair-vs-available separation: `q_fair = devig(AvgC*)`, `o_avail = Max*`. **Never de-vig `Max*`** — the maximum of N noisy prices is upward-biased by construction, and de-vigging it fabricates edge that does not exist.
- [ ] Compute `edge_prob = p_model − q_fair` and `EV = p_model · o_avail − 1`, using `model_pure` only.
- [ ] Report model edge and price edge separately — they decay differently and price edge is often the more reliable of the two.

### 5.5c — Grading (`bet_grader.py`, FR-25, FR-33)

- [ ] Implement the five sub-scores: `c_edge`, `c_robust` (from `p_std`), `c_market`, `c_calib` (from `calibration.json`), `c_support`.
- [ ] Implement the **non-monotonic edge term** — `c_edge = exp(−((edge_prob − e_peak)/σ)²)`, with an automatic **F grade above the ceiling**. Large apparent edges in an efficient market mean the model is blind to something (injury, dead rubber, rotation), not that the market is wrong. Naive systems bet hardest exactly there.
- [ ] Route F-graded large-edge bets to a review queue. These are the most valuable model-diagnostic signal the system produces — each one is a concrete instance of information the feature set is missing.
- [ ] Implement guardrails (FR-33), evaluated before grading, each forcing an F with a logged reason: unconfirmed lineup with key player doubtful, dead rubber, fixture within 72h in another competition, `p_std` above threshold, book margin above threshold, price above odds ceiling, `odds_coverage = false`.
- [ ] Tune `e_peak`, `σ`, `e_ceiling` and grade cutoffs on validation data (deferred from Design §11); persist to `decision_config.json`.

### 5.5d — Staking (`staking.py`, FR-27)

- [ ] Implement fractional Kelly with the fitted `w` from the Phase 5 checkpoint.
- [ ] Implement log-growth `g` computed over the full payoff distribution (handles pushes and quarter lines correctly).
- [ ] **Rank selections by `g` and expose the top-ranked as the match's "best bet"** (FR-24). Verify that ranking by `g` and by raw EV give different answers, and that EV-ranking crowns longshots — this is the concrete demonstration that the design choice matters.
- [ ] Implement caps: per-bet fraction, per-matchday aggregate exposure, odds ceiling.
- [ ] Sweep `λ ∈ {0.10, 0.25, 0.50, 1.00}` and produce the growth-versus-drawdown frontier.
- [ ] Implement simultaneous allocation: simulate the joint slate outcome from the per-fixture score matrices, then solve `max E[ln(1 + Σ f_k·r_k)]` under the exposure constraint (SLSQP). Reuse the Phase 4 Monte Carlo apparatus.
- [ ] Confirm simultaneous sizing is materially smaller than naive sequential sizing on a correlated slate — if it isn't, the correlation structure isn't being captured.

### 5.5e — CLV tracking (`clv_tracker.py`, FR-26, FR-29)

- [ ] Implement `bet_ledger.jsonl` append at flag time with the full record from Design §6.6, including `config_version`.
- [ ] Implement settlement: fill `odds_closing`, `result`, `clv_pct` post-match.
- [ ] Backtest CLV across ≥2 historical seasons: mean CLV, standard error, positive-CLV rate, broken down by competition and market family.
- [ ] **Label it "Friday-to-close CLV" everywhere.** The base snapshot is Friday afternoon, not a true opening line, so measured movement understates what an early bettor could capture. Valid signal, accurate name.
- [ ] Enforce the reporting rule in the summary generator: **positive ROI with negative CLV is reported as absence of demonstrated edge, not as success.**

### 5.5f — Risk & diagnostics

- [ ] `bankroll_sim.py` (FR-30): block-bootstrap the settled ledger ≥10,000 paths preserving matchday correlation; report median ROI, 5th/95th percentiles, max-drawdown distribution, risk of ruin at −50%, time-to-recovery, per `λ`. Produce the bankroll fan chart.
- [ ] `edge_map.py` (FR-31): cross-tabulate mean CLV, ROI, bet count and t-statistic by competition × market family × price bucket. **Grey out cells below the minimum sample size** rather than displaying noise as signal.
- [ ] Per-bet explainability (FR-32): SHAP top contributors plus a plain-language "why we disagree with the market" string.
- [ ] Edge decay by time to kickoff (FR-34): realized edge and CLV as a function of hours before kickoff.
- [ ] Attach bootstrap CIs and t-statistics to every reported ROI figure (NFR-10). An ROI without a dispersion estimate does not ship.

**Acceptance**: for any league fixture, the system returns ~60 priced selections with edge, EV, log-growth and grade; a single graded best bet with SHAP reasoning; a correlation-aware matchday card; a CLV backtest over ≥2 seasons with mean, SE and positive rate; a risk-of-ruin curve across four Kelly fractions; and an edge map identifying which league × market combinations show measurable edge and which do not. **Cup fixtures return predictions with `bet_recommendation: null` and a stated reason** — the `odds_coverage` gate working as designed.

---

## Phase 6 — Live context features (Design §4)

- [ ] Wire API-Football's confirmed-lineup endpoint into a pre-kickoff feature refresh (FR-18), with graceful fallback to the pre-match estimate if lineups aren't available (NFR-7). 🆕 Confirm the Phase 0 quota guard holds: one call per fixture at ~T−45min, no polling, hard stop at 90 daily calls.
- [ ] Wire injuries into squad-strength adjustment, same pattern as StatPitch v1's existing injury notebook. Batch per competition, not per fixture — 5 calls, not 50.
- [ ] Implement the live value-bet flag: compare live model probability against current odds where available, flag divergence above threshold (FR-16).
- [ ] 🆕 Wire the live path through the full Decision Layer so `/card/today` produces graded, sized recommendations rather than raw divergences.
- [ ] 🆕 Verify the lineup-delta feature actually moves grades: a confirmed lineup missing a key player should visibly change `p_model` and can legitimately flip a grade. If it doesn't, FR-18 is decorative.

**Acceptance**: a request made ~30 minutes before a real kickoff returns a prediction reflecting the confirmed lineup, falls back cleanly for matches further out, and 🆕 stays inside the 100/day quota across a full heavy Saturday — measured, not assumed.

---

## Phase 7 — API & deployment (Design §7)

- [ ] Build all endpoints in Design §7's table, including the six new Decision Layer endpoints.
- [ ] Load all `.pkl` artifacts once at startup; confirm sub-200ms response times per prediction (NFR-2). 🆕 The `/card/today` simultaneous-Kelly solve is the one genuinely expensive endpoint — precompute it in the `flag_card` scheduled job and serve the cached result rather than solving per request.
- [ ] 🆕 **Verify the API backward-compatibility contract (Design §7.1) before adding anything new.** Point the existing frontend at the v2 deployment with zero code changes and confirm every screen renders as before. v2 is additive only: no v1 route renamed or removed, no existing response field renamed, removed, or retyped.
- [ ] 🆕 Add the NFR-11 disclaimer to every response carrying a stake recommendation — as a *new* key, never replacing an existing one.
- [ ] 🆕 Confirm every fixture with `odds_coverage = false` returns `bet_recommendation: null` with an explicit reason string, rather than silently omitting the field.
- [ ] Deploy to Render.com free tier, confirm `git push` → auto-redeploy works with the full model set.
- [ ] Wire the GitHub Actions scheduled workflows to run the Phase 1/2 scraping/refresh notebooks on the required cadence.
- [ ] 🆕 Add the two new scheduled jobs: `flag_card` (pre-matchday — writes the graded card to the ledger) and `settle_ledger` (post-match — fills closing odds, results, CLV). These two jobs are what make the live CLV track record accumulate automatically; without them FR-29 is a manual chore that will not get done.

**Acceptance**: live public API serving predictions for all 12 competitions with `/today`, `/simulate`, `/value-bets/today`, `/backtest`, `/markets`, `/best-bet`, `/card/today`, `/clv/report`, `/edge-map`, `/bankroll/simulate` and `/ledger` all functional; both scheduled ledger jobs running unattended.

---

## Phase 8 — Validation & documentation

- [ ] Run the full acceptance criteria from `01_Requirements.md` §8 against the deployed system and record results.
- [ ] 🆕 **Open the untouched holdout season** (NFR-10) and evaluate against it exactly once. Report that number as the headline result regardless of what it says.
- [ ] Write a model card covering: what StatPitch is good at; where it is weakest (no cup odds coverage, lower-division strength estimates, xG unavailable for cups); what the market-benchmark numbers actually showed; 🆕 the fitted `w`; 🆕 mean CLV with standard error; 🆕 which league × market combinations showed edge and which did not.
- [ ] 🆕 State the scoping limitation plainly: the Decision Layer covers the 5 domestic leagues because **no free odds source with cup coverage exists** (the-odds-api was evaluated and rejected). This is a data-availability limit, not a modelling choice.
- [ ] 🆕 If the honest answer is "the model does not beat the closing line," write that as the conclusion. A rigorous negative result with correct methodology, proper validation, and a full risk analysis is a stronger portfolio artifact than an unfalsifiable positive one — and far more unusual.
- [ ] Update this Tasks document — check off completed items — so it stays a living artifact rather than a one-time plan.

**Acceptance**: every FR/NFR has either a passing acceptance check or a documented, honest limitation; the holdout season has been evaluated exactly once and reported as-is.

---

## Suggested build order (sequence, not calendar)

```
Phase 0 → Phase 1 → Phase 2 → Phase 3  (league matches only, simplest case)
                                  │
                                  ▼
                      Phase 5  (market evaluation + de-vig)
                                  │
                                  ▼
                      ⟨ CHECKPOINT: fit w — decide how far to build Phase 5.5 ⟩
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
             Phase 5.5 (decision layer)   Phase 4 (cup/knockout logic)
                     └────────────┬────────────┘
                                  ▼
                  Phase 6 (live) → Phase 7 (serving) → Phase 8 (validation)
```

🆕 **Reordering note.** The previous spec revision put Phase 4 (cups) before Phase 5 (market benchmark). this revision reverses it deliberately. Cup work is the largest single effort in the project and delivers **no** market benchmarking, no CLV, and no bet recommendations — because free cup odds do not exist. The market evaluation and the `w` checkpoint are cheap by comparison and tell you whether the model has any edge at all. Learn that first, then decide how much cup work to invest in.

**Milestones worth stopping at.** Each is a legitimate finished artifact on its own:

| Milestone | What you have |
|---|---|
| **M1** — Phases 0–3, leagues only | A working, calibrated multi-league prediction model |
| **M2** — + Phase 5 | An honest answer to "does this beat the closing line?", with RPS, calibration curves, and the fitted `w` |
| **M3** — + Phase 5.5 | Graded, sized, CLV-tracked bet recommendations with full risk analysis |
| **M4** — + Phase 4 | Cup and continental predictions, unbenchmarked by design |
| **M5** — + Phases 6–8 | Live lineups, full API, model card — StatPitch v2 complete |

*(Milestones are labelled M1–M5 rather than v0.5–v2.0 to avoid colliding with the product versioning: **StatPitch v1** is the shipped international model, **StatPitch v2** is this club-football system.)*

The strong recommendation is to reach **M2** before committing to anything beyond it. It is a small amount of additional work past M1, and it is the point at which you know whether the rest of the project is measuring a real edge or documenting the absence of one — both of which are worth building, but which justify very different amounts of effort.
