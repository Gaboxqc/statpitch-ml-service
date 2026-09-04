# StatPitch v2 — Requirements
## Spec-Driven Development · Artifact 1/3: Requirements (Requerimientos)

**Spec revision 2.0** — adds the Decision Layer (§5.6), corrects the data-source table against verified free-tier limits (§7), and promotes Closing Line Value to the primary edge metric (§8). Items marked 🆕 are new in this spec revision.

### Terminology used throughout these documents

| Term | Meaning |
|---|---|
| **StatPitch v1** | The shipped predecessor: international / World Cup match prediction |
| **StatPitch v2** | This system — same product, same name, same frontend, extended to club football |
| **Spec revision 1.0 / 2.0** | Versions of *these three documents*, not of the product |
| **"the first release"** | The initial v2 deployment, i.e. Phase-1 scope |

**The product name does not change.** v2 is a continuation of StatPitch, not a new product — the existing frontend consumes the same API under a backward-compatible contract (NFR-13, Design §7.1).

---

## 1. Purpose

StatPitch v2 extends StatPitch from international football into club football. Where v1 predicts World Cup and international fixtures, v2 predicts outcomes for **Europe's major domestic leagues, domestic cups, and continental club competitions** — using only free, public data sources, deployed at zero infrastructure cost, and served through the same API and frontend.

The goal is not to rebuild v1 with more leagues bolted on — it's to build something that is *structurally* better, because club football offers information international football doesn't: dense fixture calendars, confirmed lineups, transfer-market pricing, and — critically — **public betting markets**, which give us a real, externally-verifiable benchmark (closing odds) instead of just a naive baseline.

🆕 **This spec revision extends that one step further.** A probability is not a decision. The previous revision stopped at "here is our probability, and here is how it compares to the market." This one adds a **Decision Layer**: given a fixture, it enumerates every tradeable market, prices each one, sizes a stake under a risk-managed Kelly framework, grades the bet A–F, and records the decision in a ledger so the model's edge can be verified against the closing line over time — not just asserted from a backtest.

---

## 2. Vision Statement

> Given any two clubs, in any in-scope league or cup competition, on any date, StatPitch returns calibrated probabilities for every major market, identifies the single best-value selection available, sizes it under a risk-managed staking rule — and states, honestly, whether it is beating the closing line.

---

## 3. Scope

### 3.1 In-scope competitions (Phase 1)

| Type | Competition | Country |
|---|---|---|
| League | Premier League | England |
| League | La Liga | Spain |
| League | Bundesliga | Germany |
| League | Serie A | Italy |
| League | Ligue 1 | France |
| Domestic cup | FA Cup | England |
| Domestic cup | Copa del Rey | Spain |
| Domestic cup | DFB-Pokal | Germany |
| Domestic cup | Coppa Italia | Italy |
| Domestic cup | Coupe de France | France |
| Continental | UEFA Champions League (incl. qualifiers, new Swiss league phase) | Europe |
| Continental | UEFA Europa League | Europe |

### 3.2 🆕 In-scope betting markets

All derived from a single per-match Dixon-Coles score matrix — no additional model per market.

| Market family | Selections | Typical book margin | Priority |
|---|---|---|---|
| Asian Handicap | All lines −3.0 … +3.0 incl. quarter lines | ~2–3% | **Highest** — lowest margin, deepest liquidity |
| Over/Under total goals | All lines 0.5 … 5.5 incl. quarter lines | ~3–4% | **High** |
| 1X2 | Home / Draw / Away | ~5–8% | Medium |
| Double Chance / Draw No Bet | 1X, X2, 12, DNB home/away | ~5–8% | Medium |
| Both Teams To Score | Yes / No | ~5–7% | Medium |
| Team totals | Over/under per side | ~6–8% | Low |
| Correct score | Top 10 scorelines | ~15–25% | **Display only — never staked** |

**Rationale**: a 2% modelled edge survives a 2% margin and does not survive an 8% one. Asian Handicap and Over/Under are therefore where a genuine edge is realizable; 1X2 is the headline market but the hardest to beat. Correct score is returned for presentation (FR-4) but is explicitly excluded from staking because model error dominates edge at those prices.

### 3.3 Phase 2 / future extension (not required for the first release, but data sources already support it)

Eredivisie (NL), Primeira Liga (PT), Belgian Pro League, Scottish Premiership, Süper Lig (TR), Super League Greece, UEFA Conference League, second divisions (Championship, Segunda, 2. Bundesliga, Serie B, Ligue 2) for squad-strength continuity when teams are relegated/promoted.

### 3.4 Explicitly out of scope

Player-level props (cards, shots, goalscorer markets), non-European competitions, women's football (separate dataset/model effort), in-play / live-odds trading, 🆕 automated placement of real-money wagers (see NFR-11).

---

## 4. Stakeholders & Users

- **Primary user**: the project owner, for personal analysis and portfolio/demo purposes.
- **Secondary users**: anyone consuming the public REST API (developers, frontends, bots).
- **Non-goal**: this is not a licensed betting product; no real-money wagering is placed by the system itself. The staking engine (FR-27) is a **simulation and analysis tool**; its output is a recommended stake fraction, never an executed bet.

---

## 5. Functional Requirements

### 5.1 Prediction core
- **FR-1**: Given two clubs and a competition, return 1X2 probabilities.
- **FR-2**: Return Over/Under 1.5 / 2.5 / 3.5 goals probabilities.
- **FR-3**: Return Both Teams to Score (BTTS) probability.
- **FR-4**: Return top-10 most likely correct scores.
- **FR-5**: Return expected goals (xG) for each side.
- **FR-6**: Support single-match league fixtures **and** knockout cup ties.

### 5.2 Cup & knockout-specific logic
- **FR-7**: For two-legged ties, predict aggregate qualification probability by combining leg 1 and leg 2 distributions (Monte Carlo over the two independent score matrices), not just each leg in isolation.
- **FR-8**: For single-leg knockout matches (finals, single-leg rounds), model the probability of extra time and penalty-shootout outcomes, since "draw" is not a final result.
- **FR-9**: Apply a lower-division strength prior for cup entrants who play outside the 5 top leagues (e.g. a Segunda División team in Copa del Rey), sourced from Club Elo, which covers the full pyramid.
- **FR-10**: Flag and weight "cup priority" — some clubs visibly rotate squads for domestic cups vs. league/Europe; this should adjust effective squad strength, not just raw Elo.

### 5.3 Cross-competition intelligence
- **FR-11**: Maintain a single, unified cross-league strength rating so a Bundesliga club and a La Liga club can be compared even though they never meet domestically — calibrated using actual UCL/UEL results as the bridge between leagues.
- **FR-12**: League-aware modeling — goal-scoring environment (e.g., Bundesliga's higher average goals vs. Serie A/Ligue 1's lower average) must be reflected per-competition, not pooled naively.

### 5.4 Market benchmarking
- **FR-13**: For every backtested match, compare StatPitch's probability against the closing betting-market odds (converted to implied probability, de-vigged).
- **FR-14**: Report Brier score, log-loss 🆕 **and Ranked Probability Score (RPS)** of StatPitch vs. the market baseline, not just accuracy vs. a naive baseline. RPS is required because 1X2 is an *ordinal* outcome — predicting a home win when the result is a draw is a smaller error than predicting a home win when the result is an away win, and Brier score cannot express that distinction.
- **FR-15**: Run a Kelly-criterion staking backtest over historical seasons and report simulated ROI, as evidence of real edge (or lack thereof — this must be reported honestly either way).
- **FR-16**: Flag matches where StatPitch's probability diverges meaningfully from market-implied probability ("value" flags), with the divergence magnitude shown.
- 🆕 **FR-16a**: Distinguish two separate market numbers and never conflate them:
  - **Fair probability** — de-vigged **market average** closing odds (`AvgC*`), the best available consensus estimate of true probability.
  - **Available price** — **market maximum** odds (`MaxC*` / `Max*`), the price actually obtainable.

  Edge is measured as model probability vs. *fair probability*; expected value is computed at the *available price*. A large part of realizable betting edge comes from a single book being off consensus, not from the model out-thinking the market as a whole.
- 🆕 **FR-16b**: Report calibration curves and Expected Calibration Error (ECE) per competition and per probability decile, for both the model and the market.

### 5.5 Fixture & squad context
- **FR-17**: Track fixture congestion — matches played across *all* competitions in the last 14 days, not just the current competition.
- **FR-18**: Support a late "confirmed lineup" update in the ~60 minutes before kickoff, replacing the pre-match squad-average estimate with actual starting-XI strength where lineup data is available.
- **FR-19**: Track manager tenure and flag "new manager bounce" windows (first ~6 matches after a managerial change).

### 5.6 🆕 Decision layer (new in this spec revision)

This is the section that converts probabilities into ranked, sized, graded recommendations.

- **FR-23 — All-markets derivation.** From the per-match Dixon-Coles score matrix, derive probabilities for every selection in the §3.2 market table (~50–60 selections per fixture), including Asian Handicap quarter lines (which split the stake across two adjacent lines) and push/half-push outcomes.

- **FR-24 — Best bet per match.** Rank every priced selection in a fixture by **expected logarithmic growth rate** at its Kelly-optimal stake, not by raw expected value, and return the top-ranked selection as the match's "best bet."

  *Rationale*: raw EV always crowns the longshot. A nominal 20% edge on a 15.0 correct score outranks a 3% edge on Asian Handicap −0.5 at 1.95, but the first is model error and the second is the realizable bet. Log-growth ranking penalises high-variance selections automatically and is the mathematically correct objective for a repeatedly-compounded bankroll.

- **FR-25 — Bet grading (A–F).** Assign every candidate selection a letter grade from a documented composite of: edge magnitude, edge robustness (ensemble/bootstrap dispersion on the model probability), market quality (book margin, number of quoting books, line stability), calibration reliability in that probability bucket × competition, and historical realized CLV of similar bets.

  **The edge-magnitude term must be non-monotonic.** Confidence peaks at a moderate edge (~3–6 probability points) and *decreases* above ~10pp, because in a market this efficient an apparent 20-point edge almost always means the model is missing information — a key injury, a dead rubber, confirmed rotation before a midweek European tie — rather than that the market is wrong. Edges above a configurable ceiling are graded **F ("model likely blind")**, are never staked, and are routed to a review queue as a model-diagnostic signal.

  Grades map to action: **A** → full fractional-Kelly stake · **B** → half · **C** → quarter, logged only · **D** → no bet, monitor · **F** → no bet, investigate.

- **FR-26 — Closing Line Value as the primary edge metric.** For every flagged bet, compute and report CLV against the closing price:

  ```
  CLV%     = (odds_taken / odds_closing) − 1
  CLV_prob = q_fair_closing − q_fair_at_flag_time     (in probability points)
  ```

  Aggregate mean CLV, its standard error, and the share of bets with positive CLV must be reported alongside ROI, and **CLV is the headline number**. ROI over two seasons of league betting is statistically weak — a positive ROI on a few hundred bets is routinely produced by chance. Consistent positive CLV is the evidence that edge is real; the two must be reported together and any divergence between them must be stated explicitly.

- **FR-27 — Risk-managed Kelly staking engine.** Stake sizing must implement all of the following, each as a versioned, configurable parameter:
  - **Probability shrinkage toward market**: `p_used = w·p_model + (1−w)·q_fair`, with `w` fitted from backtest rather than assumed. `w` is a first-class reported result — if it fits near zero, the model adds nothing over the market, and the project must say so.
  - **Fractional Kelly**: stake = `λ · f*`, default `λ = 0.25`, with a reported growth-versus-drawdown frontier across `λ ∈ {0.10, 0.25, 0.50, 1.00}`.
  - **Simultaneous / correlated allocation**: a matchday slate contains many bets running concurrently, some on the same fixture or the same team. Sequential single-bet Kelly over-stakes in this situation. Joint allocation must be solved numerically over the simulated joint outcome distribution.
  - **Hard caps**: maximum fraction per bet, maximum aggregate exposure per matchday, and a maximum-odds ceiling above which no stake is placed.

- **FR-28 — De-vig method selection.** Implement at least three de-vigging methods — **proportional (multiplicative)**, **power**, and **Shin** — and select empirically per competition based on which produces the best-calibrated fair probabilities against realized outcomes.

  *Rationale*: this is not a footnote. Bookmakers do not spread margin evenly across outcomes; they load it onto longshots (the favourite–longshot bias). Proportional de-vig therefore systematically *overstates* the fair probability of longshots and will manufacture phantom value on draws and away underdogs — precisely where losses accumulate.

- **FR-29 — Live bet ledger.** Persist every graded recommendation: timestamp, fixture, selection, price at flag time, book offering it, grade, recommended stake fraction, closing price, and settled result. This accumulates a real, externally-checkable CLV track record over time, converting the project from a backtest into a system with an audited live record.

- **FR-30 — Bankroll simulation and risk of ruin.** Resample the settled bet log (≥10,000 runs) to produce the distribution of season outcomes: median ROI, 5th/95th percentiles, maximum-drawdown distribution, and probability of ruin at each Kelly fraction.

- **FR-31 — Market efficiency map.** Report backtested edge and CLV broken down by competition × market family × price bucket, so the system can identify where its edge actually lives and concentrate there.

  *Working hypothesis to test, not assume*: the softest prices in the universe are domestic-cup ties involving lower-division entrants, because books price them with less attention and few public models cover the full pyramid — which is exactly the coverage Club Elo uniquely provides (FR-9). If confirmed, this is the project's structural edge rather than an incidental feature. **Note the dependency**: testing this hypothesis requires cup odds, which are not currently available free (§9) — so in the first release this map is league-only and the hypothesis remains open.

- **FR-32 — Per-bet explainability.** Every flagged bet must return its top feature contributions (SHAP) plus a plain-language statement of *why* the model disagrees with the market, e.g. "away side's rolling xG is 0.4 above goals scored over 10 matches, and they have played 4 matches in 11 days."

- **FR-33 — Exclusion guardrails.** Automatically suppress recommendations where the model is structurally unreliable: unconfirmed lineups with a key player doubtful, dead rubbers late in the season, a fixture within 72 hours in another competition, model uncertainty above threshold, book margin above threshold, or price above the odds ceiling. Every suppression must be logged with its reason.

- **FR-34 — Edge decay by time to kickoff.** Measure realized edge and CLV as a function of hours before kickoff, to answer *when* a bet should be placed rather than only *whether*.

### 5.7 Simulation
- **FR-20**: Simulate full domestic cup brackets and the UEFA Champions/Europa League from the current stage to the final (Monte Carlo, ≥10,000 runs), producing per-team probabilities of reaching each round and winning the competition.

### 5.8 Serving
- **FR-21**: Expose all of the above via a REST API (predictions, rankings, today's fixtures, cup/tournament simulation, value-bet flags, backtest reports, 🆕 best bet per match, graded matchday card, CLV report, edge map, bankroll simulation).
- **FR-22**: API must support filtering/selecting by competition, since a club can appear in up to 4 competitions in a season (league, domestic cup, continental cup, and formerly a second continental competition after group-stage elimination).

---

## 6. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR-1 | **Total cost: $0.** Every data source, hosting tier, and tool must be free at the usage volumes this project needs. Verified source-by-source in §7. |
| NFR-2 | Prediction latency under ~200ms per request (models loaded in memory, not per-request retraining). |
| NFR-3 | Model must beat the naive "most common outcome" baseline by a margin comparable to or better than StatPitch v1's (~+15pp on 1X2), **and** must be evaluated against the closing-odds baseline honestly, even if it doesn't beat it. |
| NFR-4 | Fully reproducible: versioned notebooks → `.pkl` artifacts → git push → auto-redeploy, same pattern as StatPitch v1. |
| NFR-5 | Scraped sources (Understat, Transfermarkt) must be accessed politely — rate-limited, cached, respecting robots.txt — no source should be hammered or require paid/authenticated access. |
| NFR-6 | Architecture must scale from 12 competitions (Phase 1) to 20+ (Phase 2) without a redesign — via the league-embedding / unified schema approach, not one bespoke model per league. |
| NFR-7 | System should degrade gracefully: if a live data source (e.g. lineup confirmation) is unavailable, fall back to the pre-match estimate rather than failing the request. |
| NFR-8 | Update cadence must be sustainable on free infrastructure (e.g. GitHub Actions scheduled workflows, free up to 2,000 minutes/month on public repos, instead of a paid always-on worker). |
| 🆕 NFR-9 | **API-Football quota budget.** The free tier is 100 requests/day, resetting 00:00 UTC, with unused requests lost. This is the only hard quota in the stack and must be explicitly budgeted: one fixtures call per day, at most one lineup call per fixture at approximately T−45 minutes, **no polling loops**. A heavy Saturday across 5 leagues is ~50 fixtures ≈ 55 calls. NFR-7's graceful fallback is what keeps the system inside this quota, not merely a robustness nicety. |
| 🆕 NFR-10 | **Leakage-free validation.** All backtesting must use walk-forward / purged time-series splits with no feature computed from post-match information, and one untouched holdout season never used in any tuning decision. Reported ROI must carry a bootstrap confidence interval and a t-statistic on per-bet returns; an ROI figure without a dispersion estimate is not an acceptable result. |
| 🆕 NFR-11 | **Advisory only.** The staking engine outputs recommended fractions. The system must not integrate with any bookmaker account, place wagers, or hold funds. All betting outputs carry an explicit "simulation / analysis only" designation. |
| 🆕 NFR-13 | **API backward compatibility.** The existing frontend must run against v2 with zero code changes. No v1 endpoint may be renamed or removed; no existing response field may be renamed, removed, or retyped. v2 is additive only — new capability lives at new routes. See Design §7.1. |
| 🆕 NFR-12 | **Decision parameters are versioned artifacts.** `λ`, `w`, edge thresholds, grade cutoffs, caps, and the selected de-vig method per competition live in a versioned config file, not in code, so that any historical backtest result can be reproduced exactly from its parameter set. |

---

## 7. Data Requirements — verified free

Every source below was verified against its current public terms and free-tier limits. Sources that failed verification are listed in §7.2 so the decision is not silently revisited later.

### 7.1 Approved sources

| Data need | Source | Verified free-tier status | Coverage |
|---|---|---|---|
| Match results, match stats, historical betting odds | [football-data.co.uk](https://www.football-data.co.uk/data.php) | **Free, no key, no signup.** Updated twice weekly (Sun/Wed nights) | 22 European divisions, 25 seasons back to 1993/94; bookmaker odds back to 2000/01; closing odds via `C`-suffixed columns |
| League & domestic cup historical results | [openfootball](https://github.com/openfootball) (GitHub) | **CC0 public domain**, free clone, no limits | Per-country repos + Champions League / Europa League repo |
| Shot-level expected goals (xG) | [understat.com](https://understat.com) | Free public pages (polite scraping; no API, no cost) | Premier League, La Liga, Bundesliga, Serie A, Ligue 1, RFPL, 2014–present |
| Club strength ratings across the full pyramid | [clubelo.com](http://clubelo.com) API | **Free CSV API, no key** | European club football since the 1940s — includes lower-division cup entrants |
| Squad market values & ages | [Transfermarkt](https://www.transfermarkt.com) | Free public pages (polite scraping) | All clubs in scope |
| Live fixtures, injuries, lineups | [API-Football](https://www.api-football.com/pricing) (direct or RapidAPI) | **Free tier: 100 requests/day, all endpoints, no card.** Resets 00:00 UTC; unused requests lost | All in-scope competitions — see NFR-9 for the mandatory request budget |
| Supplementary event-level data | StatsBomb Open Data | Free, open license | Select competitions/seasons |

### 7.2 🆕 Sources evaluated and rejected

| Source | Why rejected |
|---|---|
| the-odds-api.com | Free "Starter" tier is **500 credits/month and restricted to NBA and MLB, h2h markets only**. Soccer is not available on the free tier at all. Additionally, a credit is charged per sport-key per region, so the effective request ceiling is below the headline 500. Does **not** solve the cup-odds gap and must not be planned around. |

### 7.3 🆕 Critical odds-data notes

**Column structure** (confirmed from football-data.co.uk `notes.txt`):
- Base odds are collected **Friday afternoons** for weekend fixtures and **Tuesday afternoons** for midweek fixtures — these are *pre-closing*, not opening, prices.
- Closing odds carry a `C` suffix: `B365CH`, `MaxCH`, `AvgCH`, etc.
- `MaxH/MaxD/MaxA` = market maximum; `AvgH/AvgD/AvgA` = market average.
- **Max and Avg columns also exist for Over/Under 2.5 and Asian Handicap**, which is what makes §3.2's low-margin market focus buildable at zero cost.

**Consequence for CLV (FR-26)**: the dataset provides a free, built-in two-point line-movement series (Friday price → closing price) for every league fixture. This must be described accurately as **"Friday-to-close CLV"** — a narrower window than true opening-line CLV, so measured movement will understate what an early bettor could capture. It remains a valid edge signal; it must not be reported as if it were full opening-line CLV.

**Pinnacle regime break (dated 23/07/2025)**: football-data.co.uk states that since that date Pinnacle's public odds API has become unreliable, their odds are systematically stale relative to other books for both pre-closing and closing prices, and **Pinnacle is no longer included in the market average and maximum calculations**.

Consequences, which must be handled explicitly rather than silently:
1. Pinnacle may be used as a gold-standard sharp benchmark **only for seasons before 2025/26**.
2. For later data, use `AvgC*` as the fair-probability source (now Pinnacle-free and therefore internally consistent) and `MaxC*` as the available price.
3. The backtest must treat 23/07/2025 as a **regime boundary**. Pooling pre- and post-break seasons without adjustment would corrupt calibration and produce misleading edge estimates.

**Constraint**: football-data.co.uk's terms allow free use of its data for personal/non-commercial analysis; if StatPitch is ever monetized, this must be revisited. This is noted here so it isn't discovered late.

---

## 8. Success Metrics / Acceptance Criteria

1. 1X2 accuracy on a held-out season ≥ StatPitch v1's benchmark uplift over naive baseline (+15pp or better).
2. Brier score, log-loss **and RPS** on 1X2 reported within an honest margin of the closing-odds market — this is the credibility bar StatPitch v1 doesn't have to clear. Calibration curves and ECE published for model and market alike.
3. 🆕 **Mean CLV reported with standard error and positive-CLV rate, over ≥2 full historical seasons — this is the primary edge criterion.** ROI is reported alongside it, with a bootstrap confidence interval and t-statistic on per-bet returns (NFR-10). A positive ROI with negative CLV must be reported as *absence* of demonstrated edge, not as success.
4. 🆕 The fitted market-shrinkage weight `w` (FR-27) is reported explicitly. A value near zero is a valid and publishable finding.
5. 🆕 Growth-versus-drawdown frontier published across all four Kelly fractions, with risk-of-ruin estimates.
6. 🆕 Market efficiency map (FR-31) published for all league × market combinations, identifying which carry measurable edge and which do not.
7. All 12 Phase-1 competitions return valid predictions via the API.
8. Two-legged cup ties return an aggregate qualification probability, not just two independent leg predictions.
9. 🆕 Every fixture returns a ranked selection list and a single graded "best bet" (FR-24, FR-25).
10. Zero paid API keys or paid hosting in the final deployed system.

---

## 9. Assumptions & Risks

- **Assumption**: Understat and Transfermarkt continue to allow scraping at low, polite request volumes; if either changes access terms, the corresponding features degrade gracefully (NFR-7) rather than breaking the system.
- **Risk — confirmed, no free mitigation available**: betting-odds history has coverage gaps for domestic cups and continental competitions on football-data.co.uk, which covers league divisions only. **No free odds source with cup coverage has been identified** — the-odds-api was evaluated and rejected (§7.2). The market benchmark, CLV tracking, and the entire Decision Layer (§5.6) are therefore **scoped to the domestic leagues football-data.co.uk publishes** — 5 at v1, 8 since 2026-09-04 with the addition of the Primeira Liga, Eredivisie and Süper Lig. This is a data-availability limit, not a modelling choice, and must be stated as such in the model card. Cup predictions remain fully supported; only their *market benchmarking* is unavailable.

  The 2026-09 extension also produced the first case where the two halves of `odds_coverage` genuinely diverge, which is what NFR-13's split was written for. Saudi Arabia has live prices with Pinnacle on the board and no historical closing odds anywhere free, so `live_odds_coverage` is true, `benchmark_coverage` is false, and it is excluded. A price with nothing to measure it against is not a recommendation.
- 🆕 **Risk — efficient markets**: the realistic expectation is that the model does **not** beat the closing line on 1X2 in the Big 5 leagues; those markets are extremely efficient. The plausible hunting grounds are Asian Handicap and totals (low margin) and off-consensus prices at individual books (FR-16a). The project's commitment is to measure and report this honestly, not to manufacture a positive result.
- 🆕 **Risk — the `w` parameter is the project's truth serum**. If market shrinkage fits near zero, the model adds nothing over the market. This should be measured early (Phase 5) rather than discovered at Phase 8, because it determines whether the Decision Layer is worth building out fully.
- 🆕 **Risk — API-Football's 100/day quota** is the only hard external limit. Poorly designed lineup polling would exhaust it in a single afternoon. Mitigated by NFR-9's explicit budget.
- 🆕 **Risk — backtest overfitting.** With ~60 selections per fixture across 5 leagues and multiple seasons, the search space is large enough to find spurious edge by chance. Mitigated by NFR-10 (purged walk-forward, untouched holdout, bootstrap CIs) and by pre-registering the market families under test rather than mining all of them.
- **Risk**: free hosting tiers (Render free tier) sleep after inactivity; a scheduled "ping" or GitHub Actions cron is needed to keep prediction latency (NFR-2) acceptable.
- **Assumption**: squad market values update slowly enough (per transfer window) that weekly/bi-weekly refresh is sufficient, matching StatPitch v1's cadence.

---

## 10. Glossary

- **1X2**: Match result market (Home win / Draw / Away win).
- **Asian Handicap (AH)**: A goal-handicap market that eliminates the draw, including quarter lines (e.g. −0.25) which split the stake across two adjacent lines. The lowest-margin, deepest-liquidity football market.
- **BTTS**: Both Teams to Score.
- **Brier score**: A proper scoring rule for probabilistic predictions; lower is better calibrated. Treats all misclassifications as equally wrong.
- **Closing odds**: The final bookmaker price available just before kickoff — widely considered the most efficient, hardest-to-beat probability estimate available.
- **CLV (Closing Line Value)**: How much better the price taken was than the closing price. The leading indicator of genuine edge, because it is far less noisy than realized ROI.
- **De-vig**: Removing the bookmaker's margin from quoted odds to recover implied "fair" probabilities. Method choice materially changes results — see FR-28.
- **Dixon-Coles**: A statistical correction to independent-Poisson goal models that fixes their systematic underestimation of low-scoring results (0-0, 1-0, 0-1, 1-1).
- **ECE (Expected Calibration Error)**: Average gap between predicted probability and observed frequency across probability bins.
- **Favourite–longshot bias**: The empirical tendency for bookmakers to load more margin onto longshot outcomes, which breaks naive proportional de-vigging.
- **Fractional Kelly**: Staking a fixed fraction λ of the full Kelly recommendation, trading a small amount of long-run growth for a large reduction in volatility and drawdown.
- **Kelly criterion**: A staking formula that sizes bets proportional to perceived edge, used here purely as a backtesting/evaluation tool, not for live wagering.
- **Log-growth rate**: The expected logarithm of bankroll multiplier for a bet at a given stake — the correct objective for a repeatedly-compounded bankroll, and the ranking criterion in FR-24.
- **Overround / margin**: The amount by which a book's implied probabilities exceed 100%.
- **Risk of ruin**: Probability that a bankroll falls below a defined threshold over a given horizon.
- **RPS (Ranked Probability Score)**: The standard proper scoring rule for *ordinal* outcomes and the conventional metric in academic football forecasting; unlike Brier, it penalises predictions by how far they are from the true outcome in outcome order.
- **Shin's method**: A de-vigging method that models the margin as arising from insider trading, distributing it non-uniformly across outcomes.
- **Swiss league phase**: The UEFA Champions/Europa League's post-2024 format — a single expanded league table before knockout rounds begin.
