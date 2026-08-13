# StatPitch v2 — Model Card

**Status:** research complete, staking disabled.
**Version:** `dec-2026.08.0-placeholder` (the decision config has never been fitted; see §7).
**Scope:** 12 European club competitions — 5 leagues, 5 domestic cups, 2 continental.

> **Advisory only (NFR-11).** This system does not integrate with any bookmaker,
> place wagers, or hold funds. Every output is a simulation or an analysis. It is
> not financial advice.

---

## 1. The headline result

**The model does not beat the closing line.** The market-shrinkage weight `w`,
which Requirements §9 names the project's truth serum, fits at **zero** on both
criteria, and both confidence intervals include zero.

Measured on 5,306 validation matches (2022/23 and 2023/24; the 2024/25 holdout
was never touched):

| criterion | `w` | 95% CI | blended | market only | model only |
|---|---|---|---|---|---|
| log-loss | **0.000** | [0.000, 0.090] | 0.9698 | 0.9698 | 0.9852 |
| log-growth | **0.000** | [0.000, 0.215] | −0.000048 | −0.000048 | −0.000564 |

`p_used = w·p_model + (1−w)·q_fair`. At `w`=0 the blend *is* the market. Adding
the model to the closing consensus does not improve on the consensus alone.

Re-fitted after the xG features landed (§4), `w` is unchanged at 0.000 with
intervals **[0.000, 0.100]** and [0.000, 0.178]. Those are the figures the API
serves at `/edge-map`, since they come from the better of the two feature sets.

Both criteria are reported because they answer different questions and can
disagree — a blend can be better calibrated on average while being worse where
bets would actually be placed. Reporting only the flattering one would be exactly
the self-deception this parameter exists to prevent. Here they agree, which makes
the finding stronger rather than weaker.

Everything below is either evidence for that result, an attempt to overturn it,
or a consequence of it.

---

## 2. What the model is

**Task.** Predict a full score matrix for a fixture, and derive ~86 market
selections from it.

**Pipeline.** Two XGBoost `count:poisson` regressors produce `λ_home` and
`λ_away`; a Dixon-Coles matrix with per-competition `rho` turns those into a
joint distribution over scorelines; every market is a summation over that one
grid. The goal environment enters as a `base_margin` offset rather than a
feature, so the model learns how a fixture departs from its competition's
baseline instead of rediscovering each league's level.

Knockout fixtures continue past 90 minutes: extra time is modelled as more
football at a rescaled rate, and a shootout as close to a coin flip (§5).

**Training data.**

| source | content | volume |
|---|---|---|
| football-data.co.uk | 5 leagues, 1993/94– | 59,079 matches, 417,631 tidy odds rows |
| openfootball | 5 domestic cups + UCL/UEL | 5,716 matches |
| Club Elo | as-of-date strength ratings | 428 clubs, 1,274,186 intervals |
| Understat | shot-based xG | 21,587 matches joined (100.0%) |

**64,795 matches** total; **61,321 feature rows** across 74 columns.

**Cost: $0.** Every source is free. No paid odds feed, no paid API, no paid
hosting. This constraint is binding and it shapes §6.

**Benchmark.** De-vigged consensus closing odds (`AvgC*`), Shin method. Pinnacle
closing is kept as a separate single-book series and never mixed into the primary
number — one book and a ~30-book consensus are different estimators.

**Windows.** Training 2019/20–2023/24. Holdout **2024/25**, untouched (NFR-10).
2025/26 sits after the 2025-07-23 Pinnacle regime break and is held entirely
separate rather than pooled (Requirements §7.3).

---

## 3. Evaluation

On the validation window, against the de-vigged closing consensus:

| | log-loss | accuracy | ECE |
|---|---|---|---|
| direct 1X2 classifier | 0.9927 | 0.5271 | 0.01433 |
| Dixon-Coles (goal model) | 0.9852 | 0.5264 | **0.00499** |
| Dixon-Coles + xG | 0.9845 | — | **0.00317** |
| **market (de-vigged close)** | **0.9698** | **0.5439** | 0.01012 |
| *what the API actually serves* | *0.9900* | *0.5172* | *0.01283* |

### Rating coverage was worth a third of the gap

The feature frame is now rebuilt by `scripts/build_features.py`, and rebuilding it
with the alias maps added for the fixture source resolved **2,214 previously-null
club ratings** — every one of them in a cup or continental fixture, and none lost.
Those matches had been falling through to the entrant prior inside the *training*
data, not merely at serving time.

Re-measured on the same window, restricted to the five odds-covered leagues so the
comparison against the market is like for like:

| | log-loss | gap to market |
|---|---|---|
| published (§3 above) | 0.9845 | +0.0147 |
| **after resolving the ratings** | **0.9794** | **+0.0096** |

Across all fixtures in the window, including cups, it is 0.9758 — better still,
but not comparable to a market number that covers only the leagues, so the
league-only figure is the one quoted.

This is a data-coverage fix, not a modelling change: no feature was added, no
parameter tuned. It does not overturn `w`=0, which is fitted against the closing
line on league fixtures and would need re-fitting to move; the gap narrowed by a
third and is still a gap.

### The deployed path is not the evaluated path

The last row is the Elo-to-goal-rate mapping in `serving/predictor.py`, measured
by `scripts/evaluate_served_path.py` over the same seasons (3,569 of the 5,306
matches carry an as-of-date rating on both sides). It had no row here until
Roadmap §2, which meant the API returned numbers whose log-loss nobody had
computed, on a page reporting 0.9845.

It costs **+0.0064 log-loss** against the fitted model scored out-of-sample on
the same window (0.9836 by walk-forward), and sits +0.0202 from the market.

Two artifacts the serving code reads are never written. `Artifacts.goal_environment`
and `Artifacts.rho` are declared, consulted at `predictor.py:204` and `:377`, and
populated by nothing — so in production every competition is priced at the pooled
1.45/1.20 rate, and **rho is 0.0 everywhere**, which makes the served matrix
independent Poisson rather than Dixon-Coles.

The obvious repair does not work. Exporting the fitted per-competition
environments and rho into those fields makes the served path **worse**, 0.9900 →
0.9996, and nearly doubles ECE:

| variant | log-loss | accuracy | ECE |
|---|---|---|---|
| deployed — pooled rates, rho = 0 | **0.9900** | 0.5172 | 0.01283 |
| + fitted environments and rho | 0.9996 | 0.5035 | 0.02424 |
| fitted goal model, out-of-sample | 0.9836 | 0.5314 | 0.01541 |

The environments are fitted as `base_margin` offsets under XGBoost's log link,
with the trees learning the residual against them. The Elo mapping is a different
functional form — a multiplicative shift on a fixed base — and the offsets do not
transfer into it. The 1.45/1.20 constants suit the mapping they were chosen for.

So the fields stay empty, deliberately, and the gap closes only by serving actual
fitted rates. That needs the rolling-form features, which serving does not have
for an arbitrary fixture — it has them only for a *known* fixture, computed ahead
of time. Roadmap §8's precompute is therefore what closes this, and now has a
measured reason rather than an architectural preference.

**A second finding, unrelated to the gap.** `predictor._rates` states that the
symmetric split "keeps total goals roughly stable as the edge grows". It does not:
for `f(s) = a·10^(s/2) + b·10^(−s/2)` the derivative at zero is `(ln10/2)·(a−b)`,
positive whenever `a > b`, and the base rates are 1.45 against 1.20. A 400-point
Elo edge lifts expected total goals from 2.65 to 3.37 — 27%, landing on the
Over/Under market. Asserted in `tests/test_elo_rates.py` as behaviour, so that
correcting it is a deliberate change rather than an accident.

Routing 1X2 through the score matrix rather than predicting it directly closed a
third of the gap to the market (0.0229 → 0.0154). Adding xG closed another 0.0007.
The remaining gap is what `w`=0 is measuring.

**Calibration was measured and then deliberately not applied.** Isotonic
calibration makes the Dixon-Coles output *worse* (0.9852 → 0.9916) even fitted
out-of-fold across 12,576 matches, so it is not a small-sample artefact. The
matrix derives probabilities from a Poisson process rather than from a
discriminative model's raw scores, so it is calibrated by construction — its ECE
is three times better than the classifier's and twice as good as the market's.
Boosted trees need calibration; this does not. The module ships anyway, because
FR-16b requires reliability curves and ECE regardless.

**Leakage.** Features are built in a single chronological pass with per-club
state, updated only *after* each row is emitted. The strongest test truncates the
future and asserts earlier feature rows come back byte-identical.

**Latency (NFR-2, budget ~200ms).** Median 3.9 ms for a league prediction, 4.4 ms
for a cup tie, 6.5 ms for the full 86-selection book — on the warm path. The free
deployment tier's cold start is a separate matter, stated in
[`DEPLOYMENT.md`](DEPLOYMENT.md).

---

## 4. What was tried against `w`=0, and what happened

Every row is a genuine attempt to overturn the headline result. None did.

| attempt | result |
|---|---|
| Understat xG features | gap 0.0154 → 0.0147; `w` stays 0.000 |
| Venue-split form, scoring streaks | real effect, redundant with rolling xG |
| Market choice (1X2 vs O/U vs AH) | AH least bad; nothing significant |
| Best market per match, ranked by EV | **−2.12%** vs +0.13% for one market |
| Best market per match, ranked by log-growth | **−3.27%** |
| Price edge at best available quotes | ~0 two-way, −4.5% on 1X2 |
| Widening to 22 divisions | +1.29% ROI, **t = 1.19**, n = 38,763 |
| **Sharp book as reference, measured by CLV** | **+0.51%, t = 3.47** ✅ |

Two of these are worth stating plainly.

**xG did not help because the market had already priced it.** xG is the
most-cited feature upgrade in football modelling, and it is genuinely used here —
`xg_diff_10` is the second most important input after `elo_diff`, and calibration
improves by a third. But the gap to the closing line moved by 0.0007. Bookmakers
use the same public shot data. Adding it reproduces information the market
already holds. That is what an efficient market means in practice.

**Picking the largest apparent edge across markets is actively harmful.**
Maximum-edge selection reliably finds the *model's* largest errors, not the
market's. The selector spent 54.5% of its picks on 1X2, the market with the worst
measured returns. This is why the bet grader's confidence in an edge *falls* once
the edge passes ~4 points, and why anything past the ceiling is graded F and
routed to review rather than staked hardest.

---

## 5. The one thing that did work

**Closing Line Value on sharp-reference selections.** Friday-to-close CLV,
pre-break, holdout excluded, 8,947 matches:

| selection rule | comparison | CLV | t | positive | n |
|---|---|---|---|---|---|
| none (whole book) | best Fri → best close | −1.19% | −23.55 | 43.2% | 26,841 |
| none (whole book) | avg Fri → avg close | −0.09% | −1.98 | 49.5% | 26,841 |
| Pinnacle edge > 2% | best Fri → best close | +0.56% | +3.46 | 53.0% | 4,929 |
| Pinnacle edge > 2% | avg Fri → avg close | **+0.51%** | **+3.47** | 52.6% | 4,929 |

The baseline matters as much as the result: taking the best Friday price with *no*
selection rule gives −1.19%, so the rule is not riding a general drift. It is
selecting bets whose prices the market moves toward by kickoff.

**Why CLV and not ROI.** Over the same selections, ROI on 38,763 settled bets
could not resolve whether an edge existed (+1.29%, t=1.19), while CLV on 4,929
priced bets resolved it clearly (t=3.47). Stripping the outcome out and asking
only whether the price moved the right way converges on roughly an eighth of the
sample. Requirements §8.3 makes CLV the headline metric for this reason.

**What this result is not.** It is not a demonstration that the *model* has edge —
the model is not what selects these bets; a sharper book's price is. It is
evidence that a stale price at a soft book, identified by reference to a sharp
one, is worth something. Whether it survives commission, limits and the fact that
football-data.co.uk's "Friday" snapshot is not a true opening line is untested,
and untestable at $0.

**The label is honest by construction.** Everything is reported as
"Friday-to-close CLV" and a test asserts that string. The base snapshot is Friday
afternoon, not a genuine opening line, so measured movement understates what an
early bettor could capture. Valid signal, accurate name.

---

## 6. Limitations

**Odds coverage is 5 of 12 competitions.** Free odds exist for the five leagues.
They do not exist for the domestic cups or for UCL/UEL. Those competitions get
predictions, brackets and simulations; they never get a bet recommendation, and
the API says so per request with a stated reason rather than omitting the field.

**No live odds.** football-data.co.uk publishes after the fact. There is no free
real-time feed, so the CLV result above cannot currently be traded forward — only
measured backward. This is the single largest gap, and it is a direct consequence
of the $0 constraint rather than an oversight.

**A club's rating is only as good as its source, and the response says which.**
Every prediction reports whether each club carried a measured Club Elo rating, a
fitted entry-round prior, the pooled entrant level, or a bare default. This is
reported because it was once silent: ratings were keyed on the league name space
alone, so 187 of 428 clubs — every club known only as a cup entrant — fell
through to a flat 1400 and two fourth-tier sides came back as equals of each
other and of the club hosting them. No error, no missing field, just a confident
wrong number. `fully_rated` is now part of every response.

**Club Elo covers only the top two tiers.** Requirements §7.1 and FR-9 both state
otherwise; verified against the API, it does not. Clubs below tier 2 return an
*empty CSV rather than a 404*, which is why the gap is easy to miss. Deep cup
entrants are handled by a fitted entry-round prior instead (§7).

**Cup history is thin.** UCL 15 seasons, DFB-Pokal 8, FA Cup 7, UEL 6, Copa del
Rey 5, Coppa Italia 5, Coupe de France 1. One season cannot train a
competition-specific model, which makes joint training with a competition
embedding load-bearing rather than elegant.

**The clean benchmark window is a quarter of the archive.** Consensus closing
columns start in 2019/20, and pooling across the Pinnacle break is forbidden, so
the Decision Layer window is 2019/20–2024/25: 10,707 matches. Earlier seasons
remain fully usable for training; it is the market benchmark that is limited.

**No lineups, no squad values, no injuries, no ensemble.** This is a mid-strength
model. The upper confidence bounds on `w` (0.09 and 0.215) bound how much room a
better one would have. What the result rules out is that *this* feature set
carries information the closing line has not already priced.

**Shootouts are close to a coin flip, and the model says so.** The measured home
rate is 55.6%, which a binomial test cannot separate from even (p=0.315), so
`SHOOTOUT_HOME_ADVANTAGE` is held at exactly 0.5. The consequence is deliberate:
the further a tie goes, the less the model knows. That is a property of the
competition, not a defect.

**The de-vig null result is underpowered, not equivalence.** Proportional, power
and Shin differ in the fifth decimal place across 8,955 matches, and all ten
paired t-tests are non-significant (p 0.17–0.98). A simulation shows power and
Shin *do* recover true probabilities better when margin sits on the longshot, but
that gap only becomes significant at ~20,000 matches. This window is short by
roughly 2×. Every competition therefore keeps the Shin default; nothing was
selected on a coin flip and written down as a measured decision.

---

## 7. Consequences for the product

Because `w`=0 and the decision config has never been fitted, several endpoints
**deliberately return nothing**, each citing the measurement behind the refusal:

- `/best-bet` — no selection, citing `w`=0.000 over 5,306 matches *and* the
  −2.12% measured for best-bet-per-match.
- `/card/today` — refuses while the config is a placeholder. `StakingEngine`
  calls `require_fitted()` in its constructor, so nothing can size a stake from
  placeholder defaults. `/health` reports `staking_enabled: false`.
- `/bankroll/simulate` — declines an empty ledger; resampling no track record is
  a simulation of nothing.

The Decision Layer is built in full — 86-selection market engine, edge
decomposition, grading with guardrails, correlation-aware fractional Kelly,
append-only CLV ledger — and is best read as a demonstration of correct
methodology rather than as a machine for extracting edge that has not been shown
to exist.

---

## 8. Corrections to the specification

Each was measured or verified against live sources, and each is now enforced in
code and tests rather than left in prose.

| spec claim | what was found |
|---|---|
| FR-9 / §7.1: Club Elo covers the pyramid | Top two tiers only; below that, empty CSV |
| §4: Understat data via `JSON.parse` in page HTML | Dead. Page is an 18KB shell; use `getLeagueData/<league>/<year>` with `X-Requested-With` |
| §7.3: closing columns run the archive's length | They start in 2019/20 |
| §3.2: margin ordering across markets | Ordering does not hold as stated |
| FR-24: rank candidates by log-growth | Growth-ranking measured −3.27% vs −2.12% for EV. Growth is right for *sizing*; the ranking claim is a different claim and does not survive |
| NFR-3: ~+15pp over naive baseline | Unreachable here. +15pp means 59% accuracy; the closing line itself manages 54.4%. Model gets +8.4pp |
| §7.1: `openfootball/europa-league` repo | Does not exist; UEL lives as `el.txt` in the champions-league repo |
| §7.1: `openfootball/france` | Redirects to consolidated `openfootball/europe` |

Two further findings that changed the model rather than the spec:

**Cup home advantage is less than half the league figure** — +24.6 Elo from 982
rated-vs-rated cup matches, against 54.4 Elo from 19,763 league matches.
Substituting a league constant into cup fixtures would over-favour the host by
~30 Elo, precisely in the lower-division-hosts-a-big-club ties the entrant prior
exists to handle.

**Extra time is more open than a pro-rata extrapolation, not less.** The usual
account of cagey, defensive extra time does not survive the data: 1.101 goals in
thirty minutes against 0.927 from the league rate, giving a multiplier of 1.39
against the fixtures that actually reach it.

---

## 9. Intended use

**Appropriate.** Probability estimates and score distributions for the 12
competitions; bracket and tie simulation; market comparison and calibration
analysis; measuring closing line value on a recorded ledger; as a worked example
of validating a betting model honestly enough to conclude it has no edge.

**Not appropriate.** Placing bets. Sizing stakes (the engine refuses).
Deriving fair probabilities from best-available prices — max-of-N sits above
consensus by construction, and de-vigging it fabricates edge (FR-16a). Treating
the cup competitions as bettable. Treating the CLV result as a live-tradeable
strategy, for the reasons in §5.

**Reproducibility.** Every ledger entry carries its `config_version`, so a
historical result is reproducible from its parameter set (NFR-12). The ledger is
append-only: the worth of a track record is that earlier entries cannot be
revised once results are known.

---

## 10. How to read the evidence

The commit history is the lab notebook — each commit message carries the
measurement that justified the change, including the ones that failed and the
bugs that produced too-good-to-be-true results before they were caught. Several
of those near-misses are instructive:

- An over/under settlement that scored losses as 0 instead of −1 produced +50%
  ROI in a 3% margin market. Caught because that is impossible, not because a
  test failed.
- Comparing a best-available Friday price against a *closing consensus* produced
  an apparent +5.4% CLV on every selection in the book, including ones chosen at
  random. It was a max-versus-mean spread, not line movement. Settlement now
  refuses cross-source comparison outright.
- The joint slate optimiser returned a flat `1e6` penalty for infeasible
  allocations. A flat penalty has zero gradient, so SLSQP never left its starting
  vector and correlated and independent slates allocated identically — silently
  defeating the entire purpose of allocating jointly.

**793 tests**, all offline; no test touches the network.
