# StatPitch v2 — Model Card

**Status:** staking ENABLED under an explicitly **experimental** selection rule.
**Version:** `dec-2026.08.1-experimental` (see §7 and §11 for exactly what
"experimental" is claiming and what it is not).
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
| Momentum features, pre-registered (below) | nothing; p = 0.43 / 0.94 / 0.90 |
| Market as an offset, model learns the residual (below) | **−0.031 log-loss, t = −8.50** — actively harmful |
| **Sharp book as reference, measured by CLV** | **+0.51%, t = 3.47** ✅ |

### Conformal prediction sets: valid, and barely informative

`statpitch.models.conformal` turns the 1X2 probabilities into sets with a
distribution-free coverage guarantee. Calibrated on 2022/23 and evaluated on
2023/24 — across a season boundary, where the exchangeability assumption is
actually tested:

| target | coverage | mean set size | size 1 | size 2 |
|---|---|---|---|---|
| 0.80 | 0.998 | 2.96 | 0.0% | 4.0% |
| 0.60 | 0.880 | 2.24 | 6.3% | 63.1% |
| 0.50 | 0.802 | 1.98 | 8.7% | 84.3% |
| 0.40 | 0.760 | 1.80 | 20.3% | 79.7% |
| 0.30 | 0.711 | 1.62 | 37.8% | 62.2% |

**At an 80% target the set is all three outcomes.** That is the method working, not
failing: an honest 80% set on a market where the best model reaches 54% accuracy
*is* "any of these three". A tighter set at that confidence would be a lie.

Coverage also **over-shoots the target** everywhere — 0.802 at a target of 0.50.
With three outcomes a set can only grow a whole outcome at a time, so coverage
moves in steps rather than continuously. Randomised APS would tighten it to the
nominal rate by dropping the marginal outcome with some probability, and that is
deliberately not done: the API contract guarantees a given fixture returns
byte-identical output on repeat calls, and a randomised set would break it. Over-
covering is the conservative direction, so the cost is width rather than validity.

And the guarantee is **marginal**, which is not a footnote. At a 0.50 target the
overall coverage is 0.802, while DFB-Pokal covers 0.688 and Coppa Italia 0.875.
Split conformal promises coverage averaged over fixtures and says nothing per
competition, so `coverage_by` reports it rather than leaving a reader to assume
the headline applies to the tie in front of them.

The measurement is `scripts/evaluate_conformal.py`. **The sets are not served.**
At the confidence levels a reader would want they carry almost no information,
and a field that always says "H, D or A" is worse than no field: it occupies a
place in the response where a reader expects something to have been narrowed
down. The module and the numbers are kept so the decision can be revisited if the
model ever sharpens.

### The market as an offset: `w`=0 was the generous reading

`w` is fitted on a **linear post-hoc blend**, `p_used = w·p_model + (1−w)·q_fair`.
That is a narrow test: one dial for every match ever played, unable to express
"the model is right about *this kind* of fixture". The nested formulation asks
directly — give the model the de-vigged closing line as a `base_margin` offset and
let it learn the residual. A model with nothing to add then scores exactly the
market, and any improvement is attributable rather than merely visible.

Pooled out-of-sample over 5,379 league matches (2021/22–2023/24):

| | log-loss |
|---|---|
| **market, used directly** | **0.9683** |
| features starting from the market (residual) | 0.9991 |
| features alone, no market | 1.0136 |

**+0.031 worse, t = −8.50, p < 0.0001.** Not "no improvement" — a strongly
significant result in the wrong direction. Allowed to adjust an efficient price
using 64 features, the model reliably makes it worse, because on this population
there is nothing left to fit but noise.

This strengthens `w`=0 rather than restating it. The blend said the optimal
weight is zero; this says any nonzero contribution is actively costly, and says
it with nonlinear interactions available that a blend cannot represent.

The market baseline reproducing 0.9683 against §3's 0.9698 is the check that the
offset passes through untouched — on a slightly shorter window, which is why the
two are close rather than identical.

A first attempt fed the market in as ordinary *features* rather than an offset and
measured the market configuration at 0.9936, a quarter-point worse than the market
it was handed. Trees are piecewise-constant and the identity map on three
continuous inputs is what they approximate worst; that wrapper cost ~0.024, an
order of magnitude more than any effect being looked for. The test would have run
through a channel lossier than its own signal.

### Squad values: a pre-registered null, and one exploratory signal

Transfermarkt squad valuations are the last free source that is a *different*
measurement rather than a restatement — a forward-looking assessment of playing
staff, where xG and momentum were both derivatives of what the model already
held. 1,562 club-seasons across the five leagues, 2010/11–2025/26, joined at 100%
after name resolution.

**Lagged by one season, deliberately.** Transfermarkt's page for a past season
does not state when within that season its figures were taken, and an
end-of-season valuation reflects the season it would be used to predict. Each
match gets its clubs' values from the *previous* season. A stale feature is
weaker; a leaky one is a wrong answer that looks like a strong one.

| test | population | improvement | t | p |
|---|---|---|---|---|
| **pre-registered** | all 61,321 rows | +0.00034 | 1.11 | **0.29** |
| exploratory | the 19,785 rows that have a valuation | +0.00105 | 3.04 | 0.014 |

**The pre-registered test is null**, and that is the result of record. Valuations
exist for 30.6% of rows — leagues only, and only after the one-season lag — so
testing on the whole frame dilutes the hypothesis by roughly three to one.

Restricting to rows that actually carry a valuation gives a positive, nominally
significant effect. It is reported as **exploratory and not as a finding**,
because the restriction was chosen after seeing the pre-registered result. That
is precisely the flexibility a pre-registration exists to remove, and the fact
that the restriction is defensible does not make the p-value confirmatory.

It is also small: +0.00105 against a remaining gap to the closing line of
+0.0096. Even taken at face value it closes about a tenth of the gap and does not
touch `w`=0.

So the columns are kept in the frame and **excluded from the model** by
`MEASURED_INERT`, on the same standard as the momentum features. This is the one
live hypothesis in the project: a confirmatory test on the restricted population,
pre-registered before the data exists, on seasons after 2023/24 — never on the
untouched 2024/25 holdout, which is reserved for a single final look.

### Momentum: three pre-registered hypotheses, three nulls

Result streaks were genuinely missing — `scoring_streak` counted goals and nothing
counted wins — so they were built along with opponent strength and Elo momentum,
and tested as a family of three under Holm–Bonferroni at α = 0.05, fixed before
the answer was known. Paired per fold, ten folds validating 2014/15–2023/24:

| group | baseline | with group | improvement | t | p | Holm |
|---|---|---|---|---|---|---|
| result streaks | 0.9808 | 0.9811 | −0.00031 | −0.82 | 0.43 | no |
| opponent strength | 0.9808 | 0.9809 | −0.00003 | −0.08 | 0.94 | no |
| Elo momentum | 0.9808 | 0.9809 | −0.00007 | −0.13 | 0.90 | no |
| *all three together* | 0.9808 | 0.9806 | +0.00026 | 0.55 | 0.60 | *not in family* |

**No group reached significance even before the correction.** The correction did
not have to do any work, which is the cleanest form this result could take: there
was nothing to correct away.

The likely reason is the same one §4 gives for xG. Club Elo already integrates
recent results — that is what a rating *is* — so a club's five-match form, the
strength it was earned against, and the drift of its rating are three views of
information the model holds twice over. `elo_diff` is the strongest feature in the
frame, and these are its derivatives.

The 25 columns are kept in `features.parquet` and excluded from the model by
`build.MEASURED_INERT`. "Unbeaten in seven" belongs on a fixture page; it does not
belong in a tree that already has the rating gap.

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

*Refined 2026-08-24.* `odds_coverage` is now the conjunction of two flags that
were always distinct: `live_odds_coverage` (a price can be obtained) and
`benchmark_coverage` (history exists to validate against). They still move
together, and they will not for long — a keyed odds feed would give the cups
prices whose closing-odds history stays empty forever. A bet needs both, and a
gate reading one flag could not tell that apart from full coverage.

**Cup fixtures were absent entirely for a period, and the model card did not
know.** openfootball stopped publishing every cup file; the fixture artifact
carried five leagues and nothing else while the FR-9 entrant prior, the FR-8
extra-time model and the FR-20 bracket simulator sat complete and idle. Two
sources now cover two of the seven competitions — OpenLigaDB keylessly for the
DFB-Pokal, and The Odds API behind a key for the rest, whose `/events` endpoint
costs no credits. Restoring them exposed two prediction defects that had never
been reachable: see §8.

**~~No live odds.~~ Live odds exist; the rule that used them does not.**
*Superseded 2026-08-24.* `football-data.co.uk/fixtures.csv` publishes pre-match
prices for the coming week — free, keyless, and in the same schema as the
archive, so a captured price and a published close are same-source and
`clv_tracker` will compare them. Captured daily since Plan §4 Phase A.

That closed the plumbing gap and opened a sharper one. §5's result is defined on
**Pinnacle**-referenced selections, and Pinnacle is not in that feed. Every
reference the feed *does* carry was measured against the same window
(`data/selection_rule_study.json`):

| reference | in the live feed | CLV, pre-break | clustered t |
|---|---|---|---|
| none (whole book) | — | −0.09% | −2.58 |
| **Pinnacle** | **no** | **+0.51%** | **+7.53** |
| B365 | yes | −0.07% | −1.09 |
| consensus | yes | −0.23% | −3.35 |

Betfair Exchange leads post-break at +2.51% (t=+7.86) and *is* in the feed, but
has one season — below Requirements line 250's ≥2-season bar, and the same
consensus rule flips sign between regimes, which is what a regime-specific
artifact looks like. So the finding still cannot be traded forward, for a
different and more precise reason than when this paragraph was written.

**Scoring choice, because it decides the answer.** Every rule selects using
`odds_max`, so scoring it on how `odds_max` then moves scores a variable on
itself: the max-vs-consensus spread narrows from +11.64% to +10.23% on selected
rows while the consensus moves 0.28% *against* the bet. Scored that way the
tradeable consensus rule reads +1.13% at t=+4.25 and is mean reversion. All
figures above are `avg → avg`.

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
| NFR-9: API-Football's 100/day free tier serves the live path | The free **plan** covers seasons 2022-2024 only. Every current-season call is refused, so lineups (FR-33) and fixture-date correction are unreachable at $0 |

Two further findings that changed the model rather than the spec:

**Cup home advantage is less than half the league figure** — +24.6 Elo from 982
rated-vs-rated cup matches, against 54.4 Elo from 19,763 league matches.
Substituting a league constant into cup fixtures would over-favour the host by
~30 Elo, precisely in the lower-division-hosts-a-big-club ties the entrant prior
exists to handle.

**Fixture dates cannot be confirmed at $0, and the schedule is provisional.**
openfootball publishes a matchday before the league confirms kickoff slots, so
88% of the fixture list sits on a nominal date — ten La Liga fixtures stacked on
one Sunday, played across four days. API-Football was the designed correction
(Roadmap §7.2), and its free plan answers a current-season request with "Free
plans do not have access to this season, try from 2022 to 2024." Verified live on
2026-08-17.

The consequence is served rather than hidden: `date_confirmed` is true only for
the 12% of fixtures whose kickoff time openfootball published, and `/today` can
return an empty list on a day that has real matches. The collector now checks the
season before spending, so a run costs nothing instead of burning five of the
ninety daily calls weekly to be told the same thing.

This also closes FR-33's lineup collection at $0. That experiment needed to start
accumulating now because it cannot be backfilled; it cannot start at all on this
plan.

**Extra time is more open than a pro-rata extrapolation, not less.** The usual
account of cagey, defensive extra time does not survive the data: 1.101 goals in
thirty minutes against 0.927 from the league rate, giving a multiplier of 1.39
against the fixtures that actually reach it.

---

**Two defects that only a cup fixture could reach** *(added 2026-08-24)*. The
offline prediction path had never seen a club Club Elo does not rate, because
cups had never reached it. When they did, it produced confident nonsense twice,
in two different ways, neither raising an error:

* A **null** rating. Hamburg Eimsbütteler BC, a fifth-tier amateur side, came
  back at **52.9% to beat Borussia Dortmund**. The fitted model does not abstain
  on a missing feature — it predicted from the rest and invented a number.
  Filling the slot with the FR-9 pooled entrant level gives 9.1%.
* **Two equal** ratings. That fix created the second case: with both clubs on the
  same prior the Elo difference is exactly zero, and the model again found
  something to discriminate on. Milton Keynes Dons, a Football League club, came
  back at **28.8% at home against an eighth-tier opponent favoured at 48.4%**.
  The Elo fallback gives 45.4/25.6/29.1 — the right shape — precisely because it
  is simpler, so precompute now declines a fixture where neither side has a
  measured rating.

Both are the failure this document already names elsewhere: no error, no missing
field, just a confident wrong number. What made them survivable is that
`fully_rated` was already part of the response contract, so every affected
prediction was at least declaring its evidence tier while being wrong.

**Guards written for leagues do not transfer to cups** *(added 2026-08-24)*.
Three separate thresholds fired on entirely correct cup behaviour once cup
fixtures existed: the club-mapping coverage floor (95.1%, one fixture from
failing the run, against a real league coverage of 100%), the live-odds
keyed-share floor (47.4% on a healthy capture, because the price feed lists
matches the fixture list has already dropped as played), and a test asserting
every fixture club carries a measured rating. Each was scoped to the population
it was actually written about. Worth expecting a fourth.

## 11. Staking, and what enabling it does and does not claim

*Added 2026-08-27. This section exists because §1 through §8 were written while
staking was off, and turning it on changes what several of them mean.*

**What changed.** `decision_config.status` moved `placeholder` -> `experimental`,
and a selection rule went live:

    reference        Pinnacle, de-vigged as its own book
    rule             back it when the best available quote beats that fair value
    market_families  1x2 only
    max_per_day      3
    threshold        0.0

**Why it became possible.** §5's finding — +0.51% Friday-to-close CLV, t=+7.53
clustered, five pre-break seasons, 7,790 matches — is defined on
*Pinnacle-referenced* selections. §6 recorded the blocker: no free live feed
publishes Pinnacle, so the one rule with multi-season evidence could be measured
backwards and never run forwards. The Odds API publishes Pinnacle. The blocker
was a data-availability limit and it is gone.

**Why 1X2 only, and why that is not a simplification.** §4 measured picking the
largest apparent edge *across* markets at **-2.12% ROI against +0.13%** for
committing to one market in advance, because maximum-edge selection finds the
model's own largest errors. A daily pick ranked over all 86 selections would be
precisely that failure, so the rule is confined to the family its evidence was
measured on and the confinement is enforced in the config rather than by
convention.

**What `experimental` is claiming.** That a rule with five seasons of measured
CLV is being run live, and that its output is worth recording.

**What it is not claiming.** That the rule has been validated *on this price
panel*. The +0.51% was measured against football-data.co.uk's `Max` column over a
7-30 book panel. The live best quote now comes from a 25-book Odds API panel —
a different estimator, so the calibration is inherited rather than re-measured.
Requirements line 250's two-season bar is met for the **rule** and not for the
**panel**. Every selection is emitted with `config_status=experimental` and
`selection_rule.blocked_by` attached, and `/bets/today` carries a
`SELECTION_RULE_EXPERIMENTAL` marker beside the bets themselves.

**What still holds, unchanged.** `w` = 0.000. The model does not beat the closing
line and contributes nothing to any of these selections: `p_used` is `q_fair`,
`model_edge` is exactly zero on every row, and every surviving pound of expected
value is price. Nothing in this section revises §1.

**What would move it to `fitted`.** The captured `live_odds` series reaching two
seasons, and `scripts/study_selection_rules.py` reproducing the result on this
panel. Until then the honest description of the output is a live test of a
measured rule on a new price panel.

---

## 12. The confidence tier, and why it is separate

*Added 2026-09-01.*

§11's rule is a threshold, and most days nothing clears it: over 48 upcoming
days, 11 carried a price at all and one produced a bet. A daily product cannot
be blank five days in six, so a second tier was added — and the whole design
effort went into keeping it distinguishable from the first.

    tier          basis           fires when            sized by
    1  rule        Pinnacle edge   price beats fair      Kelly, capped 3/day
    2  confidence  p_model         tier 1 empty          flat 0.05% stake

**Tier 2 has no measurement behind it.** It surfaces the outcome the model is
most certain about, which is a different question from the one §5 answered. The
rule asks whether a PRICE is wrong; this asks which outcome is most LIKELY, and
a heavy favourite at a fair price is extremely likely and worth nothing. §4
measured selection of this shape at -2.12% ROI against +0.13%.

Three things keep the tiers apart rather than merging into one "picks" number:

* `selection_basis` on every row, in the API, and in the ledger.
* A flat stake rather than Kelly — Kelly sizes from an edge, and inventing one
  here would be the exact failure §11 was careful to avoid.
* `/bets/today` carries a `confidence_caveat` naming the -2.12% whenever a
  tier-2 row is present.

**It also buys an experiment.** Because the tiers are tagged, `clv_tracker` can
measure them separately, and in a few months there will be a real answer to
whether confidence picks carry value. That question is currently open; tagging
is what makes it answerable rather than permanently unknown.

## 13. Every fixture carries a price

*Added 2026-09-01.* 657 upcoming fixtures carry a prediction; 30 carry a
bookmaker quote. The other 627 are not a gap in this pipeline — books open a
market roughly a week before kick-off, and 21 of them are eleven days out. No
amount of fetching creates a quote nobody has published.

They are therefore emitted at `1 / p_model` and marked `pricing="model"`, beside
`pricing="market"` for the quoted ones. A model-implied price is a real number
and is not an offer: it can be displayed, and it cannot be taken. The field is
what keeps a fixture list complete without implying that every row is bettable.

---

## 9. Intended use

**Appropriate.** Probability estimates and score distributions for the 12
competitions; bracket and tie simulation; market comparison and calibration
analysis; measuring closing line value on a recorded ledger; as a worked example
of validating a betting model honestly enough to conclude it has no edge.

**Not appropriate.** Placing bets. Treating the stakes this now sizes as a
validated edge — see §11: the rule has five seasons of evidence, the price panel
it runs on has none.
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
