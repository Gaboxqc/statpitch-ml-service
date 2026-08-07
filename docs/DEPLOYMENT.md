# Deployment

Two GitHub Actions jobs on a schedule, one Render web service. Everything on a
free tier, because the $0 constraint (NFR-1) is binding.

> **Advisory only (NFR-11).** The deployed service places no wagers, integrates
> with no bookmaker and holds no funds. Staking is disabled in code — see
> [`MODEL_CARD.md`](MODEL_CARD.md) §7.

---

## What is deployed, and what it can honestly do

`w` fits at 0.000 and the decision config has never been fitted, so the scheduled
jobs run against a system that has nothing to stake. That is not a reason to skip
building them, but it is a reason to be precise about what they are:

| piece | what it does today | what it would do fitted |
|---|---|---|
| `flag_card` | records that it ran and why it flagged nothing | grade a slate, size it, append to the ledger |
| `settle_ledger` | reports the ledger is empty | fill closing prices, compute CLV |
| Render service | serves predictions, markets, simulations | the same, plus a non-empty card |

A scheduled job that silently produces an empty result is indistinguishable from
one that is broken, so both jobs always emit a reason. The workflow log says
which of the two it was.

---

## Scheduled jobs

Logic lives in [`statpitch/ops/jobs.py`](../src/statpitch/ops/jobs.py), not in the
workflow YAML — a step written in shell cannot be unit-tested, and these carry
real correctness requirements. The workflows check out, install, call one job,
and commit if the ledger changed.

```bash
python -m statpitch.ops.jobs flag_card
```

```bash
python -m statpitch.ops.jobs settle_ledger
```

| workflow | cron (UTC) | writes |
|---|---|---|
| [`settle-ledger.yml`](../.github/workflows/settle-ledger.yml) | `0 4 * * *` | `data/bet_ledger.jsonl` |
| [`flag-card.yml`](../.github/workflows/flag-card.yml) | `0 6 * * *` | `data/bet_ledger.jsonl` |
| [`ci.yml`](../.github/workflows/ci.yml) | on push / PR | nothing |

Settlement runs before flagging so a day's results land before that day's card is
built.

### Three hazards these are built around

**A re-run must not double the book.** Workflows get re-run by hand and retried
after a flaky runner. The ledger is append-only, so a second append does not
overwrite anything — it silently doubles both the exposure and the sample size,
and the inflated sample then flows into the CLV statistics as if it were
independent evidence. `flag_card` keys on `(fixture_id, selection)` already
flagged *that day*, so a re-run skips them and a genuine re-flag on a later date
still works. `settle_ledger` only touches entries that are still pending, so
re-running it is a no-op rather than a rewrite.

**Cron is best-effort, and the CLV label depends on the clock.** GitHub's
scheduled workflows are routinely delayed at peak times and can be skipped
outright. The metric is *Friday-to-close* CLV; a snapshot mislabelled by six
hours is worse than a missing one, because nothing downstream can tell. So every
record uses the **actual** run time, never the nominal cron time, and the job
compares the two and reports a drift beyond 30 minutes as a warning.

GitHub does not expose the instant a schedule was meant to fire, so the workflow
reconstructs it (`date -u +%Y-%m-%dT06:00:00+00:00`). That hour is duplicated
between the `cron:` line and the `Derive nominal slot` step, which is exactly the
kind of pair that rots — [`test_deployment.py`](../tests/test_deployment.py)
asserts the two agree. A run delayed across midnight computes *tomorrow's* slot
and would otherwise report itself as early, so a slot in the future is flagged
too.

**A cancelled run can lose an append.** Both ledger workflows share one
`concurrency: group: ledger` with `cancel-in-progress: false`. Cancelling a run
between the append and the push destroys the runner with the file on it, and no
later run can tell the work happened. Queueing is slower and correct. Pushes
rebase and retry rather than forcing, because the ledger's whole value is that
earlier entries are never rewritten.

### Two operational facts worth knowing

- **Scheduled workflows are disabled after 60 days without repository activity.**
  If the jobs stop firing and nothing appears in the Actions tab, this is the
  first thing to check — GitHub emails the repo owner, and re-enabling is a
  button, not a config change.
- **Neither job spends the API-Football budget.** There is no live odds feed in
  this stack, so a polling job would burn the 100/day allowance to learn nothing.
  A test asserts neither job calls `quota.spend`. Anything added later goes
  through `statpitch.quota` (NFR-9).

---

## Render

[`render.yaml`](../render.yaml) is a blueprint on the **free** plan. Point Render
at the repository and it reads the file; no dashboard configuration is needed.

```bash
curl -s https://<your-service>.onrender.com/health
```

### The cold start, stated plainly

A free instance **spins down after ~15 minutes without traffic**, and the next
request pays a cold start of tens of seconds: container start, then ~2.4 s of
imports and ~0.7 s to load the artifacts.

NFR-2 allows ~200 ms. The measured warm path is 3.9 ms for a league prediction
and 6.5 ms for the full 86-selection book, so the budget is met by two orders of
magnitude — **on the warm path**. The first request after idle is not close, and
that is a property of the plan rather than of the code.

There is deliberately no keep-alive ping. Pinging a free instance to defeat its
own spin-down works against the terms of the plan and would spend the monthly
instance hours serving nobody. If the cold start matters, that is what the paid
plan is for; a test asserts this caveat stays documented rather than quietly
dropped.

### The disk is ephemeral, so the service is read-only

Anything written on a free instance is discarded on restart or redeploy. For an
append-only ledger that is the worst available failure: the write *succeeds*, the
instance recycles, and the entry is gone with no error and no gap anyone can
detect.

So the service runs with `STATPITCH_READ_ONLY=1`, and `BetLedger` raises on
`append` and `rewrite` while it is set. Reads are unaffected — `/clv/report` and
`/ledger` work normally. **The ledger is owned by the scheduled job**, which
commits it to the repository; the deployed API only ever serves what that job
has already committed.

That variable is a **floor, not a default**. `BetLedger(path, read_only=True)`
tightens it and is always honoured, but `read_only=False` on a host that has set
it *raises* rather than being obeyed or quietly ignored. Obeying it would let one
call site opt out of a guard the host set for the whole process, and the write
would then succeed and disappear — the exact failure the flag exists to prevent,
and one nothing downstream can detect. Ignoring the argument in silence would be
no better; almost every bug this project has found returned a confident wrong
answer rather than an error.

### Why the build uses a different requirements file

Nothing on a request path imports xgboost, shap, optuna, scikit-learn or the
scrapers — serving loads parquet artifacts, builds a score matrix and sums over
it. The blueprint therefore installs
[`requirements-serving.txt`](../requirements-serving.txt), which keeps hundreds of
megabytes out of a 512 MB instance.

That split is only safe because it is enforced: a test imports
`statpitch.serving.app` in a subprocess and asserts none of the training stack
appears in `sys.modules`. Add a heavy import to a request path and that test
fails, instead of the next deploy failing.

One worker, on purpose. A second buys no throughput for a CPU-bound matrix build
and would duplicate the ~12 MB Elo table in memory.

---

## Running it locally

```bash
.venv/Scripts/python.exe -m uvicorn statpitch.serving.app:app --reload
```

Interactive docs at `/docs`, machine-readable schema at `/openapi.json`.
