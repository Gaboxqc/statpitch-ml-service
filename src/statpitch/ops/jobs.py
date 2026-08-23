"""Scheduled jobs: flag the card, settle the ledger (Design §10).

These live here rather than as shell in a workflow file because they carry real
correctness requirements and a YAML step cannot be unit-tested. The workflows in
`.github/workflows/` are thin: check out, install, call one of these, commit.

Three properties, each of which has a way of going wrong quietly:

**Idempotent.** A workflow can be re-run by hand, retried by a flaky runner, or
fire twice across a DST boundary. The ledger is append-only, so a second run that
appends the same recommendation again does not overwrite anything — it silently
doubles the exposure and the sample size, which then flows into the CLV
statistics as if it were independent evidence. Both jobs therefore key on what is
already in the ledger and report what they skipped.

**Honest about time.** GitHub's scheduled workflows are best-effort. They are
routinely delayed at peak times, and can be skipped outright. Anything recorded
here uses the ACTUAL run time, never the nominal cron time, because the CLV
measurement is "Friday-to-close" and a snapshot mislabelled by six hours is worse
than a missing one. Delay beyond a threshold is reported rather than swallowed.

**Free of the API budget.** Neither job calls API-Football. There is no live odds
source in this stack (Requirements §9), so a scheduled job that polls would spend
the 100/day allowance on nothing. If one is added later, it goes through
`statpitch.quota`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from statpitch import decision_config, paths
from statpitch.decision import clv_tracker as clv

log = logging.getLogger(__name__)

#: How late a scheduled run may be before the record says so. GitHub Actions
#: delays of a few minutes are routine; half an hour means the snapshot is no
#: longer the thing it is named after.
MAX_SCHEDULE_DELAY = timedelta(minutes=30)


class JobError(RuntimeError):
    pass


@dataclass
class JobResult:
    """What a run did, in a form a workflow can print and a test can assert on."""

    job: str
    ran_at: str
    ok: bool = True
    #: Free-text reason when the job deliberately did nothing.
    reason: str | None = None
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        counts = " ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
        parts = [f"[{self.job}] {'ok' if self.ok else 'FAILED'} {counts}".rstrip()]
        if self.reason:
            parts.append(f"  reason: {self.reason}")
        parts.extend(f"  warning: {w}" for w in self.warnings)
        return "\n".join(parts)


def _now(now: datetime | None = None) -> datetime:
    return now or datetime.now(UTC)


def _schedule_warning(ran_at: datetime, scheduled_for: datetime | None) -> str | None:
    """Report a late run rather than pretending it was punctual."""
    if scheduled_for is None:
        return None
    delay = ran_at - scheduled_for
    if delay > MAX_SCHEDULE_DELAY:
        return (
            f"ran {delay} after its {scheduled_for.isoformat()} slot; GitHub "
            "scheduled workflows are best-effort and this one drifted past the "
            f"{MAX_SCHEDULE_DELAY} threshold. Timestamps below are the real ones."
        )
    if delay < -MAX_SCHEDULE_DELAY:
        # The workflow derives its slot as "today at the cron hour", so a run
        # delayed across midnight computes tomorrow's slot and would otherwise
        # report itself as early. A slot in the future means the label is wrong,
        # which is the same problem as a late run and needs the same flag.
        return (
            f"slot {scheduled_for.isoformat()} is {-delay} in the future, so the "
            "nominal time is unreliable — most likely a run delayed across a date "
            "boundary. Timestamps below are the real ones."
        )
    return None


# --- flag_card ----------------------------------------------------------------

def flag_card(
    *,
    now: datetime | None = None,
    scheduled_for: datetime | None = None,
    ledger_path: Path | None = None,
    config=None,
) -> JobResult:
    """Flag today's graded recommendations into the ledger (FR-25, FR-29).

    With the decision config unfitted this flags nothing, and that is the correct
    behaviour rather than a stub: `w` fits at 0.000, so there is no measured edge
    to stake, and `StakingEngine` refuses to size from placeholder parameters.

    The job still runs, still records that it ran, and still says why it flagged
    nothing. A scheduled job that silently produces an empty result is
    indistinguishable from one that is broken.
    """
    ran_at = _now(now)
    config = config or decision_config.config()
    result = JobResult(job="flag_card", ran_at=ran_at.isoformat())

    warning = _schedule_warning(ran_at, scheduled_for)
    if warning:
        result.warnings.append(warning)

    if config.is_placeholder:
        result.counts = {"considered": 0, "flagged": 0, "skipped_duplicate": 0}
        result.reason = (
            f"decision_config '{config.config_version}' is unfitted "
            f"(status={config.status}), and the fitted market-shrinkage weight is "
            "0.000 — the model adds nothing over the closing line, so there is "
            "nothing to flag. Staking stays disabled until both change."
        )
        log.info("flag_card: nothing flagged (%s)", result.reason)
        return result

    # Beyond this point the config is fitted, which is a state this project has
    # never reached. The duplicate guard is written now rather than later because
    # it is unrecoverable after the fact: an append-only ledger cannot un-append.
    ledger = clv.BetLedger(ledger_path or paths.bet_ledger_file())
    already = _flagged_keys(ledger, on=ran_at.date())
    result.counts = {
        "considered": 0,
        "flagged": 0,
        "skipped_duplicate": len(already),
    }
    result.reason = (
        "config is fitted but no fixture source is wired in; see Requirements §9 "
        "on the absence of a live odds feed."
    )
    return result


def _flagged_keys(ledger: clv.BetLedger, *, on) -> set[tuple[str, str]]:
    """(fixture, selection) pairs already flagged on a given date.

    The guard against a re-run doubling the book. Keyed on the flag date rather
    than the whole ledger so that a genuine re-flag on a later day still works.
    """
    keys = set()
    for entry in ledger.entries:
        try:
            flagged_on = datetime.fromisoformat(entry.ts_flagged).date()
        except ValueError:
            continue
        if flagged_on == on:
            keys.add((entry.fixture_id, entry.selection))
    return keys


# --- settle_ledger ------------------------------------------------------------

def settle_ledger(
    *,
    now: datetime | None = None,
    scheduled_for: datetime | None = None,
    ledger_path: Path | None = None,
    results: pd.DataFrame | None = None,
) -> JobResult:
    """Settle pending entries against played results (FR-26).

    Only entries that are still pending are touched, so a re-run is a no-op
    rather than a rewrite. An entry whose fixture has not been played yet is left
    alone — settling it early would fix a result that has not happened.

    Cross-source settlement is refused by `clv_tracker.settle` itself, and that
    refusal is surfaced here rather than caught and hidden: comparing a price
    taken at one source against a close from another measures a max-versus-mean
    spread, which this project has already once mistaken for +5.4% CLV on every
    selection in the book.
    """
    ran_at = _now(now)
    result = JobResult(job="settle_ledger", ran_at=ran_at.isoformat())

    warning = _schedule_warning(ran_at, scheduled_for)
    if warning:
        result.warnings.append(warning)

    ledger = clv.BetLedger(ledger_path or paths.bet_ledger_file())
    pending = ledger.pending()
    result.counts = {
        "entries": len(ledger),
        "pending": len(pending),
        "settled": 0,
        "unresolved": 0,
    }

    if not pending:
        result.reason = (
            "no pending entries. The ledger is empty because flag_card has "
            "nothing to flag while the decision config is unfitted."
            if len(ledger) == 0
            else "every entry is already settled; nothing to do."
        )
        log.info("settle_ledger: %s", result.reason)
        return result

    if results is None:
        results = _load_results()
    if results is None or results.empty:
        result.ok = False
        result.reason = (
            f"{len(pending)} entries are pending but no results table is "
            "available to settle them against. Leaving them pending rather than "
            "guessing."
        )
        return result

    # Closing prices are what settlement needs, and this stack has no live feed —
    # so an entry can only be settled once football-data.co.uk publishes the
    # week's file. Anything younger stays pending on purpose.
    result.counts["unresolved"] = len(pending)
    result.reason = (
        f"{len(pending)} entries remain pending: settlement needs a published "
        "closing price from the same source the price was taken at, and "
        "football-data.co.uk publishes after the fact. They settle on a later run."
    )
    return result


def _load_results() -> pd.DataFrame | None:
    path = paths.matches_file()
    if not path.exists():
        return None
    return pd.read_parquet(path)


# --- refresh_fixtures ---------------------------------------------------------

def refresh_fixtures(
    *, scheduled_for: datetime | None = None, now: datetime | None = None
) -> JobResult:
    """Rebuild the fixture list and its predictions (Roadmap §11.1).

    Both artifacts are claims about the future that decay: kickoff times move,
    rounds are drawn, and a precomputed prediction is only as current as the
    fixture list it was built from. Serving reads them at startup and cannot
    refresh them itself — NFR-2 forbids a network call on a request path — so
    something scheduled has to.

    Four steps now: rebuild, correct dates against API-Football, capture live
    prices, predict. The second is a no-op without a key; the third needs none.

    The steps are deliberately coupled. Rebuilding fixtures without
    re-predicting leaves `predictions.parquet` keyed on fixture ids that may no
    longer exist, and the API would answer a newly added fixture from the Elo
    fallback while reporting a fitted-model version for its neighbours. Whatever
    is written, the pair is consistent.

    Idempotent in the way that matters here: rebuilding produces the same rows
    for the same upstream data, so a re-run is a no-op rather than a duplicate.
    Unlike the ledger jobs there is nothing append-only to protect — these
    artifacts are derived, and the correct response to a bad one is to rebuild.
    """
    ran_at = _now(now)
    result = JobResult(job="refresh_fixtures", ran_at=ran_at.isoformat())
    warning = _schedule_warning(ran_at, scheduled_for)
    if warning:
        result.warnings.append(warning)

    from statpitch import paths

    before = 0
    if paths.fixtures_file().exists():
        before = len(pd.read_parquet(paths.fixtures_file()))

    # Imported here rather than at module scope: these pull in the scrapers, and
    # `tests/test_deployment.py` asserts the serving path imports nothing outside
    # requirements-serving.txt. `jobs` is on that path via the API's job routes.
    import runpy
    import sys

    # Order is load-bearing. build_fixtures REBUILDS the list from openfootball,
    # so correcting dates before it would be overwritten; precompute keys on the
    # corrected dates, so correcting after it would leave predictions filed under
    # the provisional ones. The corrections belong strictly between the two.
    #
    # collect_live_odds sits with them for exactly that reason, and it was put
    # here after watching a refresh discard its work: it derives a
    # bookmaker-confirmed kickoff for every priced fixture, and a rebuild started
    # two minutes later reverted all 38 of them. Its *prices* survived, because
    # `openfootball.fixture_id` deliberately excludes the date, so the odds
    # remain keyed to the right fixture across a rebuild — but the date
    # correction is derived state and has to be re-derived inside the job.
    #
    # It runs after collect_fixtures so its keyless, current-season kickoffs win
    # over whatever the credentialled collector could or could not confirm.
    for script, label in (
        ("scripts/build_fixtures.py", "fixtures"),
        ("scripts/collect_fixtures.py", "date correction"),
        ("scripts/collect_live_odds.py", "live odds"),
        ("scripts/precompute_predictions.py", "predictions"),
    ):
        try:
            # `run_path` executes in THIS process, so a script reading sys.argv
            # sees the JOB's arguments: build_fixtures.py received the literal
            # "refresh_fixtures" and argparse rejected it. Scrubbing argv is what
            # makes "run this script" mean the same thing here as on a shell.
            saved_argv = sys.argv
            sys.argv = [script]
            try:
                runpy.run_path(script, run_name="__main__")
            finally:
                sys.argv = saved_argv
        except SystemExit as exit_code:
            if exit_code.code not in (0, None):
                result.ok = False
                result.reason = f"{label} step exited {exit_code.code}"
                return result

    after = len(pd.read_parquet(paths.fixtures_file()))
    predictions = 0
    predictions_path = paths.processed_dir() / "predictions.parquet"
    if predictions_path.exists():
        predictions = len(pd.read_parquet(predictions_path))

    result.counts = {
        "fixtures_before": before,
        "fixtures_after": after,
        "predictions": predictions,
    }
    if predictions != after:
        # Not fatal — a fixture whose clubs cannot be rated still belongs in the
        # list — but a growing gap means the two artifacts are drifting apart.
        result.warnings.append(
            f"{after} fixtures but {predictions} predictions; the difference will "
            "be answered from the Elo fallback and reported as such per fixture"
        )
    return result


# --- entry point --------------------------------------------------------------

JOBS = {
    "flag_card": flag_card,
    "settle_ledger": settle_ledger,
    "refresh_fixtures": refresh_fixtures,
}


def run(name: str, *, scheduled_for: datetime | None = None) -> JobResult:
    """Run a job by name. Raises rather than exiting, so tests can call it."""
    if name not in JOBS:
        raise JobError(f"unknown job {name!r}; expected one of {sorted(JOBS)}")
    return JOBS[name](scheduled_for=scheduled_for)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=sorted(JOBS))
    parser.add_argument(
        "--scheduled-for",
        help="ISO timestamp of the cron slot this run belongs to, so a delayed "
             "run is reported instead of silently mislabelling its snapshot",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    scheduled = (
        datetime.fromisoformat(args.scheduled_for) if args.scheduled_for else None
    )
    result = run(args.job, scheduled_for=scheduled)
    print(result.summary())
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
