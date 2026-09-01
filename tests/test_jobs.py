"""Scheduled job behaviour (Design §10).

The properties tested here are the ones a green workflow run does not
demonstrate. A job that silently flags nothing and a job that is broken produce
the same empty ledger, and a job that runs twice produces a ledger that looks
larger and more significant than the evidence behind it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from statpitch.decision import clv_tracker as clv
from statpitch.ops import jobs

NOON = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)


@pytest.fixture
def ledger_path(tmp_path):
    return tmp_path / "bet_ledger.jsonl"


def _entry(fixture_id="ENG.PL-2026-08-06-ARS-CHE", selection="1x2_home", **kw):
    return clv.flag(
        fixture_id=fixture_id,
        competition_id="ENG.PL",
        selection=selection,
        market_family="1x2",
        odds_taken=2.10,
        price_source=clv.PriceSource.CONSENSUS,
        p_model=0.50,
        q_fair=0.47,
        grade="B",
        stake_fraction=0.01,
        kelly_lambda=0.25,
        w=0.0,
        config_version="test",
        **kw,
    )


# --- doing nothing, out loud --------------------------------------------------

def _placeholder_config():
    """A config that has never been fitted, built rather than read.

    The shipped one is `experimental` now, and these two tests are about what a
    job does when it CANNOT stake.
    """
    from dataclasses import replace

    from statpitch import decision_config

    return replace(
        decision_config.config(), status="placeholder", w_fitted=False, w=None
    )


def test_flag_card_runs_and_reports_why_it_flagged_nothing():
    """An empty result must be distinguishable from a broken job."""
    result = jobs.flag_card(now=NOON, config=_placeholder_config())
    assert result.ok
    assert result.counts["flagged"] == 0
    assert "unfitted" in result.reason
    assert "0.000" in result.reason


def test_flag_card_names_the_config_it_refused_on():
    result = jobs.flag_card(now=NOON, config=_placeholder_config())
    assert "placeholder" in result.reason


def test_settle_ledger_on_an_empty_ledger_says_so(ledger_path):
    result = jobs.settle_ledger(now=NOON, ledger_path=ledger_path)
    assert result.ok
    assert result.counts == {
        "entries": 0, "pending": 0, "settled": 0, "unresolved": 0
    }
    assert "empty" in result.reason


def test_the_summary_is_readable_in_a_workflow_log():
    summary = jobs.flag_card(now=NOON).summary()
    assert summary.startswith("[flag_card] ok")
    assert "flagged=0" in summary


# --- idempotency --------------------------------------------------------------

def test_settling_twice_does_not_change_anything(ledger_path):
    """Re-running a workflow is normal; it must not rewrite settled history."""
    ledger = clv.BetLedger(ledger_path)
    ledger.append(_entry())
    first = jobs.settle_ledger(now=NOON, ledger_path=ledger_path)
    before = ledger_path.read_text(encoding="utf-8")
    second = jobs.settle_ledger(now=NOON, ledger_path=ledger_path)
    assert ledger_path.read_text(encoding="utf-8") == before
    assert first.counts["pending"] == second.counts["pending"] == 1


def test_an_already_settled_entry_is_left_alone(ledger_path):
    ledger = clv.BetLedger(ledger_path)
    entry = _entry()
    clv.settle(
        entry, odds_closing=2.0,
        closing_price_source=clv.PriceSource.CONSENSUS, result=clv.Result.WON,
    )
    ledger.append(entry)
    result = jobs.settle_ledger(now=NOON, ledger_path=ledger_path)
    assert result.counts["pending"] == 0
    assert "already settled" in result.reason


def test_the_duplicate_guard_keys_on_fixture_and_selection(ledger_path):
    """The guard that stops a re-run doubling the book.

    An append-only ledger cannot un-append, so a second flag of the same
    selection would inflate both exposure and the CLV sample permanently.
    """
    ledger = clv.BetLedger(ledger_path)
    ledger.append(_entry(selection="1x2_home"))
    ledger.append(_entry(selection="1x2_away"))
    keys = jobs._flagged_keys(ledger, on=datetime.now(UTC).date())
    assert len(keys) == 2
    assert all(k[0] == "ENG.PL-2026-08-06-ARS-CHE" for k in keys)


def test_the_duplicate_guard_is_scoped_to_the_day(ledger_path):
    """A genuine re-flag on a later date is not a duplicate."""
    ledger = clv.BetLedger(ledger_path)
    ledger.append(_entry(now=datetime(2026, 8, 1, 9, 0, tzinfo=UTC)))
    assert jobs._flagged_keys(ledger, on=NOON.date()) == set()
    assert len(jobs._flagged_keys(ledger, on=datetime(2026, 8, 1).date())) == 1


# --- the clock ----------------------------------------------------------------

def test_a_punctual_run_carries_no_warning():
    result = jobs.flag_card(now=NOON, scheduled_for=NOON - timedelta(minutes=4))
    assert result.warnings == []


def test_a_delayed_run_is_reported_rather_than_relabelled():
    """GitHub's scheduled workflows are best-effort and drift at peak times.

    A `Friday-to-close` snapshot taken six hours late is worse than a missing
    one, because nothing downstream can tell.
    """
    result = jobs.flag_card(now=NOON, scheduled_for=NOON - timedelta(hours=6))
    assert result.warnings
    assert "best-effort" in result.warnings[0]


def test_a_slot_in_the_future_is_also_flagged():
    """The workflow derives its slot as 'today at the cron hour'.

    A run delayed across midnight therefore computes tomorrow's slot and would
    otherwise report itself as comfortably early.
    """
    result = jobs.flag_card(now=NOON, scheduled_for=NOON + timedelta(hours=3))
    assert result.warnings
    assert "future" in result.warnings[0]


def test_the_recorded_timestamp_is_the_real_one_not_the_nominal_one():
    result = jobs.flag_card(now=NOON, scheduled_for=NOON - timedelta(hours=6))
    assert result.ran_at == NOON.isoformat()


# --- the runner ---------------------------------------------------------------

def test_run_rejects_an_unknown_job():
    with pytest.raises(jobs.JobError, match="unknown job"):
        jobs.run("definitely_not_a_job")


def test_every_job_is_reachable_by_name():
    assert set(jobs.JOBS) == {"flag_card", "settle_ledger", "refresh_fixtures"}
    # The ledger jobs are pure with respect to the outside world and safe to run
    # here. `refresh_fixtures` downloads schedules and runs the fitted model, so
    # invoking it would make this a network test that rewrites committed
    # artifacts; its wiring is covered in tests/test_refresh_job.py instead.
    for name in ("flag_card", "settle_ledger"):
        assert jobs.run(name).job == name


def test_the_cli_exits_zero_on_a_deliberate_no_op(capsys):
    assert jobs.main(["flag_card"]) == 0
    assert "[flag_card] ok" in capsys.readouterr().out


def test_the_cli_accepts_a_nominal_slot(capsys):
    assert jobs.main(["flag_card", "--scheduled-for", "2026-08-06T06:00:00+00:00"]) == 0


def test_an_empty_slot_reads_as_no_slot_rather_than_raising():
    """Tolerated, but not relied on.

    An empty string is falsy, so it never reaches `fromisoformat` and the run
    reports no drift. The workflows still omit the flag rather than passing it
    empty — this is the property that makes that a preference rather than a
    requirement, and it is asserted here so the preference is not mistaken for a
    load-bearing constraint.
    """
    assert jobs.main(["flag_card", "--scheduled-for", ""]) == 0
    assert jobs.run("flag_card", scheduled_for=None).warnings == []


def test_a_result_serialises_for_a_workflow_summary():
    payload = json.loads(json.dumps(jobs.flag_card(now=NOON).as_dict()))
    assert payload["job"] == "flag_card"
    assert payload["ran_at"] == NOON.isoformat()


# --- the API budget -----------------------------------------------------------

def test_no_job_spends_the_api_football_allowance(monkeypatch, tmp_path):
    """NFR-9. A polling job would burn the 100/day allowance to learn nothing.

    `refresh_fixtures` is isolated rather than run against the real tree. It
    shells out to the collector scripts, and running them here downloaded
    schedules, fetched live odds over the network, and rewrote five committed
    artifacts — including appending a capture to the append-only odds log — on
    every single run of the suite. `test_every_job_is_reachable_by_name` above
    already excludes it for exactly that reason; this loop was reintroducing it.
    """
    import runpy

    import statpitch.quota as quota

    def explode(*args, **kwargs):
        raise AssertionError("a scheduled job must not spend the API budget")

    monkeypatch.setattr(quota.QuotaBudget, "spend", explode)

    # The pure jobs read the committed decision config, so they run as they are.
    for name in ("flag_card", "settle_ledger"):
        assert jobs.run(name).job == name

    # refresh_fixtures gets a scratch tree and stubbed scripts. The wiring it
    # would otherwise exercise is covered in tests/test_refresh_job.py, which
    # stubs the same way; what matters here is only that reaching it spends
    # nothing.
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    for name in ("fixtures", "predictions"):
        pd.DataFrame({"fixture_id": ["f0"]}).to_parquet(
            processed / f"{name}.parquet", index=False
        )
    monkeypatch.setenv("STATPITCH_DATA", str(tmp_path))
    monkeypatch.setattr(runpy, "run_path", lambda script, run_name=None: None)

    assert jobs.run("refresh_fixtures").job == "refresh_fixtures"
