"""The scheduled refresh (Roadmap §11.1).

`fixtures.parquet` and `predictions.parquet` are claims about the future that
decay — kickoff times move, rounds get drawn, and a precomputed prediction is
only as current as the fixture list it came from. Serving reads both at startup
and cannot refresh them itself, because NFR-2 forbids a network call on a request
path. Something scheduled has to.

The steps are covered end to end by actually running them elsewhere; what needs
testing here is the wiring that is easy to get wrong and invisible when it is:
the job reports what it did, refuses to claim success when a step failed, and
notices when the two artifacts drift out of step with each other.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from statpitch.ops import jobs


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """A processed tree the job can read, with the scripts stubbed out."""
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    monkeypatch.setenv("STATPITCH_DATA", str(tmp_path))

    def write(fixtures: int, predictions: int) -> None:
        pd.DataFrame({"fixture_id": [f"f{i}" for i in range(fixtures)]}).to_parquet(
            processed / "fixtures.parquet", index=False
        )
        pd.DataFrame({"fixture_id": [f"f{i}" for i in range(predictions)]}).to_parquet(
            processed / "predictions.parquet", index=False
        )

    return write


def _stub_scripts(monkeypatch, *, fail_on: str | None = None):
    calls: list[str] = []

    def fake_run_path(script, run_name=None):
        calls.append(script)
        if fail_on and fail_on in script:
            raise SystemExit(1)

    monkeypatch.setattr("runpy.run_path", fake_run_path)
    return calls


def test_it_runs_every_step_in_order(artifacts, monkeypatch):
    """The order is load-bearing, and getting it wrong loses work silently.

    build_fixtures REBUILDS the list from openfootball, so correcting dates
    before it would be overwritten. precompute keys on the corrected dates, so
    correcting after it would leave predictions filed under provisional ones.
    The corrections belong strictly between the two — including the live-odds
    capture, whose bookmaker-confirmed kickoffs a later rebuild would revert.
    """
    artifacts(10, 10)
    calls = _stub_scripts(monkeypatch)
    result = jobs.refresh_fixtures()
    assert result.ok
    assert [c.split("/")[-1] for c in calls] == [
        "build_fixtures.py", "collect_fixtures.py", "collect_live_odds.py",
        "precompute_predictions.py",
    ]


def test_it_reports_what_changed(artifacts, monkeypatch):
    artifacts(10, 10)
    _stub_scripts(monkeypatch)
    result = jobs.refresh_fixtures()
    assert result.counts["fixtures_before"] == 10
    assert result.counts["fixtures_after"] == 10
    assert result.counts["predictions"] == 10


def test_a_failing_step_is_not_reported_as_success(artifacts, monkeypatch):
    artifacts(10, 10)
    _stub_scripts(monkeypatch, fail_on="build_fixtures")
    result = jobs.refresh_fixtures()
    assert not result.ok
    assert "fixtures step exited 1" in result.reason


def test_it_stops_before_predicting_when_the_fixture_step_fails(artifacts, monkeypatch):
    """Predicting against a half-written fixture list is worse than not running."""
    artifacts(10, 10)
    calls = _stub_scripts(monkeypatch, fail_on="build_fixtures")
    jobs.refresh_fixtures()
    assert len(calls) == 1


def test_a_failing_date_correction_stops_the_run(artifacts, monkeypatch):
    """It rewrites the fixture file in place, so a half-done pass must not be
    handed to precompute as if it were complete."""
    artifacts(10, 10)
    calls = _stub_scripts(monkeypatch, fail_on="collect_fixtures")
    result = jobs.refresh_fixtures()
    assert not result.ok
    assert "date correction step exited 1" in result.reason
    assert len(calls) == 2


def test_artifacts_drifting_apart_is_warned_about(artifacts, monkeypatch):
    """Fewer predictions than fixtures means some answer from the Elo fallback."""
    artifacts(10, 7)
    _stub_scripts(monkeypatch)
    result = jobs.refresh_fixtures()
    assert result.ok
    assert any("10 fixtures but 7 predictions" in w for w in result.warnings)


def test_a_late_run_says_so(artifacts, monkeypatch):
    """GitHub's scheduled workflows are best-effort; jobs.py already assumes it."""
    artifacts(10, 10)
    _stub_scripts(monkeypatch)
    ran_at = datetime.now(UTC)
    result = jobs.refresh_fixtures(
        scheduled_for=ran_at - timedelta(hours=3), now=ran_at
    )
    assert any("after its" in w for w in result.warnings)


def test_a_dispatched_run_is_neither_early_nor_late(artifacts, monkeypatch):
    artifacts(10, 10)
    _stub_scripts(monkeypatch)
    assert jobs.refresh_fixtures().warnings == []


def test_each_script_sees_only_its_own_argv(artifacts, monkeypatch):
    """`run_path` executes in-process, so argv leaks from the job to the script.

    It did: build_fixtures.py received the literal "refresh_fixtures" from
    `python -m statpitch.ops.jobs refresh_fixtures` and argparse rejected it,
    failing the first real workflow run. The earlier tests stub the scripts, so
    the one thing they could not check was how those scripts are invoked.
    """
    import sys

    artifacts(10, 10)
    seen: list[list[str]] = []

    def record_argv(script, run_name=None):
        seen.append(list(sys.argv))

    monkeypatch.setattr("runpy.run_path", record_argv)
    monkeypatch.setattr(sys, "argv", ["-m statpitch.ops.jobs", "refresh_fixtures"])
    jobs.refresh_fixtures()

    assert seen == [
        ["scripts/build_fixtures.py"],
        ["scripts/collect_fixtures.py"],
        ["scripts/collect_live_odds.py"],
        ["scripts/precompute_predictions.py"],
    ]
    # And the job's own argv is restored, so a caller after it is unaffected.
    assert sys.argv == ["-m statpitch.ops.jobs", "refresh_fixtures"]
