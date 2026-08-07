"""Deployment contracts (render.yaml, .github/workflows).

Configuration files are not exercised by anything else in this suite, so the
mistakes in them surface in production or not at all. These tests cover the ones
that fail quietly rather than loudly:

* a serving import that pulls in the training stack — the deploy builds a
  dependency file that no longer covers it
* a ledger write on an ephemeral disk — it succeeds, then disappears
* a workflow that cancels itself mid-append
* a cron hour that no longer matches the slot the job compares against
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
RENDER = ROOT / "render.yaml"


@pytest.fixture(scope="module")
def render() -> str:
    return RENDER.read_text(encoding="utf-8")


def _workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


# --- the serving dependency split ---------------------------------------------

def test_serving_imports_nothing_outside_the_serving_requirements():
    """The check that makes requirements-serving.txt safe to keep slim.

    The deployed image installs the serving subset, so an import added to a
    request path without a matching dependency would build fine and fail at
    startup. Run in a subprocess because the rest of this suite has already
    imported the training stack into this interpreter.
    """
    probe = (
        "import sys, statpitch.serving.app;"
        "heavy = [m for m in ('xgboost','shap','optuna','sklearn','cloudscraper',"
        "'bs4','lxml','requests') if m in sys.modules];"
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, cwd=ROOT, check=True,
    )
    assert out.stdout.strip() == "", (
        f"serving now imports {out.stdout.strip()}; add it to "
        "requirements-serving.txt or keep it off the request path"
    )


def test_the_serving_requirements_cover_what_serving_needs():
    text = (ROOT / "requirements-serving.txt").read_text(encoding="utf-8")
    for package in ("fastapi", "uvicorn", "pydantic", "pandas", "pyarrow", "scipy"):
        assert package in text


def test_render_builds_from_the_serving_subset(render):
    assert "requirements-serving.txt" in render
    assert "pip install -r requirements.txt" not in render


# --- the ephemeral disk -------------------------------------------------------

def test_render_runs_the_api_read_only(render):
    """A free-plan disk is ephemeral; a write succeeds and then vanishes."""
    assert "STATPITCH_READ_ONLY" in render


def test_a_read_only_ledger_refuses_to_append(tmp_path, monkeypatch):
    from statpitch.decision import clv_tracker as clv

    monkeypatch.setenv("STATPITCH_READ_ONLY", "1")
    ledger = clv.BetLedger(tmp_path / "bet_ledger.jsonl")
    entry = clv.flag(
        fixture_id="f", competition_id="ENG.PL", selection="1x2_home",
        market_family="1x2", odds_taken=2.0,
        price_source=clv.PriceSource.CONSENSUS, p_model=0.5, q_fair=0.5,
        grade="C", stake_fraction=0.0, kelly_lambda=0.25, w=0.0,
        config_version="test",
    )
    with pytest.raises(clv.LedgerError, match="read-only"):
        ledger.append(entry)


def test_a_read_only_ledger_refuses_to_rewrite(tmp_path, monkeypatch):
    from statpitch.decision import clv_tracker as clv

    monkeypatch.setenv("STATPITCH_READ_ONLY", "1")
    with pytest.raises(clv.LedgerError, match="read-only"):
        clv.BetLedger(tmp_path / "bet_ledger.jsonl").rewrite()


def test_a_read_only_ledger_still_reads(tmp_path, monkeypatch):
    """Refusing writes must not disable /clv/report and /ledger."""
    from statpitch.decision import clv_tracker as clv

    path = tmp_path / "bet_ledger.jsonl"
    path.write_text("", encoding="utf-8")
    monkeypatch.setenv("STATPITCH_READ_ONLY", "1")
    assert clv.BetLedger(path).entries == []


def test_writes_are_allowed_by_default(tmp_path, monkeypatch):
    """The scheduled job owns the ledger and must not be locked out."""
    from statpitch.decision import clv_tracker as clv

    monkeypatch.delenv("STATPITCH_READ_ONLY", raising=False)
    assert clv.BetLedger(tmp_path / "bet_ledger.jsonl").read_only is False


def test_asking_for_a_writable_ledger_on_a_read_only_host_is_refused(
    tmp_path, monkeypatch
):
    """The environment is a floor, not a default.

    Honouring `read_only=False` here would let one call site opt out of a guard
    the host set for the whole process, and the write would then succeed and
    vanish — which is the failure the flag exists to prevent. Ignoring the
    argument silently would be no better: this project's bugs have almost all
    been confident wrong answers rather than errors.
    """
    from statpitch.decision import clv_tracker as clv

    monkeypatch.setenv("STATPITCH_READ_ONLY", "1")
    with pytest.raises(clv.LedgerError, match="STATPITCH_READ_ONLY"):
        clv.BetLedger(tmp_path / "bet_ledger.jsonl", read_only=False)


def test_an_explicit_flag_still_works_where_the_host_allows_writes(
    tmp_path, monkeypatch
):
    """Nothing is taken away from a caller running somewhere writes survive."""
    from statpitch.decision import clv_tracker as clv

    monkeypatch.delenv("STATPITCH_READ_ONLY", raising=False)
    assert clv.BetLedger(tmp_path / "a.jsonl", read_only=False).read_only is False
    assert clv.BetLedger(tmp_path / "b.jsonl", read_only=True).read_only is True


def test_the_flag_can_tighten_but_not_loosen(tmp_path, monkeypatch):
    """read_only=True is always honoured; only the relaxing direction is barred."""
    from statpitch.decision import clv_tracker as clv

    monkeypatch.setenv("STATPITCH_READ_ONLY", "1")
    assert clv.BetLedger(tmp_path / "c.jsonl", read_only=True).read_only is True


# --- the workflows ------------------------------------------------------------

@pytest.mark.parametrize("name", ["ci.yml", "flag-card.yml", "settle-ledger.yml"])
def test_every_workflow_pins_a_python_version(name):
    assert 'python-version: "3.11"' in _workflow(name)


@pytest.mark.parametrize("name", ["flag-card.yml", "settle-ledger.yml"])
def test_ledger_workflows_never_cancel_themselves(name):
    """A run cancelled between the append and the push loses the entry silently.

    The runner is destroyed with the file on it, and no later run can tell the
    work happened.
    """
    text = _workflow(name)
    assert "cancel-in-progress: false" in text
    assert "group: ledger" in text


@pytest.mark.parametrize("name", ["flag-card.yml", "settle-ledger.yml"])
def test_ledger_workflows_share_one_concurrency_group(name):
    """Flagging and settling must not interleave on the same file."""
    assert re.search(r"concurrency:\s*\n\s*group: ledger", _workflow(name))


@pytest.mark.parametrize("name", ["flag-card.yml", "settle-ledger.yml"])
def test_the_cron_hour_matches_the_slot_the_job_is_told_about(name):
    """The one duplicated value in these files.

    The job compares its real start time against this slot to decide whether the
    run drifted. If the cron hour is changed and the derived slot is not, every
    run reports itself as hours late — or worse, an hour early, and the drift
    check stops meaning anything.
    """
    text = _workflow(name)
    cron_hour = int(re.search(r'cron: "0 (\d+) \* \* \*"', text).group(1))
    slot_hour = int(re.search(r"date -u \+%Y-%m-%dT(\d+):00:00", text).group(1))
    assert cron_hour == slot_hour


@pytest.mark.parametrize("name", ["flag-card.yml", "settle-ledger.yml"])
def test_only_a_scheduled_run_is_given_a_nominal_slot(name):
    """A dispatched run is neither early nor late.

    Deriving a slot for one produced a drift warning that was correct about the
    arithmetic and meaningless about the run, on the very first manual trigger.
    A warning that fires when nothing is wrong teaches the reader to skip it on
    the scheduled path, where it is the only thing standing between a mislabelled
    snapshot and the CLV series.
    """
    assert "if: github.event_name == 'schedule'" in _workflow(name)


@pytest.mark.parametrize("name", ["flag-card.yml", "settle-ledger.yml"])
def test_the_slot_flag_is_omitted_rather_than_passed_empty(name):
    """`--scheduled-for ""` would be an argument error, not a skipped one."""
    text = _workflow(name)
    assert "${SLOT:+--scheduled-for $SLOT}" in text
    assert '--scheduled-for "${{ steps.slot.outputs.at }}"' not in text


@pytest.mark.parametrize("name", ["flag-card.yml", "settle-ledger.yml"])
def test_the_slot_reaches_the_shell_through_an_environment_variable(name):
    """Workflow expressions are interpolated before the shell sees them.

    Routing through `env:` keeps the value a value. It is our own `date` output
    today, but the habit is what stops the next one being someone else's input.
    """
    assert "SLOT: ${{ steps.slot.outputs.at }}" in _workflow(name)


def test_a_job_run_without_a_slot_reports_no_drift():
    """The other half of the same contract, on the Python side."""
    from datetime import UTC, datetime

    from statpitch.ops import jobs

    result = jobs.flag_card(now=datetime(2026, 8, 7, 3, 19, tzinfo=UTC))
    assert result.warnings == []


def test_settlement_is_scheduled_before_flagging():
    """A day's results should land before that day's card is built."""
    settle = int(re.search(r'cron: "0 (\d+)', _workflow("settle-ledger.yml")).group(1))
    flag = int(re.search(r'cron: "0 (\d+)', _workflow("flag-card.yml")).group(1))
    assert settle < flag


@pytest.mark.parametrize("name", ["flag-card.yml", "settle-ledger.yml"])
def test_a_committing_workflow_asks_for_write_permission(name):
    assert "contents: write" in _workflow(name)


@pytest.mark.parametrize("name", ["flag-card.yml", "settle-ledger.yml"])
def test_the_push_rebases_rather_than_forces(name):
    """The ledger's value is that earlier entries are never rewritten."""
    text = _workflow(name)
    assert "--rebase" in text
    assert "--force" not in text


@pytest.mark.parametrize("name", ["flag-card.yml", "settle-ledger.yml"])
def test_the_workflows_call_the_tested_module_not_inline_logic(name):
    """Logic in a YAML step cannot be unit-tested, so there is none."""
    assert "python -m statpitch.ops.jobs" in _workflow(name)


def test_ci_runs_both_the_linter_and_the_suite():
    text = _workflow("ci.yml")
    assert "ruff check" in text
    assert "pytest" in text


def test_ci_may_cancel_itself_because_it_writes_nothing():
    assert "cancel-in-progress: true" in _workflow("ci.yml")


# --- render ------------------------------------------------------------------

def test_render_stays_on_the_free_plan(render):
    """NFR-1: the zero-cost constraint is binding."""
    assert "plan: free" in render


def test_render_health_check_points_at_a_real_route(render):
    from statpitch.serving.app import app

    path = re.search(r"healthCheckPath: (\S+)", render).group(1)
    assert path in {r.path for r in app.routes}


def test_render_starts_the_app_that_exists(render):
    assert "statpitch.serving.app:app" in render


def test_render_runs_a_single_worker(render):
    """512 MB, and the Elo table expands per process."""
    assert "--workers 1" in render


def test_the_cold_start_caveat_is_documented_not_hidden():
    """Free instances spin down after ~15 minutes; the next request pays for it.

    NFR-2's 200ms budget describes the warm path. Claiming it end-to-end without
    naming the cold start would be the kind of number this project has spent the
    whole build refusing to publish.
    """
    doc = (ROOT / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")
    assert "cold start" in doc.lower()
    assert "spin" in doc.lower()
