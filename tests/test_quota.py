"""Quota guard tests (NFR-9).

Every test here uses a fake clock and a fake transport. Nothing in this file
touches the network — the point of the module is that the network cannot be
touched more than the budget allows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from statpitch.quota import DAILY_LIMIT, QuotaBudget, QuotaExhausted


class FakeClock:
    def __init__(self, start: datetime | None = None):
        self.now = start or datetime(2026, 8, 8, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


class Counter:
    """Stands in for the HTTP call; records how many times it actually fired."""

    def __init__(self, payload="ok"):
        self.calls = 0
        self.payload = payload

    def __call__(self):
        self.calls += 1
        return self.payload


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def budget(tmp_path, clock):
    return QuotaBudget(state_path=tmp_path / "quota.json", hard_stop=5, clock=clock)


# --- counting -----------------------------------------------------------------

def test_a_successful_call_consumes_exactly_one_request(budget):
    fetch = Counter()
    assert budget.spend("lineup:1", fetch) == "ok"
    assert fetch.calls == 1
    assert budget.state().used == 1


def test_second_request_for_the_same_fixture_is_served_from_cache(budget):
    fetch = Counter()
    budget.spend("lineup:1", fetch)
    budget.spend("lineup:1", fetch)
    budget.spend("lineup:1", fetch)
    # One lineup call per fixture — this is the rule that keeps a heavy Saturday
    # inside 100 requests.
    assert fetch.calls == 1
    assert budget.state().used == 1


def test_distinct_fixtures_each_cost_one(budget):
    for i in range(4):
        budget.spend(f"lineup:{i}", Counter())
    assert budget.state().used == 4


def test_batched_calls_may_declare_a_higher_cost(budget):
    budget.spend("injuries:all", Counter(), cost=3)
    assert budget.state().used == 3
    assert budget.state().remaining == 2


# --- the hard stop ------------------------------------------------------------

def test_budget_stops_at_the_hard_stop_and_returns_none(budget):
    for i in range(5):
        assert budget.spend(f"f:{i}", Counter()) == "ok"
    blocked = Counter()
    assert budget.spend("f:overflow", blocked) is None
    # The critical assertion: no HTTP call was made past the stop.
    assert blocked.calls == 0
    assert budget.state().exhausted


def test_a_batched_call_cannot_straddle_the_hard_stop(budget):
    budget.spend("a", Counter(), cost=4)
    blocked = Counter()
    assert budget.spend("big", blocked, cost=3) is None
    assert blocked.calls == 0
    assert budget.state().used == 4


def test_strict_mode_raises_instead_of_falling_back(budget):
    for i in range(5):
        budget.spend(f"f:{i}", Counter())
    with pytest.raises(QuotaExhausted, match="exhausted"):
        budget.spend("f:overflow", Counter(), strict=True)


def test_hard_stop_leaves_headroom_below_the_real_limit():
    assert QuotaBudget(hard_stop=90).hard_stop < DAILY_LIMIT


def test_hard_stop_above_the_free_tier_limit_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="daily limit"):
        QuotaBudget(state_path=tmp_path / "q.json", hard_stop=150)


# --- persistence and reset ----------------------------------------------------

def test_counter_survives_a_process_restart(tmp_path, clock):
    path = tmp_path / "quota.json"
    first = QuotaBudget(state_path=path, hard_stop=5, clock=clock)
    first.spend("a", Counter())
    first.spend("b", Counter())

    reborn = QuotaBudget(state_path=path, hard_stop=5, clock=clock)
    assert reborn.state().used == 2
    # A restart is not a way to buy more requests.
    assert reborn.state().remaining == 3


def test_counter_resets_at_the_next_utc_day(budget, clock):
    for i in range(5):
        budget.spend(f"f:{i}", Counter())
    assert budget.state().exhausted

    clock.advance(hours=13)  # 12:00 -> 01:00 the next UTC day
    assert budget.state().used == 0
    assert budget.spend("fresh", Counter()) == "ok"


def test_counter_does_not_reset_later_the_same_utc_day(budget, clock):
    budget.spend("a", Counter())
    clock.advance(hours=6)  # 12:00 -> 18:00, same UTC day
    assert budget.state().used == 1


def test_corrupt_state_file_does_not_wedge_the_pipeline(tmp_path, clock):
    path = tmp_path / "quota.json"
    path.write_text("{not json at all", encoding="utf-8")
    budget = QuotaBudget(state_path=path, hard_stop=5, clock=clock)
    assert budget.spend("a", Counter()) == "ok"


def test_reset_clears_counter_and_cache(budget):
    fetch = Counter()
    budget.spend("a", fetch)
    budget.reset()
    assert budget.state().used == 0
    budget.spend("a", fetch)
    assert fetch.calls == 2


# --- cache TTL ----------------------------------------------------------------

def test_cache_entry_is_not_refetched_inside_24h(budget, clock):
    fetch = Counter()
    budget.spend("lineup:1", fetch)
    clock.advance(hours=23)
    budget.spend("lineup:1", fetch)
    assert fetch.calls == 1


def test_cache_entry_expires_after_24h(tmp_path, clock):
    # Same UTC day would reset the counter too, so use a long TTL window and a
    # generous budget to isolate TTL behaviour from the daily reset.
    budget = QuotaBudget(state_path=tmp_path / "q.json", hard_stop=90, clock=clock)
    fetch = Counter()
    budget.spend("lineup:1", fetch)
    clock.advance(hours=25)
    budget.spend("lineup:1", fetch)
    assert fetch.calls == 2


def test_cached_returns_payload_without_spending(budget):
    budget.spend("lineup:1", Counter("XI"))
    assert budget.cached("lineup:1") == "XI"
    assert budget.cached("lineup:missing") is None
    assert budget.state().used == 1


# --- failure handling ---------------------------------------------------------

def test_a_failing_fetch_returns_none_rather_than_propagating(budget):
    def boom():
        raise ConnectionError("network down")

    assert budget.spend("lineup:1", boom) is None


def test_a_failing_fetch_still_consumes_budget_and_is_not_retried(budget):
    """Budget is reserved before the call.

    An over-count is the safe direction to be wrong in: the request may well have
    reached the API before failing, and the alternative — refunding on error — is
    exactly how a retry loop quietly drains 100 requests.
    """
    def boom():
        raise ConnectionError("network down")

    budget.spend("lineup:1", boom)
    assert budget.state().used == 1

    # The failure is not cached, so a later scheduled attempt may retry it...
    ok = Counter()
    assert budget.spend("lineup:1", ok) == "ok"
    # ...but that retry cost another request, which is what makes looping visible.
    assert budget.state().used == 2


# --- the acceptance scenario --------------------------------------------------

def test_heavy_saturday_stays_inside_the_free_tier(tmp_path, clock):
    """Design §3.2's budget, executed rather than asserted.

    1 fixtures call + 5 batched injury calls + 50 lineup calls = 56, and every
    duplicate request a matchday job might make is absorbed by the cache.
    """
    budget = QuotaBudget(state_path=tmp_path / "q.json", hard_stop=90, clock=clock)

    budget.spend("fixtures:2026-08-08", Counter())
    for comp in ("ENG.PL", "ESP.LALIGA", "GER.BUNDESLIGA", "ITA.SERIEA", "FRA.LIGUE1"):
        budget.spend(f"injuries:{comp}", Counter())
    for fixture_id in range(50):
        budget.spend(f"lineup:{fixture_id}", Counter())

    assert budget.state().used == 56

    # A second pass over the same slate — the shape a naive poller would take —
    # costs nothing at all.
    budget.spend("fixtures:2026-08-08", Counter())
    for fixture_id in range(50):
        budget.spend(f"lineup:{fixture_id}", Counter())

    assert budget.state().used == 56
    assert budget.state().used < DAILY_LIMIT


def test_state_file_is_valid_json_after_writes(budget):
    budget.spend("a", Counter())
    json.loads(budget.state_path.read_text(encoding="utf-8"))
