"""The Odds API client (Plan §4 Phase D).

Offline. No test sets a real key, and none reaches the network.

The properties worth pinning here are about the *budget*, because the free tier
is 500 credits a month rather than a day: burn it on the 3rd and there is nothing
until the 1st. Exhaustion is far less recoverable than API-Football's daily
allowance, so the guards matter more.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from statpitch.data import odds_api

FA_CUP = "ENG.FA_CUP"


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv(odds_api.ENV_KEY, "test-key")


@pytest.fixture
def keyless(monkeypatch):
    monkeypatch.delenv(odds_api.ENV_KEY, raising=False)


class FakeSession:
    """Returns a canned body and headers, counting calls."""

    def __init__(self, payload, headers=None, *, fail=False):
        self.body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        self.headers = headers or {}
        self.fail = fail
        self.calls: list[str] = []

    def get_with_headers(self, url, *, suffix=".bin", max_age=None, **kw):
        from statpitch.data.http import FetchError

        self.calls.append(url)
        if self.fail:
            raise FetchError("boom")
        return self.body, self.headers


def _event(eid, home, away, stamp):
    return {"id": eid, "home_team": home, "away_team": away, "commence_time": stamp}


@pytest.fixture
def events():
    return [
        _event("e1", "Arsenal", "Chelsea", "2026-09-05T14:00:00Z"),
        _event("e2", "Liverpool", "Everton", "2026-09-06T16:30:00Z"),
    ]


# --- no key is a capability, not a failure ------------------------------------

def test_without_a_key_nothing_is_fetched(keyless):
    assert not odds_api.configured()
    assert odds_api.fetch_events(FA_CUP) is None


def test_without_a_key_the_schedule_is_empty_and_says_what_is_missing(keyless, caplog):
    with caplog.at_level("INFO"):
        frame = odds_api.build_all_schedules()
    assert frame.empty
    assert odds_api.ENV_KEY in caplog.text
    assert "ENG.FA_CUP" in caplog.text


def test_an_unmapped_competition_is_declined(keyed):
    session = FakeSession([])
    assert odds_api.fetch_events("NOT.A.COMP", session=session) is None
    assert session.calls == []


# --- the budget ---------------------------------------------------------------

def test_the_balance_is_read_from_the_response_not_counted(keyed, events):
    """A local counter is wrong the moment the key is used from anywhere else."""
    session = FakeSession(events, {"x-requests-remaining": "412", "x-requests-used": "88"})
    budget = odds_api.Budget()
    odds_api.fetch_events(FA_CUP, session=session, budget=budget)

    assert budget.remaining == 412
    assert budget.used == 88


def test_an_unobserved_budget_is_unknown_rather_than_exhausted():
    """Refusing to call before anything is observed would mean the first call of
    every month never happens."""
    budget = odds_api.Budget()
    assert budget.remaining is None
    assert not budget.exhausted


def test_a_cache_hit_reports_no_headers_and_must_not_read_as_exhausted(keyed, events):
    """There was no response to read a balance from. Empty means unknown."""
    budget = odds_api.Budget(remaining=300)
    budget.observe({})
    assert budget.remaining == 300
    assert not budget.exhausted


def test_the_reserve_stops_spending_before_zero():
    """The free tier resets monthly, so an exhausted budget stays exhausted."""
    budget = odds_api.Budget(remaining=odds_api.DEFAULT_RESERVE, reserve=odds_api.DEFAULT_RESERVE)
    assert budget.exhausted
    assert odds_api.Budget(remaining=odds_api.DEFAULT_RESERVE + 1).exhausted is False


def test_events_are_free_so_an_exhausted_budget_still_lists_fixtures(keyed, events):
    """The whole reason a key is worth having for cups.

    `/events` costs 0 credits, so cup fixtures keep arriving even with the
    month's allowance spent.
    """
    session = FakeSession(events, {"x-requests-remaining": "0"})
    budget = odds_api.Budget(remaining=0)
    assert budget.exhausted

    fetched = odds_api.fetch_events(FA_CUP, session=session, budget=budget)
    assert fetched is not None and len(fetched) == 2
    assert budget.spent_this_run == 0


def test_an_unparseable_header_is_ignored_rather_than_crashing(keyed, events):
    budget = odds_api.Budget()
    budget.observe({"x-requests-remaining": "not-a-number"})
    assert budget.remaining is None


# --- degradation --------------------------------------------------------------

def test_a_failed_request_yields_no_fixtures(keyed):
    session = FakeSession(None, fail=True)
    assert odds_api.fetch_events(FA_CUP, session=session) is None


def test_a_malformed_payload_yields_no_fixtures(keyed):
    session = FakeSession(None)
    session.body = b"{not json"
    assert odds_api.fetch_events(FA_CUP, session=session) is None


def test_a_non_list_payload_is_rejected(keyed):
    session = FakeSession({"message": "invalid key"})
    assert odds_api.fetch_events(FA_CUP, session=session) is None


# --- parsing ------------------------------------------------------------------

def test_the_round_is_recorded_as_unknown_rather_than_guessed(events):
    """The API does not say which round a fixture belongs to.

    `taxonomy.resolve_format` keys the tie format off the stage, so guessing
    from the date would price a two-legged quarter-final as a single leg.
    """
    frame = odds_api.parse_events(events, FA_CUP)
    assert set(frame["stage"]) == {"unknown"}
    assert not frame["stage_confirmed"].any()


def test_the_format_falls_back_to_the_competition_default(events):
    frame = odds_api.parse_events(events, FA_CUP)
    assert set(frame["format"]) == {"single_leg_knockout"}


def test_kickoffs_are_confirmed_and_utc(events):
    frame = odds_api.parse_events(events, FA_CUP)
    assert frame["date_confirmed"].all()
    assert frame["date"].iloc[0] == pd.Timestamp("2026-09-05")
    assert frame["kickoff"].iloc[0] == "14:00"


def test_the_season_is_named_for_the_year_it_starts(events):
    frame = odds_api.parse_events(events, FA_CUP)
    assert set(frame["season"]) == {"2026-2027"}

    january = [_event("e9", "A", "B", "2027-01-10T15:00:00Z")]
    assert odds_api.parse_events(january, FA_CUP)["season"].iloc[0] == "2026-2027"


def test_a_knockout_fixture_id_carries_the_stage(events):
    """Even when unknown — the same pair can meet twice in one cup (FR-7)."""
    frame = odds_api.parse_events(events, FA_CUP)
    assert frame["fixture_id"].iloc[0] == "ENG.FA_CUP|2026-2027|unknown|Arsenal|Chelsea"


def test_a_league_fixture_id_omits_the_stage():
    rows = [_event("e1", "Arsenal", "Chelsea", "2026-09-05T14:00:00Z")]
    frame = odds_api.parse_events(rows, "ENG.PL")
    assert frame["fixture_id"].iloc[0] == "ENG.PL|2026-2027|Arsenal|Chelsea"


def test_an_event_missing_a_club_or_a_kickoff_is_dropped():
    rows = [
        _event("e1", "Arsenal", "", "2026-09-05T14:00:00Z"),
        _event("e2", "Arsenal", "Chelsea", None),
        _event("e3", "Arsenal", "Chelsea", "not-a-date"),
    ]
    assert odds_api.parse_events(rows, FA_CUP).empty


def test_the_source_is_named_on_every_row(events):
    frame = odds_api.parse_events(events, FA_CUP)
    assert set(frame["source"]) == {"odds_api"}


def test_cups_still_report_no_odds_coverage(events):
    """A fixture list is not a validated price. Requirements §9 is unaffected."""
    frame = odds_api.parse_events(events, FA_CUP)
    assert not frame["odds_coverage"].any()


# --- coverage claims ----------------------------------------------------------

def test_every_taxonomy_competition_has_a_sport_key():
    from statpitch import taxonomy

    for competition in taxonomy.registry():
        assert competition.competition_id in odds_api.SPORT_KEYS


def test_the_unsourced_six_are_exactly_the_cups_nothing_free_reaches():
    from statpitch.data import openfootball as of
    from statpitch.data import openligadb as old

    for competition_id in odds_api.UNSOURCED_WITHOUT_A_KEY:
        assert competition_id not in old.COMPETITIONS
        assert competition_id in of.SCHEDULE_SOURCES  # mapped, but 404 upstream


def test_describe_reports_capability_without_needing_a_key(keyless):
    report = odds_api.describe()
    assert report["configured"] is False
    assert report["events_cost_credits"] is False
    assert report["monthly_credits"] == 500
