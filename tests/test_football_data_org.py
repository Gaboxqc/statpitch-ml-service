"""football-data.org fixture source (Roadmap §7.2).

The third source tried against the same problem. The first two failed for
reasons worth not repeating: API-Football's free plan covers seasons 2022-2024,
the exact complement of the current one, and it charged budget to say so.

No test here touches the network. What needs covering is the behaviour around
the call: that an absent key costs nothing, that a source which cannot answer
degrades to "keep the provisional date" rather than raising, and that a
malformed payload drops one fixture rather than a competition.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from statpitch import taxonomy
from statpitch.data import football_data_org as fdo
from statpitch.data.http import FetchError


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv(fdo.ENV_KEY, raising=False)


PAYLOAD = {
    "matches": [
        {
            "id": 555,
            "utcDate": "2026-08-21T19:00:00Z",
            "status": "TIMED",
            "matchday": 1,
            "homeTeam": {"name": "Arsenal FC"},
            "awayTeam": {"name": "Coventry City FC"},
        },
        {"id": 556, "utcDate": "2026-08-22T14:00:00Z", "homeTeam": {"name": "X"}},
    ]
}


class _Session:
    def __init__(self, body): self.body = body
    def get_bytes(self, url, **kw): return self.body


# --- the key ------------------------------------------------------------------

def test_absent_key_is_a_normal_state():
    assert fdo.configured() is False
    assert fdo.fetch_matches("ENG.PL", date(2026, 8, 1), date(2026, 8, 21)) is None


def test_building_a_session_without_a_key_is_refused():
    with pytest.raises(fdo.FootballDataError, match=fdo.ENV_KEY):
        fdo.session()


def test_a_present_key_is_detected(monkeypatch):
    monkeypatch.setenv(fdo.ENV_KEY, "  tok  ")
    assert fdo.configured() is True
    assert fdo.api_key() == "tok"


#: Odds-covered leagues this source does not carry, with the reason.
#:
#: Unlike Transfermarkt, football-data.org's free tier is a partial list, so the
#: mapping cannot simply equal the taxonomy. Naming the gap here is what stops it
#: being read as a forgotten entry — and forces a deliberate edit if the plan
#: ever adds Turkey, rather than a silently widening hole.
NOT_OFFERED_BY_THIS_SOURCE = {
    # `/competitions` returns no Turkish competition at all — not a locked
    # TIER_TWO row, an absent one. Verified against the live listing 2026-09-04.
    "TUR.SUPERLIG",
}


def test_every_odds_covered_league_is_mapped_unless_the_source_lacks_it():
    covered = {c.competition_id for c in taxonomy.registry().with_odds_coverage()}
    assert set(fdo.COMPETITIONS) == covered - NOT_OFFERED_BY_THIS_SOURCE


def test_an_unmapped_league_degrades_to_the_provisional_date(monkeypatch):
    """Turkey's absence must cost nothing, not raise.

    `collect_fixtures` reads a None as "keep openfootball's provisional date and
    leave date_confirmed false", which is the honest state for a competition
    this source cannot see.
    """
    monkeypatch.setenv(fdo.ENV_KEY, "tok")
    assert fdo.fetch_matches(
        "TUR.SUPERLIG", date(2026, 8, 1), date(2026, 8, 21)
    ) is None


def test_an_unmapped_competition_returns_nothing(monkeypatch):
    monkeypatch.setenv(fdo.ENV_KEY, "tok")
    assert fdo.fetch_matches("ENG.FA_CUP", date(2026, 8, 1), date(2026, 8, 21)) is None


# --- parsing ------------------------------------------------------------------

def test_matches_are_flattened_with_their_kickoff():
    fixtures = fdo.parse_matches(PAYLOAD)
    assert len(fixtures) == 1
    assert fixtures[0].kickoff_utc == "2026-08-21T19:00:00Z"
    assert fixtures[0].home_team == "Arsenal FC"
    assert fixtures[0].source_id == 555


def test_an_incomplete_match_drops_one_fixture_not_the_competition():
    """A fixture left out keeps its provisional date, which everything handles."""
    assert len(fdo.parse_matches(PAYLOAD)) == 1


def test_an_empty_payload_parses_to_nothing():
    assert fdo.parse_matches({}) == []
    assert fdo.parse_matches({"matches": None}) == []


# --- degradation --------------------------------------------------------------

def test_a_successful_call_returns_fixtures(monkeypatch):
    monkeypatch.setenv(fdo.ENV_KEY, "tok")
    http = _Session(json.dumps(PAYLOAD).encode())
    assert len(fdo.fetch_matches("ENG.PL", date(2026, 8, 1), date(2026, 8, 21), http=http)) == 1


def test_a_rejected_key_degrades_instead_of_raising(monkeypatch):
    """403 and 429 both mean "keep the provisional date", not "stop the run"."""
    monkeypatch.setenv(fdo.ENV_KEY, "tok")

    class Failing:
        def get_bytes(self, url, **kw): raise FetchError("HTTP 403")

    assert fdo.fetch_matches(
        "ENG.PL", date(2026, 8, 1), date(2026, 8, 21), http=Failing()
    ) is None


def test_an_error_payload_is_not_read_as_fixtures(monkeypatch):
    """The API answers some failures with 200 and an errorCode."""
    monkeypatch.setenv(fdo.ENV_KEY, "tok")
    body = json.dumps({"errorCode": 403, "message": "restricted resource"}).encode()
    assert fdo.fetch_matches(
        "ENG.PL", date(2026, 8, 1), date(2026, 8, 21), http=_Session(body)
    ) is None


def test_malformed_json_degrades(monkeypatch):
    monkeypatch.setenv(fdo.ENV_KEY, "tok")
    assert fdo.fetch_matches(
        "ENG.PL", date(2026, 8, 1), date(2026, 8, 21), http=_Session(b"<html>")
    ) is None
