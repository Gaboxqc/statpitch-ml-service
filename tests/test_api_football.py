"""API-Football collector (Roadmap §4.5, §7.2).

`quota.py` has guarded a 100/day budget since before any caller existed. This is
the caller, and the tests are about the two ways it could go wrong quietly:
spending budget it does not have, and pairing an upstream fixture with the wrong
local one.

No test here touches the network. The client is exercised with the key absent —
which is the state every checkout is in — and the matching logic is pure.
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from statpitch.data import api_football as af


def _load():
    spec = importlib.util.spec_from_file_location(
        "collect_fixtures", Path("scripts/collect_fixtures.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cf = _load()


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv(af.ENV_KEY, raising=False)


# --- the key ------------------------------------------------------------------

def test_absent_key_is_a_capability_not_an_error():
    """A checkout with no key must behave like one with an exhausted budget."""
    assert af.configured() is False
    assert af.ApiFootball().lineups(123) is None


def test_no_call_is_made_without_a_key(monkeypatch):
    """Not merely a None return — the network and the budget are untouched."""
    def explode(*args, **kwargs):
        raise AssertionError("reached the network with no key configured")

    client = af.ApiFootball()
    monkeypatch.setattr(client, "_get", explode)
    assert client.fixtures_for_date("ENG.PL", date(2026, 8, 21), 2026) is None


def test_a_present_key_is_detected(monkeypatch):
    monkeypatch.setenv(af.ENV_KEY, "  abc123  ")
    assert af.configured() is True
    assert af.api_key() == "abc123"


def test_a_blank_key_counts_as_absent(monkeypatch):
    monkeypatch.setenv(af.ENV_KEY, "   ")
    assert af.configured() is False


def test_an_unmapped_competition_costs_nothing():
    """Cups have no league id; asking must not spend a call to find out."""
    assert af.ApiFootball().injuries("ENG.FA_CUP", 2026, date(2026, 8, 21)) is None


# --- payload shaping ----------------------------------------------------------

PAYLOAD = [
    {
        "fixture": {"id": 1234, "date": "2026-08-21T19:00:00+00:00",
                    "status": {"short": "NS"}, "venue": {"name": "Emirates"}},
        "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Coventry"}},
    },
    {"fixture": {"id": 9}, "teams": {}},
]


def test_fixtures_are_flattened():
    rows = af.parse_fixtures(PAYLOAD)
    assert len(rows) == 1
    assert rows[0]["api_fixture_id"] == 1234
    assert rows[0]["home_team"] == "Arsenal"


def test_a_malformed_entry_drops_one_fixture_not_the_round():
    """API-Football nests four levels deep; a missing branch must not fail a round."""
    assert len(af.parse_fixtures(PAYLOAD)) == 1


def test_lineups_are_flattened():
    rows = af.parse_lineups([
        {"team": {"name": "Arsenal"}, "formation": "4-3-3",
         "startXI": [{"player": {"name": "Raya"}}, {"player": {}}],
         "coach": {"name": "Arteta"}}
    ])
    assert rows[0]["formation"] == "4-3-3"
    assert rows[0]["start_xi"] == ["Raya"]


# --- matching -----------------------------------------------------------------

def _row(home="Arsenal FC", away="Coventry City FC", day="2026-08-21"):
    return pd.Series({"home_team": home, "away_team": away,
                      "date": pd.Timestamp(day)})


def _candidate(home="Arsenal", away="Coventry", stamp="2026-08-21T19:00:00+00:00"):
    return {"api_fixture_id": 1, "home_team": home, "away_team": away,
            "kickoff_utc": stamp}


def test_a_fixture_matches_across_naming_conventions():
    """openfootball writes "Arsenal FC", API-Football writes "Arsenal"."""
    assert cf.match_fixture(_row(), [_candidate()]) is not None


def test_a_fixture_matches_when_the_day_moved_inside_the_window():
    """The whole point: a provisional Sunday that is really a Friday."""
    hit = cf.match_fixture(_row(day="2026-08-23"), [_candidate()])
    assert hit is not None
    assert hit["confirmed_at"].normalize() == pd.Timestamp("2026-08-21")


def test_a_move_beyond_the_window_is_not_the_same_fixture():
    """A postponement of two weeks is not a reschedule of this round."""
    assert cf.match_fixture(_row(day="2026-09-30"), [_candidate()]) is None


def test_an_ambiguous_match_is_refused():
    """Two candidates fit, so the fixture keeps its provisional date.

    The same rule the Elo and Transfermarkt resolvers follow: resolving to the
    first candidate is how a club's data ends up on a different club.
    """
    twins = [_candidate(), _candidate(stamp="2026-08-22T15:00:00+00:00")]
    assert cf.match_fixture(_row(), twins) is None


def test_a_different_opponent_does_not_match():
    assert cf.match_fixture(_row(), [_candidate(away="Chelsea")]) is None


def test_legal_forms_do_not_decide_a_match():
    """"Real" and "Borussia" are noise; several clubs in a league carry each."""
    assert cf.normalise("Real Sociedad") == cf.normalise("Sociedad")


# --- the free-plan season window ----------------------------------------------

def test_the_free_plan_window_is_what_the_api_reported():
    """Verified 2026-08-17: "try from 2022 to 2024"."""
    assert af.FREE_PLAN_SEASONS == (2022, 2024)
    assert af.season_available(2023)
    assert not af.season_available(2026)


def test_an_out_of_plan_season_costs_no_budget(monkeypatch):
    """The point of checking before spending.

    API-Football answers an out-of-plan request with HTTP 200 and an `errors`
    object, and quota reserves budget BEFORE the call — so a request that cannot
    succeed still costs one of the ninety. The first real run burned five that
    way, weekly, forever.
    """
    monkeypatch.setenv(af.ENV_KEY, "abc123")
    client = af.ApiFootball()

    def explode(*args, **kwargs):
        raise AssertionError("spent budget on a season the plan cannot see")

    monkeypatch.setattr(client.budget, "spend", explode)
    assert client.fixtures_in_range("ENG.PL", date(2026, 8, 1), date(2026, 8, 21), 2026) is None


def test_an_in_plan_season_is_attempted(monkeypatch):
    monkeypatch.setenv(af.ENV_KEY, "abc123")
    client = af.ApiFootball()
    monkeypatch.setattr(client.budget, "spend", lambda key, fetch: ["reached"])
    assert client.fixtures_in_range(
        "ENG.PL", date(2023, 8, 1), date(2023, 8, 21), 2023
    ) == ["reached"]


def test_a_plan_restriction_is_its_own_error(monkeypatch):
    """Not transient: retrying spends budget to be told the same thing."""
    monkeypatch.setenv(af.ENV_KEY, "abc123")
    client = af.ApiFootball()

    class FakeResponse:
        def raise_for_status(self): pass
        def json(self):
            return {"errors": {"plan": "Free plans do not have access to this season"}}

    monkeypatch.setattr(client.session, "get", lambda *a, **k: FakeResponse())
    with pytest.raises(af.PlanRestricted, match="Free plans"):
        client._get("/fixtures", {"league": 39})
