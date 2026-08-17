"""Fixture listing (Roadmap §7).

The fixture source is the daily entry point for the consuming API: without it
`/today` has nothing to return and there is no integration to build. Three
things are worth testing beyond "it parses".

**The results path must not move.** `parse_football_txt` gained the ability to
emit unplayed fixtures, and every existing caller builds training data where a
scoreless row is actively harmful — it would join the match log as a real fixture
with null goals and enter feature windows. So the new behaviour is opt-in, and
that default is asserted rather than assumed.

**A scheduled fixture is not a match with missing data.** Both have null scores.
Only the `played` flag separates them, so anything keyed on a null check would
conflate a postponed cup tie with next Saturday's league game.

**An empty list is not a missing source.** `/today` returning `[]` means there is
no football today; the artifact being absent is a refusal. A consumer that reads
those as the same thing records a broken deploy as a quiet Tuesday.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from statpitch.data import openfootball as of
from statpitch.serving import contract
from statpitch.serving.app import app

# A league schedule, in openfootball's published form: `v` separator, no score,
# times only on some lines, and a date line without a year after the first.
SCHEDULE = """\
= English Premier League 2026/27

▪ Matchday 1
  Fri Aug 21 2026
    20:00  Arsenal FC              v Coventry City FC
  Sat Aug 22
    12:30  Hull City AFC           v Manchester United FC
           Everton FC              v Crystal Palace FC
"""

# The same file once the round has been played.
PLAYED = """\
= English Premier League 2026/27

▪ Matchday 1
  Fri Aug 21 2026
    20:00  Arsenal FC  2-1 (1-0)  Coventry City FC
"""


# --- parser -------------------------------------------------------------------

def test_unplayed_fixtures_are_skipped_by_default():
    """The training path must be byte-identical to before this feature existed."""
    assert of.parse_football_txt(SCHEDULE, 2026) == []


def test_unplayed_fixtures_are_emitted_when_asked():
    matches = of.parse_football_txt(SCHEDULE, 2026, include_unplayed=True)
    assert len(matches) == 3
    assert [m.home_team for m in matches] == ["Arsenal FC", "Hull City AFC", "Everton FC"]
    assert all(not m.played for m in matches)
    assert all(m.home_goals is None and m.away_goals is None for m in matches)


def test_scheduled_fixtures_carry_dates_including_the_undated_line():
    """Only the first date line has a year; the rest inherit the season."""
    matches = of.parse_football_txt(SCHEDULE, 2026, include_unplayed=True)
    assert matches[0].date == pd.Timestamp(2026, 8, 21)
    assert matches[1].date == pd.Timestamp(2026, 8, 22)
    # A line with no kickoff time belongs to the date above it, not to no date.
    assert matches[2].date == pd.Timestamp(2026, 8, 22)


def test_played_matches_still_parse_with_unplayed_enabled():
    """Turning the flag on must not change how a played match is read."""
    with_flag = of.parse_football_txt(PLAYED, 2026, include_unplayed=True)
    without = of.parse_football_txt(PLAYED, 2026)
    assert with_flag == without
    assert with_flag[0].played is True
    assert (with_flag[0].home_goals, with_flag[0].away_goals) == (2, 1)


def test_a_line_that_is_neither_is_still_skipped():
    """Absence of a score is not licence to treat any line as a fixture."""
    noise = "▪ Matchday 1\n  Fri Aug 21 2026\n    (attendance 60,000)\n"
    assert of.parse_football_txt(noise, 2026, include_unplayed=True) == []


# --- fixture_id ---------------------------------------------------------------

def test_fixture_id_excludes_the_date_so_a_postponement_keeps_its_identity():
    """A date-keyed id would record a rearranged match as new and the old as gone."""
    a = of.fixture_id("ENG.PL", "2026-2027", "Arsenal FC", "Chelsea FC",
                      "matchday_1", knockout=False)
    b = of.fixture_id("ENG.PL", "2026-2027", "Arsenal FC", "Chelsea FC",
                      "matchday_12", knockout=False)
    assert a == b


def test_knockout_fixture_id_keeps_the_stage_because_a_pair_can_meet_twice():
    """FR-7: both legs of a tie are the same pair in the same season."""
    first = of.fixture_id("UEFA.UCL", "2026-2027", "Arsenal FC", "Real Madrid CF",
                          "quarter_final", knockout=True)
    other = of.fixture_id("UEFA.UCL", "2026-2027", "Arsenal FC", "Real Madrid CF",
                          "semi_final", knockout=True)
    assert first != other


# --- the built artifact -------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def fixtures(client):
    from statpitch.serving.app import predictor

    frame = predictor().artifacts.fixtures
    if frame is None or frame.empty:
        pytest.skip("no fixtures artifact built in this checkout")
    return frame


def test_artifact_contains_only_unplayed_fixtures(fixtures):
    assert "home_goals" not in fixtures.columns


def test_artifact_fixture_ids_are_unique(fixtures):
    """A duplicate id would collapse two fixtures into one row downstream."""
    assert fixtures["fixture_id"].is_unique


def test_every_fixture_club_resolves_to_a_measured_rating(client, fixtures):
    """The whole point of the alias map.

    Before it, 55% of an upcoming five-league list rated at the pooled prior —
    Manchester City, Bayern and Paris Saint-Germain among them — and every
    response said so via `fully_rated` while still returning a confident number.
    """
    body = client.get("/fixtures/upcoming?limit=1000&include_predictions=true").json()
    unrated = [
        f"{f['home_team']} v {f['away_team']}"
        for f in body["fixtures"]
        if not (f.get("prediction") or {}).get("fully_rated")
    ]
    assert not unrated, f"{len(unrated)} fixture(s) not fully rated: {unrated[:5]}"


# --- API ----------------------------------------------------------------------

def test_upcoming_returns_fixtures_without_predictions_by_default(client, fixtures):
    body = client.get("/fixtures/upcoming?limit=3").json()
    assert body["count"] == 3
    assert body["total"] >= 3
    assert "prediction" not in body["fixtures"][0]
    assert body["generated_at_source"]


def test_upcoming_attaches_predictions_on_request(client, fixtures):
    body = client.get("/fixtures/upcoming?limit=1&include_predictions=true").json()
    prediction = body["fixtures"][0]["prediction"]
    assert set(prediction["probabilities"]) == {"home", "draw", "away"}


def test_upcoming_filters_by_competition_and_date(client, fixtures):
    competition = fixtures["competition_id"].iloc[0]
    body = client.get(f"/fixtures/upcoming?competition_id={competition}&limit=500").json()
    assert {f["competition_id"] for f in body["fixtures"]} == {competition}

    day = str(fixtures["date"].min().date())
    same_day = client.get(f"/fixtures/upcoming?from={day}&to={day}&limit=500").json()
    assert {f["date"] for f in same_day["fixtures"]} == {day}


def test_upcoming_rejects_an_unknown_competition(client, fixtures):
    assert client.get("/fixtures/upcoming?competition_id=NOPE").status_code == 404


def test_upcoming_paginates(client, fixtures):
    first = client.get("/fixtures/upcoming?limit=2").json()
    second = client.get("/fixtures/upcoming?limit=2&offset=2").json()
    assert first["total"] == second["total"]
    ids = {f["fixture_id"] for f in first["fixtures"]}
    assert ids.isdisjoint({f["fixture_id"] for f in second["fixtures"]})


def test_today_keeps_its_v1_shape(client, fixtures):
    """NFR-13: the keys do not move just because the list can now be non-empty."""
    body = client.get("/today").json()
    assert {"date", "fixtures", "note"} <= set(body)


def test_today_without_the_artifact_refuses_rather_than_returning_empty(client):
    """The distinction a consumer's sync job depends on."""
    from statpitch.serving.app import predictor

    artifacts = predictor().artifacts
    saved = artifacts.fixtures
    artifacts.fixtures = None
    try:
        body = client.get("/today").json()
        assert body["fixtures"] == []
        assert body["refusal"]["reason_code"] == str(
            contract.ReasonCode.NO_FIXTURE_SOURCE
        )

        upcoming = client.get("/fixtures/upcoming").json()
        assert upcoming["refusal"]["reason_code"] == str(
            contract.ReasonCode.NO_FIXTURE_SOURCE
        )
    finally:
        artifacts.fixtures = saved


def test_today_with_the_artifact_present_does_not_refuse(client, fixtures):
    body = client.get("/today").json()
    assert "refusal" not in body


# --- provisional dates (openfootball publishes matchdays before slots) --------

SCHEDULE_MIXED = """\
▪ Matchday 1
  Fri Aug 21 2026
    20:00  Arsenal FC              v Coventry City FC
  Sat Aug 22
           Everton FC              v Crystal Palace FC
"""


def test_a_published_kickoff_time_is_captured():
    matches = of.parse_football_txt(SCHEDULE_MIXED, 2026, include_unplayed=True)
    assert matches[0].kickoff == "20:00"


def test_a_fixture_without_a_time_reports_no_kickoff():
    """Which is how a provisional matchday date announces itself.

    openfootball publishes a matchday before the league confirms slots: every
    fixture lands on one nominal date with a time only on the first line. La Liga
    matchday 1 2026/27 stacked ten fixtures on Sunday 16 August that were played
    across the 14th to the 17th. Only 12% of the current fixture list carries a
    confirmed time, so a consumer that treats every date as final will show
    matches on the wrong day.
    """
    matches = of.parse_football_txt(SCHEDULE_MIXED, 2026, include_unplayed=True)
    assert matches[1].kickoff is None


def test_the_schedule_marks_which_dates_are_confirmed(fixtures):
    assert "date_confirmed" in fixtures.columns
    assert fixtures["date_confirmed"].dtype == bool


def test_upcoming_excludes_past_fixtures_by_default(client, fixtures):
    """The artifact is filtered to the future when BUILT, so without a request-time
    floor the window of already-played fixtures grows every day it ages."""
    from datetime import UTC, datetime

    body = client.get("/fixtures/upcoming?limit=500").json()
    today = str(datetime.now(UTC).date())
    assert body["from"] == today
    assert all(f["date"] >= today for f in body["fixtures"])


def test_an_explicit_from_still_reaches_back(client, fixtures):
    body = client.get("/fixtures/upcoming?from=2000-01-01&limit=5").json()
    assert body["from"] == "2000-01-01"


def test_each_fixture_reports_whether_its_date_is_confirmed(client, fixtures):
    body = client.get("/fixtures/upcoming?limit=20").json()
    for fixture in body["fixtures"]:
        assert isinstance(fixture["date_confirmed"], bool)
        # A confirmed date carries the time it was confirmed at.
        if fixture["date_confirmed"]:
            assert fixture["kickoff"]


def test_the_artifact_keeps_recently_past_fixtures(fixtures):
    """A provisional date that has just passed does not mean the match was played.

    La Liga matchday 1 2026/27 sat on a nominal Sunday and was played across four
    days. Building with `date >= today` dropped the whole matchday the morning
    after that Sunday — including fixtures still to be played — and `/today`
    returned nothing on a day with real matches. The artifact now keeps a few
    days of lookback so the date correction can still move them forward, and
    `/fixtures/upcoming` hides genuinely past ones at request time instead.
    """
    from datetime import UTC, datetime

    today = datetime.now(UTC).date()
    earliest = fixtures["date"].min().date()
    assert (today - earliest).days <= 10, (
        "the artifact starts too far in the past to be a lookback window"
    )
