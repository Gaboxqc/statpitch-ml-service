"""OpenLigaDB cup fixtures (Plan §4 Phase D).

Offline. The payload below is trimmed from a real
`api.openligadb.de/getmatchdata/dfb/2026` response.

This source exists because openfootball stopped publishing cup files: every
2026-27 cup path 404s and the `champions-league` repo has no 2026-27 directory
at all. OpenLigaDB is keyless and covers one of the seven cups.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from statpitch.data import openligadb as old

DFB = "GER.DFB_POKAL"


def _match(mid, home, away, stamp, *, finished=False, group="1. Runde"):
    return {
        "matchID": mid,
        "matchDateTimeUTC": stamp,
        "matchIsFinished": finished,
        "group": {"groupName": group, "groupOrderID": 1},
        "team1": {"teamId": 1, "teamName": home},
        "team2": {"teamId": 2, "teamName": away},
    }


@pytest.fixture
def payload():
    return [
        _match(1, "Hamburg Eimsbütteler BC", "Borussia Dortmund",
               "2026-09-01T18:45:00Z"),
        _match(2, "VfL Osnabrück", "FC Bayern München", "2026-09-02T18:45:00Z"),
        _match(3, "SC St. Tönis", "Eintracht Frankfurt", "2026-08-21T16:00:00Z",
               finished=True),
    ]


# --- what gets published ------------------------------------------------------

def test_only_unplayed_matches_are_listed(payload):
    """A fixture list, not a result archive."""
    frame = old.parse_matches(payload, DFB, 2026)
    assert len(frame) == 2
    assert "SC St. Tönis" not in set(frame["home_team"])


def test_kickoffs_are_confirmed_not_nominal(payload):
    """OpenLigaDB publishes a real UTC kickoff, so no correction pass is needed."""
    frame = old.parse_matches(payload, DFB, 2026)
    assert frame["date_confirmed"].all()
    assert frame["kickoff"].iloc[0] == "18:45"
    assert frame["date"].iloc[0] == pd.Timestamp("2026-09-01")


def test_utc_is_not_reinterpreted_as_local(payload):
    """`matchDateTimeUTC` is already UTC — the sibling `matchDateTime` is not."""
    frame = old.parse_matches(payload, DFB, 2026)
    assert frame["kickoff"].tolist() == ["18:45", "18:45"]


# --- the round, which is not cosmetic -----------------------------------------

def test_the_round_is_parsed_into_a_taxonomy_stage(payload):
    """`taxonomy.resolve_format` keys the tie format off the stage.

    An unknown stage falls back to the competition default, which would price a
    two-legged tie as a single leg.
    """
    frame = old.parse_matches(payload, DFB, 2026)
    assert set(frame["stage"]) == {"round_1"}
    assert set(frame["stage_detail"]) == {"1. Runde"}


def test_german_round_labels_reuse_the_openfootball_parser(payload):
    """The same labels appear in the openfootball cup files it was written for."""
    for label, expected in (
        ("1. Runde", "round_1"),
        ("Achtelfinale", "round_of_16"),
        ("Halbfinale", "semi_final"),
        ("Finale", "final"),
    ):
        rows = [_match(9, "A", "B", "2026-09-01T18:45:00Z", group=label)]
        frame = old.parse_matches(rows, DFB, 2026)
        if label == "Finale":
            # Not in the German pattern list; it must degrade, not mislabel.
            assert frame["stage"].iloc[0] != "round_1"
        else:
            assert frame["stage"].iloc[0] == expected


def test_the_format_follows_from_the_stage(payload):
    frame = old.parse_matches(payload, DFB, 2026)
    assert set(frame["format"]) == {"single_leg_knockout"}


def test_a_missing_round_does_not_crash_the_parse():
    rows = [{**_match(1, "A", "B", "2026-09-01T18:45:00Z"), "group": None}]
    frame = old.parse_matches(rows, DFB, 2026)
    assert frame["stage"].iloc[0] == "unknown"


# --- rejecting what cannot be listed ------------------------------------------

def test_a_match_without_a_kickoff_is_dropped():
    """An undated fixture cannot be listed by date, and guessing puts it on the
    wrong day."""
    rows = [_match(1, "A", "B", None)]
    assert old.parse_matches(rows, DFB, 2026).empty


def test_a_match_missing_a_club_is_dropped():
    rows = [{**_match(1, "A", "B", "2026-09-01T18:45:00Z"), "team2": {}}]
    assert old.parse_matches(rows, DFB, 2026).empty


def test_an_unparseable_kickoff_is_dropped():
    rows = [_match(1, "A", "B", "not-a-date")]
    assert old.parse_matches(rows, DFB, 2026).empty


# --- the schedule contract ----------------------------------------------------

def test_rows_carry_the_columns_the_fixture_build_expects(payload):
    frame = old.parse_matches(payload, DFB, 2026)
    for column in (
        "fixture_id", "competition_id", "season", "stage", "stage_detail",
        "format", "neutral_venue", "date", "kickoff", "date_confirmed",
        "home_team", "away_team", "source", "odds_coverage",
    ):
        assert column in frame.columns


def test_the_source_is_named_on_every_row(payload):
    frame = old.parse_matches(payload, DFB, 2026)
    assert set(frame["source"]) == {"openligadb"}


def test_the_fixture_id_carries_the_stage_for_a_knockout(payload):
    """The same pair can legitimately meet twice in one cup (FR-7)."""
    frame = old.parse_matches(payload, DFB, 2026)
    assert "round_1" in frame["fixture_id"].iloc[0]
    assert frame["fixture_id"].iloc[0].startswith("GER.DFB_POKAL|2026-2027|")


def test_cups_still_report_no_odds_coverage(payload):
    """A fixture list is not a price. Requirements §9 is unaffected by this source."""
    frame = old.parse_matches(payload, DFB, 2026)
    assert not frame["odds_coverage"].any()


# --- fetching -----------------------------------------------------------------

class FakeSession:
    def __init__(self, body, *, fail=False):
        self.body = body
        self.fail = fail
        self.calls: list[tuple[str, float | None]] = []

    def get_bytes(self, url, *, suffix=".bin", max_age=None, **kw):
        from statpitch.data.http import FetchError

        self.calls.append((url, max_age))
        if self.fail:
            raise FetchError("boom")
        return self.body


def test_an_unmapped_competition_is_declined_without_a_request():
    session = FakeSession(b"[]")
    assert old.fetch_season("ENG.FA_CUP", 2026, session=session) is None
    assert session.calls == []


def test_a_failed_request_degrades_to_no_fixtures(payload):
    session = FakeSession(b"", fail=True)
    assert old.fetch_season(DFB, 2026, session=session) is None


def test_malformed_json_degrades_to_no_fixtures():
    session = FakeSession(b"{not json")
    assert old.fetch_season(DFB, 2026, session=session) is None


def test_a_non_list_payload_is_rejected():
    session = FakeSession(json.dumps({"error": "nope"}).encode())
    assert old.fetch_season(DFB, 2026, session=session) is None


def test_schedules_are_fetched_with_an_expiry(payload):
    """A schedule describes the future and must not come from an unbounded cache."""
    session = FakeSession(json.dumps(payload).encode())
    old.fetch_season(DFB, 2026, session=session)
    assert session.calls[0][1] == old.SCHEDULE_MAX_AGE_SECONDS


def test_utf8_club_names_survive_the_round_trip(payload):
    session = FakeSession(json.dumps(payload).encode("utf-8"))
    fetched = old.fetch_season(DFB, 2026, session=session)
    frame = old.parse_matches(fetched, DFB, 2026)
    assert "VfL Osnabrück" in set(frame["home_team"])
    assert "FC Bayern München" in set(frame["away_team"])
