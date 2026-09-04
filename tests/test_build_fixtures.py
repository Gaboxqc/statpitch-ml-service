"""Fixture-artifact assembly (`scripts/build_fixtures.py`).

The script concatenates three schedule sources, and the failure mode worth a
test is not that a source breaks — an empty frame is visible immediately — but
that two sources both succeed for one competition. The de-dup that was there
keyed on `fixture_id`, which is built from club names, and the sources spell
clubs differently: The Odds API writes "Galatasaray" where openfootball writes
"Galatasaray SK". Two disjoint id sets both survive, and the artifact carries
every Turkish fixture twice with no error anywhere.

That went from hypothetical to reachable when TUR.SUPERLIG was added. It has no
openfootball file for 2026-27, so it is served from the Odds API today and would
double the day openfootball publishes one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_fixtures", Path("scripts/build_fixtures.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bf = _load()


def _rows(*specs) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fixture_id": f"{cid}|{home}|{away}",
                "competition_id": cid,
                "source": source,
                "home_team": home,
                "away_team": away,
            }
            for cid, source, home, away in specs
        ]
    )


def test_a_single_source_competition_is_left_alone():
    frame = _rows(
        ("ENG.PL", "openfootball", "Arsenal FC", "Chelsea FC"),
        ("ENG.PL", "openfootball", "Everton FC", "Fulham FC"),
    )
    assert len(bf._one_source_per_competition(frame)) == 2


def test_two_sources_for_one_competition_collapse_to_the_better_one():
    """The names differ, so `fixture_id` cannot collapse these — only source can."""
    frame = _rows(
        ("TUR.SUPERLIG", "odds_api", "Galatasaray", "Besiktas"),
        ("TUR.SUPERLIG", "openfootball", "Galatasaray SK", "Besiktas JK"),
    )
    out = bf._one_source_per_competition(frame)
    assert list(out["source"]) == ["openfootball"]
    assert list(out["home_team"]) == ["Galatasaray SK"]


def test_openfootball_wins_over_odds_api_for_the_round_label():
    """Priority is by what the row carries, not by which source has more rows.

    The Odds API cannot supply a stage, so its rows are emitted with
    `stage_confirmed=false` and fall back to the competition default format.
    Preferring it because it happened to return more fixtures would trade a
    resolvable format for an unresolvable one.
    """
    frame = _rows(
        ("UEFA.UCL", "odds_api", "A", "B"),
        ("UEFA.UCL", "odds_api", "C", "D"),
        ("UEFA.UCL", "odds_api", "E", "F"),
        ("UEFA.UCL", "openfootball", "A FC", "B FC"),
    )
    assert set(bf._one_source_per_competition(frame)["source"]) == {"openfootball"}


def test_competitions_are_collapsed_independently_of_each_other():
    frame = _rows(
        ("ENG.PL", "openfootball", "Arsenal FC", "Chelsea FC"),
        ("TUR.SUPERLIG", "odds_api", "Galatasaray", "Besiktas"),
        ("UEFA.UCL", "odds_api", "AEK Athens", "LASK"),
    )
    out = bf._one_source_per_competition(frame)
    assert len(out) == 3
    assert set(out["competition_id"]) == {"ENG.PL", "TUR.SUPERLIG", "UEFA.UCL"}


def test_an_unknown_source_ranks_last_rather_than_crashing():
    """A source added later must not win by default just for being unlisted."""
    frame = _rows(
        ("ENG.PL", "some_new_source", "Arsenal", "Chelsea"),
        ("ENG.PL", "openfootball", "Arsenal FC", "Chelsea FC"),
    )
    assert list(bf._one_source_per_competition(frame)["source"]) == ["openfootball"]


def test_every_priority_entry_is_a_source_something_actually_emits():
    """A typo here would silently demote a real source to unknown-rank."""
    assert set(bf.SOURCE_PRIORITY) == {"openfootball", "openligadb", "odds_api"}
