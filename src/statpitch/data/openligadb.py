"""OpenLigaDB — keyless DFB-Pokal fixtures (Plan §4 Phase D).

openfootball stopped publishing cup files. Verified across two days: every
2026-27 cup path 404s, `england/2025-26/facup.txt` has been gone since before
that, and the `champions-league` repo has no 2026-27 directory at all. The cup
modelling stack — the FR-9 entrant prior, FR-8 extra time and shootouts, the
FR-20 bracket simulator — has been complete and idle for want of a fixture.

This restores one competition of the seven, and it is the only one available at
$0. `football_data_org.py` already records the survey that found it:

    OpenLigaDB is the fallback worth keeping in mind: it needs no credential at
    all and returned 306 Bundesliga fixtures with real UTC kickoff times.

It was passed over as the *league* source because it covers one country of five.
For the DFB-Pokal that objection does not apply — one country is the whole
competition.

What it gives that a generic odds feed would not
================================================

**The round.** `group.groupName` is "1. Runde", "Achtelfinale", "Halbfinale" —
which `openfootball.normalise_stage` already parses, because the same German
labels appear in the openfootball cup files it was written for. Stage is not
cosmetic here: `taxonomy.resolve_format` keys the tie format off it, and
`is_neutral_venue` keys the final's venue off it. A fixture with an unknown
stage falls back to the competition default and would price a two-legged tie as
a single leg.

**Real UTC kickoffs.** `matchDateTimeUTC` is confirmed, not a nominal matchday,
so these arrive with `date_confirmed=true` and need no correction pass.

What it does not give
=====================

Everything except Germany. The FA Cup, Copa del Rey, Coppa Italia, Coupe de
France and both UEFA competitions have no keyless source, and the API says so per
competition rather than serving an empty list that looks like "no matches today".
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from statpitch import taxonomy
from statpitch.data import openfootball as of
from statpitch.data.http import FetchError, PoliteSession

log = logging.getLogger(__name__)

BASE_URL = "https://api.openligadb.de"

#: competition_id -> OpenLigaDB league shortcut. Deliberately only the cup: the
#: German league is ingested from football-data.co.uk, which also carries the
#: odds the Decision Layer needs, and taking it from here as well would duplicate
#: every match under two naming conventions.
COMPETITIONS: dict[str, str] = {
    "GER.DFB_POKAL": "dfb",
}

#: Schedules describe the future and must not be served from an unbounded cache.
#: Shared with the openfootball schedule path so both expire together.
SCHEDULE_MAX_AGE_SECONDS = of.SCHEDULE_MAX_AGE_SECONDS


def matchdata_url(shortcut: str, season: int) -> str:
    """Season is the starting year: 2026 is the 2026/27 competition."""
    return f"{BASE_URL}/getmatchdata/{shortcut}/{season}"


def fetch_season(
    competition_id: str,
    season: int,
    *,
    session: PoliteSession | None = None,
    max_age: float | None = SCHEDULE_MAX_AGE_SECONDS,
) -> list[dict] | None:
    """Every match OpenLigaDB holds for one competition-season.

    Returns None when the source cannot answer — an unmapped competition, a
    season that does not exist yet, or a failed request. Every caller's fallback
    is to publish no fixtures for that competition, which is the state the API
    already reports honestly.
    """
    shortcut = COMPETITIONS.get(competition_id)
    if shortcut is None:
        return None
    url = matchdata_url(shortcut, season)
    try:
        body = (session or PoliteSession()).get_bytes(
            url, suffix=".json", max_age=max_age
        )
    except FetchError as exc:
        log.warning("openligadb: %s — %s", competition_id, exc)
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.warning("openligadb: %s — malformed payload: %s", competition_id, exc)
        return None
    if not isinstance(payload, list):
        log.warning("openligadb: %s — expected a list of matches", competition_id)
        return None
    return payload


def _team(match: dict, key: str) -> str | None:
    team = match.get(key) or {}
    name = team.get("teamName")
    return str(name).strip() if name else None


def parse_matches(payload: list[dict], competition_id: str, season: int) -> pd.DataFrame:
    """Flatten OpenLigaDB matches into scheduled-fixture rows.

    Finished matches are dropped: this is a fixture list, and a played match
    belongs in the result archive rather than on a matchday card. A match without
    both clubs or without a kickoff is dropped too — an undated fixture cannot be
    listed by date, and inventing one would put it on a day it is not on.
    """
    competition = taxonomy.get(competition_id)
    label = of.season_label(of.season_dir(season))
    rows: list[dict] = []
    finished = skipped = 0

    for match in payload:
        if match.get("matchIsFinished"):
            finished += 1
            continue
        home, away = _team(match, "team1"), _team(match, "team2")
        stamp = match.get("matchDateTimeUTC")
        if not (home and away and stamp):
            skipped += 1
            continue
        kickoff = pd.to_datetime(stamp, utc=True, errors="coerce")
        if pd.isna(kickoff):
            skipped += 1
            continue
        kickoff = kickoff.tz_convert(None)

        raw_stage = ((match.get("group") or {}).get("groupName")) or ""
        stage = of.normalise_stage(raw_stage) if raw_stage else "unknown"
        rows.append(
            {
                "fixture_id": of.fixture_id(
                    competition_id, label, home, away, stage,
                    knockout=competition.is_knockout,
                ),
                "competition_id": competition_id,
                "season": label,
                "stage": stage,
                "stage_detail": raw_stage or None,
                "format": competition.resolve_format(stage=stage, season=label),
                "neutral_venue": competition.is_neutral_venue(stage),
                "date": kickoff.normalize(),
                "kickoff": kickoff.strftime("%H:%M"),
                # OpenLigaDB publishes a confirmed UTC kickoff, not a nominal
                # matchday, so these need no correction pass.
                "date_confirmed": True,
                "home_team": home,
                "away_team": away,
                "home_country": None,
                "away_country": None,
                "source": "openligadb",
                "odds_coverage": competition.odds_coverage,
            }
        )

    if skipped:
        log.warning(
            "openligadb: %s — dropped %d match(es) missing a club or a kickoff",
            competition_id, skipped,
        )
    log.info(
        "openligadb: %s %s — %d scheduled, %d already played",
        competition_id, label, len(rows), finished,
    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_schedule(
    competition_id: str,
    seasons: list[int],
    *,
    session: PoliteSession | None = None,
    max_age: float | None = SCHEDULE_MAX_AGE_SECONDS,
) -> pd.DataFrame:
    """Scheduled fixtures for one mapped competition across seasons."""
    session = session or PoliteSession()
    frames = []
    for season in seasons:
        payload = fetch_season(
            competition_id, season, session=session, max_age=max_age
        )
        if not payload:
            continue
        frame = parse_matches(payload, competition_id, season)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="fixture_id", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_all_schedules(
    seasons: list[int],
    *,
    session: PoliteSession | None = None,
    max_age: float | None = SCHEDULE_MAX_AGE_SECONDS,
) -> pd.DataFrame:
    """Scheduled fixtures across every competition this source covers."""
    session = session or PoliteSession()
    frames = []
    for competition_id in COMPETITIONS:
        try:
            frame = build_schedule(
                competition_id, seasons, session=session, max_age=max_age
            )
        except Exception:
            log.exception("openligadb: failed to build schedule for %s", competition_id)
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
