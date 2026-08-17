"""football-data.org fixtures — confirmed kickoff times (Roadmap §7.2).

The third attempt at the same problem, and the reason the first two failed is
worth carrying: openfootball publishes a matchday before the league confirms
kickoff slots, so 88% of the fixture list sits on a nominal date. API-Football
was the designed correction and its free plan turned out to cover seasons
2022-2024 only, which is the current season's exact complement.

This source was chosen after measuring the alternatives rather than from a
listing:

    ESPN               keyless, all leagues, 403 from this host
    TheSportsDB        free test key returns ONE event per league
    OpenLigaDB         keyless and complete, German leagues only
    football-data.org  free key, all five leagues          <- this

OpenLigaDB is the fallback worth keeping in mind: it needs no credential at all
and returned 306 Bundesliga fixtures with real UTC kickoff times. It covers one
league of five, which is why it is not the primary.

Rate limit
==========

Ten requests a minute on the free tier, which is the binding constraint rather
than a daily budget. One call per competition covers a whole date range, so a
full correction is five calls — but they go through `PoliteSession` with a
deliberate delay anyway, because a 429 costs more than waiting does.

Unlike API-Football there is no per-day allowance to protect, so this does not
route through `statpitch.quota`. That module exists to ration a hard 100/day
ceiling; applying it here would ration something that is not scarce.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date as Date

from statpitch.data.http import FetchError, PoliteSession

log = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"

ENV_KEY = "STATPITCH_FOOTBALL_DATA_ORG_KEY"

#: competition_id -> football-data.org competition code.
COMPETITIONS: dict[str, str] = {
    "ENG.PL": "PL",
    "ESP.LALIGA": "PD",
    "GER.BUNDESLIGA": "BL1",
    "ITA.SERIEA": "SA",
    "FRA.LIGUE1": "FL1",
}

#: Ten requests a minute, so seven seconds apart leaves headroom for a retry
#: without ever approaching the limit.
MIN_INTERVAL = 7.0


class FootballDataError(RuntimeError):
    pass


def api_key() -> str | None:
    key = os.environ.get(ENV_KEY, "").strip()
    return key or None


def configured() -> bool:
    """Whether a key is present. False is a normal state, not a failure."""
    return api_key() is not None


def session() -> PoliteSession:
    """A rate-limited session carrying the auth header."""
    key = api_key()
    if key is None:
        raise FootballDataError(f"{ENV_KEY} is not set")
    return PoliteSession(min_interval=MIN_INTERVAL, headers={"X-Auth-Token": key})


@dataclass(frozen=True, slots=True)
class Fixture:
    home_team: str
    away_team: str
    #: ISO-8601 UTC, e.g. "2026-08-21T19:00:00Z".
    kickoff_utc: str
    status: str | None
    matchday: int | None
    source_id: int | None


def parse_matches(payload: dict) -> list[Fixture]:
    """Flatten the `matches` array, dropping anything without the essentials.

    A missing branch drops one fixture rather than failing a competition: the
    correction is best-effort by design, and a fixture left uncorrected keeps its
    provisional date, which is the state everything already handles.
    """
    out: list[Fixture] = []
    for match in (payload or {}).get("matches", []) or []:
        home = ((match.get("homeTeam") or {}).get("name"))
        away = ((match.get("awayTeam") or {}).get("name"))
        stamp = match.get("utcDate")
        if not (home and away and stamp):
            continue
        out.append(
            Fixture(
                home_team=home,
                away_team=away,
                kickoff_utc=stamp,
                status=match.get("status"),
                matchday=match.get("matchday"),
                source_id=match.get("id"),
            )
        )
    return out


def fetch_matches(
    competition_id: str,
    start: Date,
    end: Date,
    *,
    http: PoliteSession | None = None,
) -> list[Fixture] | None:
    """Confirmed fixtures for one competition across a date range — one call.

    Returns None rather than raising when the source cannot answer: no key, an
    unmapped competition, or a failed request. Every caller's fallback is to keep
    the provisional date, so a None is a degraded answer rather than an error.
    """
    if not configured():
        log.debug("football-data.org: no key configured")
        return None
    code = COMPETITIONS.get(competition_id)
    if code is None:
        return None

    url = (
        f"{BASE_URL}/competitions/{code}/matches"
        f"?dateFrom={start.isoformat()}&dateTo={end.isoformat()}"
    )
    try:
        body = (http or session()).get_bytes(url, suffix=".json")
    except FetchError as exc:
        # 403 means the key is rejected or the competition is outside the free
        # tier; 429 means the minute limit was hit. Both leave dates provisional.
        log.warning("football-data.org: %s — %s", competition_id, exc)
        return None

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        log.warning("football-data.org: %s — malformed JSON: %s", competition_id, exc)
        return None

    if payload.get("errorCode") or payload.get("message"):
        log.warning(
            "football-data.org: %s — %s",
            competition_id, payload.get("message") or payload.get("errorCode"),
        )
        return None
    return parse_matches(payload)
