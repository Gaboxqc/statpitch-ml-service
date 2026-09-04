"""API-Football client (Roadmap §4.5, §7.2).

`statpitch.quota` has enforced a 100/day budget since before any caller existed —
tested, with a hard stop at 90, a fixture-keyed cache and a graceful `None` on
exhaustion. This is the caller it was waiting for.

Two jobs, one budget
====================

**Correcting fixture dates.** The larger of the two, and it was not the original
motivation. openfootball publishes a matchday before the league confirms kickoff
slots, so 88% of the fixture list sits on a nominal date rather than a real one —
ten La Liga fixtures stacked on one Sunday, played across four days. API-Football
has confirmed dates. Correcting them is what makes `/today` trustworthy, and it
pays off immediately.

**Collecting lineups and injuries.** The original motivation, and the one with a
clock on it. Confirmed XI lands ~1h before kickoff and the free tier has **no
historical archive**, so this cannot be backfilled and cannot re-test `w` — it can
only accumulate forward. Its evaluation is a season away, and that season starts
when collection does.

Spending the budget
===================

90 usable calls a day against ~50 fixtures in a five-league weekend round. Date
correction is one call per competition-round rather than one per fixture, so it
costs a handful; lineups are one per fixture and only worth spending close to
kickoff, when the XI is actually confirmed.

Everything goes through `QuotaBudget.spend`, which returns `None` rather than
raising when the budget is gone. A caller that gets `None` keeps the openfootball
date and the pre-match estimate — degraded, never wrong.

The key
=======

Read from `STATPITCH_API_FOOTBALL_KEY`. Absent, `configured()` is False and every
call returns `None` without touching the network or the budget, so a checkout with
no key behaves exactly like one whose budget is exhausted. That is deliberate:
absence of a credential is not an error condition, it is a capability this
deployment does not have.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any

import requests

from statpitch import quota

log = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"

ENV_KEY = "STATPITCH_API_FOOTBALL_KEY"

#: competition_id -> API-Football league id, for the leagues whose schedules
#: openfootball publishes provisionally. The cups are absent because their draws
#: are not published far enough ahead for a date to need correcting.
LEAGUE_IDS: dict[str, int] = {
    "ENG.PL": 39,
    "ESP.LALIGA": 140,
    "GER.BUNDESLIGA": 78,
    "ITA.SERIEA": 135,
    "FRA.LIGUE1": 61,
    "POR.PRIMEIRA": 94,
    "NED.EREDIVISIE": 88,
    "TUR.SUPERLIG": 203,
}

TIMEOUT = 30

#: Seasons the FREE plan can see, inclusive. Verified against the live API on
#: 2026-08-17, which answered a request for season 2026 with:
#:
#:     {'plan': 'Free plans do not have access to this season,
#:               try from 2022 to 2024.'}
#:
#: This is the constraint that decides what §4.5 and §7.2 can be. Neither
#: correcting an upcoming fixture's date nor collecting a confirmed XI is
#: possible at $0, because both concern the CURRENT season and the free plan
#: cannot see it. NFR-1 makes that a limit rather than a bug to fix.
#:
#: Checked before spending rather than after failing: the API answers an
#: out-of-plan request with HTTP 200 and an `errors` object, and the budget is
#: reserved before the call, so a request that cannot succeed still costs one of
#: the ninety. The first real run burned five that way.
FREE_PLAN_SEASONS = (2022, 2024)


class PlanRestricted(RuntimeError):
    """The requested season is outside what the configured plan can see."""


class ApiFootballError(RuntimeError):
    pass


def api_key() -> str | None:
    key = os.environ.get(ENV_KEY, "").strip()
    return key or None


def configured() -> bool:
    """Whether a key is present. False is a normal state, not a failure."""
    return api_key() is not None


def season_available(season: int) -> bool:
    """Whether the free plan can see this season at all."""
    low, high = FREE_PLAN_SEASONS
    return low <= season <= high


@dataclass
class ApiFootball:
    """Budget-guarded access to the endpoints this project uses."""

    budget: quota.QuotaBudget = field(default_factory=quota.budget_from_env)
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        key = api_key()
        if key is None:
            raise ApiFootballError(f"{ENV_KEY} is not set")
        response = self.session.get(
            f"{BASE_URL}{path}",
            params=params,
            headers={"x-apisports-key": key},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        # API-Football answers 200 with an `errors` object rather than a 4xx, so
        # a bad key or an exhausted plan looks like success to `raise_for_status`.
        errors = payload.get("errors")
        if errors:
            # A plan restriction is not a transient failure, so it is raised as
            # its own type: retrying it spends budget to be told the same thing.
            if isinstance(errors, dict) and "plan" in errors:
                raise PlanRestricted(str(errors["plan"]))
            raise ApiFootballError(f"api-football returned errors: {errors}")
        return payload.get("response", [])

    def _spend(self, cache_key: str, path: str, params: dict[str, Any]) -> Any | None:
        if not configured():
            log.debug("api-football: no key configured; skipping %s", cache_key)
            return None

        season = params.get("season")
        if season is not None and not season_available(int(season)):
            low, high = FREE_PLAN_SEASONS
            log.warning(
                "api-football: season %s is outside the free plan's %d-%d window, "
                "so %s cannot succeed; skipping without spending a call",
                season, low, high, cache_key,
            )
            return None

        return self.budget.spend(cache_key, lambda: self._get(path, params))

    # --- fixtures ---------------------------------------------------------

    def fixtures_for_date(
        self, competition_id: str, on: Date, season: int
    ) -> list[dict] | None:
        """Confirmed fixtures for one league on one date.

        Keyed per (league, date) so a re-run inside the TTL is free. One call
        covers every fixture that day, which is what keeps date correction cheap
        enough to run daily inside a 90-call budget.
        """
        league = LEAGUE_IDS.get(competition_id)
        if league is None:
            return None
        return self._spend(
            f"fixtures:{competition_id}:{on.isoformat()}",
            "/fixtures",
            {"league": league, "season": season, "date": on.isoformat()},
        )

    def fixtures_in_range(
        self, competition_id: str, start: Date, end: Date, season: int
    ) -> list[dict] | None:
        """Confirmed fixtures for one league across a date range — one call."""
        league = LEAGUE_IDS.get(competition_id)
        if league is None:
            return None
        return self._spend(
            f"fixtures:{competition_id}:{start.isoformat()}:{end.isoformat()}",
            "/fixtures",
            {
                "league": league,
                "season": season,
                "from": start.isoformat(),
                "to": end.isoformat(),
            },
        )

    def lineups(self, fixture_id: int) -> list[dict] | None:
        """Confirmed XI for one fixture. Empty until ~1h before kickoff."""
        return self._spend(
            f"lineups:{fixture_id}", "/fixtures/lineups", {"fixture": fixture_id}
        )

    def injuries(self, competition_id: str, season: int, on: Date) -> list[dict] | None:
        """Reported injuries for one league on one date."""
        league = LEAGUE_IDS.get(competition_id)
        if league is None:
            return None
        return self._spend(
            f"injuries:{competition_id}:{on.isoformat()}",
            "/injuries",
            {"league": league, "season": season, "date": on.isoformat()},
        )


# --- shaping ------------------------------------------------------------------

def parse_fixtures(payload: list[dict]) -> list[dict]:
    """Flatten the fixture payload to what date correction needs.

    Defensive about shape: API-Football nests four levels deep and a missing
    branch should drop one fixture rather than fail a whole round.
    """
    out: list[dict] = []
    for item in payload or []:
        fixture = item.get("fixture") or {}
        teams = item.get("teams") or {}
        home = (teams.get("home") or {}).get("name")
        away = (teams.get("away") or {}).get("name")
        stamp = fixture.get("date")
        if not (home and away and stamp):
            continue
        out.append(
            {
                "api_fixture_id": fixture.get("id"),
                "home_team": home,
                "away_team": away,
                # ISO-8601 with offset, e.g. "2026-08-21T19:00:00+00:00".
                "kickoff_utc": stamp,
                "status": ((fixture.get("status") or {}).get("short")),
                "venue": ((fixture.get("venue") or {}).get("name")),
            }
        )
    return out


def parse_lineups(payload: list[dict]) -> list[dict]:
    """Flatten a lineup payload to one row per club."""
    out: list[dict] = []
    for item in payload or []:
        team = (item.get("team") or {}).get("name")
        if not team:
            continue
        starters = [
            ((p or {}).get("player") or {}).get("name")
            for p in (item.get("startXI") or [])
        ]
        out.append(
            {
                "team": team,
                "formation": item.get("formation"),
                "start_xi": [n for n in starters if n],
                "coach": ((item.get("coach") or {}).get("name")),
            }
        )
    return out
