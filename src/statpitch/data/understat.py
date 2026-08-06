"""Understat expected-goals ingestion (Design §4, Notebook 07).

xG is the single most valuable feature addition available here. Goals are a noisy
realisation of chance quality: a side that creates 2.1 xG and scores once is
better than its scoreline, and rolling xG separates that from a side that created
0.4 and got lucky. Rolling *goals* cannot.

How the data is actually reached
================================

The spec assumes the well-known approach: fetch the league page and regex the
`JSON.parse('...datesData...')` block out of the HTML. **That no longer works.**
The season page now ships as an 18KB shell with the table markup and no data at
all; the only `JSON.parse` left in it belongs to an advertisement.

The data comes from a JSON route instead:

    https://understat.com/getLeagueData/<league>/<season_start_year>

with one non-obvious requirement: it returns **404 without an
`X-Requested-With: XMLHttpRequest` header**. With it, one request per
league-season yields every match's xG for both sides — cleaner and far lighter
than parsing HTML ever was.

Coverage is 2014/15 onward for the Big 5 (plus RFPL, which is out of scope), so
xG features exist for roughly the last third of the match archive and are absent
before it. That is a real limitation for the long training window, and the
feature builder emits nulls rather than zeroes so a model can tell "no chance
created" from "not measured".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import pandas as pd

from statpitch import taxonomy
from statpitch.data.http import FetchError, PoliteSession

log = logging.getLogger(__name__)

BASE_URL = "https://understat.com"

#: Without this the JSON routes 404. It is the one thing that makes them usable.
AJAX_HEADERS = {"X-Requested-With": "XMLHttpRequest"}

#: Understat's coverage begins here.
FIRST_SEASON = 2014


class UnderstatError(RuntimeError):
    pass


def session(**kwargs) -> PoliteSession:
    """A polite session carrying the header Understat's JSON routes require."""
    kwargs.setdefault("min_interval", 2.0)   # a scraped source, so tread lightly
    return PoliteSession(headers=dict(AJAX_HEADERS), **kwargs)


def league_url(understat_code: str, season_start: int) -> str:
    return f"{BASE_URL}/getLeagueData/{understat_code}/{season_start}"


@dataclass(frozen=True, slots=True)
class SeasonXG:
    competition_id: str
    season_start: int
    matches: pd.DataFrame


def fetch_season(
    competition_id: str,
    season_start: int,
    *,
    http: PoliteSession | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Per-match xG for one competition-season. Empty frame when unavailable."""
    competition = taxonomy.get(competition_id)
    code = competition.understat_code
    if not code:
        raise UnderstatError(f"{competition_id} has no Understat code")
    if season_start < FIRST_SEASON:
        return pd.DataFrame()

    http = http or session()
    try:
        body = http.get_bytes(
            league_url(code, season_start), suffix=".json", force=force
        )
    except FetchError as exc:
        log.warning("understat: %s %s unavailable (%s)", competition_id, season_start, exc)
        return pd.DataFrame()

    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        # A 404 shell rather than JSON — almost always the missing AJAX header.
        log.warning(
            "understat: %s %s did not return JSON; the X-Requested-With header is "
            "required and its absence yields an HTML 404 page",
            competition_id, season_start,
        )
        return pd.DataFrame()

    rows = payload.get("dates") or payload.get("datesData") or []
    records = []
    for match in rows:
        if not match.get("isResult"):
            continue          # fixture not yet played
        xg = match.get("xG") or {}
        goals = match.get("goals") or {}
        records.append(
            {
                "competition_id": competition_id,
                "season_start_year": season_start,
                "understat_id": match.get("id"),
                "date": pd.to_datetime(match.get("datetime"), errors="coerce"),
                "understat_home": (match.get("h") or {}).get("title"),
                "understat_away": (match.get("a") or {}).get("title"),
                "home_xg": pd.to_numeric(xg.get("h"), errors="coerce"),
                "away_xg": pd.to_numeric(xg.get("a"), errors="coerce"),
                "home_goals": pd.to_numeric(goals.get("h"), errors="coerce"),
                "away_goals": pd.to_numeric(goals.get("a"), errors="coerce"),
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame = frame[frame["date"].notna() & frame["home_xg"].notna()]
    return frame.reset_index(drop=True)


def fetch_all(
    competitions: list[str] | None = None,
    first_season: int = FIRST_SEASON,
    last_season: int | None = None,
    *,
    http: PoliteSession | None = None,
) -> pd.DataFrame:
    """Every available competition-season of xG."""
    registry = taxonomy.registry()
    if competitions is None:
        competitions = [c.competition_id for c in registry if c.understat_code]
    if last_season is None:
        today = pd.Timestamp.today()
        last_season = today.year if today.month >= 7 else today.year - 1

    http = http or session()
    frames = []
    for competition_id in competitions:
        for season_start in range(max(first_season, FIRST_SEASON), last_season + 1):
            frame = fetch_season(competition_id, season_start, http=http)
            if not frame.empty:
                frames.append(frame)
                log.info(
                    "understat: %s %s — %d matches",
                    competition_id, season_start, len(frame),
                )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --- deriving the club-name map from fixtures ---------------------------------

#: A name pairing must be corroborated this many times before it is accepted, so
#: one coincidental date-and-scoreline collision cannot mint a mapping.
MIN_NAME_SUPPORT = 3


def derive_name_map(
    xg: pd.DataFrame, matches: pd.DataFrame, *, tolerance_days: int = 2
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Map Understat club names onto football-data ones **without using names**.

    Both sources contain the same fixtures, so a match is identified by its
    competition, date and scoreline. Once a fixture is identified, the club names
    on either side must refer to the same two clubs — which yields the mapping as
    a by-product of the join rather than as a guess about spelling.

    This is deliberately not fuzzy matching. Names like "Manchester City" ->
    "Man City" are easy, but the same fuzzy pass proposes "Real Madrid" for
    "Ath Madrid" and "Cercle Brugge" for "Club Brugge", which attach the wrong
    club's data while looking entirely plausible. Fixture identity cannot make
    that error: either the two sources agree a match happened between those clubs
    on that date with that score, or they do not.

    Returns the accepted mapping and the full vote tally, so a contested name is
    visible rather than silently resolved by ordering.
    """
    if xg.empty or matches.empty:
        return {}, {}

    left = xg.copy()
    left["date_only"] = left["date"].dt.normalize()
    right = matches.copy()
    right["date_only"] = right["date"].dt.normalize()

    votes: dict[str, dict[str, int]] = {}

    for offset in range(-tolerance_days, tolerance_days + 1):
        shifted = left.assign(date_only=left["date_only"] + pd.Timedelta(days=offset))
        joined = shifted.merge(
            right,
            on=["competition_id", "date_only", "home_goals", "away_goals"],
            how="inner",
            suffixes=("_u", "_f"),
        )
        # A scoreline can repeat on a matchday, so only unambiguous fixtures vote.
        counts = joined.groupby(
            ["competition_id", "date_only", "home_goals", "away_goals"]
        )["match_id"].transform("size")
        unique = joined[counts == 1]

        for source, target in (
            ("understat_home", "home_team"), ("understat_away", "away_team")
        ):
            for u, f in zip(unique[source], unique[target], strict=True):
                votes.setdefault(str(u), {}).setdefault(str(f), 0)
                votes[str(u)][str(f)] += 1

    mapping: dict[str, str] = {}
    contested: dict[str, dict[str, int]] = {}
    for understat_name, tally in votes.items():
        best, support = max(tally.items(), key=lambda kv: kv[1])
        if support < MIN_NAME_SUPPORT:
            continue
        runner_up = sorted(tally.values(), reverse=True)
        # Require a clear winner: a name splitting its votes evenly between two
        # clubs is a collision, not a mapping.
        if len(runner_up) > 1 and runner_up[1] > support * 0.34:
            contested[understat_name] = tally
            continue
        mapping[understat_name] = best

    if contested:
        log.warning(
            "understat: %d club name(s) had contested fixture evidence and were "
            "left unmapped: %s", len(contested), sorted(contested),
        )
    log.info("understat: derived %d club name mappings from fixture identity", len(mapping))
    return mapping, votes


# --- joining to the match log -------------------------------------------------

def attach_to_matches(
    xg: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    tolerance_days: int = 2,
    name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Map Understat rows onto `match_id`.

    Names are translated through a map derived from fixture identity (see
    `derive_name_map`) before joining, because the two sources spell 43 of 168
    clubs differently — "Manchester City" against "Man City", "Nottingham Forest"
    against "Nott'm Forest" — and a raw name join silently drops them.

    A date tolerance is allowed because Understat timestamps kickoff in UTC while
    football-data.co.uk records the local match date, so a late kickoff can land
    on the following day.
    """
    from statpitch.data.club_elo import normalise

    if xg.empty or matches.empty:
        return pd.DataFrame()

    if name_map is None:
        name_map, _ = derive_name_map(xg, matches, tolerance_days=tolerance_days)

    left = xg.copy()
    left["home_name"] = left["understat_home"].map(lambda n: name_map.get(str(n), n))
    left["away_name"] = left["understat_away"].map(lambda n: name_map.get(str(n), n))
    left["home_key"] = left["home_name"].map(normalise)
    left["away_key"] = left["away_name"].map(normalise)
    left["date_only"] = left["date"].dt.normalize()

    right = matches[["match_id", "competition_id", "date", "home_team", "away_team"]].copy()
    right["home_key"] = right["home_team"].map(normalise)
    right["away_key"] = right["away_team"].map(normalise)
    right["date_only"] = right["date"].dt.normalize()

    merged = left.merge(
        right,
        on=["competition_id", "home_key", "away_key"],
        how="inner",
        suffixes=("_xg", "_match"),
    )
    gap = (merged["date_only_xg"] - merged["date_only_match"]).dt.days.abs()
    merged = merged[gap <= tolerance_days]

    out = merged[["match_id", "home_xg", "away_xg"]].drop_duplicates(subset="match_id")
    log.info(
        "understat: matched %d of %d xG rows onto the match log (%.1f%%)",
        len(out), len(xg), 100.0 * len(out) / max(len(xg), 1),
    )
    return out.reset_index(drop=True)
