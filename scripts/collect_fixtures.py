"""Correct fixture dates and collect lineups from API-Football (Roadmap §4.5, §7.2).

    python scripts/collect_fixtures.py [--horizon-days 21]

Two jobs against one 100/day budget, and they are worth different things.

**Neither works on the free plan.** Verified 2026-08-17: API-Football answers a
current-season request with "Free plans do not have access to this season, try
from 2022 to 2024." Both jobs below concern the *current* season, so both are
blocked at $0 (NFR-1). The season guard means a run now costs nothing and says
so, rather than burning five calls to be told the same thing weekly. Everything
below describes what the collector does when a plan can see the season.

**Date correction** is the immediate one. openfootball publishes a matchday
before the league confirms kickoff slots, so 88% of the fixture list sits on a
nominal date — ten La Liga fixtures stacked on one Sunday, played across four
days. This rewrites those dates where API-Football has a confirmed one, and marks
them `date_confirmed`. One call per league per horizon window, so the whole
correction costs about five calls.

**Lineups** are the deferred one, and this run is what makes them possible.
Matching a fixture yields API-Football's own `api_fixture_id`, which is the key
`/fixtures/lineups` needs; it is stored on the fixture row here so a later
collector does not have to re-derive it. Collecting the XI itself is not done
yet — it costs one call per fixture and is worthless outside the hour before
kickoff, so it belongs to a job that runs on a different clock.

Without a key
=============

Every call returns `None` and the run is a no-op that reports itself as skipped.
openfootball's provisional dates continue to be served, flagged
`date_confirmed=false`. A missing credential is a capability this deployment does
not have, not an error.

Matching
========

API-Football writes its own club names, so a corrected fixture has to be matched
back to the openfootball one. Matching is constrained to the same competition and
the same ±3 day window, and requires a unique candidate — an ambiguous match is
left uncorrected rather than resolved to a guess. The same rule the Elo and
Transfermarkt resolvers already follow, for the same reason.
"""

from __future__ import annotations

import argparse
import logging
import unicodedata
from datetime import UTC, datetime, timedelta

import pandas as pd

from statpitch import paths
from statpitch.data import api_football as af

log = logging.getLogger("collect")

#: How far ahead to correct. Leagues confirm slots a few weeks out, so a longer
#: horizon spends budget on dates that are still provisional upstream anyway.
DEFAULT_HORIZON_DAYS = 21

#: How far a corrected date may move before the match is rejected as a different
#: fixture. A postponement of more than this is not a reschedule of the same
#: round, and pairing across it would silently retitle a fixture.
MAX_DATE_SHIFT_DAYS = 3


def normalise(name: str) -> frozenset[str]:
    """Distinctive lowercase tokens, accents folded, legal forms dropped."""
    ascii_only = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in ascii_only)
    noise = {"fc", "afc", "cf", "sc", "ac", "as", "ss", "ssc", "us", "ud", "cd",
             "sd", "ca", "rc", "rcd", "sv", "vfb", "vfl", "tsg", "bsc", "de",
             "club", "calcio", "real", "atletico", "athletic", "borussia"}
    return frozenset(w for w in cleaned.split() if w not in noise)


def match_fixture(
    row: pd.Series, candidates: list[dict]
) -> dict | None:
    """Find the API-Football fixture that is this openfootball one.

    Requires a unique token match on both clubs within the date window. Returns
    None when zero or several candidates fit, which leaves the fixture on its
    provisional date — the honest outcome, and the one that cannot mislabel.
    """
    home, away = normalise(row["home_team"]), normalise(row["away_team"])
    window = timedelta(days=MAX_DATE_SHIFT_DAYS)
    hits = []
    for candidate in candidates:
        stamp = pd.Timestamp(candidate["kickoff_utc"]).tz_convert(None)
        if abs(stamp.normalize() - row["date"]) > window:
            continue
        c_home, c_away = normalise(candidate["home_team"]), normalise(candidate["away_team"])
        if not (home & c_home) or not (away & c_away):
            continue
        hits.append((candidate, stamp))
    if len(hits) != 1:
        return None
    candidate, stamp = hits[0]
    return {**candidate, "confirmed_at": stamp}


def correct_dates(fixtures: pd.DataFrame, horizon_days: int) -> tuple[pd.DataFrame, dict]:
    """Rewrite provisional dates where API-Football has a confirmed one."""
    client = af.ApiFootball()
    today = pd.Timestamp(datetime.now(UTC).date())
    horizon = today + pd.Timedelta(days=horizon_days)

    in_window = fixtures[(fixtures["date"] >= today) & (fixtures["date"] <= horizon)]
    stats: dict = {"in_window": int(len(in_window)), "corrected": 0, "moved": 0,
                   "unmatched": 0, "calls_skipped": 0, "seasons": set()}
    if in_window.empty:
        return fixtures, stats

    out = fixtures.copy()
    if "api_fixture_id" not in out.columns:
        out["api_fixture_id"] = pd.NA
    for competition_id, group in in_window.groupby("competition_id"):
        if competition_id not in af.LEAGUE_IDS:
            continue
        season = int(str(group["season"].iloc[0]).split("-")[0])
        stats["seasons"].add(season)
        payload = client.fixtures_in_range(
            str(competition_id), today.date(), horizon.date(), season
        )
        if payload is None:
            stats["calls_skipped"] += 1
            continue
        candidates = af.parse_fixtures(payload)
        log.info("%s — %d confirmed fixtures upstream", competition_id, len(candidates))

        for index, row in group.iterrows():
            hit = match_fixture(row, candidates)
            if hit is None:
                stats["unmatched"] += 1
                continue
            stamp = hit["confirmed_at"]
            if stamp.normalize() != row["date"]:
                stats["moved"] += 1
            out.loc[index, "date"] = stamp.normalize()
            out.loc[index, "kickoff"] = stamp.strftime("%H:%M")
            out.loc[index, "date_confirmed"] = True
            # The id lineup collection will need, captured while we have it.
            out.loc[index, "api_fixture_id"] = hit.get("api_fixture_id")
            stats["corrected"] += 1
    return out, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not af.configured():
        log.warning(
            "%s is not set — nothing to collect. openfootball's provisional dates "
            "are served as-is and flagged date_confirmed=false.", af.ENV_KEY,
        )
        return 0

    path = paths.fixtures_file()
    if not path.exists():
        log.error("no fixtures — run scripts/build_fixtures.py first")
        return 1

    fixtures = pd.read_parquet(path)
    before = int(fixtures["date_confirmed"].sum())
    fixtures, stats = correct_dates(fixtures, args.horizon_days)
    after = int(fixtures["date_confirmed"].sum())

    log.info(
        "%d fixtures in the %d-day window: %d corrected (%d moved to a different "
        "day), %d unmatched; confirmed dates %d -> %d",
        stats["in_window"], args.horizon_days, stats["corrected"], stats["moved"],
        stats["unmatched"], before, after,
    )
    if stats["calls_skipped"]:
        seasons = sorted(stats["seasons"])
        out_of_plan = [s for s in seasons if not af.season_available(s)]
        if out_of_plan:
            low, high = af.FREE_PLAN_SEASONS
            log.warning(
                "%d competition(s) skipped: season(s) %s are outside the free "
                "plan's %d-%d window, so no current-season fixture can be "
                "confirmed at $0. Dates stay provisional and are flagged "
                "date_confirmed=false. See MODEL_CARD §8.",
                stats["calls_skipped"], out_of_plan, low, high,
            )
        else:
            log.warning(
                "%d competition(s) skipped — budget exhausted or the call failed. "
                "Those fixtures keep their provisional dates.",
                stats["calls_skipped"],
            )

    fixtures.to_parquet(path, index=False)
    log.info("wrote %s", path)

    from statpitch import quota

    log.info("budget: %s", quota.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
