"""Fetch Club Elo histories for the clubs of the odds-covered leagues.

    python scripts/fetch_league_club_elo.py [--competitions POR.PRIMEIRA ...]
                                            [--limit N] [--retries N]

The sibling `fetch_cup_club_elo.py` does the same job for cup entrants and reads
its club list from `cup_club_elo_map.json`. There is no equivalent map for
leagues and there does not need to be: a league club's spelling is reconciled by
`club_elo.NAME_ALIASES` plus country-constrained matching, so the club list can
be derived from the match log itself rather than stored as a third artifact that
can go stale against it.

Shard per club, for the reason the cup fetcher records
=====================================================

Each club's history is written to its own parquet under `elo_shards/` as it
arrives, and the combined artifact is rebuilt from the shards at the end. The
first version of the cup fetch built one large in-memory frame, was killed
part-way, and lost everything. An interrupted run here resumes instead.

That matters more than it did there. Club Elo's per-club endpoint has been
observed returning 502 for sustained stretches (2026-09-04), and a run that dies
on the fortieth club of a hundred and twenty must not start again from zero.

A club that cannot be fetched is skipped and reported, never guessed
===================================================================

An unrated club is a normal state, not a failure: Club Elo rates a second tier
for the Big 5 only, so a club newly promoted into the Primeira Liga, Eredivisie
or Süper Lig has no history until it appears in a tier-1 snapshot. Those clubs
fall through to the FR-9 pooled entrant prior, which is the documented fallback.
The run therefore reports what it could not get and exits 0; it exits non-zero
only when it got nothing at all, which is the signature of an outage rather than
of a genuinely unrated club.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import pandas as pd

from statpitch import paths, taxonomy
from statpitch.data import club_elo as ce
from statpitch.data.http import FetchError, PoliteSession

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("fetch_league_club_elo")

#: Spacing between per-club requests. Slower than the cup fetcher's 0.2s because
#: that pace is what preceded the observed 502s.
MIN_INTERVAL = 0.5

#: Backoff between attempts at one club, in seconds. Three tries, widening — an
#: outage is not distinguishable from a slow moment on the first failure.
RETRY_BACKOFF = (2.0, 8.0, 20.0)


def clubs_needed(competition_ids: list[str]) -> dict[str, str]:
    """Club Elo names required by these competitions -> the competition seen in.

    Resolution mirrors `build_features.load_aliases` exactly: apply
    `NAME_ALIASES` first, then match against the roster constrained by the
    country in the `competition_id` prefix.
    """
    matches = pd.read_parquet(paths.matches_file())
    roster = pd.read_parquet(paths.processed_dir() / "clubelo_roster_full.parquet")

    needed: dict[str, str] = {}
    for competition_id in competition_ids:
        country = competition_id.split(".")[0]
        sub = matches[matches["competition_id"] == competition_id]
        if sub.empty:
            log.warning("%s: no matches in the log", competition_id)
            continue
        # Season-aware, and every window is taken rather than only the current
        # one: a name that means two clubs across eras needs BOTH histories, or
        # the seasons pointing at the one that was skipped go unrated.
        targets: dict[str, str] = {}
        for name in set(sub["home_team"]) | set(sub["away_team"]):
            windows = ce.SEASON_ALIASES.get(str(name))
            if windows:
                for _, target in windows:
                    targets[target] = country
            else:
                targets[ce.resolve_alias(str(name), aliases=ce.NAME_ALIASES)] = country
        resolution = ce.resolve_cup_clubs(targets, roster)
        if resolution.ambiguous:
            raise SystemExit(
                f"{competition_id}: {len(resolution.ambiguous)} ambiguous club name(s) "
                f"— resolve them in club_elo.NAME_ALIASES before fetching: "
                f"{dict(list(resolution.ambiguous.items())[:5])}"
            )
        for target in resolution.mapping.values():
            needed.setdefault(target, competition_id)
        if resolution.unmatched:
            log.warning(
                "%s: %d club(s) with no Club Elo entry, which for a non-Big-5 "
                "league usually means newly promoted: %s",
                competition_id, len(resolution.unmatched),
                ", ".join(sorted(resolution.unmatched)),
            )
    return needed


def fetch_one(target: str, session: PoliteSession, retries: int) -> pd.DataFrame | None:
    for attempt in range(retries):
        try:
            return ce.fetch_club_history(target, session=session)
        except (FetchError, ce.ClubEloError) as exc:
            if attempt + 1 == retries:
                log.warning("no history for %s after %d attempt(s): %s",
                            target, retries, exc)
                return None
            time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--competitions", nargs="+", default=None,
        help="Competition ids. Defaults to every odds-covered league.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many new clubs (for a smoke test).")
    parser.add_argument("--retries", type=int, default=len(RETRY_BACKOFF))
    args = parser.parse_args()

    competition_ids = args.competitions or [
        c.competition_id for c in taxonomy.registry().with_odds_coverage()
    ]

    shard_dir = paths.processed_dir() / "elo_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    combined_path = paths.processed_dir() / "elo_ratings_all.parquet"
    existing = pd.read_parquet(combined_path)
    have = set(existing["clubelo_name"])

    needed = clubs_needed(competition_ids)
    todo = sorted(t for t in needed if t not in have)
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(needed)} club(s) needed; {len(needed) - len(todo)} already held; "
          f"fetching {len(todo)}", flush=True)
    if not todo:
        print("nothing to fetch", flush=True)
        return 0

    session = PoliteSession(min_interval=MIN_INTERVAL)
    fetched = failed = 0
    for i, target in enumerate(todo, 1):
        shard = shard_dir / f"{ce.club_slug(target)}.parquet"
        if shard.exists():
            continue
        history = fetch_one(target, session, args.retries)
        if history is None or history.empty:
            failed += 1
            continue
        history.to_parquet(shard, index=False)
        fetched += 1
        if i % 20 == 0:
            print(f"  {i}/{len(todo)} ({fetched} fetched, {failed} unavailable)",
                  flush=True)

    # Rebuilt from the shards rather than appended to, so a resumed run and a
    # single run produce the same artifact.
    shards = sorted(shard_dir.glob("*.parquet"))
    frames = [existing] + [pd.read_parquet(s) for s in shards]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["clubelo_name", "valid_from"])
    combined = combined.sort_values(["clubelo_name", "valid_from"]).reset_index(drop=True)
    combined.to_parquet(combined_path, index=False)

    print(f"fetched {fetched}, unavailable {failed}; combined {len(combined)} rows "
          f"across {combined['clubelo_name'].nunique()} clubs -> {combined_path.name}",
          flush=True)

    if fetched == 0 and failed:
        print("nothing was fetched at all — this reads as an upstream outage "
              "rather than a set of genuinely unrated clubs", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
