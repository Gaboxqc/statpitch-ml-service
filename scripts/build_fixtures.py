"""Build the upcoming-fixtures artifact (Roadmap §7).

    python scripts/build_fixtures.py [--seasons 2026 2027] [--horizon-days 120]

Serving cannot fetch this itself. NFR-2 forbids a network call on a request
path, and `serving/app.py` loads its artifacts once at startup precisely so a
prediction is never the thing that pays for a download. So the fixture list is
built here, offline, and committed as a parquet the way every other processed
artifact is.

**This artifact goes stale, and that is visible rather than silent.** A fixture
list is a claim about the future, and kickoff times move. `generated_at` is
written into the file and reported by `/fixtures/upcoming`, so a consumer can see
how old the answer is instead of inferring freshness from the fact that a
response arrived. Roadmap §11 puts the rebuild on the weekly schedule alongside
the retrain; until then this is run by hand.

Cups are expected to be missing. At the time of writing every 2026-27 league
schedule is published while every domestic cup and continental file still 404s,
because those draws have not been made. An empty cup is normal, not a failure,
and the script reports what it found per competition rather than failing on
absence.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime

import pandas as pd

from statpitch import paths
from statpitch.data import odds_api
from statpitch.data import openfootball as of
from statpitch.data import openligadb as old
from statpitch.data.http import PoliteSession
from statpitch.env import load_dotenv

log = logging.getLogger("build_fixtures")

#: How far ahead to keep. A full season of fixtures is published at once, and
#: serving the lot would hand a consumer 380 rows per league of which the far end
#: is the least reliable — kickoff times beyond a few months are provisional.
DEFAULT_HORIZON_DAYS = 120

#: How far BACK to keep fixtures, and why this is not zero.
#:
#: A provisional date is a nominal matchday date, and the real kickoffs spread
#: around it — La Liga matchday 1 2026/27 sat on Sunday 16 August and was played
#: from the 14th to the 17th. Filtering to `date >= today` therefore drops
#: fixtures that have not been played yet, because their *placeholder* is in the
#: past while their *actual* kickoff is ahead. That is exactly what happened: the
#: whole of matchday 1 vanished from the artifact on the 17th, and /today
#: returned nothing on a day with real matches.
#:
#: Wider than `collect_fixtures.MAX_DATE_SHIFT_DAYS` (3), so any fixture the
#: correction could still move forward survives long enough to be corrected.
#: Genuinely past fixtures are hidden at request time instead, where
#: /fixtures/upcoming defaults `from` to today.
LOOKBACK_DAYS = 5


def build(
    seasons: list[int], horizon_days: int, lookback_days: int = LOOKBACK_DAYS
) -> pd.DataFrame:
    session = PoliteSession()
    sources = [of.build_all_schedules(seasons, session=session)]

    # openfootball stopped publishing cup files entirely, so the cup competitions
    # come from wherever else they can be had. OpenLigaDB is keyless and covers
    # the DFB-Pokal with confirmed UTC kickoffs and a round label; the other six
    # cups have no free source and are reported as absent per competition.
    sources.append(old.build_all_schedules(seasons, session=session))

    # The six cups nothing free can reach. Costs no credits — the Odds API's
    # /events endpoint is free and only /odds is metered — but it needs a key,
    # and without one this returns nothing and says which competitions that
    # leaves uncovered.
    sources.append(odds_api.build_all_schedules(session=session))

    populated = [f for f in sources if not f.empty]
    if not populated:
        return pd.DataFrame()
    frame = pd.concat(populated, ignore_index=True)

    # A fixture published by two sources keeps the row with a confirmed kickoff.
    # Only one source covers any given competition today, so this is a guard
    # against a future overlap rather than something that currently fires.
    frame = (
        frame.sort_values(["fixture_id", "date_confirmed"])
        .drop_duplicates(subset="fixture_id", keep="last")
    )

    today = pd.Timestamp(datetime.now(UTC).date())
    floor = today - pd.Timedelta(days=lookback_days)
    horizon = today + pd.Timedelta(days=horizon_days)
    upcoming = frame[(frame["date"] >= floor) & (frame["date"] <= horizon)]
    return upcoming.sort_values(["date", "competition_id"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=None,
        help="Season start years. Defaults to the current and next season.",
    )
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Before anything asks whether a credential is configured. `.env.example`
    # has always told people to put one here; nothing read it until now.
    load_dotenv()

    if args.seasons is None:
        # A season is named for the year it starts, and starts in July.
        now = datetime.now(UTC)
        current = now.year if now.month >= 7 else now.year - 1
        args.seasons = [current, current + 1]

    frame = build(args.seasons, args.horizon_days, args.lookback_days)
    if frame.empty:
        log.error("no fixtures found for seasons %s — artifact not written", args.seasons)
        return 1

    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    frame = frame.copy()
    frame["generated_at"] = generated_at

    paths.ensure_dirs()
    destination = paths.fixtures_file()
    frame.to_parquet(destination, index=False)

    by_competition = frame.groupby("competition_id").size().sort_values(ascending=False)
    log.info(
        "wrote %d fixtures to %s (%s .. %s)",
        len(frame), destination,
        frame["date"].min().date(), frame["date"].max().date(),
    )
    log.info("by competition: %s", json.dumps(by_competition.to_dict()))

    log.info("by source: %s", json.dumps(frame.groupby("source").size().to_dict()))

    missing = sorted(set(of.SCHEDULE_SOURCES) - set(by_competition.index))
    if missing:
        # Two different absences, and conflating them is how a dead upstream
        # source hides for a fortnight. A cup with no published draw genuinely
        # has nothing to publish; a cup with no source at all will never publish
        # anything, however long you wait.
        sourced = set(of.SCHEDULE_SOURCES) | set(old.COMPETITIONS)
        undrawn = [c for c in missing if c in sourced]
        unsourced = [c for c in missing if c not in sourced]
        if undrawn:
            log.info("no fixtures published yet for: %s", ", ".join(undrawn))
        if unsourced:
            log.warning(
                "no fixture source at all for: %s — these will stay empty until "
                "one is added, which is not the same as an undrawn round",
                ", ".join(unsourced),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
