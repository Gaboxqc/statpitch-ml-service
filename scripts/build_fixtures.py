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
from statpitch.data import openfootball as of
from statpitch.data.http import PoliteSession

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
    frame = of.build_all_schedules(seasons, session=session)
    if frame.empty:
        return frame

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

    missing = sorted(set(of.SCHEDULE_SOURCES) - set(by_competition.index))
    if missing:
        # Not an error: a cup with no published draw has no fixtures to publish.
        log.info("no fixtures published yet for: %s", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
