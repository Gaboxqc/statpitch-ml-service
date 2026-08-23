"""Capture pre-match prices for the upcoming fixture list (Plan §4 Phase A).

    python scripts/collect_live_odds.py [--dry-run] [--no-correct-dates]

Free and keyless. `football-data.co.uk/fixtures.csv` is the same publisher and
the same modern-era schema as the archive the model was measured against, which
is what lets a Friday price and an `AvgC*` close be compared without
`clv_tracker` refusing the pair as cross-source.

Run it on a schedule, not once
==============================

Each run appends a capture; it never replaces one. Two captures of the same
selection are what a CLV number is made of, so the value of this job is in the
series, and a run that overwrites its predecessor produces a file that looks
current and measures nothing.

The intended cadence is a Friday-afternoon snapshot — matching the baseline
MODEL_CARD §5's +0.51% was measured on — plus one close to kickoff. Both come
from the same source, so both ends of the measurement share a ruler.

What it writes
==============

`data/processed/live_odds.parquet`, keyed on the fixture list's own
`fixture_id`, with `selection_key` already mapped onto `market_engine`'s
namespace so Phase B is a lookup rather than a translation.

Date correction, for free
=========================

The odds rows carry a bookmaker-confirmed kickoff, which is exactly what
`collect_fixtures.py` spends an API-Football or football-data.org call to get —
and which that collector cannot get at all on API-Football's free plan, because
it does not cover the current season. Any fixture priced here therefore arrives
with a confirmed date attached, so the correction is applied to
`fixtures.parquet` as a side effect. `--no-correct-dates` turns that off.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime

import pandas as pd

from statpitch import paths
from statpitch.data import football_data_live as live

log = logging.getLogger("collect_live_odds")

#: Below this share of priced fixtures resolving to a fixture_id, the run still
#: writes but says loudly that the club map has fallen behind — a promoted club
#: nobody has aliased looks exactly like a quiet 5% coverage drop.
MIN_KEYED_SHARE = 0.90


def _write_club_map(resolutions: dict[str, live.ClubResolution]) -> None:
    """Record the reconciliation beside the other name maps, for audit.

    Written every run rather than cached and reused: the fixture list changes
    with promotion and relegation, and a stale map is how a club silently stops
    being priced.
    """
    destination = paths.processed_dir() / "fixture_odds_map.json"
    payload = {
        "matched": {
            competition: dict(sorted(resolution.mapping.items()))
            for competition, resolution in sorted(resolutions.items())
        },
        "curated": {
            competition: dict(sorted(resolution.curated.items()))
            for competition, resolution in sorted(resolutions.items())
            if resolution.curated
        },
        "unmatched": {
            competition: resolution.unmatched
            for competition, resolution in sorted(resolutions.items())
            if resolution.unmatched
        },
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log.info("wrote %s", destination)


def correct_dates(fixtures: pd.DataFrame, keyed: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply bookmaker-confirmed kickoffs to the fixture list.

    One row per fixture is enough — every selection for a fixture carries the
    same kickoff — so the odds frame is reduced before the join. A fixture with
    no parseable time keeps its provisional date rather than being moved to
    midnight.
    """
    confirmed = (
        keyed[keyed["kickoff_utc"].notna()]
        .drop_duplicates(subset="fixture_id")
        .set_index("fixture_id")[["kickoff_utc"]]
    )
    stats = {"confirmed": 0, "moved": 0}
    if confirmed.empty:
        return fixtures, stats

    out = fixtures.copy()
    for column, default in (("date_confirmed", False), ("kickoff", pd.NA)):
        if column not in out.columns:
            out[column] = default

    for index, fixture_id in out["fixture_id"].items():
        if fixture_id not in confirmed.index:
            continue
        stamp = pd.Timestamp(confirmed.loc[fixture_id, "kickoff_utc"])
        if pd.isna(stamp):
            continue
        if stamp.normalize() != out.at[index, "date"]:
            stats["moved"] += 1
        out.at[index, "date"] = stamp.normalize()
        out.at[index, "kickoff"] = stamp.strftime("%H:%M")
        out.at[index, "date_confirmed"] = True
        stats["confirmed"] += 1
    return out, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and report, but write nothing.",
    )
    parser.add_argument(
        "--no-correct-dates", action="store_true",
        help="Leave fixtures.parquet alone even where a confirmed kickoff exists.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    fixtures_path = paths.fixtures_file()
    if not fixtures_path.exists():
        log.error("no fixtures — run scripts/build_fixtures.py first")
        return 1
    fixtures = pd.read_parquet(fixtures_path)

    paths.ensure_dirs()
    path, cid = live.fetch()
    odds = live.parse(path, cid=cid)
    if odds.empty:
        log.warning(
            "no priced fixtures in capture %s for any competition in the "
            "taxonomy — normal outside a matchday window, and nothing is written",
            cid,
        )
        return 0

    by_competition = odds.drop_duplicates(
        subset=["competition_id", "fd_home", "fd_away"]
    ).groupby("competition_id").size()
    log.info(
        "capture %s: %d priced fixtures, %d selection rows — %s",
        cid, int(by_competition.sum()), len(odds),
        json.dumps(by_competition.to_dict()),
    )

    resolutions = live.resolve_all(odds, fixtures)
    for competition, resolution in sorted(resolutions.items()):
        log.info(
            "%s — %d/%d clubs mapped (%d curated)%s",
            competition, len(resolution.mapping),
            len(resolution.mapping) + len(resolution.unmatched),
            len(resolution.curated),
            f"; UNMATCHED: {', '.join(resolution.unmatched)}"
            if resolution.unmatched else "",
        )
        if resolution.unmatched:
            log.warning(
                "%s: %d club(s) have prices that cannot be keyed to a fixture. "
                "Add them to football_data_live.CLUB_ALIASES after checking the "
                "spelling in fixtures.parquet — do not guess.",
                competition, len(resolution.unmatched),
            )

    mapping = {competition: r.mapping for competition, r in resolutions.items()}
    keyed, stats = live.attach_fixture_ids(odds, fixtures, mapping)
    share = stats["keyed"] / stats["priced"] if stats["priced"] else 0.0
    log.info(
        "keyed %d/%d selection rows (%.1f%%) — %d dropped for an unmapped club, "
        "%d for a fixture not in the list",
        stats["keyed"], stats["priced"], share * 100,
        stats["unmapped_club"], stats["unlisted"],
    )
    if share < MIN_KEYED_SHARE:
        log.warning(
            "only %.1f%% of priced rows keyed, below the %.0f%% floor — the club "
            "map or the fixture list is behind the season",
            share * 100, MIN_KEYED_SHARE * 100,
        )

    shifted = keyed[keyed["date_shift_days"].fillna(0) != 0]
    if not shifted.empty:
        moved = shifted.drop_duplicates(subset="fixture_id")
        log.info(
            "%d fixture(s) are priced for a different day than the list has them "
            "on — openfootball's provisional matchday, which the odds confirm",
            len(moved),
        )

    if args.dry_run:
        log.info("dry run — nothing written")
        return 0

    _write_club_map(resolutions)
    destination, appended = live.append_snapshot(keyed)
    log.info("appended %d row(s) to %s", appended, destination)

    if not args.no_correct_dates:
        fixtures, date_stats = correct_dates(fixtures, keyed)
        fixtures.to_parquet(fixtures_path, index=False)
        log.info(
            "fixtures: %d kickoff(s) confirmed from the odds feed, %d moved to a "
            "different day; wrote %s",
            date_stats["confirmed"], date_stats["moved"], fixtures_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
