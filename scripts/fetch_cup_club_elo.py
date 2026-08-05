"""Fetch Club Elo histories for cup entrants that resolved to a rated club.

Separate from the league fetch because it is incremental and restartable: the
first attempt built one large in-memory frame and was killed part-way, losing
everything. This writes each club's history to its own parquet shard as it
arrives, so an interrupted run resumes instead of restarting.
"""

from __future__ import annotations

import json
import logging
import sys

import pandas as pd

from statpitch import paths
from statpitch.data import club_elo as ce
from statpitch.data.http import PoliteSession

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("fetch_cup_club_elo")


def main() -> int:
    shard_dir = paths.processed_dir() / "elo_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    existing = pd.read_parquet(paths.elo_file())
    have = set(existing["clubelo_name"])

    mapping = json.loads(
        (paths.processed_dir() / "cup_club_elo_map.json").read_text(encoding="utf-8")
    )["matched"]

    todo = {src: tgt for src, tgt in mapping.items() if tgt not in have}
    targets = sorted(set(todo.values()))
    print(f"already have {len(have)} clubs; fetching {len(targets)}", flush=True)

    session = PoliteSession(min_interval=0.2)
    fetched = 0
    for i, target in enumerate(targets, 1):
        shard = shard_dir / f"{ce.club_slug(target)}.parquet"
        if shard.exists():
            continue
        try:
            history = ce.fetch_club_history(target, session=session)
        except Exception as exc:  # noqa: BLE001 - a missing club must not stop the run
            log.warning("no history for %s: %s", target, exc)
            continue
        history.to_parquet(shard, index=False)
        fetched += 1
        if i % 50 == 0:
            print(f"  {i}/{len(targets)}", flush=True)

    shards = sorted(shard_dir.glob("*.parquet"))
    frames = [existing] + [pd.read_parquet(s) for s in shards]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["clubelo_name", "valid_from"])
    combined = combined.sort_values(["clubelo_name", "valid_from"]).reset_index(drop=True)

    out = paths.processed_dir() / "elo_ratings_all.parquet"
    combined.to_parquet(out, index=False)
    print(
        f"fetched {fetched} new; combined {len(combined)} rows across "
        f"{combined['clubelo_name'].nunique()} clubs -> {out.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
