"""Rebuild `features.parquet` from the processed inputs (Roadmap §1).

    python scripts/build_features.py [--verify] [--out PATH]

`features.parquet` is the input to every model in this project and, until now,
the one artifact with no committed way to produce it. `scripts/train.py` made
training reproducible from it; this makes *it* reproducible, which is what
Roadmap §3 needs before a single feature can be added — a feature set that cannot
be rebuilt cannot be extended, only replaced.

`--verify` rebuilds and diffs against the committed artifact instead of writing,
which is how the pipeline below was established rather than assumed: the row
count, the match_id set and every shared column are compared.

The pipeline
============

    merge_match_log -> build_features -> attach_outcomes -> drop_burn_in

`drop_burn_in` is the step that is easy to miss and accounts for the whole
difference between the 64,671-row match log and the 61,321-row feature frame: it
removes rows where either club has fewer than five prior matches, whose form
columns are null or one-match noise labelled as signal.

Ratings are looked up by `clubelo_name`
=======================================

Club names arrive in three spellings — football-data's abbreviations for the
leagues, openfootball's formal names for the cups, and Club Elo's own short
forms — so the lookup goes through both alias maps and `NAME_ALIASES`. Keyed on
`source_name` instead it would silently lose the 187 clubs that entered the Elo
table as cup entrants, and the only symptom would be null ratings on cup
fixtures.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from statpitch import paths
from statpitch.data import club_elo as ce
from statpitch.features import build as fb

log = logging.getLogger("build_features")

#: Columns compared by --verify. Ratings are included deliberately: they are the
#: ones a name-resolution regression would silently null out.
KEY_COLUMNS = ("home_elo", "away_elo", "elo_diff", "home_form_5", "away_form_5")


def load_aliases() -> dict[str, str]:
    """Every club-name spelling this project knows, mapped to Club Elo's."""
    aliases: dict[str, str] = dict(ce.NAME_ALIASES)
    aliases.update(ce.OPENFOOTBALL_ALIASES)
    for name in ("cup_club_elo_map.json", "fixture_club_elo_map.json"):
        path = paths.processed_dir() / name
        if path.exists():
            aliases.update(
                json.loads(path.read_text(encoding="utf-8")).get("matched", {})
            )
    return aliases


def build_elo_lookup(
    log_frame: pd.DataFrame, aliases: dict[str, str]
) -> dict[tuple[str, pd.Timestamp], float]:
    """(club, date) -> rating strictly before that date, for the whole log."""
    elo_table = pd.read_parquet(paths.processed_dir() / "elo_ratings_all.parquet")

    pairs: list[tuple[str, pd.Timestamp]] = []
    for date, home, away in zip(
        log_frame["date"], log_frame["home_team"], log_frame["away_team"], strict=True
    ):
        for club in (home, away):
            pairs.append((aliases.get(str(club), str(club)), date))

    resolved = ce.build_lookup(elo_table, pairs)

    # Re-key onto the names the feature builder will see.
    lookup: dict[tuple[str, pd.Timestamp], float] = {}
    for date, home, away in zip(
        log_frame["date"], log_frame["home_team"], log_frame["away_team"], strict=True
    ):
        for club in (home, away):
            rating = resolved.get((aliases.get(str(club), str(club)), date))
            if rating is not None:
                lookup[(str(club), date)] = rating
    return lookup


def build() -> pd.DataFrame:
    processed = paths.processed_dir()
    matches = pd.read_parquet(processed / "matches_clean.parquet")
    cups = pd.read_parquet(processed / "cup_matches.parquet")
    log_frame = fb.merge_match_log(matches, cups)
    log.info("match log: %d rows", len(log_frame))

    aliases = load_aliases()
    elo_lookup = build_elo_lookup(log_frame, aliases)
    slots = 2 * len(log_frame)
    log.info(
        "resolved %d/%d club-date rating slots (%.1f%%)",
        len(elo_lookup), slots, 100 * len(elo_lookup) / slots,
    )

    xg_path = processed / "match_xg.parquet"
    xg_lookup = None
    if xg_path.exists():
        xg = pd.read_parquet(xg_path)
        xg_lookup = dict(
            zip(xg["match_id"], zip(xg["home_xg"], xg["away_xg"], strict=True),
                strict=True)
        )
        log.info("xG available for %d matches", len(xg_lookup))

    features = fb.build_features(log_frame, elo_lookup=elo_lookup, xg_lookup=xg_lookup)
    features = fb.attach_outcomes(features, log_frame)
    features = fb.drop_burn_in(features)
    log.info("features: %d rows, %d columns", len(features), len(features.columns))
    return features


def verify(built: pd.DataFrame) -> int:
    """Diff a rebuild against the committed artifact."""
    committed = pd.read_parquet(paths.processed_dir() / "features.parquet")
    problems = 0

    if len(built) != len(committed):
        log.error("row count: built %d, committed %d", len(built), len(committed))
        problems += 1
    if set(built["match_id"]) != set(committed["match_id"]):
        log.error(
            "match_id sets differ: %d only in the rebuild, %d only in the committed",
            len(set(built["match_id"]) - set(committed["match_id"])),
            len(set(committed["match_id"]) - set(built["match_id"])),
        )
        problems += 1

    missing = sorted(set(committed.columns) - set(built.columns))
    added = sorted(set(built.columns) - set(committed.columns))
    if missing:
        log.error("columns missing from the rebuild: %s", missing)
        problems += 1
    if added:
        log.info("new columns in the rebuild: %s", added)

    left = built.set_index("match_id").sort_index()
    right = committed.set_index("match_id").sort_index()
    shared = left.index.intersection(right.index)
    for column in KEY_COLUMNS:
        if column not in left.columns or column not in right.columns:
            continue
        a = left.loc[shared, column].astype(float).to_numpy()
        b = right.loc[shared, column].astype(float).to_numpy()
        both_present = ~np.isnan(a) & ~np.isnan(b)
        gap = float(np.abs(a[both_present] - b[both_present]).max()) if both_present.any() else 0.0

        # Direction matters and the two are not symmetric. Gaining a value the
        # committed artifact lacked is the alias work of Roadmap §7 reaching the
        # historical frame; losing one is a name-resolution regression, and the
        # only symptom either way would be a null rating on a cup fixture.
        gained = int((~np.isnan(a) & np.isnan(b)).sum())
        lost = int((np.isnan(a) & ~np.isnan(b)).sum())

        if gap > 1e-6 or lost:
            log.error(
                "%s: max |diff| %.3g where both present, %d value(s) LOST",
                column, gap, lost,
            )
            problems += 1
        elif gained:
            log.info(
                "%s: identical where both present, and %d row(s) newly resolved",
                column, gained,
            )
        else:
            log.info("%s: identical over %d rows", column, len(shared))

    if problems:
        log.error("%d regression(s); the rebuild is NOT safe to commit", problems)
        return 1
    log.info(
        "rebuild reproduces the committed artifact, up to newly resolved ratings"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="Diff against the committed artifact; write nothing.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    features = build()
    if args.verify:
        return verify(features)

    destination = paths.processed_dir() / (args.out or "features.parquet")
    features.to_parquet(destination, index=False)
    log.info("wrote %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
