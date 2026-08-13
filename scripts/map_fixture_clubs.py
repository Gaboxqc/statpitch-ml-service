"""Map fixture club names onto Club Elo (Roadmap §7).

    python scripts/map_fixture_clubs.py

openfootball writes the formal registered name — "Manchester City FC",
"FC Bayern München" — while Club Elo uses short forms: "Man City", "Bayern".
Without a mapping, a fixture list looks fine and predicts badly: the club falls
through to the pooled entrant prior and the response carries a confident number
built on nothing in particular.

That is not hypothetical. Before this script, 55% of an upcoming five-league
fixture list rated at the pooled prior, including Manchester City, Bayern and
Paris Saint-Germain. `fully_rated` marked every one of them, which is exactly
what MODEL_CARD §6 describes as the failure mode worth catching early: "no
error, no missing field, just a confident wrong number."

The country constraint is doing the work
========================================

Matching is delegated to `club_elo.resolve_cup_clubs`, which indexes the roster
by (country, normalised name) and refuses to guess across borders. The country
comes from the competition_id prefix — ENG.PL, ESP.LALIGA — which is already
spelled the way Club Elo spells countries. Unconstrained, "Union", "Atletico"
and "Racing" each match clubs in half a dozen leagues.

Ambiguity is reported, never resolved
=====================================

A name matching two roster clubs is written to the `ambiguous` section and left
out of the mapping. "FC Bayern München" matches both `Bayern` and `Muenchen 60`;
picking the closest string would attach a title-winning club's rating to a
different club for every fixture it plays. Those cases are fixed by hand in
`club_elo.OPENFOOTBALL_ALIASES` and confirmed against the roster.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import pandas as pd

from statpitch import paths
from statpitch.data import club_elo as ce

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("map_fixture_clubs")

#: Where the mapping lands. Kept apart from `cup_club_elo_map.json` so it stays
#: obvious which source a given alias was needed for, and so rebuilding one
#: cannot silently drop the other.
OUTPUT_NAME = "fixture_club_elo_map.json"

#: Below this, the fixture list is not worth serving predictions from and the
#: run fails rather than writing a mapping that looks complete.
MIN_COVERAGE = 0.95


def main() -> int:
    fixtures_path = paths.fixtures_file()
    if not fixtures_path.exists():
        log.error("no fixtures at %s — run scripts/build_fixtures.py first", fixtures_path)
        return 1

    fixtures = pd.read_parquet(fixtures_path)
    roster = pd.read_parquet(paths.processed_dir() / "clubelo_roster_full.parquet")

    # The competition_id prefix is the Club Elo country code already.
    names: dict[str, str | None] = {}
    for row in fixtures.itertuples():
        country = str(row.competition_id).split(".")[0]
        names[str(row.home_team)] = country
        names[str(row.away_team)] = country

    resolved = ce.resolve_cup_clubs(names, roster)
    mapping = dict(resolved.mapping)

    # Curated entries win: they exist precisely where automatic matching was
    # wrong or ambiguous, and re-deriving them would reintroduce the error.
    curated_used = {
        name: target
        for name, target in ce.OPENFOOTBALL_ALIASES.items()
        if name in names
    }
    mapping.update(curated_used)

    still_missing = sorted(set(names) - set(mapping))
    coverage = len(mapping) / len(names) if names else 0.0

    log.info(
        "%d/%d clubs mapped (%.1f%%) — %d automatic, %d curated",
        len(mapping), len(names), coverage * 100,
        len(resolved.mapping), len(curated_used),
    )
    if resolved.ambiguous:
        log.info(
            "ambiguous, resolved by hand where used: %s",
            json.dumps({k: list(v) for k, v in resolved.ambiguous.items()}),
        )
    if still_missing:
        log.warning(
            "%d club(s) still unmapped — they will rate at the pooled prior and "
            "report fully_rated=false: %s",
            len(still_missing), ", ".join(still_missing),
        )

    if coverage < MIN_COVERAGE:
        log.error(
            "coverage %.1f%% is below the %.0f%% floor; add the missing clubs to "
            "club_elo.OPENFOOTBALL_ALIASES rather than shipping this mapping",
            coverage * 100, MIN_COVERAGE * 100,
        )
        return 1

    destination = paths.processed_dir() / OUTPUT_NAME
    destination.write_text(
        json.dumps(
            {
                "matched": dict(sorted(mapping.items())),
                "ambiguous": {k: list(v) for k, v in resolved.ambiguous.items()},
                "unmatched": still_missing,
                "stats": {
                    "clubs": len(names),
                    "matched": len(mapping),
                    "coverage": round(coverage, 4),
                    "automatic": len(resolved.mapping),
                    "curated": len(curated_used),
                },
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("wrote %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
