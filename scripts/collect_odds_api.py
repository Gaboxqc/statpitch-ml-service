"""Capture prices from The Odds API — 1X2 daily, everything on a matchday.

    python scripts/collect_odds_api.py [--dry-run] [--markets h2h totals spreads]

Why this exists alongside `collect_live_odds.py`
================================================

football-data.co.uk is free and keyless and covers the five leagues, but it
publishes **one matchday block at a time**: between blocks it lists nothing but
matches already played, so midweek there is no price for anything. It also never
covered the cups.

This fills both gaps, and carries one thing football-data.co.uk does not.

**Pinnacle.** MODEL_CARD §5's +0.51% CLV (t=+7.53 clustered, five pre-break
seasons, 7,790 matches) is defined on Pinnacle-referenced selections, and Phase C
recorded the blocker as *"Pinnacle is not published in the live fixture feed"* —
so the only rule with multi-season evidence could be measured backwards and never
traded forwards. This API publishes Pinnacle. That removes the blocker Phase C
named; it does not by itself fit the config, which stays a deliberate decision.

The budget, which is the whole design constraint
================================================

500 credits a **month**. Billing is per request per market per region — not per
fixture — so twenty fixtures cost exactly what one does, and the cost is driven
entirely by how many competitions x markets are asked for.

    daily 1X2 sweep, competitions with fixtures      ~7 credits   ~210/month
    + totals and handicaps, matchday competitions    ~2-6 extra   ~90/month
                                                                  ~300/month

`/events` is free, so the sweep asks which competitions have anything upcoming
before spending a credit on any of them. Five of twelve had nothing when this was
written; asking about them anyway would have burned 150 credits a month to learn
that a cup is not being played.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from statpitch import paths
from statpitch.data import football_data_live as live
from statpitch.data import odds_api
from statpitch.env import load_dotenv

log = logging.getLogger("collect_odds_api")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report, write nothing.")
    parser.add_argument(
        "--markets", nargs="+", default=None,
        help="Override the daily/matchday split and ask for these everywhere.",
    )
    parser.add_argument(
        "--competitions", nargs="+", default=None,
        help="Limit to these competition ids.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()

    if not odds_api.configured():
        log.error(
            "%s is not set. Put it in .env — see .env.example.", odds_api.ENV_KEY
        )
        return 1

    fixtures_path = paths.fixtures_file()
    fixtures = pd.read_parquet(fixtures_path) if fixtures_path.exists() else None
    if fixtures is None:
        log.warning("no fixture list — every competition will be treated as non-matchday")

    budget = odds_api.Budget()
    cid = live.capture_id()
    wanted = args.competitions or list(odds_api.SPORT_KEYS)

    # /events is free, so find out where the fixtures are before spending
    # anything. Asking a cup that is not being played costs a credit and learns
    # nothing.
    with_events: list[str] = []
    for competition_id in wanted:
        events = odds_api.fetch_events(competition_id, budget=budget)
        if events:
            with_events.append(competition_id)
    log.info(
        "%d/%d competition(s) have upcoming events (free to ask): %s",
        len(with_events), len(wanted), ", ".join(with_events) or "none",
    )
    if not with_events:
        log.info("nothing upcoming anywhere; no credits spent")
        return 0

    frames: list[pd.DataFrame] = []
    plan: dict[str, list[str]] = {}
    for competition_id in with_events:
        markets = (
            tuple(args.markets) if args.markets
            else odds_api.markets_for(competition_id, fixtures)
        )
        plan[competition_id] = list(markets)
        payload = odds_api.fetch_odds(competition_id, markets=markets, budget=budget)
        if not payload:
            continue
        frame = odds_api.parse_odds(payload, competition_id, cid=cid)
        if not frame.empty:
            frames.append(frame)
            log.info(
                "%s — %d selection row(s) over %d fixture(s), markets %s",
                competition_id, len(frame),
                frame.drop_duplicates(subset=["fd_home", "fd_away"]).shape[0],
                ",".join(markets),
            )

    log.info("plan: %s", json.dumps(plan))
    log.info("budget: %s", json.dumps(budget.describe()))

    if not frames:
        log.warning("no prices returned; nothing to write")
        return 0

    priced = pd.concat(frames, ignore_index=True)

    resolutions = live.resolve_all(priced, fixtures) if fixtures is not None else {}
    for competition_id, resolution in sorted(resolutions.items()):
        if resolution.unmatched:
            log.warning(
                "%s: %d club(s) unmatched — their prices cannot be keyed to a "
                "fixture: %s",
                competition_id, len(resolution.unmatched),
                ", ".join(resolution.unmatched),
            )
    mapping = {c: r.mapping for c, r in resolutions.items()}
    keyed, stats = live.attach_fixture_ids(priced, fixtures, mapping)
    log.info("keyed %d/%d row(s): %s", stats["keyed"], stats["priced"], json.dumps(stats))

    if args.dry_run:
        log.info("dry run — nothing written")
        return 0

    if keyed.empty:
        log.warning("nothing keyed to a fixture; not appending")
        return 0

    paths.ensure_dirs()
    destination, appended = live.append_snapshot(keyed)
    log.info("appended %d row(s) to %s", appended, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
