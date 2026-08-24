"""Build the matchday card from the current fixtures, predictions and prices.

    python scripts/build_card.py [--dry-run]

Offline, like every other artifact. Serving reads `card.parquet`; it does not
derive 86 selections per fixture and solve a joint Kelly allocation on a request
path, which is both outside NFR-2's ~200 ms budget and the reason the market
engine is absent from `requirements-serving.txt`'s import surface.

Order matters: this runs after `collect_live_odds` (which supplies the prices)
and after `precompute_predictions` (which supplies the goal rates), so it is the
last step of `refresh_fixtures`.

Expect an empty card today, with a reason
=========================================

`w` fits at 0.000, so the shrunk probability is the market's own and `model_edge`
is zero for every selection. `decision_config` is a placeholder, so
`StakingEngine` refuses to size anything. Both are real findings rather than
missing work, and the card records them per row instead of returning nothing.

That distinction is the whole point of the exercise: before this, `/card/today`
returned a hardcoded empty list, which is the same JSON as a computed empty card
and a completely different fact.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from statpitch import decision_config, paths
from statpitch.decision import card as card_builder

log = logging.getLogger("build_card")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report, write nothing.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    processed = paths.processed_dir()
    required = {
        "fixtures": paths.fixtures_file(),
        "predictions": processed / "predictions.parquet",
        "odds": paths.live_odds_file(),
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        log.error(
            "cannot build a card without %s — run refresh_fixtures first",
            " and ".join(missing),
        )
        return 1

    fixtures = pd.read_parquet(required["fixtures"])
    predictions = pd.read_parquet(required["predictions"])
    odds = pd.read_parquet(required["odds"])
    config = decision_config.config()

    model_version = "unknown"
    if "model_version" in predictions.columns and not predictions.empty:
        model_version = str(predictions["model_version"].iloc[0])

    card, stats = card_builder.build_card(
        fixtures, predictions, odds, config, model_version=model_version
    )

    log.info("card: %s", json.dumps(stats.as_dict()))
    if stats.arbitrage_fixtures:
        log.warning(
            "%d fixture(s) quote a best-of-N book summing below 1.0, so those "
            "prices were not all available at once and any edge computed from "
            "them is overstated. Carried as `max_book_sum` for Phase C: %s",
            len(stats.arbitrage_fixtures), ", ".join(stats.arbitrage_fixtures[:3]),
        )

    if config.is_placeholder:
        log.warning(
            "nothing staked: decision_config '%s' is unfitted (status=%s, w=%s). "
            "The card is computed in full and every stake_fraction is 0.0.",
            config.config_version, config.status, config.w,
        )

    if card.empty:
        log.warning("card is empty — no fixture had both a prediction and a priced market")

    if args.dry_run:
        log.info("dry run — nothing written")
        return 0

    paths.ensure_dirs()
    destination = paths.card_file()
    card.to_parquet(destination, index=False)
    log.info("wrote %d row(s) to %s", len(card), destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
