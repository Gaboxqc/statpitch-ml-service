"""Train the goal model end to end and register the artifact (Roadmap §1).

    python scripts/train.py [--dry-run] [--params '{"max_depth": 5}'] [--notes "..."]

Before this script the project had a real model and no way to produce one:
`GoalModel.fit` was called only from `tests/test_goals.py`, nothing was saved,
and the deployed API served a different inference path entirely. One command now
goes from `features.parquet` to a versioned, registered artifact.

The logic lives in `statpitch.models.training`; this file is argument parsing,
file IO and registry bookkeeping. Everything with a correctness requirement is
importable and tested.

**Registering is not promoting.** This script never promotes. An artifact is
recorded with its scores and left for a deliberate decision, because a pipeline
that promotes whatever it just built is a mechanism for shipping a regression
quietly — which is the thing Roadmap §11.2's gate exists to prevent. Promote with
`scripts/promote_model.py` once the numbers have been looked at.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime

import pandas as pd

from statpitch import decision_config, paths
from statpitch.features import build as fb
from statpitch.models import registry, training
from statpitch.models.goals import DEFAULT_PARAMS

log = logging.getLogger("train")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate and report; write no artifact.")
    parser.add_argument("--params", default=None,
                        help="JSON overrides for the XGBoost parameters.")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    features_path = paths.processed_dir() / "features.parquet"
    if not features_path.exists():
        log.error("no features at %s", features_path)
        return 1

    frame = fb.drop_burn_in(pd.read_parquet(features_path))
    frame = frame[frame["result"].notna()].reset_index(drop=True)

    holdout = decision_config.config().benchmark.holdout_season
    seasons = training.eligible_seasons(frame, holdout)
    training.assert_holdout_untouched(seasons, holdout)
    if len(seasons) <= training.DEFAULT_MIN_TRAIN_SEASONS:
        log.error("only %d eligible seasons; too few for a fold", len(seasons))
        return 1

    columns = fb.feature_columns(frame)
    params = {**DEFAULT_PARAMS, **(json.loads(args.params) if args.params else {})}
    eligible = frame[frame["season"].isin(seasons)].reset_index(drop=True)

    log.info(
        "training on %d rows, %d features, seasons %s..%s (holdout %s excluded)",
        len(eligible), len(columns), seasons[0], seasons[-1], holdout,
    )

    folds = training.walk_forward(eligible, seasons, columns, params)
    summary = training.aggregate(folds)
    if not summary:
        log.error("no fold produced a usable score")
        return 1
    log.info(
        "walk-forward: log_loss %.4f +/- %.4f over %d folds (%d rows)",
        summary["mean_log_loss"], summary["std_log_loss"],
        summary["folds"], summary["total_validation_rows"],
    )

    card = training.card_comparison(folds)
    if card is not None:
        log.info(
            "MODEL_CARD window %s: log_loss %.4f vs published %.4f (%+.4f) — %s. "
            "Market %.4f, gap %+.4f.",
            "+".join(card["seasons"]), card["log_loss"], card["card_log_loss"],
            card["difference"],
            "reproduces" if card["reproduces"] else "DOES NOT REPRODUCE",
            card["market_log_loss"], card["gap_to_market"],
        )
        if not card["reproduces"]:
            log.warning(
                "the published figure was not reproduced within %.3f. That is a "
                "finding about this pipeline or the card, not a rounding issue — "
                "do not promote until it is explained.",
                training.CARD_TOLERANCE,
            )

    # The shipped artifact sees every eligible season. The folds above are how it
    # is judged; this is what would get served.
    final = training.fit_model(eligible, columns, params)
    in_sample = training.score(final, eligible)
    log.info(
        "final model in-sample log_loss %.4f (not a generalisation estimate)",
        in_sample["log_loss"],
    )

    if args.dry_run:
        log.info("dry run — no artifact written")
        return 0

    sha, dirty = registry.git_sha()
    version = registry.version_for(sha)
    destination = paths.models_dir() / version
    final.save(destination)

    entry = registry.Entry(
        version=version,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=sha,
        git_dirty=dirty,
        train_seasons=seasons,
        validation_seasons=[f["validation_season"] for f in folds],
        holdout_season=holdout,
        holdout_touched=False,
        feature_columns=columns,
        n_features=len(columns),
        n_train_rows=int(len(eligible)),
        params=params,
        input_checksums={features_path.name: registry.checksum(features_path)},
        metrics={
            "walk_forward": summary,
            "model_card_window": card,
            "folds": folds,
            "final_in_sample": in_sample,
        },
        notes=args.notes,
    )

    store = registry.Registry.load(paths.models_dir())
    store.add(entry)
    store.save()

    log.info("wrote %s and registered %s (not promoted)", destination, version)
    if dirty:
        log.warning(
            "the working tree was dirty; this artifact cannot be reproduced from "
            "%s alone, and the registry records that", sha[:8],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
