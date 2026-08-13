"""Publish a registered artifact as a GitHub Release asset (Roadmap §10.1).

    python scripts/publish_model.py                 # the promoted model
    python scripts/publish_model.py goals-2026...   # a specific version

Boosters are gitignored at ~1.7 MB a run, so a fresh checkout has a registry
describing artifacts nobody has. Publishing closes that gap: the weekly refresh
downloads the model instead of spending 2.5 minutes retraining one that has not
changed.

Only registered versions can be published. An artifact with no registry entry has
no recorded training window, metrics or input checksums, and a release asset
without those is a binary nobody can interpret later.
"""

from __future__ import annotations

import argparse
import json
import logging

from statpitch import paths
from statpitch.models import registry, release

log = logging.getLogger("publish")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?", help="Defaults to the promoted model.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    store = registry.Registry.load(paths.models_dir())
    if args.version:
        entry = store.get(args.version)
    else:
        entry = store.promoted
        if entry is None:
            log.error(
                "nothing is promoted, so there is no default to publish. Name a "
                "version, or promote one with scripts/promote_model.py."
            )
            return 1

    if entry.git_dirty:
        # Not refused: a run that produced numbers is worth keeping. But a
        # published asset outlives the working tree that made it, and nobody
        # downloading it later can tell it was unreproducible unless it says so.
        log.warning(
            "%s was built from a dirty tree and cannot be rebuilt from %s alone",
            entry.version, entry.git_sha[:8],
        )

    summary = entry.metrics.get("walk_forward", {})
    notes = "\n".join([
        f"Trained {entry.created_at} from `{entry.git_sha[:8]}`"
        + (" (dirty tree)" if entry.git_dirty else ""),
        "",
        f"- seasons `{entry.train_seasons[0]}`..`{entry.train_seasons[-1]}`, "
        f"holdout `{entry.holdout_season}` excluded",
        f"- {entry.n_train_rows} rows, {entry.n_features} features",
        f"- walk-forward log-loss {summary.get('mean_log_loss', float('nan')):.4f} "
        f"± {summary.get('std_log_loss', float('nan')):.4f} "
        f"over {summary.get('folds', '?')} folds",
        "",
        "Input checksums:",
        "```json",
        json.dumps(entry.input_checksums, indent=2),
        "```",
    ])

    tag = release.publish(entry.version, notes=notes)
    log.info("published %s as %s", entry.version, tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
