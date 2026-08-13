"""Mark a registered artifact as the one to serve (Roadmap §1.2).

    python scripts/promote_model.py --list
    python scripts/promote_model.py goals-20260813-36b455c0

Separate from training on purpose. `scripts/train.py` records an artifact and its
scores and stops; promotion is the decision that it should be served, and a
pipeline that makes that decision automatically is a mechanism for shipping a
regression quietly. Roadmap §11.2 replaces this manual step with a gate that
compares against the incumbent across folds — the separation is what gives that
gate somewhere to say no.

Exactly one entry is promoted at a time. Two would make "which model produced
this number" unanswerable, which is the question the registry exists to answer.
"""

from __future__ import annotations

import argparse
import logging

from statpitch import paths
from statpitch.models import registry

log = logging.getLogger("promote")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?", help="Registered version to promote.")
    parser.add_argument("--list", action="store_true", help="Show the registry.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    store = registry.Registry.load(paths.models_dir())
    if not store.entries:
        log.error("registry is empty — run scripts/train.py first")
        return 1

    if args.list or not args.version:
        current = store.promoted
        for entry in store.entries:
            summary = entry.metrics.get("walk_forward", {})
            card = entry.metrics.get("model_card_window") or {}
            log.info(
                "%s %s  log_loss %.4f +/- %.4f over %s folds%s%s",
                "*" if entry.promoted else " ",
                entry.version,
                summary.get("mean_log_loss", float("nan")),
                summary.get("std_log_loss", float("nan")),
                summary.get("folds", "?"),
                "" if not entry.git_dirty else "  [dirty tree]",
                "" if card.get("reproduces", True) else "  [does not reproduce card]",
            )
        log.info("promoted: %s", current.version if current else "none")
        return 0 if args.list else 1

    entry = store.promote(args.version)
    store.save()
    log.info("promoted %s (trained %s, %d rows)",
             entry.version, entry.created_at, entry.n_train_rows)
    if entry.git_dirty:
        log.warning(
            "%s was built from a dirty tree and cannot be reproduced from its "
            "recorded commit alone", entry.version,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
