"""Pre-registered ablation of the momentum features (Roadmap §3, §1.5).

    python scripts/ablate_features.py [--frame features.parquet]

Three feature groups were added. The question is whether any of them carries
information the model did not already have, and the honest way to ask it is
decided **before** the answer is known.

The pre-registration
====================

`HYPOTHESES` below is the family. Three tests, one per group, each comparing
walk-forward log-loss against the same baseline. The family-wise error rate is
controlled at 0.05 by Holm–Bonferroni, which Roadmap §1.5 fixed in advance for a
reason worth restating: testing twenty features at p < 0.05 against one
validation window manufactures roughly one false positive per twenty tests, and
that is exactly how a recorded null result gets overturned by accident.

Holm rather than plain Bonferroni because it is uniformly more powerful at the
same guarantee — there is no reason to be more conservative than necessary when
the prior is already pessimistic.

Why paired, per fold
====================

Each group is scored on the *same* folds as the baseline, and the test is on the
per-fold differences. Season-to-season variation is large — the fold spread is
~0.018 — and dwarfs any plausible feature effect, so an unpaired comparison would
be measuring which seasons landed in validation. Pairing removes it.

The combined configuration is reported alongside but is **not** part of the
family: it is not an independent hypothesis, and counting it would inflate the
correction against the three that are.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from scipy import stats

from statpitch import decision_config, paths
from statpitch.features import build as fb
from statpitch.models import training

log = logging.getLogger("ablate")

#: The pre-registered family. Prefixes, matched against column names.
HYPOTHESES: dict[str, tuple[str, ...]] = {
    "result_streaks": (
        "home_win_streak", "away_win_streak", "home_loss_streak", "away_loss_streak",
        "home_unbeaten_run", "away_unbeaten_run", "home_winless_run",
        "away_winless_run", "home_since_win", "away_since_win", "home_since_loss",
        "away_since_loss", "unbeaten_run_diff",
    ),
    "opponent_strength": (
        "home_opponent_elo_5", "away_opponent_elo_5", "home_opponent_elo_10",
        "away_opponent_elo_10", "opponent_elo_diff_5", "opponent_elo_diff_10",
    ),
    "elo_momentum": (
        "home_elo_delta_5", "away_elo_delta_5", "home_elo_delta_10",
        "away_elo_delta_10", "elo_delta_diff_5", "elo_delta_diff_10",
    ),
}

FAMILY_ALPHA = 0.05

#: Folds are restricted to the seasons the model is actually judged on. The full
#: 28-fold sweep reaches back to 1996, and four configurations across it is an
#: hour of compute to answer a question about the modern game. Stated rather than
#: quietly assumed, because it does change the baseline's absolute level.
FIRST_VALIDATION_SEASON = "2014-2015"


def holm(p_values: dict[str, float], alpha: float = FAMILY_ALPHA) -> dict[str, dict]:
    """Holm–Bonferroni: sort ascending, compare p_(i) against alpha/(m−i)."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, dict] = {}
    still_rejecting = True
    for index, (name, p) in enumerate(ordered):
        threshold = alpha / (m - index)
        # Once one test fails, every larger p-value fails too — that step-down is
        # what makes Holm valid, and skipping it would be plain Bonferroni with
        # extra steps.
        rejected = still_rejecting and p <= threshold
        still_rejecting = rejected
        out[name] = {
            "p": p, "threshold": threshold, "rank": index + 1, "significant": rejected
        }
    return out


def run_folds(frame: pd.DataFrame, seasons: list[str], columns: list[str]) -> list[dict]:
    start = seasons.index(FIRST_VALIDATION_SEASON)
    return training.walk_forward(frame, seasons, columns, min_train_seasons=start)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", default="features.parquet")
    parser.add_argument("--out", default="ablation.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    frame = fb.drop_burn_in(pd.read_parquet(paths.processed_dir() / args.frame))
    frame = frame[frame["result"].notna()].reset_index(drop=True)

    holdout = decision_config.config().benchmark.holdout_season
    seasons = training.eligible_seasons(frame, holdout)
    training.assert_holdout_untouched(seasons, holdout)

    every_column = fb.feature_columns(frame, include_inert=True)
    added = {c for group in HYPOTHESES.values() for c in group}
    missing = sorted(added - set(every_column))
    if missing:
        log.error("frame is missing pre-registered columns: %s", missing)
        return 1
    baseline_columns = [c for c in every_column if c not in added]

    log.info(
        "baseline %d features, %d added across %d pre-registered groups; folds "
        "validate %s..%s",
        len(baseline_columns), len(added), len(HYPOTHESES),
        FIRST_VALIDATION_SEASON, seasons[-1],
    )

    configurations = {"baseline": baseline_columns}
    for name, group in HYPOTHESES.items():
        configurations[name] = baseline_columns + [c for c in every_column if c in group]
    configurations["all_groups"] = every_column

    results: dict[str, list[dict]] = {}
    for name, columns in configurations.items():
        log.info("--- %s (%d features)", name, len(columns))
        results[name] = run_folds(frame, seasons, columns)

    baseline_by_season = {
        f["validation_season"]: f["log_loss"] for f in results["baseline"]
    }
    comparisons: dict[str, dict] = {}
    p_values: dict[str, float] = {}

    for name in [*HYPOTHESES, "all_groups"]:
        paired = [
            (baseline_by_season[f["validation_season"]], f["log_loss"])
            for f in results[name]
            if f["validation_season"] in baseline_by_season
        ]
        base = np.array([p[0] for p in paired])
        variant = np.array([p[1] for p in paired])
        # Positive = the variant scored lower log-loss, i.e. better.
        delta = base - variant
        test = stats.ttest_rel(base, variant)
        comparisons[name] = {
            "folds": len(paired),
            "mean_baseline": float(base.mean()),
            "mean_variant": float(variant.mean()),
            "mean_improvement": float(delta.mean()),
            "std_improvement": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
            "t": float(test.statistic),
            "p_uncorrected": float(test.pvalue),
        }
        if name in HYPOTHESES:
            p_values[name] = float(test.pvalue)

    corrected = holm(p_values)
    for name, verdict in corrected.items():
        comparisons[name].update(verdict)

    log.info("")
    log.info(
        "%-18s %9s %9s %10s %8s %10s %8s",
        "group", "baseline", "variant", "improve", "t", "p", "Holm",
    )
    for name in [*HYPOTHESES, "all_groups"]:
        c = comparisons[name]
        verdict = (
            "n/a" if name not in corrected
            else ("SIGNIFICANT" if c["significant"] else "no")
        )
        log.info(
            "%-18s %9.4f %9.4f %+10.5f %8.2f %10.4f %8s",
            name, c["mean_baseline"], c["mean_variant"], c["mean_improvement"],
            c["t"], c["p_uncorrected"], verdict,
        )

    any_significant = any(v["significant"] for v in corrected.values())
    log.info("")
    log.info(
        "family-wise alpha %.2f over %d hypotheses: %s",
        FAMILY_ALPHA, len(p_values),
        "at least one group survives" if any_significant
        else "no group survives correction",
    )

    destination = paths.processed_dir() / args.out
    destination.write_text(
        json.dumps(
            {
                "first_validation_season": FIRST_VALIDATION_SEASON,
                "family_alpha": FAMILY_ALPHA,
                "hypotheses": {k: list(v) for k, v in HYPOTHESES.items()},
                "comparisons": comparisons,
                "folds": {k: v for k, v in results.items()},
            },
            indent=2, default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("wrote %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
