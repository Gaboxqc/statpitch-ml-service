"""Measure what conformal prediction sets are worth here (Roadmap §6.2).

    python scripts/evaluate_conformal.py [--alpha 0.2 0.4 0.6]

Split conformal turns probabilities into sets with a coverage guarantee. The
guarantee is free; whether the sets are *informative* is not, and on a
three-outcome market where the best available model reaches 54% accuracy it is
the open question.

Fit on seasons up to the calibration year, calibrate on one season, evaluate on
the next — so the reported coverage is on data neither the model nor the
threshold has seen, and across a season boundary, which is where the
exchangeability assumption is actually tested. Football is not stationary; a
number quoted from the calibration set itself would be a description of the
calibration set.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from statpitch import paths
from statpitch.features import build as fb
from statpitch.models import conformal, training

log = logging.getLogger("conformal-eval")

DEFAULT_ALPHAS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
CALIBRATION_SEASON = "2022-2023"
EVALUATION_SEASON = "2023-2024"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--out", default="conformal.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    frame = fb.drop_burn_in(pd.read_parquet(paths.processed_dir() / "features.parquet"))
    frame = frame[frame["result"].notna()]
    columns = fb.feature_columns(frame)

    cutoff = training.season_start(CALIBRATION_SEASON) - 1
    train = frame[frame["season"].map(training.season_start) <= cutoff]
    calibration = frame[frame["season"] == CALIBRATION_SEASON]
    evaluation = frame[frame["season"] == EVALUATION_SEASON]
    if calibration.empty or evaluation.empty:
        log.error("calibration or evaluation season missing from the frame")
        return 1

    model = training.fit_model(train, columns)

    def labels(rows: pd.DataFrame) -> np.ndarray:
        return np.array([training.CLASSES.index(r) for r in rows["result"]])

    calibration_p = model.predict_one_x_two(calibration)
    evaluation_p = model.predict_one_x_two(evaluation)
    calibration_y, evaluation_y = labels(calibration), labels(evaluation)

    log.info(
        "trained on %d, calibrated on %d (%s), evaluated on %d (%s)",
        len(train), len(calibration), CALIBRATION_SEASON,
        len(evaluation), EVALUATION_SEASON,
    )
    log.info(
        "%6s %8s %10s %10s %8s %8s", "alpha", "target", "coverage", "set size",
        "size 1", "size 2",
    )

    rows = []
    for alpha in args.alpha:
        fitted = conformal.calibrate(calibration_p, calibration_y, alpha=alpha)
        result = conformal.evaluate(evaluation_p, evaluation_y, fitted)
        sizes = np.array(
            [len(s) for s in conformal.prediction_sets(evaluation_p, fitted)]
        )
        row = {
            **fitted.as_dict(),
            **result,
            "share_size_1": float((sizes == 1).mean()),
            "share_size_2": float((sizes == 2).mean()),
        }
        rows.append(row)
        log.info(
            "%6.2f %8.2f %10.3f %10.2f %7.1f%% %7.1f%%",
            alpha, result["target"], result["coverage"], result["mean_set_size"],
            100 * row["share_size_1"], 100 * row["share_size_2"],
        )

    # Coverage is marginal. A set covering 80% overall can cover far less in one
    # competition, and the guarantee says nothing about that — so it is measured.
    middle = conformal.calibrate(calibration_p, calibration_y, alpha=0.5)
    by_competition = conformal.coverage_by(
        evaluation_p, evaluation_y, middle, evaluation["competition_id"]
    )
    log.info("")
    log.info("coverage by competition at alpha=0.50:")
    for line in by_competition.round(3).to_string().splitlines():
        log.info("  %s", line)

    destination = paths.processed_dir() / args.out
    destination.write_text(
        json.dumps(
            {
                "calibration_season": CALIBRATION_SEASON,
                "evaluation_season": EVALUATION_SEASON,
                "n_train": int(len(train)),
                "curve": rows,
                "by_competition_alpha_0.5": by_competition.reset_index().to_dict(
                    orient="records"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("wrote %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
