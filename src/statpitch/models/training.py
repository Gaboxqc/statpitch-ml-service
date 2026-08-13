"""Walk-forward training and evaluation (Roadmap §1).

This lives in the package rather than in `scripts/train.py` for the reason
`ops/jobs.py` already gives for the scheduled jobs: it carries real correctness
requirements, and a script is not unit-testable. The holdout guard in particular
is a claim the project makes about itself (NFR-10) and needs a test proving it
raises, not a comment saying it should.

Walk-forward, not a single split
================================

A model that will be retrained weekly has to be evaluated the way it will run:
fit on everything up to a season, score that season, expand, repeat. One held-out
slice cannot separate a better model from a luckier one, and the promotion gate
of Roadmap §11.2 has to decide exactly that. The per-fold spread is the point —
it is what says whether a difference between two models is larger than the
disagreement between seasons.

The holdout is untouched, and provably so
=========================================

Every season from the holdout onwards is excluded from training and validation.
That also drops 2025/26, which sits after the 2025-07-23 Pinnacle regime break
and is held separately, and any future season whose results do not exist yet.
`assert_holdout_untouched` raises before a single tree is fitted.

What this deliberately does not do
==================================

It does not calibrate. MODEL_CARD §3 measured isotonic calibration making this
model *worse* — 0.9852 to 0.9916, fitted out-of-fold across 12,576 matches —
because the matrix derives probabilities from a Poisson process rather than from
a discriminative model's raw scores. Adding a calibration step because it is
conventional would undo a measured result.

It does not tune. `DEFAULT_PARAMS` are hand-set and stay so until Roadmap §5.1
tunes them under these folds with the search budget recorded.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from statpitch.models import calibration
from statpitch.models.goals import DEFAULT_PARAMS, GoalModel

log = logging.getLogger(__name__)

#: 1X2 column order, fixed by `ScoreMatrix.one_x_two`.
CLASSES = ("H", "D", "A")

#: Folds are only meaningful where the validation season has enough matches to
#: separate models. Below this a fold is reported but excluded from the mean.
MIN_FOLD_ROWS = 500

#: Seasons before this have too little history behind them to fit on.
DEFAULT_MIN_TRAIN_SEASONS = 3

#: The seasons MODEL_CARD §1 measured on — "5,306 validation matches (2022/23 and
#: 2023/24)". Scored separately so reproducing the published figure is a computed
#: comparison rather than something a reader eyeballs against a table. The mean
#: over *all* folds is a different and worse number, because it reaches back into
#: seasons with less data and no xG.
CARD_VALIDATION_SEASONS = ("2022-2023", "2023-2024")

#: MODEL_CARD §3, Dixon-Coles + xG, and the de-vigged closing consensus it trails.
CARD_MODEL_LOG_LOSS = 0.9845
CARD_MARKET_LOG_LOSS = 0.9698

#: How far the reproduction may drift before it stops being one. Fold-to-fold
#: spread is ~0.017, so a tighter tolerance would fail on noise.
CARD_TOLERANCE = 0.010


class HoldoutViolation(RuntimeError):
    """A season at or after the holdout reached training or validation."""


def season_start(season: str) -> int:
    return int(str(season).split("-")[0])


def assert_holdout_untouched(seasons, holdout: str) -> None:
    """NFR-10, enforced before anything expensive happens."""
    offenders = sorted({s for s in seasons if season_start(s) >= season_start(holdout)})
    if offenders:
        raise HoldoutViolation(
            f"{offenders} include the untouched holdout {holdout} or a season "
            "after it. NFR-10 reserves it for a single look at the end, and a "
            "model that has seen it cannot provide one."
        )


def eligible_seasons(frame: pd.DataFrame, holdout: str) -> list[str]:
    """Every season strictly before the holdout, oldest first."""
    seasons = sorted(frame["season"].dropna().unique(), key=season_start)
    return [s for s in seasons if season_start(s) < season_start(holdout)]


def score(model: GoalModel, frame: pd.DataFrame) -> dict[str, float]:
    """1X2 log-loss, accuracy and ECE for one set of rows."""
    probabilities = model.predict_one_x_two(frame)
    labels = np.array([CLASSES.index(r) for r in frame["result"]])
    predicted = probabilities.argmax(axis=1)
    return {
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "accuracy": float((predicted == labels).mean()),
        "ece": float(
            calibration.expected_calibration_error(
                probabilities, calibration.one_hot(labels)
            )
        ),
        "n": int(len(frame)),
    }


def fit_model(
    frame: pd.DataFrame, columns: list[str], params: dict | None = None
) -> GoalModel:
    """Fit the paired regressors and then rho, which depends on their rates."""
    model = GoalModel(feature_columns=columns).fit(
        frame, frame["home_goals"], frame["away_goals"],
        params={**DEFAULT_PARAMS, **(params or {})},
    )
    model.fit_rho(frame, frame["home_goals"], frame["away_goals"])
    return model


def walk_forward(
    frame: pd.DataFrame,
    seasons: list[str],
    columns: list[str],
    params: dict | None = None,
    *,
    min_train_seasons: int = DEFAULT_MIN_TRAIN_SEASONS,
) -> list[dict]:
    """Expanding-window folds: fit on everything earlier, score one season."""
    folds = []
    for index in range(min_train_seasons, len(seasons)):
        validation_season = seasons[index]
        train_seasons = seasons[:index]
        train = frame[frame["season"].isin(train_seasons)]
        validation = frame[frame["season"] == validation_season]
        if train.empty or validation.empty:
            continue

        model = fit_model(train, columns, params)
        metrics = score(model, validation)
        metrics.update(
            validation_season=validation_season,
            train_seasons=train_seasons,
            n_train=int(len(train)),
            counted=metrics["n"] >= MIN_FOLD_ROWS,
        )
        log.info(
            "fold %s: log_loss %.4f  accuracy %.4f  ece %.5f  (train %d, val %d)",
            validation_season, metrics["log_loss"], metrics["accuracy"],
            metrics["ece"], len(train), metrics["n"],
        )
        folds.append(metrics)
    return folds


def aggregate(folds: list[dict]) -> dict[str, float]:
    """Mean and spread across counted folds.

    The standard deviation is the point. A promotion gate comparing two models on
    a mean alone promotes noise; the spread says whether a difference is larger
    than the disagreement between seasons.
    """
    counted = [f for f in folds if f.get("counted")]
    if not counted:
        return {}
    out: dict[str, float] = {"folds": len(counted)}
    for metric in ("log_loss", "accuracy", "ece"):
        values = np.array([f[metric] for f in counted], dtype=float)
        out[f"mean_{metric}"] = float(values.mean())
        out[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    out["total_validation_rows"] = int(sum(f["n"] for f in counted))
    return out


def card_comparison(folds: list[dict]) -> dict | None:
    """Row-weighted score over the seasons MODEL_CARD §1 measured on.

    Row-weighted rather than a mean of means: the card reports one figure over
    5,306 matches, and averaging two seasons of unequal size answers a slightly
    different question than the one being compared against.

    Returns None when the folds do not cover both seasons, because a partial
    comparison against a published number is worse than none.
    """
    subset = [f for f in folds if f["validation_season"] in CARD_VALIDATION_SEASONS]
    if len(subset) != len(CARD_VALIDATION_SEASONS):
        return None

    rows = sum(f["n"] for f in subset)
    weighted = sum(f["log_loss"] * f["n"] for f in subset) / rows
    return {
        "seasons": [f["validation_season"] for f in subset],
        "n": rows,
        "log_loss": float(weighted),
        "card_log_loss": CARD_MODEL_LOG_LOSS,
        "difference": float(weighted - CARD_MODEL_LOG_LOSS),
        "reproduces": bool(abs(weighted - CARD_MODEL_LOG_LOSS) <= CARD_TOLERANCE),
        "market_log_loss": CARD_MARKET_LOG_LOSS,
        "gap_to_market": float(weighted - CARD_MARKET_LOG_LOSS),
    }
