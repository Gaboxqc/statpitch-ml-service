"""Measure what the API actually serves (Roadmap §2.1).

    python scripts/evaluate_served_path.py [--artifact models/goals-...]

MODEL_CARD §3 measures a Dixon-Coles matrix driven by fitted XGBoost goal rates.
`serving/predictor.py` derives its rates from an Elo difference instead, and that
path had no row in any evaluation table — the deployed API returned numbers whose
log-loss nobody had computed, next to a card reporting 0.9845.

Worse, two artifacts the code reads are never written. `Artifacts.goal_environment`
and `Artifacts.rho` are declared, consulted at `predictor.py:204` and `:377`, and
populated by nothing. In production that means:

  * every competition is priced at the pooled 1.45/1.20 rate, including the
    Bundesliga, whose 3.07 goals per match is the exact case `models/goals.py`
    says pooling would misprice "squarely on the Over/Under market"; and
  * rho is 0.0 everywhere, so the Dixon-Coles low-score correction — the entire
    reason `models/dixon_coles.py` exists — is inert, and the served matrix is
    independent Poisson.

This script scores three variants on the seasons MODEL_CARD §1 measures, so the
cost of each gap is a number rather than an argument:

    deployed   Elo mapping, pooled rates, rho = 0   (what the API returns today)
    exported   Elo mapping + fitted environments and rho
    fitted     the full XGBoost goal model          (what the card measures)

Ratings come from the feature frame's as-of-date `home_elo`/`away_elo`, not from
today's table. Scoring history with current ratings would credit the mapping with
information it will never have at prediction time.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from statpitch import decision_config, paths
from statpitch.features import build as fb
from statpitch.models import calibration, elo_rates, training
from statpitch.models.entrant_prior import ELO_SCALE
from statpitch.models.goals import GoalModel
from statpitch.serving.predictor import (
    CUP_HOME_ADVANTAGE_ELO,
    DEFAULT_AWAY_RATE,
    DEFAULT_HOME_RATE,
    ELO_GOAL_SENSITIVITY,
    LEAGUE_HOME_ADVANTAGE_ELO,
)

log = logging.getLogger("evaluate_served_path")

#: The five leagues are round-robin, so the league home-advantage figure applies.
#: Cup fixtures in the window are a small minority and take the cup constant.
LEAGUE_COMPETITIONS = frozenset(
    {"ENG.PL", "ESP.LALIGA", "GER.BUNDESLIGA", "ITA.SERIEA", "FRA.LIGUE1"}
)


def _metrics(probabilities: np.ndarray, frame: pd.DataFrame) -> dict[str, float]:
    labels = np.array([training.CLASSES.index(r) for r in frame["result"]])
    return {
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "accuracy": float((probabilities.argmax(axis=1) == labels).mean()),
        "ece": float(
            calibration.expected_calibration_error(
                probabilities, calibration.one_hot(labels)
            )
        ),
        "n": int(len(frame)),
    }


def score_elo_path(
    frame: pd.DataFrame,
    *,
    environments: dict[str, tuple[float, float]] | None,
    rho: dict[str, float] | None,
) -> dict[str, float]:
    advantage = np.where(
        frame["competition_id"].isin(LEAGUE_COMPETITIONS),
        LEAGUE_HOME_ADVANTAGE_ELO,
        CUP_HOME_ADVANTAGE_ELO,
    )
    lambda_home, lambda_away = elo_rates.rates_for_frame(
        frame,
        environments=environments,
        default_base=(DEFAULT_HOME_RATE, DEFAULT_AWAY_RATE),
        home_advantage=advantage,
        sensitivity=ELO_GOAL_SENSITIVITY,
        elo_scale=ELO_SCALE,
    )
    rho_values = np.array(
        [(rho or {}).get(str(c), 0.0) for c in frame["competition_id"]], dtype=float
    )
    return _metrics(elo_rates.one_x_two(lambda_home, lambda_away, rho_values), frame)


def _fitted_out_of_sample() -> dict[str, float]:
    """Score the goal model on seasons it did not see.

    The shipped artifact is fitted on every eligible season, **including** the two
    being scored here, so evaluating it directly would report an in-sample number
    — it lands around 0.956, and comparing that against the Elo path would
    manufacture a gap out of memorised training rows.

    So each card season is scored by a model fitted only on seasons strictly
    before it, which is what `training.walk_forward` already builds and what
    MODEL_CARD §1's 0.9845 refers to.
    """
    frame = fb.drop_burn_in(pd.read_parquet(paths.processed_dir() / "features.parquet"))
    frame = frame[frame["result"].notna()].reset_index(drop=True)
    holdout = decision_config.config().benchmark.holdout_season
    seasons = training.eligible_seasons(frame, holdout)
    training.assert_holdout_untouched(seasons, holdout)

    columns = fb.feature_columns(frame)
    last_card_season = max(training.CARD_VALIDATION_SEASONS, key=training.season_start)
    start = seasons.index(min(training.CARD_VALIDATION_SEASONS,
                              key=training.season_start))
    folds = training.walk_forward(
        frame[frame["season"].isin(seasons)].reset_index(drop=True),
        seasons[: seasons.index(last_card_season) + 1],
        columns,
        min_train_seasons=start,
    )
    card = training.card_comparison(folds)
    if card is None:
        raise SystemExit("walk-forward did not cover both card seasons")
    return {
        "log_loss": card["log_loss"],
        "accuracy": float(
            sum(f["accuracy"] * f["n"] for f in folds
                if f["validation_season"] in training.CARD_VALIDATION_SEASONS)
            / card["n"]
        ),
        "ece": float(
            sum(f["ece"] * f["n"] for f in folds
                if f["validation_season"] in training.CARD_VALIDATION_SEASONS)
            / card["n"]
        ),
        "n": card["n"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=None,
                        help="Trained model directory. Defaults to the newest.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    frame = fb.drop_burn_in(pd.read_parquet(paths.processed_dir() / "features.parquet"))
    frame = frame[
        frame["result"].notna()
        & frame["season"].isin(training.CARD_VALIDATION_SEASONS)
        & frame["home_elo"].notna()
        & frame["away_elo"].notna()
    ].reset_index(drop=True)
    log.info(
        "scoring %d matches over %s",
        len(frame), " + ".join(training.CARD_VALIDATION_SEASONS),
    )

    if args.artifact:
        artifact_dir = paths.models_dir() / args.artifact
    else:
        candidates = sorted(paths.models_dir().glob("goals-*"))
        if not candidates:
            log.error("no trained artifact found — run scripts/train.py first")
            return 1
        artifact_dir = candidates[-1]
    model = GoalModel.load(artifact_dir)
    log.info("loaded %s (%d environments, %d rho)",
             artifact_dir.name, len(model.environments), len(model.rho))

    results = {
        "deployed": score_elo_path(frame, environments=None, rho=None),
        "exported": score_elo_path(
            frame, environments=model.environments, rho=model.rho
        ),
        "fitted": _fitted_out_of_sample(),
    }

    log.info("%-10s %-10s %-10s %-9s", "variant", "log_loss", "accuracy", "ece")
    for name, metrics in results.items():
        log.info(
            "%-10s %-10.4f %-10.4f %-9.5f",
            name, metrics["log_loss"], metrics["accuracy"], metrics["ece"],
        )

    deployed = results["deployed"]["log_loss"]
    exported = results["exported"]["log_loss"]
    fitted = results["fitted"]["log_loss"]
    log.info(
        "exporting environments and rho recovers %+.4f; the remaining gap to the "
        "fitted model is %+.4f, and to the market (%.4f) %+.4f",
        exported - deployed, fitted - exported,
        training.CARD_MARKET_LOG_LOSS, exported - training.CARD_MARKET_LOG_LOSS,
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
