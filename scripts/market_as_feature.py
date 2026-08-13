"""Does the model know anything the market does not? (Roadmap §5.4)

    python scripts/market_as_feature.py

MODEL_CARD §1 answers this with a **linear post-hoc blend**:
`p_used = w·p_model + (1−w)·q_fair`, with `w` fitted at 0.000. That is a strong
result but a narrow test. It asks whether a fixed-weight mixture of two finished
probability vectors beats one of them, and it cannot express "the model is right
about *this kind* of fixture" — a blend has one dial for every match ever played.

The nested formulation asks the question directly. Give the model the de-vigged
closing line as a `base_margin` **offset** and let it learn the residual:

    market      the de-vigged closing consensus, used directly
    features    the shipped feature set, no market input
    residual    the same features, starting from the market

`residual` versus `market` is the test. A model with nothing to add scores exactly
the market, so any improvement is attributable rather than merely visible, and
nonlinear interactions a blend cannot represent are available to it.

Why an offset and not just more features
========================================

The first version of this experiment fed the market in as three ordinary features
and measured that configuration at 0.9936 — a quarter-point *worse* than the
market it had been handed. Trees are piecewise-constant, and the identity map on
three continuous inputs is exactly what they approximate worst. That wrapper cost
~0.024 log-loss, an order of magnitude more than any effect being looked for, so
the test would have run through a channel lossier than its own signal.

The test is paired **per match**, not per fold. Only three seasons have consensus
closing odds and enough history behind them to train on, and three paired folds
cannot resolve anything; per match gives thousands of pairs. Each match
contributes the difference between the two models' log-loss on the outcome that
actually happened.

Scope, and why it is smaller than it looks
==========================================

Consensus closing columns start in 2019/20 and the Pinnacle regime break at
2025-07-23 is not pooled across, so the window is 2019/20–2023/24 minus the
untouched holdout. Only the five leagues have odds at all. Whatever this finds
applies to league fixtures with a closing line, which is exactly the population
`w` was fitted on — and therefore the right population to challenge it on.

This is an information-content measurement, not a shippable model. It fits a
direct 1X2 classifier rather than the goal model, because market probabilities are
1X2 quantities and injecting them into a Poisson rate model would answer a
different question badly. MODEL_CARD §3 already records that a direct classifier
is worse than the score matrix; the comparison here is between configurations of
the same classifier, where that penalty cancels.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.metrics import log_loss

from statpitch import decision_config, paths
from statpitch.decision import devig
from statpitch.features import build as fb
from statpitch.models import training

log = logging.getLogger("market")

MARKET_COLUMNS = ["market_home", "market_draw", "market_away"]

#: The five competitions with a free odds source (Requirements §9).
LEAGUES = frozenset(
    {"ENG.PL", "ESP.LALIGA", "GER.BUNDESLIGA", "ITA.SERIEA", "FRA.LIGUE1"}
)

#: Matching the shipped model's shape as closely as a classifier can, so the
#: comparison is between feature sets rather than between hyperparameters.
PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": 0,
}


def market_probabilities() -> pd.DataFrame:
    """De-vigged closing consensus per match, as a (match_id, H, D, A) frame."""
    odds = pd.read_parquet(paths.odds_file())
    close = odds[
        (odds["snapshot"] == "close")
        & (odds["market"] == "1x2")
        & (odds["odds_regime"] == "pre_2025_07_23")
        & odds["odds_avg"].notna()
    ]
    wide = close.pivot_table(
        index="match_id", columns="selection", values="odds_avg", aggfunc="first"
    )
    wide = wide.dropna(subset=["home", "draw", "away"])
    # Shin is the fitted default; MODEL_CARD §6 records the de-vig comparison as
    # underpowered rather than resolved, so this inherits the choice rather than
    # relitigating it inside a different experiment.
    fair = devig.devig_many(wide[["home", "draw", "away"]].to_numpy(), method="shin")
    return pd.DataFrame(
        {
            "match_id": wide.index,
            "market_home": fair[:, 0],
            "market_draw": fair[:, 1],
            "market_away": fair[:, 2],
        }
    )


def _labels(frame: pd.DataFrame) -> np.ndarray:
    return np.array([training.CLASSES.index(r) for r in frame["result"]])


def _margin(frame: pd.DataFrame) -> np.ndarray:
    """The market as a starting point, in the model's own link space."""
    return np.log(np.clip(frame[MARKET_COLUMNS].to_numpy(dtype=float), 1e-15, 1.0))


def fit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    *,
    offset: bool,
) -> np.ndarray:
    """Fit one configuration and return validation probabilities.

    With `offset`, the de-vigged market is supplied as `base_margin` and the trees
    learn the **residual** against it. That is the formulation this experiment
    needs: handed the market as ordinary features instead, a tree ensemble cannot
    reproduce it — trees are piecewise-constant and the identity map on three
    continuous inputs is exactly what they approximate worst. Measured, that
    wrapper costs ~0.024 log-loss, which is an order of magnitude larger than any
    effect being looked for, so the test would run through a channel far lossier
    than its own signal.

    As an offset the market passes through untouched, and a model that has nothing
    to add scores exactly the market. Improvement is then attributable rather than
    merely visible.
    """
    labels = _labels(train)
    train_matrix = xgb.DMatrix(train[columns], label=labels)
    validation_matrix = xgb.DMatrix(validation[columns])
    if offset:
        train_matrix.set_base_margin(_margin(train))
        validation_matrix.set_base_margin(_margin(validation))

    booster = xgb.train(
        {k: v for k, v in PARAMS.items() if k != "n_estimators"},
        train_matrix,
        num_boost_round=PARAMS["n_estimators"],
    )
    return booster.predict(validation_matrix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="market_as_feature.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    frame = fb.drop_burn_in(pd.read_parquet(paths.processed_dir() / "features.parquet"))
    frame = frame[frame["result"].notna() & frame["competition_id"].isin(LEAGUES)]
    frame = frame.merge(market_probabilities(), on="match_id", how="inner")

    holdout = decision_config.config().benchmark.holdout_season
    seasons = training.eligible_seasons(frame, holdout)
    training.assert_holdout_untouched(seasons, holdout)
    frame = frame[frame["season"].isin(seasons)].reset_index(drop=True)

    feature_columns = [
        c for c in fb.feature_columns(frame) if c not in MARKET_COLUMNS
    ]
    #: `features` is the shipped feature set with no market input; `residual` is
    #: the same features starting from the market and learning what it missed.
    configurations = {
        "features": (feature_columns, False),
        "residual": (feature_columns, True),
    }

    log.info(
        "%d matches with a de-vigged closing line, seasons %s..%s",
        len(frame), seasons[0], seasons[-1],
    )

    # Walk forward, pooling out-of-sample predictions so the paired test has
    # matches rather than folds to work with.
    pooled: dict[str, list[np.ndarray]] = {
        name: [] for name in ("market", *configurations)
    }
    labels: list[np.ndarray] = []
    validated: list[str] = []

    for index in range(2, len(seasons)):
        season = seasons[index]
        train = frame[frame["season"].isin(seasons[:index])]
        validation = frame[frame["season"] == season]
        if len(train) < 500 or validation.empty:
            continue
        validated.append(season)
        labels.append(_labels(validation))
        # The market itself, used directly rather than through any model. This is
        # the baseline the residual configuration must beat, and it is what
        # MODEL_CARD §3's 0.9698 refers to.
        pooled["market"].append(validation[MARKET_COLUMNS].to_numpy(dtype=float))
        for name, (columns, offset) in configurations.items():
            pooled[name].append(fit_predict(train, validation, columns, offset=offset))
        log.info("fold %s: train %d, validate %d", season, len(train), len(validation))

    if not validated:
        log.error("no fold had enough history to train on")
        return 1

    y = np.concatenate(labels)
    scores: dict[str, float] = {}
    per_match: dict[str, np.ndarray] = {}
    for name in ("market", *configurations):
        probabilities = np.vstack(pooled[name])
        scores[name] = float(log_loss(y, probabilities, labels=[0, 1, 2]))
        # Per-match log-loss, for the paired test.
        per_match[name] = -np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-15, 1))

    delta = per_match["market"] - per_match["residual"]
    test = stats.ttest_1samp(delta, 0.0)

    log.info("")
    log.info("pooled out-of-sample over %d matches (%s)", len(y), ", ".join(validated))
    for name in ("market", "features", "residual"):
        log.info("  %-9s log-loss %.4f", name, scores[name])
    log.info("")
    log.info(
        "residual vs market: %+.5f log-loss, t = %.2f, p = %.4f",
        scores["residual"] - scores["market"], float(test.statistic),
        float(test.pvalue),
    )
    improved = scores["residual"] < scores["market"] and test.pvalue < 0.05
    log.info(
        "the feature set %s beyond the closing line",
        "ADDS information" if improved else "adds nothing detectable",
    )

    destination = paths.processed_dir() / args.out
    destination.write_text(
        json.dumps(
            {
                "matches": int(len(y)),
                "validated_seasons": validated,
                "log_loss": scores,
                "residual_minus_market": scores["residual"] - scores["market"],
                "t": float(test.statistic),
                "p": float(test.pvalue),
                "adds_information": bool(improved),
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
