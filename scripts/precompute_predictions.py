"""Predict every upcoming fixture with the fitted model (Roadmap §8, §2).

    python scripts/precompute_predictions.py [--artifact goals-...]

This is what closes the gap MODEL_CARD §3 records. The API derives goal rates
from an Elo difference and costs +0.0064 log-loss against the fitted model,
because serving cannot run the fitted model: that needs rolling-form features,
and serving has none for an arbitrary fixture.

It has them for a *known* fixture. Rolling form depends only on matches already
played, so appending scheduled fixtures to the match log and running the same
chronological pass produces a real feature row for each — computed here, offline,
where xgboost is allowed to exist. Serving reads the result.

Three properties this inherits rather than invents
==================================================

**No leakage.** `build_features` emits a scheduled fixture's row and then skips
the state update, because there is no result to record. A fixture contributes
nothing to any club's form, including its own.

**No xgboost in the deployed image.** `requirements-serving.txt` excludes it and
`tests/test_deployment.py` enforces that. The model runs here; the API reads a
parquet of finished predictions.

**Honest provenance.** Every row carries the `model_version` that produced it, so
a consumer storing predictions can tell fitted-model rows from the Elo fallback
the API still uses for fixtures that were never precomputed.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from statpitch import paths
from statpitch.data import club_elo as ce
from statpitch.features import build as fb
from statpitch.models import explain, registry
from statpitch.models.goals import GoalModel

log = logging.getLogger("precompute")


def _select_artifact(name: str | None):
    store = registry.Registry.load(paths.models_dir())
    if name:
        return paths.models_dir() / name, store.get(name).version
    promoted = store.promoted
    if promoted is not None:
        return paths.models_dir() / promoted.version, promoted.version
    candidates = sorted(paths.models_dir().glob("goals-*"))
    if not candidates:
        raise SystemExit("no trained artifact — run scripts/train.py first")
    log.warning(
        "no promoted model in the registry; falling back to the newest artifact "
        "%s. Promote deliberately with scripts/promote_model.py.",
        candidates[-1].name,
    )
    return candidates[-1], candidates[-1].name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    processed = paths.processed_dir()
    fixtures_path = paths.fixtures_file()
    if not fixtures_path.exists():
        log.error("no fixtures — run scripts/build_fixtures.py first")
        return 1

    fixtures = pd.read_parquet(fixtures_path)
    matches = pd.read_parquet(processed / "matches_clean.parquet")
    cups = pd.read_parquet(processed / "cup_matches.parquet")

    log_frame = fb.merge_match_log(matches, cups, fixtures=fixtures)
    scheduled = set(fixtures["fixture_id"])
    log.info(
        "match log: %d played + %d scheduled", len(log_frame) - len(scheduled),
        len(scheduled),
    )

    # Ratings are only needed for the scheduled rows; historical feature values
    # are rebuilt here but discarded, and looking every one of them up would cost
    # far more than it buys.
    elo_table = pd.read_parquet(processed / "elo_ratings_all.parquet")
    aliases: dict[str, str] = {}
    for name in ("cup_club_elo_map.json", "fixture_club_elo_map.json"):
        path = processed / name
        if path.exists():
            import json

            aliases.update(json.loads(path.read_text(encoding="utf-8")).get("matched", {}))

    upcoming = log_frame[log_frame["match_id"].isin(scheduled)]
    pairs = [
        (aliases.get(str(club), str(club)), date)
        for date, home, away in zip(
            upcoming["date"], upcoming["home_team"], upcoming["away_team"], strict=True
        )
        for club in (home, away)
    ]
    raw_lookup = ce.build_lookup(elo_table, pairs)
    # Re-key onto the fixture's own club names, which is what build_features sees.
    elo_lookup = {
        (str(club), date): raw_lookup[(aliases.get(str(club), str(club)), date)]
        for date, home, away in zip(
            upcoming["date"], upcoming["home_team"], upcoming["away_team"], strict=True
        )
        for club in (home, away)
        if (aliases.get(str(club), str(club)), date) in raw_lookup
    }
    log.info("resolved %d/%d club ratings for scheduled fixtures",
             len(elo_lookup), len(set(pairs)))

    # match_xg is the Understat frame already joined onto match_id; understat_xg
    # is keyed on Understat's own ids and cannot be looked up by match.
    xg_path = processed / "match_xg.parquet"
    xg_lookup = None
    if xg_path.exists():
        xg = pd.read_parquet(xg_path)
        xg_lookup = dict(
            zip(xg["match_id"], zip(xg["home_xg"], xg["away_xg"], strict=True),
                strict=True)
        )
        log.info("xG available for %d matches", len(xg_lookup))

    features = fb.build_features(log_frame, elo_lookup=elo_lookup, xg_lookup=xg_lookup)
    rows = features[features["match_id"].isin(scheduled)].reset_index(drop=True)
    if rows.empty:
        log.error("no feature rows produced for scheduled fixtures")
        return 1

    artifact_dir, version = _select_artifact(args.artifact)
    model = GoalModel.load(artifact_dir)
    registry.verify_features(model.feature_columns, fb.feature_columns(rows))

    lambda_home, lambda_away = model.predict(rows)
    probabilities = model.predict_one_x_two(rows)

    out = pd.DataFrame(
        {
            "fixture_id": rows["match_id"],
            "competition_id": rows["competition_id"],
            "date": rows["date"],
            "home_team": rows["home_team"],
            "away_team": rows["away_team"],
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            # The fitted per-competition rho travels with the rates. Serving
            # applies no correction of its own (Artifacts.rho is empty by
            # measurement), so without this a precomputed fixture would get
            # fitted rates inside an independent-Poisson matrix.
            "rho": [
                float(model.rho.get(str(c), 0.0)) for c in rows["competition_id"]
            ],
            "prob_home": probabilities[:, 0],
            "prob_draw": probabilities[:, 1],
            "prob_away": probabilities[:, 2],
            "home_elo": rows["home_elo"],
            "away_elo": rows["away_elo"],
            "model_version": version,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )

    unrated = int(out["home_elo"].isna().sum() + out["away_elo"].isna().sum())
    if unrated:
        log.warning(
            "%d club slot(s) had no rating; those rows predict from form alone "
            "and their elo features are null", unrated,
        )

    destination = processed / "predictions.parquet"
    out.to_parquet(destination, index=False)

    # FR-32. Computed here for the same reason the rates are: `shap` is part of
    # the training stack, and requirements-serving.txt excludes it. The
    # explanation is written beside the prediction it explains, from the same
    # model in the same run, so the two cannot describe different fixtures.
    explanations = explain.explanations_frame(model, rows, rows["match_id"])
    explanations["model_version"] = version
    explanations.to_parquet(processed / "explanations.parquet", index=False)
    log.info(
        "wrote %d explanation rows (%d per fixture per side)",
        len(explanations), explain.DEFAULT_TOP_N,
    )
    log.info(
        "wrote %d predictions to %s using %s (mean lambda %.2f v %.2f)",
        len(out), destination, version,
        float(np.mean(lambda_home)), float(np.mean(lambda_away)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
