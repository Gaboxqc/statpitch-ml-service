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
from statpitch.models import entrant_prior as ep
from statpitch.models import explain, registry, release
from statpitch.models.goals import GoalModel

log = logging.getLogger("precompute")


def _select_artifact(name: str | None):
    """Locate the artifact to predict with, downloading it if it is not here.

    Boosters are gitignored, so a fresh checkout has a registry describing files
    nobody has. `release.ensure_local` fetches the published asset rather than
    retraining to get it back — local copies win, so a developer who has just
    trained is not overwritten by whatever was last published.
    """
    store = registry.Registry.load(paths.models_dir())
    version = None
    if name:
        version = store.get(name).version
    elif store.promoted is not None:
        version = store.promoted.version
    elif store.entries:
        # Nothing promoted. Falling back to the newest *registered* entry rather
        # than the newest local directory matters in CI, where the boosters are
        # gitignored and there is no local directory to find — the previous
        # fallback would have failed the refresh outright.
        version = store.entries[-1].version
        log.warning(
            "no promoted model; falling back to the newest registered artifact "
            "%s. Promote deliberately with scripts/promote_model.py.", version,
        )

    if version is not None:
        try:
            return release.ensure_local(version), version
        except release.ReleaseError as exc:
            local = paths.models_dir() / version
            if (local / "model.json").exists():
                return local, version
            raise SystemExit(
                f"{version} is registered but neither present locally nor "
                f"downloadable ({exc}). Rebuild it with `python scripts/train.py`."
            ) from exc

    candidates = sorted(paths.models_dir().glob("goals-*"))
    if not candidates:
        raise SystemExit(
            "the registry is empty and no artifact is present — run "
            "`python scripts/train.py` first"
        )
    log.warning(
        "registry is empty; using the unregistered local artifact %s, whose "
        "training window and metrics are therefore unknown", candidates[-1].name,
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

    # FR-9. An unrated club must not reach the model as a null — see
    # `entrant_prior.fill_missing_ratings` for what that produced when it did.
    pooled_elo = ep.pooled_elo_from_file(paths.data_root() / "entrant_prior.json")
    if pooled_elo is None:
        log.error(
            "no entrant_prior.json — unrated clubs will reach the model as nulls, "
            "which produces confident nonsense rather than an abstention"
        )
    slots = [
        (str(club), date)
        for date, home, away in zip(
            upcoming["date"], upcoming["home_team"], upcoming["away_team"], strict=True
        )
        for club in (home, away)
    ]
    elo_lookup, rating_source, filled = ep.fill_missing_ratings(
        elo_lookup, slots, pooled_elo
    )
    if filled:
        log.warning(
            "%d club slot(s) had no Club Elo rating and were given the pooled "
            "entrant level %.1f (FR-9). Those fixtures report a prior rather "
            "than a measured rating.", filled, pooled_elo,
        )

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
            # Which tier of evidence supplied each rating, so a consumer can tell
            # a measured club from one carrying the pooled entrant prior.
            "home_rating_source": [
                rating_source.get((str(c), d), "unknown")
                for c, d in zip(rows["home_team"], rows["date"], strict=True)
            ],
            "away_rating_source": [
                rating_source.get((str(c), d), "unknown")
                for c, d in zip(rows["away_team"], rows["date"], strict=True)
            ],
            "model_version": version,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )

    explanations = explain.explanations_frame(model, rows, rows["match_id"])
    explanations["model_version"] = version

    unrated = int(out["home_elo"].isna().sum() + out["away_elo"].isna().sum())
    if unrated:
        log.error(
            "%d club slot(s) STILL have no rating after the entrant-prior fill; "
            "those rows predict from form alone and will invent a number", unrated,
        )
    priors = int(
        (out["home_rating_source"] == "pooled_prior").sum()
        + (out["away_rating_source"] == "pooled_prior").sum()
    )
    if priors:
        log.info("%d club slot(s) rated from the pooled entrant prior", priors)

    # A fixture where NEITHER club has a measured rating is not predicted here.
    #
    # Both sides take the same pooled prior, so the Elo difference is exactly
    # zero and the fitted model has no rating signal at all — it then
    # discriminates on form and xG features that do not exist for either club and
    # returns a confident, asymmetric answer built on noise. Measured: Milton
    # Keynes Dons, a Football League club, came back at 28.8% at home with an
    # eighth-tier opponent favoured at 48.4%.
    #
    # Serving's Elo mapping is strictly better here precisely because it is
    # simpler: equal ratings plus home advantage gives 45.4/25.6/29.1, which is
    # the right shape. Omitting the row lets that fallback answer, which is the
    # documented behaviour for any fixture that was never precomputed.
    #
    # One prior and one measured rating is left alone: there the Elo difference
    # is real and the model has something to work with.
    both_priors = (
        (out["home_rating_source"] == "pooled_prior")
        & (out["away_rating_source"] == "pooled_prior")
    )
    if both_priors.any():
        dropped = out[both_priors]
        log.warning(
            "%d fixture(s) have no measured rating on EITHER side and are left "
            "to the Elo fallback, which handles equal ratings correctly: %s",
            len(dropped),
            "; ".join(
                f"{r.home_team} v {r.away_team}" for r in dropped.itertuples()
            ),
        )
        out = out[~both_priors].reset_index(drop=True)
        explanations = explanations[
            explanations["fixture_id"].isin(set(out["fixture_id"]))
        ]

    destination = processed / "predictions.parquet"
    out.to_parquet(destination, index=False)

    # FR-32. Computed here for the same reason the rates are: `shap` is part of
    # the training stack, and requirements-serving.txt excludes it. The
    # explanation is written beside the prediction it explains, from the same
    # model in the same run, so the two cannot describe different fixtures.
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
