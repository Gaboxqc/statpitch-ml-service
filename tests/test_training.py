"""Training, persistence and the registry (Roadmap §1).

Three claims are load-bearing enough to need a test rather than a docstring.

**NFR-10 is enforced, not remembered.** The 2024/25 holdout is reserved for a
single look at the end. A guard that only exists as a comment is a guard that
gets removed by someone reading the comment as advice.

**A saved model reloads identically.** The boosters are only part of the model:
the per-competition environments the `base_margin` offsets come from, and the
fitted rho, live in a sidecar. A reload that silently dropped the environments
would predict against the pooled goal rate for every competition and look merely
slightly wrong — the failure mode that does not announce itself.

**A feature mismatch fails loudly.** Missing columns raise deep inside a
prediction; extra ones do not raise at all, and the caller quietly gets a model
evaluated on a different feature set than the one it is being fed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from statpitch.models import registry, training
from statpitch.models.goals import GoalModel

FEATURES = ["elo_diff", "form_diff_5", "rest_diff"]


#: Above `training.MIN_FOLD_ROWS`, so folds count toward the aggregate. A season
#: below that floor is reported and excluded, which is correct behaviour and not
#: what these tests are here to exercise.
ROWS_PER_SEASON = 600


def _frame(
    seasons: tuple[str, ...], per_season: int = ROWS_PER_SEASON, seed: int = 0
) -> pd.DataFrame:
    """A synthetic match log with real structure: stronger sides score more."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for _ in range(per_season):
            elo_diff = float(rng.normal(0, 120))
            lam_home = float(np.clip(1.45 * np.exp(elo_diff / 400), 0.2, 4.0))
            lam_away = float(np.clip(1.20 * np.exp(-elo_diff / 400), 0.2, 4.0))
            home_goals = int(rng.poisson(lam_home))
            away_goals = int(rng.poisson(lam_away))
            rows.append(
                {
                    "season": season,
                    "competition_id": "ENG.PL",
                    "elo_diff": elo_diff,
                    "form_diff_5": float(rng.normal(0, 1)),
                    "rest_diff": float(rng.normal(0, 2)),
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "result": (
                        "H" if home_goals > away_goals
                        else "D" if home_goals == away_goals else "A"
                    ),
                }
            )
    return pd.DataFrame(rows)


SEASONS = ("2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024")


@pytest.fixture(scope="module")
def frame():
    return _frame(SEASONS)


@pytest.fixture(scope="module")
def fitted(frame):
    return training.fit_model(frame, FEATURES)


# --- the holdout guard (NFR-10) -----------------------------------------------

def test_holdout_season_is_refused():
    with pytest.raises(training.HoldoutViolation, match="2024-2025"):
        training.assert_holdout_untouched(["2023-2024", "2024-2025"], "2024-2025")


def test_seasons_after_the_holdout_are_refused_too():
    """2025/26 sits after the Pinnacle regime break and is held separately."""
    with pytest.raises(training.HoldoutViolation):
        training.assert_holdout_untouched(["2025-2026"], "2024-2025")


def test_eligible_seasons_stop_before_the_holdout(frame):
    extended = pd.concat([frame, _frame(("2024-2025", "2025-2026"), 10, seed=9)])
    seasons = training.eligible_seasons(extended, "2024-2025")
    assert seasons == list(SEASONS)
    training.assert_holdout_untouched(seasons, "2024-2025")


def test_a_clean_window_passes_the_guard():
    training.assert_holdout_untouched(list(SEASONS), "2024-2025")


# --- walk-forward -------------------------------------------------------------

def test_walk_forward_never_trains_on_its_validation_season(frame):
    folds = training.walk_forward(frame, list(SEASONS), FEATURES, min_train_seasons=2)
    assert folds
    for fold in folds:
        assert fold["validation_season"] not in fold["train_seasons"]


def test_walk_forward_windows_expand(frame):
    folds = training.walk_forward(frame, list(SEASONS), FEATURES, min_train_seasons=2)
    sizes = [fold["n_train"] for fold in folds]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_aggregate_reports_spread_not_just_a_mean(frame):
    folds = training.walk_forward(frame, list(SEASONS), FEATURES, min_train_seasons=2)
    summary = training.aggregate(folds)
    assert summary["folds"] >= 2
    assert "std_log_loss" in summary
    assert summary["mean_log_loss"] > 0


def test_aggregate_ignores_folds_too_small_to_mean_anything():
    tiny = [
        {"log_loss": 1.0, "accuracy": 0.5, "ece": 0.01, "n": 10, "counted": False},
        {"log_loss": 0.9, "accuracy": 0.5, "ece": 0.01, "n": 900, "counted": True},
    ]
    assert training.aggregate(tiny)["folds"] == 1


def test_card_comparison_returns_nothing_when_a_season_is_missing():
    """A partial comparison against a published number is worse than none."""
    folds = [{"validation_season": "2022-2023", "log_loss": 0.98, "n": 2000}]
    assert training.card_comparison(folds) is None


def test_card_comparison_is_row_weighted():
    folds = [
        {"validation_season": "2022-2023", "log_loss": 1.00, "n": 3000},
        {"validation_season": "2023-2024", "log_loss": 0.90, "n": 1000},
    ]
    result = training.card_comparison(folds)
    assert result["log_loss"] == pytest.approx((1.00 * 3000 + 0.90 * 1000) / 4000)


# --- scoring ------------------------------------------------------------------

def test_score_reports_a_proper_distribution(fitted, frame):
    metrics = training.score(fitted, frame)
    assert metrics["n"] == len(frame)
    assert 0.0 < metrics["log_loss"] < 2.0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_one_x_two_columns_are_home_draw_away(fitted, frame):
    """`score` indexes labels by this order; a silent change would invert it."""
    strong_home = frame.assign(elo_diff=600.0).head(20)
    probabilities = fitted.predict_one_x_two(strong_home)
    assert (probabilities[:, 0] > probabilities[:, 2]).all()


# --- persistence --------------------------------------------------------------

def test_saved_model_predicts_identically_after_reload(fitted, frame, tmp_path):
    before = fitted.predict(frame)
    fitted.save(tmp_path / "artifact")
    reloaded = GoalModel.load(tmp_path / "artifact")
    after = reloaded.predict(frame)
    np.testing.assert_allclose(before[0], after[0], rtol=1e-6)
    np.testing.assert_allclose(before[1], after[1], rtol=1e-6)


def test_reload_preserves_the_sidecar_not_just_the_boosters(fitted, tmp_path):
    """Environments and rho are as load-bearing as the trees."""
    fitted.save(tmp_path / "artifact")
    reloaded = GoalModel.load(tmp_path / "artifact")
    assert reloaded.feature_columns == fitted.feature_columns
    assert reloaded.environments == fitted.environments
    assert reloaded.pooled_environment == fitted.pooled_environment
    assert reloaded.rho == fitted.rho


def test_artifact_is_not_a_pickle(fitted, tmp_path):
    """The format has to survive a runtime upgrade, which a pickle does not."""
    fitted.save(tmp_path / "artifact")
    written = {p.name for p in (tmp_path / "artifact").iterdir()}
    assert written == {"home.json", "away.json", "model.json"}


def test_load_refuses_an_unknown_schema(fitted, tmp_path):
    fitted.save(tmp_path / "artifact")
    meta_path = tmp_path / "artifact" / "model.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["schema"] = 99
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        GoalModel.load(tmp_path / "artifact")


def test_saving_an_unfitted_model_raises(tmp_path):
    with pytest.raises(ValueError, match="unfitted"):
        GoalModel(feature_columns=FEATURES).save(tmp_path / "artifact")


# --- registry -----------------------------------------------------------------

def _entry(version: str, **overrides) -> registry.Entry:
    base = {
        "version": version,
        "created_at": "2026-08-13T00:00:00+00:00",
        "git_sha": "abc123",
        "git_dirty": False,
        "train_seasons": list(SEASONS),
        "validation_seasons": ["2023-2024"],
        "holdout_season": "2024-2025",
        "holdout_touched": False,
        "feature_columns": FEATURES,
        "n_features": len(FEATURES),
        "n_train_rows": 1000,
        "params": {},
        "input_checksums": {"features.parquet": "deadbeef"},
        "metrics": {"walk_forward": {"mean_log_loss": 0.99}},
    }
    return registry.Entry(**{**base, **overrides})


def test_registry_round_trips(tmp_path):
    store = registry.Registry.load(tmp_path)
    store.add(_entry("goals-1"))
    store.save()

    reloaded = registry.Registry.load(tmp_path)
    assert [e.version for e in reloaded.entries] == ["goals-1"]
    assert reloaded.get("goals-1").train_seasons == list(SEASONS)


def test_a_version_identifies_one_artifact(tmp_path):
    store = registry.Registry.load(tmp_path)
    store.add(_entry("goals-1"))
    with pytest.raises(registry.RegistryError, match="already registered"):
        store.add(_entry("goals-1"))


def test_registering_does_not_promote(tmp_path):
    """Roadmap §11.2: promoting whatever was just built ships regressions."""
    store = registry.Registry.load(tmp_path)
    store.add(_entry("goals-1"))
    assert store.promoted is None


def test_promotion_is_exclusive(tmp_path):
    store = registry.Registry.load(tmp_path)
    store.add(_entry("goals-1"))
    store.add(_entry("goals-2"))
    store.promote("goals-1")
    store.promote("goals-2")
    assert [e.version for e in store.entries if e.promoted] == ["goals-2"]


def test_promoting_an_unknown_version_raises(tmp_path):
    store = registry.Registry.load(tmp_path)
    store.add(_entry("goals-1"))
    with pytest.raises(registry.RegistryError, match="no registered model"):
        store.promote("goals-nope")


def test_registry_refuses_a_schema_it_cannot_read(tmp_path):
    (tmp_path / registry.REGISTRY_NAME).write_text(
        json.dumps({"schema": 99, "entries": []}), encoding="utf-8"
    )
    with pytest.raises(registry.RegistryError, match="schema"):
        registry.Registry.load(tmp_path)


# --- feature verification -----------------------------------------------------

def test_verify_features_accepts_an_exact_match():
    registry.verify_features(FEATURES, list(FEATURES))


def test_verify_features_rejects_a_missing_column():
    with pytest.raises(registry.RegistryError, match="missing"):
        registry.verify_features(FEATURES, FEATURES[:-1])


def test_verify_features_rejects_an_extra_column():
    """The quiet one: an extra column does not raise anywhere else."""
    with pytest.raises(registry.RegistryError, match="unexpected"):
        registry.verify_features(FEATURES, [*FEATURES, "surprise"])


def test_verify_features_rejects_a_reordering():
    with pytest.raises(registry.RegistryError, match="order"):
        registry.verify_features(FEATURES, list(reversed(FEATURES)))
