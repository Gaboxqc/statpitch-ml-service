"""Goal-model tests (Design §5.1). Offline, synthetic data with a known answer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statpitch.models import dixon_coles as dc
from statpitch.models.goals import (
    LAMBDA_BOUNDS,
    MIN_COMPETITION_MATCHES,
    GoalModel,
)

FEATURES = ["strength", "noise"]


def _dataset(n=1500, seed=0, home_env=1.6, away_env=1.3, competition_id="ENG.PL"):
    """Goals generated from a known competition environment and a strength signal."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 1, n)
    frame = pd.DataFrame({
        "competition_id": competition_id,
        "strength": strength,
        "noise": rng.normal(0, 1, n),
    })
    lambda_home = home_env * np.exp(0.3 * strength)
    lambda_away = away_env * np.exp(-0.3 * strength)
    home_goals = pd.Series(rng.poisson(lambda_home))
    away_goals = pd.Series(rng.poisson(lambda_away))
    return frame, home_goals, away_goals


@pytest.fixture(scope="module")
def fitted():
    frame, home_goals, away_goals = _dataset()
    model = GoalModel(feature_columns=FEATURES)
    model.fit(frame, home_goals, away_goals)
    model.fit_rho(frame, home_goals, away_goals)
    return model, frame, home_goals, away_goals


# --- goal environments --------------------------------------------------------

def test_environment_is_learned_per_competition(fitted):
    model, _, home_goals, away_goals = fitted
    home_env, away_env = model.environment("ENG.PL")
    assert home_env == pytest.approx(float(home_goals.mean()), abs=1e-9)
    assert away_env == pytest.approx(float(away_goals.mean()), abs=1e-9)


def test_two_competitions_get_different_environments():
    """The measured spread this exists for: Bundesliga runs ~3.07 goals a match."""
    a, ha, aa = _dataset(n=1200, home_env=1.5, away_env=1.2, competition_id="ITA.SERIEA")
    b, hb, ab = _dataset(n=1200, seed=2, home_env=1.9, away_env=1.5,
                         competition_id="GER.BUNDESLIGA")
    frame = pd.concat([a, b], ignore_index=True)
    home_goals = pd.concat([ha, hb], ignore_index=True)
    away_goals = pd.concat([aa, ab], ignore_index=True)

    model = GoalModel(feature_columns=FEATURES).fit(frame, home_goals, away_goals)
    serie_a = model.environment("ITA.SERIEA")
    bundesliga = model.environment("GER.BUNDESLIGA")
    assert bundesliga[0] > serie_a[0]
    assert bundesliga[1] > serie_a[1]


def test_a_thin_competition_falls_back_to_the_pooled_environment():
    frame, home_goals, away_goals = _dataset(n=1200)
    thin, thin_home, thin_away = _dataset(n=20, seed=5, competition_id="FRA.COUPE_DE_FRANCE")
    combined = pd.concat([frame, thin], ignore_index=True)

    model = GoalModel(feature_columns=FEATURES).fit(
        combined,
        pd.concat([home_goals, thin_home], ignore_index=True),
        pd.concat([away_goals, thin_away], ignore_index=True),
    )
    assert "FRA.COUPE_DE_FRANCE" not in model.environments
    assert model.environment("FRA.COUPE_DE_FRANCE") == model.pooled_environment


def test_an_unseen_competition_uses_the_pooled_environment(fitted):
    model, _, _, _ = fitted
    assert model.environment("NEW.LEAGUE") == model.pooled_environment


def test_the_same_features_get_different_rates_in_different_competitions():
    """The offset must differentiate competitions at *inference*, not only in fit.

    Worth pinning explicitly: with a single competition in the training data,
    XGBoost stores the constant training margin as the model's base score, so
    predicting without `base_margin` returns the same number and the offset looks
    like it does nothing. Two competitions is what actually exercises it.
    """
    a, ha, aa = _dataset(n=900, home_env=1.4, away_env=1.1, competition_id="ITA.SERIEA")
    b, hb, ab = _dataset(n=900, seed=8, home_env=2.0, away_env=1.6,
                         competition_id="GER.BUNDESLIGA")
    model = GoalModel(feature_columns=FEATURES).fit(
        pd.concat([a, b], ignore_index=True),
        pd.concat([ha, hb], ignore_index=True),
        pd.concat([aa, ab], ignore_index=True),
    )

    # Identical feature rows, differing only in which competition they belong to.
    rows = a.head(60).copy()
    as_serie_a = model.predict(rows.assign(competition_id="ITA.SERIEA"))[0].mean()
    as_bundesliga = model.predict(rows.assign(competition_id="GER.BUNDESLIGA"))[0].mean()
    assert as_bundesliga > as_serie_a * 1.2


def test_the_offset_is_applied_at_prediction_time():
    """A higher environment must raise predictions for identical features.

    Uses its own model rather than the shared fixture: mutating a module-scoped
    model here once leaked a doubled home environment into every later test,
    which showed up as a 69% implied home-win rate.
    """
    frame, home_goals, away_goals = _dataset(n=800, seed=11)
    model = GoalModel(feature_columns=FEATURES).fit(frame, home_goals, away_goals)

    baseline = model.predict(frame.head(50))[0].mean()
    model.environments["ENG.PL"] = (
        model.environments["ENG.PL"][0] * 2.0,
        model.environments["ENG.PL"][1],
    )
    raised = model.predict(frame.head(50))[0].mean()
    assert raised > baseline * 1.5


# --- prediction ---------------------------------------------------------------

def test_predicted_rates_track_the_true_ones(fitted):
    model, frame, home_goals, away_goals = fitted
    lambda_home, lambda_away = model.predict(frame)
    assert lambda_home.mean() == pytest.approx(float(home_goals.mean()), rel=0.1)
    assert lambda_away.mean() == pytest.approx(float(away_goals.mean()), rel=0.1)


def test_the_strength_signal_is_recovered(fitted):
    """Stronger home sides must be given higher rates than weaker ones."""
    model, frame, _, _ = fitted
    lambda_home, lambda_away = model.predict(frame)
    strong = frame["strength"] > 1.0
    weak = frame["strength"] < -1.0
    assert lambda_home[strong].mean() > lambda_home[weak].mean()
    assert lambda_away[strong].mean() < lambda_away[weak].mean()


def test_rates_are_clipped_to_a_plausible_range(fitted):
    model, frame, _, _ = fitted
    lambda_home, lambda_away = model.predict(frame)
    low, high = LAMBDA_BOUNDS
    assert lambda_home.min() >= low and lambda_home.max() <= high
    assert lambda_away.min() >= low and lambda_away.max() <= high


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(ValueError, match="not fitted"):
        GoalModel(feature_columns=FEATURES).predict(pd.DataFrame())


def test_fitting_an_empty_frame_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        GoalModel(feature_columns=FEATURES).fit(
            pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)
        )


# --- score matrices -----------------------------------------------------------

def test_every_matrix_sums_to_one(fitted):
    model, frame, _, _ = fitted
    for matrix in model.score_matrices(frame.head(40)):
        assert float(matrix.matrix.sum()) == pytest.approx(1.0, abs=1e-12)


def test_implied_one_x_two_sums_to_one(fitted):
    model, frame, _, _ = fitted
    probabilities = model.predict_one_x_two(frame.head(40))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_implied_one_x_two_matches_the_observed_result_rates(fitted):
    """The Phase 3 acceptance check, in miniature.

    Averaged over the sample, the matrix-implied 1X2 must agree with how often
    each result actually occurred. If the matrix were wrong the disagreement
    would show up here before it reached sixty markets.
    """
    model, frame, home_goals, away_goals = fitted
    probabilities = model.predict_one_x_two(frame)
    observed = np.array([
        float((home_goals > away_goals).mean()),
        float((home_goals == away_goals).mean()),
        float((home_goals < away_goals).mean()),
    ])
    assert probabilities.mean(axis=0) == pytest.approx(observed, abs=0.03)


def test_rho_is_clamped_per_match_not_only_per_competition(fitted):
    """A competition-level rho can be invalid for a high-scoring fixture.

    Left unclamped the matrix raises mid-slate, which would take out a whole
    matchday rather than one fixture.
    """
    model, frame, _, _ = fitted
    model.rho["ENG.PL"] = 0.19  # fine on average, not at the top of the range
    high = frame.head(5).copy()
    high["strength"] = 4.0
    for matrix in model.score_matrices(high):
        assert float(matrix.matrix.sum()) == pytest.approx(1.0, abs=1e-12)
        assert np.all(matrix.matrix >= 0)


def test_rho_is_fitted_per_competition(fitted):
    model, _, _, _ = fitted
    assert "ENG.PL" in model.rho
    low, high = dc.rho_bounds(*model.environment("ENG.PL"))
    assert low <= model.rho["ENG.PL"] <= high


def test_thin_competitions_get_no_rho_and_default_to_independence():
    frame, home_goals, away_goals = _dataset(n=1200)
    thin, thin_home, thin_away = _dataset(n=30, seed=6, competition_id="ITA.COPPA_ITALIA")
    combined = pd.concat([frame, thin], ignore_index=True)
    combined_home = pd.concat([home_goals, thin_home], ignore_index=True)
    combined_away = pd.concat([away_goals, thin_away], ignore_index=True)

    model = GoalModel(feature_columns=FEATURES)
    model.fit(combined, combined_home, combined_away)
    model.fit_rho(combined, combined_home, combined_away)

    assert "ITA.COPPA_ITALIA" not in model.rho
    # Falls back to rho=0, i.e. independent Poisson, rather than borrowing another
    # competition's dependence structure.
    assert len(model.score_matrices(thin.head(3))) == 3


def test_min_competition_matches_is_a_meaningful_threshold():
    assert MIN_COMPETITION_MATCHES >= 100
