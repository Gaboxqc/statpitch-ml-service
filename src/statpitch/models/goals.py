"""Expected-goals model feeding the Dixon-Coles matrix (Design §5.1).

Two Poisson regressors — one per side — turn a feature row into the goal rates
`lambda_home` and `lambda_away`. Those rates are the only inputs the score matrix
needs, and the matrix is what every market is derived from, so this model sits
upstream of the entire Decision Layer.

The per-competition goal-environment offset
===========================================

Design §5.1 requires the goal environment to be modelled per competition rather
than pooled, and the measured spread justifies it: Bundesliga runs 3.07 goals per
match against roughly 2.7 in the other four leagues. Pooling would systematically
under-predict German totals and over-predict everyone else's — a bias that lands
squarely on the Over/Under market.

It is implemented as an **offset**, not a feature. XGBoost's `count:poisson`
objective uses a log link, so passing `log(competition mean goals)` as
`base_margin` makes the model learn the *ratio* by which a fixture departs from
its competition's baseline, rather than having to rediscover each league's level
from scratch. A one-hot competition feature could in principle learn the same
thing, but it would spend tree splits doing it and would extrapolate badly to a
competition with few matches — which, after Phase 1, describes most of the cups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from statpitch.models import dixon_coles as dc

log = logging.getLogger(__name__)

#: Fallback goal environment for a competition with too little history to
#: estimate one. Roughly the Big-5 average per side.
DEFAULT_GOAL_ENVIRONMENT = 1.35

#: Below this many matches a competition inherits the pooled environment rather
#: than its own noisy estimate.
MIN_COMPETITION_MATCHES = 200

#: Goal rates are clipped before reaching the score matrix. A tree ensemble can
#: emit an implausible rate on a strange feature row, and the matrix would then
#: return a confident nonsense distribution to sixty markets.
LAMBDA_BOUNDS = (0.15, 5.0)

DEFAULT_PARAMS = {
    "objective": "count:poisson",
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": 0,
}


@dataclass
class GoalModel:
    """Paired Poisson regressors plus the per-competition offsets they assume."""

    feature_columns: list[str]
    home_model: XGBRegressor | None = None
    away_model: XGBRegressor | None = None
    environments: dict[str, tuple[float, float]] = field(default_factory=dict)
    pooled_environment: tuple[float, float] = (
        DEFAULT_GOAL_ENVIRONMENT,
        DEFAULT_GOAL_ENVIRONMENT,
    )
    rho: dict[str, float] = field(default_factory=dict)

    # --- offsets ---------------------------------------------------------

    def environment(self, competition_id: str) -> tuple[float, float]:
        return self.environments.get(competition_id, self.pooled_environment)

    def _offsets(self, competitions: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """log-link base margins, one per row."""
        pairs = [self.environment(c) for c in competitions]
        home = np.log([max(h, 1e-6) for h, _ in pairs])
        away = np.log([max(a, 1e-6) for _, a in pairs])
        return home, away

    # --- fitting ---------------------------------------------------------

    def fit(
        self,
        features: pd.DataFrame,
        home_goals: pd.Series,
        away_goals: pd.Series,
        *,
        params: dict | None = None,
    ) -> GoalModel:
        if features.empty:
            raise ValueError("cannot fit a goal model on an empty frame")

        counts = features["competition_id"].value_counts()
        self.environments = {}
        for competition_id, n in counts.items():
            if n < MIN_COMPETITION_MATCHES:
                continue
            mask = features["competition_id"] == competition_id
            self.environments[str(competition_id)] = (
                float(home_goals[mask].mean()),
                float(away_goals[mask].mean()),
            )
        self.pooled_environment = (float(home_goals.mean()), float(away_goals.mean()))

        skipped = [c for c, n in counts.items() if n < MIN_COMPETITION_MATCHES]
        if skipped:
            log.info(
                "goals: %d competition(s) below %d matches use the pooled "
                "environment: %s",
                len(skipped), MIN_COMPETITION_MATCHES, sorted(map(str, skipped)),
            )

        home_offset, away_offset = self._offsets(features["competition_id"])
        x = features[self.feature_columns]
        settings = {**DEFAULT_PARAMS, **(params or {})}

        self.home_model = XGBRegressor(**settings)
        self.home_model.fit(x, home_goals, base_margin=home_offset)

        self.away_model = XGBRegressor(**settings)
        self.away_model.fit(x, away_goals, base_margin=away_offset)
        return self

    def fit_rho(
        self, features: pd.DataFrame, home_goals: pd.Series, away_goals: pd.Series
    ) -> GoalModel:
        """Fit the Dixon-Coles rho per competition, given this model's rates.

        Must run after `fit`, because rho is defined relative to the goal rates
        and shifts if those change.
        """
        lambda_home, lambda_away = self.predict(features)
        self.rho = {}
        for competition_id, group in features.groupby("competition_id"):
            idx = features.index.get_indexer(group.index)
            if len(idx) < MIN_COMPETITION_MATCHES:
                continue
            self.rho[str(competition_id)] = dc.fit_rho(
                home_goals.to_numpy()[idx],
                away_goals.to_numpy()[idx],
                lambda_home[idx],
                lambda_away[idx],
            )
        return self

    # --- prediction ------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Per-match goal rates, clipped to a plausible range."""
        if self.home_model is None or self.away_model is None:
            raise ValueError("goal model is not fitted")

        home_offset, away_offset = self._offsets(features["competition_id"])
        x = features[self.feature_columns]

        low, high = LAMBDA_BOUNDS
        home = np.clip(self.home_model.predict(x, base_margin=home_offset), low, high)
        away = np.clip(self.away_model.predict(x, base_margin=away_offset), low, high)
        return home, away

    def score_matrices(self, features: pd.DataFrame) -> list[dc.ScoreMatrix]:
        """One Dixon-Coles matrix per row, using that competition's rho."""
        lambda_home, lambda_away = self.predict(features)
        competitions = features["competition_id"].to_numpy()

        matrices = []
        for lh, la, competition_id in zip(lambda_home, lambda_away, competitions, strict=True):
            rho = self.rho.get(str(competition_id), 0.0)
            # A rho fitted on the competition's average rates can still be out of
            # range for an unusually high-scoring fixture, so clamp per match
            # rather than letting the matrix raise mid-slate.
            low, high = dc.rho_bounds(float(lh), float(la))
            matrices.append(
                dc.score_matrix(float(lh), float(la), float(np.clip(rho, low, high)))
            )
        return matrices

    def predict_one_x_two(self, features: pd.DataFrame) -> np.ndarray:
        """1X2 probabilities implied by the score matrices, as (n, 3)."""
        return np.array([m.one_x_two() for m in self.score_matrices(features)])
