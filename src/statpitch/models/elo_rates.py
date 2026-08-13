"""The deployed Elo-to-goal-rate mapping, as a measurable function (Roadmap §2).

`serving/predictor.py` derives goal rates from an Elo difference rather than from
the fitted XGBoost model MODEL_CARD §3 evaluates. That mapping had no row in any
evaluation table: the API served numbers whose log-loss had never been measured,
while shipping alongside a card reporting 0.9845.

This module is the same arithmetic, vectorised over a feature frame, so it can be
scored on the card's validation window. Deliberately pure — every constant is an
argument. `scripts/evaluate_served_path.py` supplies them from `predictor`, and
`tests/test_elo_rates.py` asserts this agrees with the real `Predictor` on real
fixtures, because two implementations of one formula drift the moment nobody is
checking.

Why the rates are split symmetrically
=====================================

A stronger side both scores more and concedes less, so the Elo edge is applied as
`+shift/2` to the home rate and `-shift/2` to the away rate. That keeps expected
total goals roughly constant as the edge grows, which is what stops a lopsided
fixture drifting into the Over/Under market as a side effect of being lopsided.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from statpitch.models import dixon_coles as dc


def rates_from_elo(
    elo_home: np.ndarray,
    elo_away: np.ndarray,
    *,
    base_home: np.ndarray | float,
    base_away: np.ndarray | float,
    home_advantage: np.ndarray | float,
    sensitivity: float,
    elo_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Goal rates implied by a rating difference, as the deployed path computes them."""
    edge = np.asarray(elo_home, dtype=float) - np.asarray(elo_away, dtype=float)
    edge = edge + home_advantage
    shift = sensitivity * edge / elo_scale
    return (
        np.asarray(base_home, dtype=float) * 10.0 ** (shift / 2.0),
        np.asarray(base_away, dtype=float) * 10.0 ** (-shift / 2.0),
    )


def one_x_two(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: np.ndarray | float = 0.0,
    *,
    max_goals: int = dc.DEFAULT_MAX_GOALS,
) -> np.ndarray:
    """1X2 probabilities per row, clamping rho into the range each rate pair allows.

    A rho fitted on a competition's average rates can fall outside the bounds of
    an unusually high-scoring fixture, where it would drive a matrix cell
    negative. Clamping per row is what `predictor._safe_rho` already does on the
    serving path; doing anything else here would measure a different model than
    the one deployed.
    """
    rho_values = np.broadcast_to(np.asarray(rho, dtype=float), np.shape(lambda_home))
    out = np.empty((len(np.asarray(lambda_home)), 3), dtype=float)
    for i, (lh, la, r) in enumerate(zip(lambda_home, lambda_away, rho_values, strict=True)):
        low, high = dc.rho_bounds(float(lh), float(la))
        matrix = dc.score_matrix(
            float(lh), float(la), float(min(max(r, low), high)), max_goals=max_goals
        )
        out[i] = matrix.one_x_two()
    return out


def rates_for_frame(
    frame: pd.DataFrame,
    *,
    environments: dict[str, tuple[float, float]] | None,
    default_base: tuple[float, float],
    home_advantage: float,
    sensitivity: float,
    elo_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the mapping across a feature frame using its as-of-date ratings.

    `home_elo` / `away_elo` are the ratings as they stood before the match, which
    is what makes this a fair evaluation rather than an anachronism: the live
    predictor reads today's rating for a future fixture, and scoring history with
    today's ratings would flatter it with information it will never have.
    """
    environments = environments or {}
    base = np.array(
        [environments.get(str(c), default_base) for c in frame["competition_id"]],
        dtype=float,
    )
    return rates_from_elo(
        frame["home_elo"].to_numpy(dtype=float),
        frame["away_elo"].to_numpy(dtype=float),
        base_home=base[:, 0],
        base_away=base[:, 1],
        home_advantage=home_advantage,
        sensitivity=sensitivity,
        elo_scale=elo_scale,
    )
