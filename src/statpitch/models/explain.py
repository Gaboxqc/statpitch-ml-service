"""Per-prediction explanations (FR-32, Roadmap §6.1).

`shap>=0.46` has been a declared dependency since the first commit, with the
comment "FR-32 per-bet explainability", and nothing imported it. This is that
requirement, wired.

A probability on its own is not reviewable. "Home 72%" cannot be argued with;
"the rating gap is worth +0.31 in log-rate, the away side's winless run +0.06,
and the home side's extra rest +0.02" can be — and a reader who disagrees with
the model can now say where.

What the numbers mean, exactly
==============================

The goal model is XGBoost `count:poisson`, so the trees work on a **log link**.
SHAP values are therefore additive in log-rate space and multiplicative on goals:
a contribution of +0.31 multiplies the rate by e^0.31 ≈ 1.36. Reporting them as
"+0.31 goals" would be wrong by however far the fixture sits from the baseline,
so `multiplier` is carried alongside and the units are named in the output.

The base value is the competition's goal environment
====================================================

`GoalModel` passes `log(competition mean goals)` as `base_margin`, and XGBoost
uses base_margin **in place of** `base_score` rather than in addition to it, so

    log λ = Σ shap + log(competition baseline)

which is precisely what `models/goals.py` says the model is built to learn — "the
*ratio* by which a fixture departs from its competition's baseline, rather than
having to rediscover each league's level from scratch". An explanation therefore
reads as: this competition scores 1.52 at home, and this fixture is 1.36× that.

In practice the two sides of that identity differ by a **constant 4.5e-4** in
log-rate (spread 2.5e-7 across 400 rows), because XGBoost carries its intercept
in float32 while the environment is recomputed in float64. That is 0.045% on a
goal rate and of no consequence to a reader — but it matters to how the check is
built.

So the base is derived as `log λ − Σ shap`, which makes the contributions
reconstruct the prediction *exactly*, and the interpretive claim — that they are
measured against the competition's own baseline — is asserted separately by
`check_base_is_per_competition`. Folding the residual into an additivity check
instead would force a tolerance loose enough to hide a real mismatch, and a SHAP
implementation quietly disagreeing with the model is worse than no explanation:
it is confidently wrong about the reasoning rather than about the answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from statpitch.models.goals import GoalModel

log = logging.getLogger(__name__)

#: How many contributions to keep per side. Enough to explain a prediction,
#: short enough to render. The remainder is summed into `other` so the parts
#: still add up — dropping it would make the explanation not reconstruct.
DEFAULT_TOP_N = 6

#: How much the derived base may vary *within* one competition, in log-rate. The
#: measured spread is 2e-7 — float32 round-trip — so this is two orders of
#: magnitude of headroom and still far below anything a real misalignment causes.
BASE_TOLERANCE = 1e-5

SIDES = ("home", "away")


class ExplanationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Contribution:
    feature: str
    #: The feature's value for this fixture, for a reader who wants to check it.
    feature_value: float
    #: Additive contribution in log-rate space.
    value: float

    @property
    def multiplier(self) -> float:
        return float(np.exp(self.value))


def _side_model(model: GoalModel, side: str):
    if side not in SIDES:
        raise ExplanationError(f"side must be one of {SIDES}, not {side!r}")
    booster = model.home_model if side == "home" else model.away_model
    if booster is None:
        raise ExplanationError("goal model is not fitted")
    return booster


def shap_values(model: GoalModel, frame: pd.DataFrame, side: str) -> np.ndarray:
    """Raw SHAP values for one side, as (rows, features) in log-rate space."""
    import shap

    booster = _side_model(model, side)
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(frame[model.feature_columns])
    return np.asarray(values, dtype=float)


def environment_log_rates(
    model: GoalModel, frame: pd.DataFrame, side: str
) -> np.ndarray:
    """log of each row's competition goal environment."""
    index = 0 if side == "home" else 1
    return np.log(
        [max(model.environment(str(c))[index], 1e-6) for c in frame["competition_id"]]
    )


def base_log_rates(
    model: GoalModel, frame: pd.DataFrame, side: str, values: np.ndarray
) -> np.ndarray:
    """The value the contributions add to, per row — exact by construction.

    Defined as `log λ − Σ shap` rather than assembled from XGBoost's intercept.
    That is a deliberate choice, and the reason is worth recording.

    `GoalModel` passes `log(competition mean goals)` as `base_margin`, and
    XGBoost uses base_margin **in place of** `base_score`, so the base *should* be
    the competition's goal environment exactly. Measured, it sits a constant
    4.5e-4 in log-rate away from it — spread 2.5e-7 across 400 rows, so systematic
    rather than noise, and traceable to XGBoost carrying its intercept in float32
    while the environment is recomputed in float64. It is 0.045% on a goal rate.

    Assembling the base from the intercept would leave that residual in the
    additivity check, forcing a tolerance loose enough to hide a genuine mismatch.
    Deriving it from the prediction instead makes additivity exact, and moves the
    real claim — that these contributions are measured against the competition's
    own baseline — into `check_base_is_per_competition`, where it is asserted on
    its own terms.
    """
    predicted = model.predict(frame)[0 if side == "home" else 1]
    return np.log(predicted) - values.sum(axis=1)


def check_base_is_per_competition(
    model: GoalModel, frame: pd.DataFrame, side: str, values: np.ndarray
) -> float:
    """Assert the contributions are measured from a per-competition baseline.

    This is the claim an explanation makes when it says "this fixture is 1.36× its
    competition's rate". If SHAP were describing a different model, or the feature
    columns were misaligned, the derived base would wander from row to row while
    the contributions still summed to the prediction — every one of them measured
    from a different starting point, and nothing about the totals would show it.

    **Constancy within a competition is the invariant, not closeness to the fitted
    environment.** The two sit a constant distance apart — 4.5e-4 for the shipped
    model, 5.9e-3 for a small synthetic one — because XGBoost carries its
    intercept in float32 and applies it where the environment is recomputed in
    float64. That distance is model-dependent and of no consequence at 0.045% on a
    rate, so pinning it would only produce a tolerance that has to be loosened
    until it catches nothing. The spread *within* a competition is 2e-7, and a
    misalignment moves it by orders of magnitude.

    Rows where `predict` clipped to `LAMBDA_BOUNDS` are skipped: the clip is
    applied after the model rather than by it, so a clipped row legitimately fails
    to reconstruct and reporting it would blame the guardrail.
    """
    from statpitch.models.goals import LAMBDA_BOUNDS

    predicted = model.predict(frame)[0 if side == "home" else 1]
    low, high = LAMBDA_BOUNDS
    unclipped = (predicted > low * 1.001) & (predicted < high * 0.999)
    if not unclipped.any():
        return 0.0

    derived = base_log_rates(model, frame, side, values)
    grouped = pd.DataFrame(
        {"competition_id": frame["competition_id"].to_numpy(), "base": derived}
    )[unclipped]
    spread = grouped.groupby("competition_id")["base"].agg(lambda s: s.max() - s.min())
    worst = float(spread.max()) if len(spread) else 0.0

    if worst > BASE_TOLERANCE:
        offender = spread.idxmax()
        raise ExplanationError(
            f"{side} contributions are not measured from a per-competition "
            f"baseline: within {offender!r} the derived base varies by {worst:.2e} "
            f"in log-rate, tolerance {BASE_TOLERANCE:.0e}. The contributions would "
            "each be measured from a different starting point while still summing "
            "to the prediction."
        )
    return worst


def top_contributions(
    model: GoalModel,
    frame: pd.DataFrame,
    side: str,
    *,
    top_n: int = DEFAULT_TOP_N,
    verify: bool = True,
) -> list[list[Contribution]]:
    """The `top_n` largest contributions per row, plus an `other` remainder.

    Ranked by absolute value: a feature arguing strongly *against* the rate is as
    much a part of the explanation as one arguing for it, and dropping negatives
    would turn an explanation into a justification.
    """
    values = shap_values(model, frame, side)
    if verify:
        check_base_is_per_competition(model, frame, side, values)

    columns = list(model.feature_columns)
    feature_values = frame[columns].to_numpy(dtype=float)

    out: list[list[Contribution]] = []
    for row_index in range(values.shape[0]):
        row = values[row_index]
        order = np.argsort(-np.abs(row))[:top_n]
        kept = [
            Contribution(
                feature=columns[i],
                feature_value=float(feature_values[row_index, i]),
                value=float(row[i]),
            )
            for i in order
        ]
        remainder = float(row.sum() - sum(c.value for c in kept))
        if abs(remainder) > 1e-9:
            # Kept so the parts still reconstruct the whole. An explanation whose
            # listed contributions silently omit a third of the movement invites
            # exactly the wrong conclusion about what drove it.
            kept.append(
                Contribution(feature="other", feature_value=float("nan"), value=remainder)
            )
        out.append(kept)
    return out


def explanations_frame(
    model: GoalModel,
    frame: pd.DataFrame,
    keys: pd.Series,
    *,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Long-format explanations for both sides, ready to persist."""
    records: list[dict] = []
    for side in SIDES:
        values = shap_values(model, frame, side)
        base = base_log_rates(model, frame, side, values)
        per_row = top_contributions(model, frame, side, top_n=top_n)
        for row_index, contributions in enumerate(per_row):
            for rank, contribution in enumerate(contributions):
                records.append(
                    {
                        "fixture_id": str(keys.iloc[row_index]),
                        "side": side,
                        "rank": rank,
                        "feature": contribution.feature,
                        "feature_value": contribution.feature_value,
                        "value": contribution.value,
                        "multiplier": contribution.multiplier,
                        "base_log_rate": float(base[row_index]),
                    }
                )
    return pd.DataFrame.from_records(records)
