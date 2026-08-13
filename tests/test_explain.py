"""Per-prediction explanations (FR-32, Roadmap §6.1).

An explanation is a claim about *why* a number came out, and a wrong one is worse
than none: it is confidently wrong about the reasoning rather than about the
answer, and it invites exactly the wrong conclusion about what drove a fixture.

Two properties carry that weight.

**The parts reconstruct the whole.** Contributions that sum to something other
than the prediction describe a different model. Keeping only the top few would
break this, which is why the remainder is retained as `other` rather than
dropped.

**The base is per competition, not per row.** That is what makes "this fixture is
1.36× its league's rate" true rather than merely plausible. If SHAP were
describing a different model, or the feature columns were misaligned, the derived
base would wander from row to row while the contributions still summed correctly
— every one of them measured from a different starting point, with nothing in the
totals to show it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statpitch.models import explain, training
from statpitch.models.goals import GoalModel

FEATURES = ["elo_diff", "form_diff_5", "rest_diff"]


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(7)
    n = 900
    elo_diff = rng.normal(0, 120, n)
    return pd.DataFrame(
        {
            "season": ["2023-2024"] * n,
            "competition_id": rng.choice(["ENG.PL", "GER.BUNDESLIGA"], n),
            "elo_diff": elo_diff,
            "form_diff_5": rng.normal(0, 1, n),
            "rest_diff": rng.normal(0, 2, n),
            "home_goals": rng.poisson(np.clip(1.45 * np.exp(elo_diff / 400), 0.2, 4)),
            "away_goals": rng.poisson(np.clip(1.20 * np.exp(-elo_diff / 400), 0.2, 4)),
        }
    ).assign(
        result=lambda d: np.where(
            d.home_goals > d.away_goals, "H",
            np.where(d.home_goals == d.away_goals, "D", "A"),
        )
    )


@pytest.fixture(scope="module")
def model(frame):
    return training.fit_model(frame, FEATURES)


@pytest.mark.parametrize("side", ["home", "away"])
def test_contributions_reconstruct_the_prediction(model, frame, side):
    values = explain.shap_values(model, frame, side)
    base = explain.base_log_rates(model, frame, side, values)
    predicted = model.predict(frame)[0 if side == "home" else 1]
    np.testing.assert_allclose(
        np.exp(values.sum(axis=1) + base), predicted, rtol=1e-5
    )


@pytest.mark.parametrize("side", ["home", "away"])
def test_the_base_is_flat_within_a_competition(model, frame, side):
    """The interpretive claim, asserted on its own terms.

    Constancy within a competition is the invariant. The offset between the
    derived base and the fitted environment is a model-dependent float32
    artefact, so pinning it would only buy a tolerance loose enough to catch
    nothing.
    """
    values = explain.shap_values(model, frame, side)
    spread = explain.check_base_is_per_competition(model, frame, side, values)
    assert spread < explain.BASE_TOLERANCE


def test_a_wandering_base_is_caught(model, frame):
    """What a mismatched model or misaligned features would look like.

    Corrupting the values makes the derived base move per row while the
    contributions still sum to something — which is precisely the failure the
    check exists to catch, and which additivity alone cannot see.
    """
    values = explain.shap_values(model, frame, "home")
    corrupted = values + np.linspace(0.0, 0.5, values.shape[0])[:, None]
    with pytest.raises(explain.ExplanationError, match="per-competition"):
        explain.check_base_is_per_competition(model, frame, "home", corrupted)


def test_top_contributions_still_sum_to_the_whole(model, frame):
    """`other` is what keeps a truncated list honest."""
    values = explain.shap_values(model, frame, "home")
    per_row = explain.top_contributions(model, frame, "home", top_n=2)
    for row_index, contributions in enumerate(per_row):
        assert sum(c.value for c in contributions) == pytest.approx(
            values[row_index].sum(), abs=1e-9
        )


def test_no_remainder_row_when_nothing_was_dropped(model, frame):
    per_row = explain.top_contributions(model, frame, "home", top_n=len(FEATURES))
    assert all(c.feature != "other" for c in per_row[0])


def test_contributions_are_ranked_by_magnitude_not_sign(model, frame):
    """A feature arguing against the rate is part of the explanation.

    Ranking by signed value would turn an explanation into a justification.
    """
    contributions = explain.top_contributions(model, frame, "home", top_n=3)[0]
    magnitudes = [abs(c.value) for c in contributions if c.feature != "other"]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_the_multiplier_is_the_exponent_of_the_contribution(model, frame):
    """The log link is the whole reason these are not "goals"."""
    contribution = explain.top_contributions(model, frame, "home", top_n=1)[0][0]
    assert contribution.multiplier == pytest.approx(np.exp(contribution.value))


def test_elo_diff_dominates(model, frame):
    """MODEL_CARD §4 names it the strongest input; an explanation should agree.

    The synthetic fixture generates goals from `elo_diff` alone — the other two
    features are noise — so it should lead more often than both of them together.
    A threshold much above that would be measuring this fixture's sample size
    rather than the model's attribution.
    """
    leading = [c[0].feature for c in explain.top_contributions(model, frame, "home", top_n=1)]
    others = len(leading) - leading.count("elo_diff")
    assert leading.count("elo_diff") > others


def test_explanations_frame_covers_both_sides(model, frame):
    keys = pd.Series([f"f{i}" for i in range(len(frame))])
    out = explain.explanations_frame(model, frame, keys, top_n=3)
    assert set(out["side"]) == {"home", "away"}
    assert set(out.columns) >= {
        "fixture_id", "side", "rank", "feature", "value", "multiplier",
    }
    assert out["fixture_id"].nunique() == len(frame)


def test_an_unfitted_model_cannot_be_explained(frame):
    with pytest.raises(explain.ExplanationError, match="not fitted"):
        explain.shap_values(GoalModel(feature_columns=FEATURES), frame, "home")


def test_an_unknown_side_is_refused(model, frame):
    with pytest.raises(explain.ExplanationError, match="side must be"):
        explain.shap_values(model, frame, "neutral")
