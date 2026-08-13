"""The promotion gate (Roadmap §11.2).

Once a retrain runs unattended, something has to decide whether what it built
should be served. Promoting whatever was just trained is a mechanism for shipping
a regression quietly; refusing to promote anything makes the loop pointless.

The rule is **not worse**, not better. Requiring an improvement every week would
pin the served model to whichever week got a lucky validation split, and a model
that merely matches the incumbent on fresher data is worth having. "Worse" is
measured against the incumbent's own fold-to-fold spread, because a difference
smaller than the disagreement between seasons is not a difference — which is why
`aggregate` reports a standard deviation at all.
"""

from __future__ import annotations

from statpitch.models import registry


def _entry(version: str, mean: float | None = 0.99, std: float = 0.02, **overrides):
    metrics = {}
    if mean is not None:
        metrics["walk_forward"] = {"mean_log_loss": mean, "std_log_loss": std}
    base = {
        "version": version,
        "created_at": "2026-08-13T00:00:00+00:00",
        "git_sha": "abc123",
        "git_dirty": False,
        "train_seasons": ["2022-2023"],
        "validation_seasons": ["2023-2024"],
        "holdout_season": "2024-2025",
        "holdout_touched": False,
        "feature_columns": ["elo_diff"],
        "n_features": 1,
        "n_train_rows": 1000,
        "params": {},
        "input_checksums": {},
        "metrics": metrics,
    }
    return registry.Entry(**{**base, **overrides})


def test_the_first_scored_artifact_is_promoted():
    decision = registry.gate(_entry("goals-1"), None)
    assert decision.promote
    assert "no incumbent" in decision.reason


def test_a_clearly_better_candidate_is_promoted():
    decision = registry.gate(_entry("goals-2", mean=0.95), _entry("goals-1", mean=0.99))
    assert decision.promote


def test_a_clearly_worse_candidate_is_refused():
    decision = registry.gate(_entry("goals-2", mean=1.10), _entry("goals-1", mean=0.99))
    assert not decision.promote
    assert "worse" in decision.reason


def test_a_candidate_worse_only_within_fold_noise_is_promoted():
    """Not worse, rather than better — otherwise the loop chases lucky splits."""
    incumbent = _entry("goals-1", mean=0.9900, std=0.0200)
    decision = registry.gate(_entry("goals-2", mean=0.9950), incumbent)
    assert decision.promote
    assert decision.noise == 0.0200


def test_the_margin_is_the_incumbents_own_spread():
    """A model with stable folds gets a tighter bar than one with noisy folds."""
    noisy = registry.gate(
        _entry("c", mean=1.00), _entry("i", mean=0.99, std=0.05)
    )
    stable = registry.gate(
        _entry("c", mean=1.00), _entry("i", mean=0.99, std=0.001)
    )
    assert noisy.promote
    assert not stable.promote


def test_a_candidate_that_saw_the_holdout_is_refused_outright():
    """NFR-10. No score can rescue this, so the score is never consulted."""
    candidate = _entry("goals-2", mean=0.5, holdout_touched=True)
    decision = registry.gate(candidate, _entry("goals-1"))
    assert not decision.promote
    assert "holdout" in decision.reason


def test_an_unscored_candidate_is_refused():
    decision = registry.gate(_entry("goals-2", mean=None), _entry("goals-1"))
    assert not decision.promote
    assert "no walk-forward score" in decision.reason


def test_an_unscored_incumbent_blocks_automatic_promotion():
    """Nothing to compare against is not the same as passing the comparison."""
    decision = registry.gate(_entry("goals-2"), _entry("goals-1", mean=None))
    assert not decision.promote
    assert "nothing to compare" in decision.reason


def test_the_reason_is_recorded_whether_or_not_it_promotes():
    """A refusal must be readable months later without rerunning it."""
    for candidate_mean in (0.90, 1.20):
        decision = registry.gate(
            _entry("c", mean=candidate_mean), _entry("i", mean=0.99)
        )
        assert decision.reason
        assert decision.candidate_log_loss == candidate_mean
        assert decision.incumbent_log_loss == 0.99


def test_gate_decides_but_does_not_promote():
    """Deciding and doing are separate, so a caller can log a refusal and stop."""
    incumbent = _entry("goals-1", promoted=True)
    candidate = _entry("goals-2", mean=0.5)
    decision = registry.gate(candidate, incumbent)
    assert decision.promote
    # Nothing moved: the gate reports, the caller acts.
    assert candidate.promoted is False
    assert incumbent.promoted is True
