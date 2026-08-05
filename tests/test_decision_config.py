"""Decision config tests (NFR-12).

The important behaviour here is negative: the shipped config is a placeholder, and
placeholder parameters must be structurally incapable of sizing a stake.
"""

from __future__ import annotations

import json

import pytest

from statpitch import decision_config
from statpitch.decision_config import DecisionConfigError, load


@pytest.fixture(scope="module")
def cfg():
    return load()


def test_shipped_config_is_a_placeholder_and_refuses_to_stake(cfg):
    assert cfg.is_placeholder
    assert cfg.w is None
    with pytest.raises(DecisionConfigError, match="placeholder"):
        cfg.require_fitted()


def test_config_version_is_present(cfg):
    # Every ledger row records this, so a backtest is reproducible from its params.
    assert cfg.config_version


def test_devig_default_is_not_proportional(cfg):
    # Design §6.2: proportional de-vig overstates longshot fair probability and
    # points value flags at exactly the bets that lose money. It stays available
    # for the FR-28 comparison, but it is not what an unfitted competition falls
    # back to.
    assert cfg.devig_default_method != "proportional"
    assert "proportional" in cfg.raw["devig"]["methods_implemented"]


def test_devig_method_falls_back_to_default_until_notebook_12_fits_it(cfg):
    assert cfg.devig_method("ENG.PL") == cfg.devig_default_method
    assert cfg.devig_method("SOMETHING.UNSEEN") == cfg.devig_default_method


def test_correct_score_is_non_stakeable(cfg):
    assert "correct_score" in cfg.market_engine.non_stakeable_markets


def test_grade_thresholds_are_ordered_and_d_f_never_stake(cfg):
    c = cfg.grading.cutoffs
    assert c["A"] > c["B"] > c["C"] > c["D"]
    assert cfg.grading.stake_multiplier["D"] == 0.0
    assert cfg.grading.stake_multiplier["F"] == 0.0


def test_edge_ceiling_sits_above_the_confidence_peak(cfg):
    # Design §6.4: the peak is where confidence is highest (~4pp); the ceiling is
    # where an apparent edge becomes evidence of model blindness (~12pp).
    assert cfg.grading.e_ceiling > cfg.grading.e_peak


def test_regime_pooling_is_forbidden_by_default(cfg):
    # Requirements §7.3 — pooling pre/post Pinnacle-break odds is a correctness bug.
    assert cfg.allow_pooling_across_regimes is False
    assert cfg.pinnacle_break_date == "2025-07-23"


def test_clv_label_states_the_friday_snapshot_honestly(cfg):
    assert cfg.clv_label == "Friday-to-close CLV"


def test_lambda_frontier_covers_all_four_kelly_fractions(cfg):
    assert cfg.staking.lambda_frontier == (0.10, 0.25, 0.50, 1.00)


# --- validation ---------------------------------------------------------------

def _write(tmp_path, cfg_dict):
    path = tmp_path / "decision_config.json"
    path.write_text(json.dumps(cfg_dict), encoding="utf-8")
    return path


def _base(cfg):
    return json.loads(json.dumps(cfg.raw))


def test_fitted_config_permits_staking(tmp_path, cfg):
    raw = _base(cfg)
    raw["status"] = "fitted"
    raw["market_shrinkage"] = {"w": 0.22, "w_fitted": True}
    loaded = load(_write(tmp_path, raw))
    assert not loaded.is_placeholder
    loaded.require_fitted()  # must not raise


def test_w_outside_unit_interval_is_rejected(tmp_path, cfg):
    raw = _base(cfg)
    raw["market_shrinkage"] = {"w": 1.4, "w_fitted": True}
    with pytest.raises(DecisionConfigError, match=r"\[0, 1\]"):
        load(_write(tmp_path, raw))


def test_w_marked_fitted_but_left_null_is_still_a_placeholder(tmp_path, cfg):
    raw = _base(cfg)
    raw["status"] = "fitted"
    raw["market_shrinkage"] = {"w": None, "w_fitted": True}
    assert load(_write(tmp_path, raw)).is_placeholder


def test_missing_config_version_is_rejected(tmp_path, cfg):
    raw = _base(cfg)
    del raw["config_version"]
    with pytest.raises(DecisionConfigError, match="config_version"):
        load(_write(tmp_path, raw))


def test_misordered_grade_cutoffs_are_rejected(tmp_path, cfg):
    raw = _base(cfg)
    raw["grading"]["cutoffs"] = {"A": 0.5, "B": 0.7, "C": 0.4, "D": 0.3}
    with pytest.raises(DecisionConfigError, match="decrease"):
        load(_write(tmp_path, raw))


def test_ceiling_below_peak_is_rejected(tmp_path, cfg):
    raw = _base(cfg)
    raw["grading"]["e_ceiling"] = 0.01
    with pytest.raises(DecisionConfigError, match="e_ceiling"):
        load(_write(tmp_path, raw))


def test_grade_d_with_a_nonzero_stake_multiplier_is_rejected(tmp_path, cfg):
    raw = _base(cfg)
    raw["grading"]["stake_multiplier"]["D"] = 0.1
    with pytest.raises(DecisionConfigError, match="zero stake multiplier"):
        load(_write(tmp_path, raw))


def test_per_bet_cap_above_matchday_cap_is_rejected(tmp_path, cfg):
    raw = _base(cfg)
    raw["staking"]["cap_per_bet"] = 0.5
    with pytest.raises(DecisionConfigError, match="cap_per_matchday"):
        load(_write(tmp_path, raw))


def test_making_correct_score_stakeable_is_rejected(tmp_path, cfg):
    raw = _base(cfg)
    raw["market_engine"]["non_stakeable_markets"] = []
    with pytest.raises(DecisionConfigError, match="correct_score"):
        load(_write(tmp_path, raw))


def test_unknown_devig_method_for_a_competition_is_rejected(tmp_path, cfg):
    raw = _base(cfg)
    raw["devig"]["method_per_competition"]["ENG.PL"] = "vibes"
    with pytest.raises(DecisionConfigError, match="unknown de-vig method"):
        load(_write(tmp_path, raw))


def test_lambda_out_of_range_is_rejected(tmp_path, cfg):
    raw = _base(cfg)
    raw["staking"]["kelly_lambda"] = 1.5
    with pytest.raises(DecisionConfigError, match="kelly_lambda"):
        load(_write(tmp_path, raw))


def test_reset_cache_reloads(cfg):
    decision_config.reset_cache()
    assert decision_config.config().config_version == cfg.config_version
