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


def test_a_placeholder_config_refuses_to_stake(cfg):
    """The invariant, asserted on a constructed placeholder.

    This used to assert the SHIPPED config was one. That was a fact about the
    committed file rather than a property of the code, and the file moved to
    `experimental` when the Pinnacle-referenced rule went live. The guarantee
    that an unfitted config cannot size a stake did not move with it.
    """
    from dataclasses import replace

    placeholder = replace(cfg, status="placeholder", w_fitted=False, w=None)
    assert placeholder.is_placeholder
    with pytest.raises(DecisionConfigError, match="placeholder"):
        placeholder.require_fitted()


def test_an_unfitted_w_alone_is_enough_to_refuse(cfg):
    """Both halves of `is_placeholder` are load-bearing.

    A config could be marked fitted while `w` had never been estimated; that is
    still a placeholder for staking purposes.
    """
    from dataclasses import replace

    assert replace(cfg, status="fitted", w_fitted=False).is_placeholder


def test_the_shipped_config_records_which_state_it_is_in(cfg):
    """Whatever state it is in, it must be a known one and name itself."""
    assert cfg.status in {"placeholder", "experimental", "fitted"}
    assert cfg.config_version.endswith(cfg.status)


def test_a_shipped_config_that_stakes_must_carry_a_selection_rule(cfg):
    """Staking without a rule would size whatever graded highest, which is the
    max-edge selection MODEL_CARD §4 measured at -2.12% ROI."""
    if cfg.is_placeholder:
        return
    assert cfg.selection_rule.is_active
    assert cfg.selection_rule.market_families


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


# --- benchmark (decided: consensus closing is primary) ------------------------

def test_primary_benchmark_is_the_consensus_closing_price(cfg):
    assert cfg.benchmark.primary == "consensus_closing"
    assert cfg.benchmark.primary_price_column == "odds_avg"


def test_primary_window_starts_where_consensus_closing_odds_do(cfg):
    # Verified against the live archive: no AvgC*/MaxC* columns before 2019/20.
    assert cfg.benchmark.primary_first_season == "2019-2020"
    assert cfg.benchmark.covers("2019-2020")
    assert not cfg.benchmark.covers("2018-2019")


def test_primary_window_stops_at_the_pinnacle_regime_break(cfg):
    assert not cfg.benchmark.covers("2025-2026")
    assert cfg.benchmark.post_break_season_held_separately == "2025-2026"


def test_a_post_break_row_is_excluded_even_inside_the_season_range(cfg):
    assert cfg.benchmark.covers("2024-2025", odds_regime="pre_2025_07_23")
    assert not cfg.benchmark.covers("2024-2025", odds_regime="post_2025_07_23")


def test_pinnacle_is_secondary_and_reaches_further_back(cfg):
    assert cfg.benchmark.secondary == "pinnacle_closing"
    assert cfg.benchmark.secondary_price_column == "odds_pinnacle"
    assert cfg.benchmark.secondary_first_season == "2012-2013"


def test_holdout_is_inside_the_window_and_excluded_from_training(cfg):
    b = cfg.benchmark
    assert b.covers(b.holdout_season)
    assert b.is_holdout("2024-2025")
    assert "2024-2025" not in b.training_seasons()
    assert len(b.training_seasons()) >= 2   # Requirements §8.3


def test_training_seasons_are_contiguous_pre_break_seasons(cfg):
    assert cfg.benchmark.training_seasons() == [
        "2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024",
    ]


def test_max_odds_may_never_be_the_benchmark_price(tmp_path, cfg):
    """FR-16a, enforced rather than documented.

    Max-of-N is above consensus by construction; de-vigging it produces fair
    probabilities that look like free money and are not.
    """
    raw = _base(cfg)
    raw["benchmark"]["primary_price_column"] = "odds_max"
    with pytest.raises(DecisionConfigError, match="above consensus by construction"):
        load(_write(tmp_path, raw))


def test_holdout_outside_the_window_is_rejected(tmp_path, cfg):
    raw = _base(cfg)
    raw["benchmark"]["holdout_season"] = "2015-2016"
    with pytest.raises(DecisionConfigError, match="outside the primary benchmark window"):
        load(_write(tmp_path, raw))


def test_missing_holdout_is_rejected(tmp_path, cfg):
    raw = _base(cfg)
    del raw["benchmark"]["holdout_season"]
    with pytest.raises(DecisionConfigError, match="holdout_season is required"):
        load(_write(tmp_path, raw))


def test_a_window_that_is_only_the_holdout_is_rejected(tmp_path, cfg):
    raw = _base(cfg)
    raw["benchmark"]["primary_window"]["first_season"] = "2024-2025"
    with pytest.raises(DecisionConfigError, match="nothing but the holdout"):
        load(_write(tmp_path, raw))


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
