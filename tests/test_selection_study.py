"""Sharp-reference selection study (Plan §4 Phase C).

Offline, on synthetic prices. The point of these is not to re-derive the
measurement — that is `scripts/study_selection_rules.py` against the committed
archive — but to pin the properties that decide whether the measurement means
anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statpitch.data import football_data_live as live
from statpitch.decision import selection_study as st

ORDER = ["home", "draw", "away"]


def _rows(match, *, avg, close, ref, mx=None, season="2025-2026"):
    """One match's three selections, preclose and close.

    The best quote sits 10% above the consensus so that it clears the reference's
    de-vigged fair value; at 3% every edge is negative and the rules select
    nothing, which is a property of real efficient books but makes for a fixture
    that cannot exercise anything.
    """
    mx = mx or [p * 1.10 for p in avg]
    out = []
    for i, sel in enumerate(ORDER):
        out.append(
            {
                "match_id": match, "selection": sel, "season": season,
                "competition_id": "ENG.PL", "market": "1x2",
                "odds_schema_era": "modern", "odds_regime": "post_2025_07_23",
                "snapshot": "preclose",
                "odds_avg": avg[i], "odds_max": mx[i], "odds_bfe": ref[i],
                "odds_pinnacle": ref[i], "odds_b365": ref[i],
            }
        )
        out.append(
            {
                "match_id": match, "selection": sel, "season": season,
                "competition_id": "ENG.PL", "market": "1x2",
                "odds_schema_era": "modern", "odds_regime": "post_2025_07_23",
                "snapshot": "close",
                "odds_avg": close[i], "odds_max": close[i] * 1.03,
                "odds_bfe": close[i], "odds_pinnacle": close[i], "odds_b365": close[i],
            }
        )
    return out


@pytest.fixture
def odds():
    rows = []
    for i in range(60):
        rows += _rows(f"m{i}", avg=[2.10, 3.40, 3.60], close=[2.05, 3.45, 3.65],
                      ref=[2.12, 3.38, 3.58])
    return pd.DataFrame(rows)


# --- which references can actually be used ------------------------------------

def test_pinnacle_is_not_in_the_live_feed():
    """The whole reason this study exists — the rule that works cannot be traded."""
    assert not st.in_live_feed("odds_pinnacle")
    assert "PS" not in live.LIVE_BOOK_PREFIXES


def test_the_exchange_and_b365_are_in_the_live_feed():
    assert st.in_live_feed("odds_bfe")
    assert st.in_live_feed("odds_b365")


def test_the_consensus_aggregates_count_as_available():
    """`Avg` and `Max` are columns of fixtures.csv, not books."""
    assert st.in_live_feed("odds_avg")
    assert st.in_live_feed("odds_max")


def test_feed_membership_is_derived_not_restated():
    """Checked against the live module, so it cannot drift out of date."""
    for book in ("BFE", "B365"):
        assert book in live.LIVE_BOOK_PREFIXES


# --- clustering ---------------------------------------------------------------

def test_clustering_widens_the_error_when_observations_repeat():
    """Three selections on one match settle from one scoreline.

    Treating them as independent overstates t; the clustered statistic must be
    the smaller of the two on perfectly correlated groups.
    """
    values = np.array([0.02, 0.02, 0.02, -0.01, -0.01, -0.01] * 10, dtype=float)
    groups = np.repeat(np.arange(20), 3)
    naive = values.mean() / (values.std(ddof=1) / np.sqrt(len(values)))
    assert abs(st.clustered_t(values, groups)) < abs(naive)


def test_clustering_matches_the_naive_error_when_every_group_is_a_singleton():
    values = np.array([0.01, -0.02, 0.03, 0.00, 0.015, -0.005] * 8, dtype=float)
    groups = np.arange(len(values))
    naive = values.mean() / (values.std(ddof=1) / np.sqrt(len(values)))
    assert st.clustered_t(values, groups) == pytest.approx(naive, rel=0.05)


def test_a_single_group_yields_no_statistic():
    values = np.array([0.01, 0.02, 0.03])
    assert np.isnan(st.clustered_t(values, np.array([1, 1, 1])))


# --- the scoring choice that separates signal from artifact -------------------

def test_the_default_scoring_is_consensus_to_consensus(odds):
    """Scored on the column the rule does NOT select with.

    Every rule selects using `odds_max`. Scoring `max -> max` would score a
    variable on itself and reward mean reversion of the max-vs-consensus spread.
    """
    frame = st.wide_frame(odds, regime="post_2025_07_23")
    frame = frame.merge(
        st.devigged(frame, "odds_bfe"),
        left_on=["match_id", "selection"], right_index=True, how="left",
    )
    result = st.evaluate(frame, "odds_bfe", 0.0)
    assert result is not None
    # The consensus shortened on home, so backing home has positive CLV.
    assert result.mean_clv != 0.0


def test_a_rule_that_selects_nothing_returns_none(odds):
    frame = st.wide_frame(odds, regime="post_2025_07_23")
    frame = frame.merge(
        st.devigged(frame, "odds_bfe"),
        left_on=["match_id", "selection"], right_index=True, how="left",
    )
    assert st.evaluate(frame, "odds_bfe", 10.0) is None


def test_an_unknown_reference_returns_none(odds):
    frame = st.wide_frame(odds, regime="post_2025_07_23")
    assert st.evaluate(frame, "odds_nonexistent", 0.0) is None


# --- the baseline control -----------------------------------------------------

def test_the_study_always_reports_an_unselected_baseline(odds):
    """Without it, a rule inheriting a market-wide drift looks like skill."""
    results, _ = st.study(odds, regime="post_2025_07_23")
    assert results
    assert results[0].reference == "none (whole book)"
    assert results[0].n_matches == 60


def test_the_baseline_covers_every_match_not_just_selected_ones(odds):
    frame = st.wide_frame(odds, regime="post_2025_07_23")
    control = st.baseline(frame)
    assert control is not None
    assert control.n_selections == len(frame)


# --- regimes and the holdout --------------------------------------------------

def test_regimes_are_never_pooled(odds):
    """Pinnacle left the published aggregates on 2025-07-23.

    That changes what "the best quote" means, so the two regimes are different
    experiments and `allow_pooling_across_regimes` is false for a reason.
    """
    pre, _ = st.study(odds, regime="pre_2025_07_23")
    assert pre == []


def test_the_holdout_season_can_be_excluded(odds):
    _, meta = st.study(odds, regime="post_2025_07_23", exclude_seasons=("2025-2026",))
    assert meta["matches"] == 0
    assert meta["excluded_seasons"] == ["2025-2026"]


def test_significance_uses_the_clustered_statistic():
    result = st.CLVResult(
        reference="odds_bfe", in_live_feed=True, threshold=0.0,
        n_selections=100, n_matches=90, mean_clv=0.02,
        clustered_t=1.5, naive_t=9.9, positive_rate=0.6,
    )
    # Naive t would clear the bar; the clustered one is what counts.
    assert not result.is_significant


def test_a_devigged_book_sums_to_one(odds):
    frame = st.wide_frame(odds, regime="post_2025_07_23")
    probabilities = st.devigged(frame, "odds_avg")
    totals = probabilities.groupby(level="match_id").sum()
    assert totals.min() == pytest.approx(1.0)
    assert totals.max() == pytest.approx(1.0)


def test_an_incomplete_book_is_dropped_rather_than_half_devigged(odds):
    holed = odds.copy()
    mask = (holed.match_id == "m0") & (holed.selection == "draw")
    holed.loc[mask, "odds_avg"] = np.nan
    frame = st.wide_frame(holed, regime="post_2025_07_23")
    probabilities = st.devigged(frame, "odds_avg")
    assert "m0" not in set(probabilities.index.get_level_values("match_id"))


def test_devigged_output_is_named_for_what_evaluate_reads(odds):
    """These two must compose without an intervening rename.

    They did not: `devigged` named its output after the wide frame's column
    (`odds_bfe_pre_q`) while `evaluate` looked for `odds_bfe_q`, so composing
    them selected nothing at all unless the caller renamed in between.
    """
    frame = st.wide_frame(odds, regime="post_2025_07_23")
    probabilities = st.devigged(frame, "odds_bfe")
    assert probabilities.name == "odds_bfe_q"

    joined = frame.merge(
        probabilities, left_on=["match_id", "selection"], right_index=True, how="left"
    )
    assert st.evaluate(joined, "odds_bfe", 0.0) is not None
