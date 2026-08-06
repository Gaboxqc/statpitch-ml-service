"""CLV tracker tests (FR-26, FR-29, Design §6.6).

Two invariants carry the weight: CLV must compare like with like, and positive
ROI with negative CLV must be reported as absence of edge rather than success.
"""

from __future__ import annotations

import pytest

from statpitch.decision import clv_tracker as ct


def _entry(odds=2.10, stake=0.01, source=ct.PriceSource.BEST, **kwargs):
    base = {
        "fixture_id": "ENG.PL-2024-08-17-ARS-CHE",
        "competition_id": "ENG.PL",
        "selection": "ah_home_-0.5",
        "market_family": "asian_handicap",
        "odds_taken": odds,
        "price_source": source,
        "p_model": 0.52,
        "q_fair": 0.49,
        "grade": "B",
        "stake_fraction": stake,
        "kelly_lambda": 0.25,
        "w": 0.0,
        "config_version": "test-1",
    }
    return ct.flag(**{**base, **kwargs})


# --- flagging -----------------------------------------------------------------

def test_flagging_records_the_full_decision(ledger_path=None):
    e = _entry()
    assert e.edge_prob == pytest.approx(0.03)
    assert e.config_version == "test-1"
    assert e.price_source == "best"
    assert not e.is_settled


def test_config_version_is_recorded_so_a_backtest_is_reproducible():
    """NFR-12: any historical result must be reproducible from its parameters."""
    assert _entry().config_version


def test_impossible_odds_are_rejected():
    with pytest.raises(ct.LedgerError, match="odds_taken"):
        _entry(odds=0.9)


# --- the like-for-like invariant ----------------------------------------------

def test_settling_against_a_different_price_source_is_refused():
    """The mistake that manufactured +5.4% CLV on every selection in the book.

    A price taken at the best quote and closed against the consensus measures the
    spread between two sources, not the movement of the line.
    """
    e = _entry(source=ct.PriceSource.BEST)
    with pytest.raises(ct.LedgerError, match="like with like"):
        ct.settle(
            e, odds_closing=2.0,
            closing_price_source=ct.PriceSource.CONSENSUS,
            result=ct.Result.WON,
        )


def test_settling_against_the_same_source_is_allowed():
    e = ct.settle(
        _entry(source=ct.PriceSource.BEST), odds_closing=2.0,
        closing_price_source=ct.PriceSource.BEST, result=ct.Result.WON,
    )
    assert e.is_settled


def test_sharp_reference_prices_settle_against_sharp_closing():
    e = ct.settle(
        _entry(source=ct.PriceSource.SHARP), odds_closing=2.0,
        closing_price_source=ct.PriceSource.SHARP, result=ct.Result.LOST,
    )
    assert e.clv_pct == pytest.approx(0.05)


# --- CLV arithmetic -----------------------------------------------------------

def test_clv_is_the_ratio_of_taken_to_closing():
    e = ct.settle(
        _entry(odds=2.10), odds_closing=2.00,
        closing_price_source=ct.PriceSource.BEST, result=ct.Result.LOST,
    )
    assert e.clv_pct == pytest.approx(0.05)


def test_clv_is_negative_when_the_price_improved_after_flagging():
    e = ct.settle(
        _entry(odds=2.00), odds_closing=2.20,
        closing_price_source=ct.PriceSource.BEST, result=ct.Result.WON,
    )
    assert e.clv_pct < 0


def test_clv_does_not_depend_on_the_result():
    """CLV strips the outcome out, which is why it converges so much faster."""
    won = ct.settle(_entry(), odds_closing=2.0,
                    closing_price_source=ct.PriceSource.BEST, result=ct.Result.WON)
    lost = ct.settle(_entry(), odds_closing=2.0,
                     closing_price_source=ct.PriceSource.BEST, result=ct.Result.LOST)
    assert won.clv_pct == lost.clv_pct


def test_clv_in_probability_points_is_recorded_when_available():
    e = ct.settle(
        _entry(), odds_closing=2.0, closing_price_source=ct.PriceSource.BEST,
        result=ct.Result.WON, q_fair_closing=0.52,
    )
    assert e.clv_prob == pytest.approx(0.03)


# --- settlement outcomes ------------------------------------------------------

@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (ct.Result.WON, 1.10), (ct.Result.HALF_WON, 0.55),
        (ct.Result.PUSH, 0.0), (ct.Result.VOID, 0.0),
        (ct.Result.HALF_LOST, -0.5), (ct.Result.LOST, -1.0),
    ],
)
def test_every_outcome_settles_to_the_right_profit(result, expected):
    e = ct.settle(
        _entry(odds=2.10, stake=1.0), odds_closing=2.0,
        closing_price_source=ct.PriceSource.BEST, result=result,
    )
    assert e.profit == pytest.approx(expected)


def test_a_push_returns_the_stake_rather_than_losing_it():
    e = ct.settle(
        _entry(stake=1.0), odds_closing=2.0,
        closing_price_source=ct.PriceSource.BEST, result=ct.Result.PUSH,
    )
    assert e.profit == 0.0


# --- the ledger ---------------------------------------------------------------

def test_ledger_appends_and_reloads(tmp_path):
    path = tmp_path / "bets.jsonl"
    ledger = ct.BetLedger(path)
    ledger.append(_entry())
    ledger.append(_entry(odds=1.95))
    assert len(ledger) == 2

    reopened = ct.BetLedger(path)
    assert len(reopened) == 2
    assert reopened.entries[0].selection == "ah_home_-0.5"


def test_ledger_separates_settled_from_pending(tmp_path):
    ledger = ct.BetLedger(tmp_path / "bets.jsonl")
    ledger.append(_entry())
    ledger.append(
        ct.settle(_entry(), odds_closing=2.0,
                  closing_price_source=ct.PriceSource.BEST, result=ct.Result.WON)
    )
    assert len(ledger.pending()) == 1
    assert len(ledger.settled()) == 1


def test_ledger_persists_settlements(tmp_path):
    path = tmp_path / "bets.jsonl"
    ledger = ct.BetLedger(path)
    ledger.append(_entry())
    ct.settle(ledger.entries[0], odds_closing=2.0,
              closing_price_source=ct.PriceSource.BEST, result=ct.Result.WON)
    ledger._entries[0] = ct.settle(
        ledger._entries[0], odds_closing=2.0,
        closing_price_source=ct.PriceSource.BEST, result=ct.Result.WON,
    )
    ledger.rewrite()
    assert ct.BetLedger(path).settled()


def test_an_empty_ledger_starts_clean(tmp_path):
    assert len(ct.BetLedger(tmp_path / "nothing.jsonl")) == 0


# --- reporting ----------------------------------------------------------------

def _settled(clv_values, results=None, stake=0.01):
    out = []
    for i, clv in enumerate(clv_values):
        taken = 2.0 * (1 + clv)
        result = (results or [ct.Result.LOST] * len(clv_values))[i]
        out.append(
            ct.settle(_entry(odds=taken, stake=stake), odds_closing=2.0,
                      closing_price_source=ct.PriceSource.BEST, result=result)
        )
    return out


def test_report_aggregates_clv_with_dispersion():
    r = ct.report(_settled([0.01, 0.02, 0.03, -0.01]))
    assert r.n == 4
    assert r.mean_clv == pytest.approx(0.0125)
    assert r.clv_se > 0
    assert r.positive_rate == 0.75


def test_report_labels_itself_friday_to_close():
    """Design §6.6: the base snapshot is Friday, not a true opening line."""
    assert ct.report(_settled([0.01])).label == "Friday-to-close CLV"
    assert "Friday" in ct.CLV_LABEL


def test_significant_positive_clv_is_called_evidence():
    r = ct.report(_settled([0.02] * 40 + [0.019] * 40))
    assert r.clv_is_significant
    assert "evidence of genuine edge" in r.verdict()


def test_directionally_positive_but_noisy_clv_says_so():
    r = ct.report(_settled([0.05, -0.04, 0.06, -0.05, 0.02]))
    assert not r.clv_is_significant
    assert "not yet distinguishable from zero" in r.verdict()


def test_positive_roi_with_negative_clv_is_reported_as_no_edge():
    """Design §6.6's reporting rule, applied rather than described.

    Over a few hundred bets a positive ROI is routinely produced by luck. The
    prices are what say whether the selections were right.
    """
    entries = _settled(
        [-0.02] * 6,
        results=[ct.Result.WON] * 5 + [ct.Result.LOST],
    )
    r = ct.report(entries)
    assert r.mean_roi > 0
    assert r.mean_clv < 0
    assert "ABSENCE of demonstrated edge" in r.verdict()


def test_an_empty_report_says_so():
    assert ct.report([]).verdict() == "no settled bets"


def test_unsettled_entries_are_excluded_from_the_report():
    assert ct.report([_entry(), _entry()]).n == 0


def test_report_breaks_down_by_competition():
    a = _settled([0.02])
    b = _settled([-0.01])
    b[0].competition_id = "ESP.LALIGA"
    by = ct.report_by(a + b, "competition_id")
    assert set(by) == {"ENG.PL", "ESP.LALIGA"}
    assert by["ENG.PL"].mean_clv > 0
    assert by["ESP.LALIGA"].mean_clv < 0


def test_report_breaks_down_by_market_family():
    by = ct.report_by(_settled([0.01, 0.02]), "market_family")
    assert "asian_handicap" in by


def test_summary_reports_both_metrics_and_a_verdict():
    text = ct.summarise(_settled([0.01, 0.02, 0.015]))
    assert "Friday-to-close CLV" in text
    assert "ROI" in text
