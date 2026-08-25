"""Odds API price capture (matchday build).

Offline. The payload shape is trimmed from a real
`/v4/sports/soccer_epl/odds?regions=eu&markets=h2h` response.

The reason this exists at all: the bookmaker panel here includes **pinnacle**.
MODEL_CARD §5's +0.51% CLV is defined on Pinnacle-referenced selections, and
Phase C recorded the blocker as "Pinnacle is not published in the live fixture
feed" — true of football-data.co.uk, and not of this source.
"""

from __future__ import annotations

import pandas as pd
import pytest

from statpitch.data import odds_api


def _event(home, away, stamp="2026-08-28T19:00:00Z", *, books=None):
    return {
        "id": "e1", "home_team": home, "away_team": away, "commence_time": stamp,
        "bookmakers": books if books is not None else [
            {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
                {"name": home, "price": 4.70},
                {"name": away, "price": 1.58},
                {"name": "Draw", "price": 4.10},
            ]}]},
            {"key": "betfair_ex_eu", "markets": [{"key": "h2h", "outcomes": [
                {"name": home, "price": 4.90},
                {"name": away, "price": 1.62},
                {"name": "Draw", "price": 4.30},
            ]}]},
            {"key": "williamhill", "markets": [{"key": "h2h", "outcomes": [
                {"name": home, "price": 4.50},
                {"name": away, "price": 1.55},
                {"name": "Draw", "price": 4.00},
            ]}]},
        ],
    }


@pytest.fixture
def payload():
    return [_event("Crystal Palace", "Manchester City")]


# --- the reference Phase C could not reach ------------------------------------

def test_pinnacle_is_carried_in_its_own_column(payload):
    """The blocker Phase C named, removed.

    A rule defined on Pinnacle could be measured backwards and never traded
    forwards, because the free fixture feed does not publish it. This does.
    """
    frame = odds_api.parse_odds(payload, "ENG.PL", cid="T1")
    home = frame[frame["selection_key"] == "1x2_home"].iloc[0]
    assert home["odds_pinnacle"] == pytest.approx(4.70)


def test_the_exchange_is_carried_too(payload):
    frame = odds_api.parse_odds(payload, "ENG.PL", cid="T1")
    home = frame[frame["selection_key"] == "1x2_home"].iloc[0]
    assert home["odds_bfe"] == pytest.approx(4.90)


def test_a_named_book_missing_from_the_panel_is_null_not_zero():
    """Absent is not the same as priced at nothing."""
    thin = [_event("A", "B", books=[
        {"key": "williamhill", "markets": [{"key": "h2h", "outcomes": [
            {"name": "A", "price": 2.0}, {"name": "B", "price": 3.5},
            {"name": "Draw", "price": 3.4},
        ]}]},
    ])]
    frame = odds_api.parse_odds(thin, "ENG.PL", cid="T1")
    assert frame["odds_pinnacle"].isna().all()


# --- FR-16a: the two market numbers stay apart --------------------------------

def test_consensus_is_the_mean_and_the_price_is_the_best(payload):
    frame = odds_api.parse_odds(payload, "ENG.PL", cid="T1")
    home = frame[frame["selection_key"] == "1x2_home"].iloc[0]
    assert home["odds_avg"] == pytest.approx((4.70 + 4.90 + 4.50) / 3)
    assert home["odds_max"] == pytest.approx(4.90)
    assert home["odds_max"] >= home["odds_avg"]


def test_the_book_count_is_recorded(payload):
    frame = odds_api.parse_odds(payload, "ENG.PL", cid="T1")
    assert set(frame["n_books"]) == {3}


def test_an_impossible_price_is_dropped():
    bad = [_event("A", "B", books=[
        {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "A", "price": 1.0}, {"name": "B", "price": 0.5},
            {"name": "Draw", "price": 3.4},
        ]}]},
    ])]
    frame = odds_api.parse_odds(bad, "ENG.PL", cid="T1")
    assert set(frame["selection_key"]) == {"1x2_draw"}


# --- market and selection mapping ---------------------------------------------

def test_outcomes_map_onto_the_projects_selection_names(payload):
    frame = odds_api.parse_odds(payload, "ENG.PL", cid="T1")
    assert set(frame["selection_key"]) == {"1x2_home", "1x2_draw", "1x2_away"}


def test_totals_carry_their_line():
    rows = [_event("A", "B", books=[
        {"key": "pinnacle", "markets": [{"key": "totals", "outcomes": [
            {"name": "Over", "price": 1.9, "point": 2.5},
            {"name": "Under", "price": 1.95, "point": 2.5},
        ]}]},
    ])]
    frame = odds_api.parse_odds(rows, "ENG.PL", cid="T1")
    assert set(frame["selection_key"]) == {"over_2.5", "under_2.5"}


def test_each_handicap_side_is_quoted_at_its_own_line():
    """Unlike football-data's single `AHh`, there is nothing to negate here."""
    rows = [_event("A", "B", books=[
        {"key": "pinnacle", "markets": [{"key": "spreads", "outcomes": [
            {"name": "A", "price": 1.95, "point": -0.5},
            {"name": "B", "price": 1.9, "point": 0.5},
        ]}]},
    ])]
    frame = odds_api.parse_odds(rows, "ENG.PL", cid="T1")
    assert set(frame["selection_key"]) == {"ah_home_-0.5", "ah_away_0.5"}


def test_an_unrecognised_market_is_ignored():
    rows = [_event("A", "B", books=[
        {"key": "pinnacle", "markets": [{"key": "btts", "outcomes": [
            {"name": "Yes", "price": 1.8}, {"name": "No", "price": 2.0},
        ]}]},
    ])]
    assert odds_api.parse_odds(rows, "ENG.PL", cid="T1").empty


def test_an_event_without_a_kickoff_is_dropped():
    rows = [_event("A", "B", stamp=None)]
    assert odds_api.parse_odds(rows, "ENG.PL", cid="T1").empty


# --- the daily / matchday split -----------------------------------------------

@pytest.fixture
def fixtures():
    from datetime import UTC, datetime, timedelta

    today = datetime.now(UTC).date()
    return pd.DataFrame({
        "competition_id": ["ENG.PL", "ITA.SERIEA"],
        "date": [pd.Timestamp(today), pd.Timestamp(today + timedelta(days=4))],
        "home_team": ["A", "C"], "away_team": ["B", "D"],
    })


def test_a_competition_playing_today_gets_every_market(fixtures):
    assert odds_api.markets_for("ENG.PL", fixtures) == odds_api.MATCHDAY_MARKETS


def test_a_competition_not_playing_today_gets_one_market(fixtures):
    """One credit keeps a price series running; three would not be affordable
    across every competition every day."""
    assert odds_api.markets_for("ITA.SERIEA", fixtures) == odds_api.DAILY_MARKETS
    assert len(odds_api.DAILY_MARKETS) == 1


def test_no_fixture_list_falls_back_to_the_cheap_sweep():
    assert odds_api.markets_for("ENG.PL", None) == odds_api.DAILY_MARKETS
    assert odds_api.markets_for("ENG.PL", pd.DataFrame()) == odds_api.DAILY_MARKETS


# --- the budget must not undercount -------------------------------------------

def test_a_multi_market_request_costs_more_than_one_credit():
    """Billing is per market per region. Counting requests said 7 for a sweep
    the API charged 11 for, and undercounting is the direction that empties a
    monthly budget."""
    budget = odds_api.Budget()

    class Session:
        def get_with_headers(self, url, **kw):
            return b"[]", {"x-requests-remaining": "400", "x-requests-used": "100"}

    odds_api._get(
        "https://x.test/odds", Session(), budget,
        max_age=0, costs_credits=True, cost=3,
    )
    assert budget.spent_this_run == 3


def test_the_matchday_sweep_stays_inside_the_monthly_allowance():
    """Seven competitions, two of them playing, is eleven credits a day.

    Thirty days of that is 330 against a 500 allowance — the arithmetic the
    daily/matchday split exists to satisfy.
    """
    playing, idle = 2, 5
    daily = idle * len(odds_api.DAILY_MARKETS) + playing * len(odds_api.MATCHDAY_MARKETS)
    assert daily * 30 < odds_api.MONTHLY_CREDITS
