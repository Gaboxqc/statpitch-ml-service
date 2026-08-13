"""Squad market values (Roadmap §4.1).

The scraper's risk is not that it breaks — a layout change raises loudly — but
that it parses *plausibly wrong*. A misread amount is a number, and a number
flows into a feature without complaint.
"""

from __future__ import annotations

import pytest

from statpitch.data import transfermarkt as tm

PAGE = """
<table class="items"><tbody>
<tr><td></td><td>Manchester City</td><td>36</td><td>25.7</td><td>21</td>
    <td>&euro;40.63m</td><td>&euro;1.46bn</td></tr>
<tr><td></td><td>Promoted FC</td><td>24</td><td>26.1</td><td>5</td>
    <td>-</td><td>-</td></tr>
</tbody></table>
"""


@pytest.mark.parametrize(
    "text,expected",
    [
        ("€1.46bn", 1_460_000_000.0),
        ("€955.65m", 955_650_000.0),
        ("€500k", 500_000.0),
        ("€1,200k", 1_200_000.0),
    ],
)
def test_amounts_parse_with_their_suffix(text, expected):
    assert tm.parse_amount(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["-", "", "?", "   "])
def test_a_missing_amount_is_none_not_zero(text):
    """Zero reads as "this squad is worthless", which is wrong in the one
    direction that biases a model: newly promoted clubs."""
    assert tm.parse_amount(text) is None


def test_a_row_is_parsed_into_its_columns():
    values = tm.parse_season(PAGE)
    assert len(values) == 2
    city = values[0]
    assert city.club == "Manchester City"
    assert city.squad_size == 36
    assert city.average_age == pytest.approx(25.7)
    assert city.total_value_eur == pytest.approx(1_460_000_000.0)


def test_an_unvalued_club_still_yields_a_row():
    """Dropping it would silently shorten a season's club list."""
    promoted = tm.parse_season(PAGE)[1]
    assert promoted.club == "Promoted FC"
    assert promoted.squad_size == 24
    assert promoted.total_value_eur is None


def test_a_layout_change_raises_rather_than_returning_nothing():
    """An empty list would read as "this season had no clubs"."""
    with pytest.raises(tm.TransfermarktError, match="layout"):
        tm.parse_season("<html><body>no table here</body></html>")


def test_only_the_odds_covered_leagues_are_mapped():
    assert set(tm.COMPETITIONS) == {
        "ENG.PL", "ESP.LALIGA", "GER.BUNDESLIGA", "ITA.SERIEA", "FRA.LIGUE1"
    }


def test_an_unmapped_competition_is_refused():
    with pytest.raises(tm.TransfermarktError, match="no Transfermarkt code"):
        tm.fetch_season("ENG.FA_CUP", 2023)


def test_the_season_url_carries_the_season():
    assert "saison_id=2023" in tm.season_url("GB1", 2023)
