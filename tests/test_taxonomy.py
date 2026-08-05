"""Taxonomy tests (Design §2).

Format resolution is load-bearing: Design §5.3 branches inference on `format`, so
resolving a UCL final as `swiss_league_phase` would silently run the wrong
sub-model rather than raise. These tests pin the cases that actually vary.
"""

from __future__ import annotations

import json

import pytest

from statpitch import taxonomy
from statpitch.taxonomy import TaxonomyError, load_registry

PHASE_1_COMPETITIONS = {
    "ENG.PL", "ESP.LALIGA", "GER.BUNDESLIGA", "ITA.SERIEA", "FRA.LIGUE1",
    "ENG.FA_CUP", "ESP.COPA_DEL_REY", "GER.DFB_POKAL", "ITA.COPPA_ITALIA",
    "FRA.COUPE_DE_FRANCE", "UEFA.UCL", "UEFA.UEL",
}


@pytest.fixture(scope="module")
def reg():
    return load_registry()


# --- coverage of the shipped taxonomy -----------------------------------------

def test_all_twelve_phase_one_competitions_present(reg):
    assert set(reg.competitions) == PHASE_1_COMPETITIONS


def test_odds_coverage_is_true_for_leagues_and_false_for_everything_else(reg):
    # Requirements §9: football-data.co.uk covers league divisions only, and no free
    # odds source with cup coverage exists. This flag is how that limit is enforced
    # in code rather than in prose.
    for comp in reg:
        assert comp.odds_coverage is (comp.competition_type == "league"), comp.competition_id
    assert len(reg.with_odds_coverage()) == 5


def test_every_league_maps_to_a_football_data_division_code(reg):
    codes = {c.football_data_code for c in reg.of_type("league")}
    assert codes == {"E0", "SP1", "D1", "I1", "F1"}


def test_cups_have_null_tier_but_admit_lower_tiers(reg):
    for comp in reg:
        if comp.competition_type == "domestic_cup":
            assert comp.tier is None
            assert comp.admits_lower_tiers, comp.competition_id
        if comp.competition_type == "league":
            assert comp.tier == 1


def test_lookup_by_football_data_code(reg):
    assert reg.by_football_data_code("E0").competition_id == "ENG.PL"
    with pytest.raises(TaxonomyError):
        reg.by_football_data_code("XX9")


def test_unknown_competition_raises_with_a_useful_message(reg):
    with pytest.raises(TaxonomyError, match="unknown competition_id"):
        reg["ENG.CHAMPIONSHIP"]


# --- format resolution --------------------------------------------------------

def test_league_format_is_round_robin_regardless_of_stage(reg):
    pl = reg["ENG.PL"]
    assert pl.resolve_format() == "round_robin"
    assert pl.resolve_format(stage="matchweek_38", season="2024-2025") == "round_robin"


def test_copa_del_rey_semi_final_is_two_legged_but_other_rounds_are_not(reg):
    copa = reg["ESP.COPA_DEL_REY"]
    assert copa.resolve_format(stage="round_of_16", season="2024-2025") == "single_leg_knockout"
    assert copa.resolve_format(stage="semi_final", season="2024-2025") == "two_leg_knockout"


def test_copa_del_rey_was_two_legged_throughout_before_the_2019_restructure(reg):
    copa = reg["ESP.COPA_DEL_REY"]
    assert copa.resolve_format(stage="round_of_16", season="2016-2017") == "two_leg_knockout"
    # ...and reverts to single-leg once the historical window closes.
    assert copa.resolve_format(stage="round_of_16", season="2019-2020") == "single_leg_knockout"


def test_ucl_league_phase_replaced_the_group_stage_in_2024(reg):
    ucl = reg["UEFA.UCL"]
    assert ucl.resolve_format(stage="league_phase", season="2024-2025") == "swiss_league_phase"
    assert ucl.resolve_format(stage="group_stage", season="2021-2022") == "round_robin"


def test_ucl_final_is_a_single_leg_at_a_neutral_venue(reg):
    ucl = reg["UEFA.UCL"]
    assert ucl.resolve_format(stage="final", season="2024-2025") == "single_leg_knockout"
    assert ucl.is_neutral_venue("final")
    assert not ucl.is_neutral_venue("quarter_final")


def test_ucl_knockout_rounds_are_two_legged(reg):
    ucl = reg["UEFA.UCL"]
    for stage in ("round_of_16", "quarter_final", "semi_final", "knockout_playoff"):
        assert ucl.resolve_format(stage=stage, season="2024-2025") == "two_leg_knockout", stage


def test_coppa_italia_semi_finals_are_two_legged_and_stay_that_way(reg):
    coppa = reg["ITA.COPPA_ITALIA"]
    for season in ("2022-2023", "2024-2025", "2025-2026"):
        assert coppa.resolve_format(stage="semi_final", season=season) == "two_leg_knockout"
    # ...while every other round is a single leg.
    assert coppa.resolve_format(stage="quarter_final", season="2025-2026") == "single_leg_knockout"
    assert coppa.resolve_format(stage="final", season="2025-2026") == "single_leg_knockout"


def test_both_domestic_cups_with_two_legged_semis_are_covered(reg):
    """FR-7's aggregate-tie engine is needed for domestic cups, not only Europe."""
    two_legged_semis = {
        c.competition_id
        for c in reg.of_type("domestic_cup")
        if c.resolve_format(stage="semi_final", season="2025-2026") == "two_leg_knockout"
    }
    assert two_legged_semis == {"ESP.COPA_DEL_REY", "ITA.COPPA_ITALIA"}


def test_stage_names_normalise_across_spelling_variants(reg):
    ucl = reg["UEFA.UCL"]
    for spelling in ("round_of_16", "Round of 16", "ROUND-OF-16"):
        assert ucl.resolve_format(stage=spelling, season="2024-2025") == "two_leg_knockout"


def test_dfb_pokal_is_single_leg_in_every_round(reg):
    pokal = reg["GER.DFB_POKAL"]
    for stage in ("round_1", "semi_final", "final"):
        assert pokal.resolve_format(stage=stage, season="2024-2025") == "single_leg_knockout"


# --- season handling ----------------------------------------------------------

@pytest.mark.parametrize(
    ("season", "expected"),
    [("2024-2025", 2024), ("2024-25", 2024), ("2024", 2024), (" 1993-94 ", 1993)],
)
def test_season_start_year_accepts_the_formats_used_across_sources(season, expected):
    assert taxonomy.season_start_year(season) == expected


@pytest.mark.parametrize("bad", ["not-a-season", "", "20xx-21", "1700-01"])
def test_season_start_year_rejects_junk(bad):
    with pytest.raises(TaxonomyError):
        taxonomy.season_start_year(bad)


def test_away_goals_rule_applies_only_before_2021_22(reg):
    ucl = reg["UEFA.UCL"]
    assert ucl.away_goals_rule_applies("2020-2021")
    assert not ucl.away_goals_rule_applies("2021-2022")
    assert not ucl.away_goals_rule_applies("2024-2025")


# --- validation ---------------------------------------------------------------

def _write(tmp_path, competitions, **top):
    payload = {"competitions": competitions, **top}
    path = tmp_path / "competitions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


BASE_ROW = {
    "competition_id": "X.Y",
    "name": "X",
    "competition_type": "league",
    "format": "round_robin",
    "tier": 1,
    "odds_coverage": True,
}


def test_odds_coverage_may_not_be_omitted(tmp_path):
    row = {k: v for k, v in BASE_ROW.items() if k != "odds_coverage"}
    with pytest.raises(TaxonomyError, match="odds_coverage"):
        load_registry(_write(tmp_path, [row]))


def test_unknown_format_is_rejected(tmp_path):
    with pytest.raises(TaxonomyError, match="unknown format"):
        load_registry(_write(tmp_path, [BASE_ROW | {"format": "best_of_three"}]))


def test_unknown_competition_type_is_rejected(tmp_path):
    with pytest.raises(TaxonomyError, match="competition_type"):
        load_registry(_write(tmp_path, [BASE_ROW | {"competition_type": "friendly"}]))


def test_duplicate_competition_id_is_rejected(tmp_path):
    with pytest.raises(TaxonomyError, match="duplicate"):
        load_registry(_write(tmp_path, [BASE_ROW, BASE_ROW]))


def test_malformed_format_history_is_rejected(tmp_path):
    row = BASE_ROW | {"format_history": [{"format": "round_robin"}]}
    with pytest.raises(TaxonomyError, match="until_season"):
        load_registry(_write(tmp_path, [row]))


def test_empty_taxonomy_is_rejected(tmp_path):
    with pytest.raises(TaxonomyError, match="no competitions"):
        load_registry(_write(tmp_path, []))
