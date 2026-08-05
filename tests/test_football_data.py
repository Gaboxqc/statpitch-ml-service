"""football-data.co.uk ingestion tests (Design §3.1).

Fixtures are hand-written CSVs reproducing the three real schema eras, so the
suite runs offline and pins the era behaviour that the live archive actually
exhibits (verified against the real files during development).
"""

from __future__ import annotations

import pandas as pd
import pytest

from statpitch.data import football_data as fd
from statpitch.data.football_data import OddsEra, SeasonFile

# --- fixtures -----------------------------------------------------------------

MODERN_CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,HS,AS,HST,AST,HF,AF,HC,AC,HY,AY,HR,AR,B365H,B365D,B365A,PSH,PSD,PSA,WHH,WHD,WHA,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA,B365>2.5,B365<2.5,Max>2.5,Max<2.5,Avg>2.5,Avg<2.5,AHh,B365AHH,B365AHA,MaxAHH,MaxAHA,AvgAHH,AvgAHA,B365CH,B365CD,B365CA,PSCH,PSCD,PSCA,MaxCH,MaxCD,MaxCA,AvgCH,AvgCD,AvgCA,MaxC>2.5,MaxC<2.5,AvgC>2.5,AvgC<2.5,AHCh,MaxCAHH,MaxCAHA,AvgCAHH,AvgCAHA
E0,17/08/2024,15:00,Arsenal,Chelsea,2,1,H,1,0,H,M Oliver,15,9,6,3,10,12,7,4,2,3,0,0,1.90,3.60,4.20,1.95,3.65,4.30,1.91,3.60,4.10,2.00,3.75,4.50,1.92,3.62,4.15,1.80,2.05,1.88,2.10,1.83,2.02,-0.5,1.98,1.92,2.05,1.98,2.00,1.90,1.85,3.70,4.40,1.90,3.72,4.45,1.95,3.80,4.60,1.88,3.68,4.35,1.90,2.08,1.85,2.00,-0.5,2.02,2.00,1.97,1.92
E0,18/08/2024,14:00,Everton,Brighton,0,3,A,0,1,A,A Taylor,8,17,2,9,14,9,3,8,1,2,0,0,3.10,3.40,2.30,3.20,3.45,2.35,3.05,3.40,2.32,3.30,3.55,2.45,3.12,3.42,2.33,1.95,1.90,2.05,1.95,1.98,1.88,0.25,1.90,2.00,1.95,2.08,1.92,2.02,3.30,3.50,2.20,3.35,3.55,2.22,3.45,3.60,2.30,3.28,3.48,2.21,2.00,1.92,1.95,1.90,0.25,1.93,2.05,1.90,2.00
"""

BETBRAIN_CSV = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,B365H,B365D,B365A,PSH,PSD,PSA,WHH,WHD,WHA,PSCH,PSCD,PSCA,Bb1X2,BbMxH,BbAvH,BbMxD,BbAvD,BbMxA,BbAvA,BbOU,BbMx>2.5,BbAv>2.5,BbMx<2.5,BbAv<2.5,BbAH,BbAHh,BbMxAHH,BbAvAHH,BbMxAHA,BbAvAHA
E0,08/08/2015,Man United,Tottenham,1,0,H,0,0,D,1.80,3.60,4.75,1.85,3.65,4.80,1.83,3.60,4.60,1.88,3.70,4.90,39,1.90,1.82,3.80,3.62,5.00,4.70,35,2.10,2.02,1.85,1.78,22,-0.75,1.98,1.92,1.98,1.90
E0,08/08/2015,Bournemouth,Aston Villa,0,1,A,0,0,D,2.20,3.30,3.40,2.25,3.35,3.45,2.20,3.30,3.30,2.30,3.40,3.50,39,2.30,2.22,3.45,3.32,3.55,3.38,35,2.05,1.98,1.90,1.82,22,-0.25,2.00,1.94,1.95,1.88
"""

LEGACY_CSV = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR,Referee,HS,AS,HST,AST,GBH,GBD,GBA,IWH,IWD,IWA,LBH,LBD,LBA,SBH,SBD,SBA,WHH,WHD,WHA
E0,19/08/00,Charlton,Man City,4,0,H,2,0,H,Rob Harris,17,8,14,4,2.00,3.00,3.20,2.20,2.90,2.70,2.20,3.25,2.75,2.20,3.25,2.88,2.10,3.20,3.10
E0,19/08/00,Chelsea,West Ham,4,2,H,1,0,H,Graham Barber,17,12,10,5,1.50,3.40,5.00,1.55,3.30,4.50,1.53,3.40,5.00,1.50,3.50,5.50,1.50,3.40,5.00
"""

MESSY_CSV = """Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA,
E0,17/08/2024,Arsenal,Chelsea,2,1,H,1.90,3.60,4.20,2.00,3.75,4.50,1.92,3.62,4.15,
E0,18/08/2024,Leeds,Burnley,,,,2.10,3.40,3.60,2.20,3.50,3.70,2.12,3.42,3.62,
,,,,,,,,,,,,,,,,
"""


def _season_file(tmp_path, name, text, start_year, competition_id="ENG.PL"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return SeasonFile(competition_id, "E0", start_year, path)


@pytest.fixture
def modern(tmp_path):
    return _season_file(tmp_path, "E0_modern.csv", MODERN_CSV, 2024)


@pytest.fixture
def betbrain(tmp_path):
    return _season_file(tmp_path, "E0_bb.csv", BETBRAIN_CSV, 2015)


@pytest.fixture
def legacy(tmp_path):
    return _season_file(tmp_path, "E0_legacy.csv", LEGACY_CSV, 2000)


# --- era classification -------------------------------------------------------

@pytest.mark.parametrize(
    ("year", "era"),
    [
        (1993, OddsEra.LEGACY), (2004, OddsEra.LEGACY),
        (2005, OddsEra.BETBRAIN), (2018, OddsEra.BETBRAIN),
        (2019, OddsEra.MODERN), (2025, OddsEra.MODERN),
    ],
)
def test_era_boundaries_match_the_live_archive(year, era):
    assert fd.era_for_season(year) is era


def test_only_the_modern_era_has_consensus_closing_odds():
    # This is the constraint that confines CLV and the whole Decision Layer to
    # 2019/20 onward — it is a data fact, not a modelling choice.
    assert not fd.has_consensus_closing(2018)
    assert fd.has_consensus_closing(2019)


def test_decision_layer_window_covers_at_least_two_seasons():
    seasons = fd.decision_layer_seasons(last=2024)
    assert seasons[0] == 2019
    assert len(seasons) >= 2  # Requirements §8.3


@pytest.mark.parametrize(
    ("year", "code"), [(2019, "1920"), (2024, "2425"), (1999, "9900"), (1993, "9394")]
)
def test_season_code_matches_the_url_scheme(year, code):
    assert fd.season_code(year) == code
    assert fd.csv_url(year, "E0").endswith(f"/{code}/E0.csv")


def test_season_label_is_the_canonical_project_form():
    assert fd.season_label(2024) == "2024-2025"


# --- match parsing ------------------------------------------------------------

def test_modern_matches_parse_with_scores_and_kickoff(modern):
    m = fd.parse_matches(modern)
    assert len(m) == 2
    row = m.iloc[0]
    assert row["home_team"] == "Arsenal"
    assert row["away_team"] == "Chelsea"
    assert row["home_goals"] == 2 and row["away_goals"] == 1
    assert row["result"] == "H"
    assert row["date"] == pd.Timestamp("2024-08-17")
    assert row["kickoff_local"] == "15:00"
    assert row["season"] == "2024-2025"
    assert row["format"] == "round_robin"


def test_two_digit_years_parse_correctly(legacy):
    m = fd.parse_matches(legacy)
    assert m.iloc[0]["date"] == pd.Timestamp("2000-08-19")


def test_kickoff_time_is_absent_before_the_modern_era(betbrain):
    # FR-34's edge-decay-by-hours-to-kickoff shares the Decision Layer's window.
    m = fd.parse_matches(betbrain)
    assert m["kickoff_local"].isna().all()


def test_match_stats_are_carried_through(modern):
    m = fd.parse_matches(modern)
    row = m.iloc[0]
    assert row["home_shots"] == 15
    assert row["away_shots"] == 9
    assert row["home_shots_target"] == 6
    assert row["home_corners"] == 7


def test_match_ids_are_unique_and_deterministic(modern):
    a = fd.parse_matches(modern)
    b = fd.parse_matches(modern)
    assert a["match_id"].is_unique
    assert list(a["match_id"]) == list(b["match_id"])
    assert a.iloc[0]["match_id"].startswith("ENG.PL|2024-08-17|Arsenal|Chelsea")


def test_unplayed_rows_and_padding_are_dropped(tmp_path):
    sf = _season_file(tmp_path, "messy.csv", MESSY_CSV, 2024)
    m = fd.parse_matches(sf)
    # One real match; the scoreless fixture and the all-empty padding row go.
    assert len(m) == 1
    assert m.iloc[0]["home_team"] == "Arsenal"


def test_trailing_unnamed_columns_do_not_break_parsing(tmp_path):
    sf = _season_file(tmp_path, "messy.csv", MESSY_CSV, 2024)
    assert not fd.parse_odds(sf).empty


# --- odds regime --------------------------------------------------------------

def test_odds_regime_splits_on_the_pinnacle_break(modern):
    m = fd.parse_matches(modern)
    assert set(m["odds_regime"]) == {"pre_2025_07_23"}


def test_a_post_break_match_is_tagged_separately(tmp_path):
    text = MODERN_CSV.replace("17/08/2024", "16/08/2025").replace("18/08/2024", "17/08/2025")
    sf = _season_file(tmp_path, "E0_2526.csv", text, 2025)
    m = fd.parse_matches(sf)
    assert set(m["odds_regime"]) == {"post_2025_07_23"}


# --- odds parsing: the fair-vs-available separation ---------------------------

def test_modern_odds_cover_both_snapshots_and_all_three_markets(modern):
    o = fd.parse_odds(modern)
    assert set(o["snapshot"]) == {"preclose", "close"}
    assert set(o["market"]) == {"1x2", "ou", "ah"}
    # 2 matches x 2 snapshots x (3 + 2 + 2) selections
    assert len(o) == 2 * 2 * 7


def test_avg_and_max_are_kept_in_separate_columns(modern):
    """FR-16a / Design §3.1 — fair probability comes from Avg, price from Max.

    Max-of-N is above consensus by construction, so de-vigging it fabricates edge.
    Keeping them in distinct columns is what makes that mistake hard to commit.
    """
    o = fd.parse_odds(modern)
    home = o[(o.market == "1x2") & (o.selection == "home") & (o.snapshot == "close")]
    row = home.iloc[0]
    assert float(row["odds_avg"]) == 1.88
    assert float(row["odds_max"]) == 1.95
    assert row["odds_max"] > row["odds_avg"]


def test_closing_odds_differ_from_preclosing_giving_the_line_movement_series(modern):
    o = fd.parse_odds(modern)
    sel = (o.market == "1x2") & (o.selection == "home")
    pre = o[sel & (o.snapshot == "preclose")].iloc[0]["odds_avg"]
    close = o[sel & (o.snapshot == "close")].iloc[0]["odds_avg"]
    assert float(pre) != float(close)


def test_asian_handicap_line_is_captured_per_snapshot(modern):
    o = fd.parse_odds(modern)
    ah = o[(o.market == "ah") & (o.selection == "ah_home")]
    pre = ah[ah.snapshot == "preclose"].sort_values("match_id").iloc[0]
    close = ah[ah.snapshot == "close"].sort_values("match_id").iloc[0]
    assert float(pre["line"]) == -0.5
    assert float(close["line"]) == -0.5


def test_over_under_line_defaults_to_the_published_2_5(modern):
    o = fd.parse_odds(modern)
    ou = o[o.market == "ou"]
    assert set(ou["selection"]) == {"over", "under"}
    assert (ou["line"].astype(float) == 2.5).all()


def test_betbrain_era_has_no_consensus_closing_only_pinnacle(betbrain):
    o = fd.parse_odds(betbrain)
    close = o[o.snapshot == "close"]
    assert not close.empty
    # The single-book benchmark exists...
    assert close["odds_pinnacle"].notna().all()
    # ...but consensus closing does not, and is not faked.
    assert close["odds_avg"].isna().all()
    assert close["odds_max"].isna().all()


def test_betbrain_aggregates_map_from_the_bb_prefix(betbrain):
    o = fd.parse_odds(betbrain)
    home = o[
        (o.market == "1x2") & (o.selection == "home") & (o.snapshot == "preclose")
        & o.match_id.str.contains("ManUnited")
    ]
    row = home.iloc[0]
    assert float(row["odds_avg"]) == 1.82   # BbAvH
    assert float(row["odds_max"]) == 1.90   # BbMxH


def test_betbrain_publishes_the_book_count_directly(betbrain):
    o = fd.parse_odds(betbrain)
    assert (o["n_books"].dropna() == 39).all()   # Bb1X2


def test_legacy_era_yields_no_published_consensus(legacy):
    o = fd.parse_odds(legacy)
    assert not o.empty
    assert o["odds_avg"].isna().all()
    assert o["odds_max"].isna().all()
    assert set(o["snapshot"]) == {"preclose"}


# --- the book panel -----------------------------------------------------------

def test_panel_reconstruction_covers_the_legacy_era(legacy):
    """The panel is what extends a price series across all 25 seasons."""
    o = fd.parse_odds(legacy)
    # Charlton: GB 2.00, IW 2.20, LB 2.20, SB 2.20, WH 2.10
    home = o[(o.selection == "home") & o.match_id.str.contains("Charlton")]
    row = home.iloc[0]
    assert float(row["odds_panel_max"]) == pytest.approx(2.20)
    assert float(row["odds_panel_avg"]) == pytest.approx(2.14)
    assert int(row["n_panel_books"]) == 5


def test_panel_is_never_written_into_the_published_consensus_columns(modern):
    """A 5-book panel and a 30-book consensus are different estimators.

    Measured against the live 2024/25 archive the panel average runs ~0.04 above
    the published Avg, so conflating them would bias every fair probability.
    """
    o = fd.parse_odds(modern)
    row = o[(o.selection == "home") & (o.snapshot == "close")].iloc[0]
    assert float(row["odds_avg"]) == 1.88          # published AvgCH
    assert float(row["odds_panel_avg"]) != float(row["odds_avg"])


def test_panel_is_snapshot_aware(modern):
    o = fd.parse_odds(modern)
    sel = (o.selection == "home") & (o.market == "1x2")
    pre = o[sel & (o.snapshot == "preclose")].iloc[0]["odds_panel_avg"]
    close = o[sel & (o.snapshot == "close")].iloc[0]["odds_panel_avg"]
    assert float(pre) != float(close)


def test_panel_only_applies_to_1x2(modern):
    o = fd.parse_odds(modern)
    assert o[o.market != "1x2"]["odds_panel_avg"].isna().all()


# --- data hygiene -------------------------------------------------------------

def test_impossible_prices_are_discarded(tmp_path):
    text = MODERN_CSV.replace(",1.90,3.60,4.20,", ",0.85,3.60,4.20,", 1)
    sf = _season_file(tmp_path, "bad.csv", text, 2024)
    o = fd.parse_odds(sf)
    row = o[(o.selection == "home") & (o.snapshot == "preclose")].sort_values("match_id").iloc[0]
    assert pd.isna(row["odds_b365"])       # 0.85 rejected
    assert float(row["odds_avg"]) == 1.92  # neighbours untouched


def test_odds_rows_only_reference_known_matches(tmp_path):
    sf = _season_file(tmp_path, "messy.csv", MESSY_CSV, 2024)
    m = fd.parse_matches(sf)
    o = fd.parse_odds(sf, matches=m)
    assert set(o["match_id"]) <= set(m["match_id"])


# --- ragged rows --------------------------------------------------------------

# Nine files in the real archive (2002/03-2004/05) have rows wider than their
# header, always from trailing empty commas. Left unhandled they cost ~3,200
# matches, silently, because the parse failure is caught per-file.
RAGGED_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,,\n"
    "F1,09/08/03,Lyon,Monaco,3,1,H,1.90,3.00,3.75,,,,,,,\n"
    "F1,09/08/03,Nantes,Lens,2,0,H,2.00,3.00,3.65,,,,,,,\n"
)

RAGGED_WITH_REAL_DATA = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    "F1,09/08/03,Lyon,Monaco,3,1,H\n"
    "F1,09/08/03,Nantes,Lens,2,0,H,SURPRISE\n"
)


def test_ragged_rows_with_empty_extras_are_recovered(tmp_path):
    sf = _season_file(tmp_path, "ragged.csv", RAGGED_CSV, 2003)
    m = fd.parse_matches(sf)
    assert len(m) == 2
    assert set(m["home_team"]) == {"Lyon", "Nantes"}


def test_ragged_rows_carrying_real_data_are_dropped_not_silently_trimmed(tmp_path, caplog):
    sf = _season_file(tmp_path, "ragged2.csv", RAGGED_WITH_REAL_DATA, 2003)
    m = fd.parse_matches(sf)
    # Lyon survives; the Nantes row would have lost a populated field, so it goes
    # — and says so, rather than being quietly trimmed.
    assert list(m["home_team"]) == ["Lyon"]
    assert any("beyond the header width" in r.message for r in caplog.records)


def test_build_concatenates_and_sorts(modern, betbrain):
    matches, odds = fd.build([modern, betbrain])
    assert len(matches) == 4
    assert matches["date"].is_monotonic_increasing
    assert matches["match_id"].is_unique
    assert set(odds["odds_schema_era"]) == {"modern", "betbrain"}


def test_build_survives_an_unreadable_file(tmp_path, modern):
    broken = _season_file(tmp_path, "broken.csv", "not,a,valid\ncsv", 2024)
    matches, _ = fd.build([modern, broken])
    assert len(matches) == 2  # the good file still lands


def test_normalise_team_collapses_whitespace():
    assert fd.normalise_team("  Man   United ") == "Man United"
