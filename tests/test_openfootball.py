"""openfootball parser tests. Offline — fixtures reproduce the real DSL.

Every quirk pinned here was found in the live repos, not imagined: two different
match-line layouts, mixed-language stage labels inside a single file, and
qualifier files whose stage labels are indistinguishable from a domestic cup's.
"""

from __future__ import annotations

import pandas as pd
import pytest

from statpitch import taxonomy
from statpitch.data import openfootball as of

CUP_TXT = """= DFB Pokal 2023/24

# Teams      64

▪ Round 1
Fri Aug 11
  18:00  1. FC Saarbrücken        2-1 (0-0)  Karlsruher SC
         TuS Bersenbrück          0-7 (0-4)  Bor. Mönchengladbach
  18:00  SV Sandhausen            4-2 pen. 3-3 a.e.t. (3-3, 1-2)  Hannover 96
Sat Aug 12
         Rot-Weiss Essen          3-4 a.e.t. (3-3, 1-1)  Hamburger SV
         FC Teutonia 05           0-8 (0-3)  Borussia Dortmund
▪ Final
Sat May 25 2024
  20:00  1. FC Kaiserslautern      0-1 (0-0)  Bayer Leverkusen
"""

CONTINENTAL_TXT = """= UEFA Champions League 2023/24

▪ Group, Matchday 1
  Tue Sep 19 2023
    18:45  AC Milan (ITA)          v Newcastle United FC (ENG)  0-0
    21:00  SS Lazio (ITA)          v Atletico Madrid (ESP)  1-1 (0-1)
▪ Finals, Round of 16
  Tue Feb 13 2024
    21:00  FC Kobenhavn (DEN)      v Manchester City FC (ENG)  1-3 (1-2)
▪ Finals, Final
  Sat Jun 1 2024
    21:00  Borussia Dortmund (GER) v Real Madrid CF (ESP)  0-2 (0-0)
"""


@pytest.fixture(scope="module")
def cup():
    return of.parse_football_txt(CUP_TXT, 2023)


@pytest.fixture(scope="module")
def continental():
    return of.parse_football_txt(CONTINENTAL_TXT, 2023)


# --- the two match-line layouts -----------------------------------------------

def test_cup_layout_puts_the_score_between_the_teams(cup):
    m = cup[0]
    assert m.home_team == "1. FC Saarbrücken"
    assert m.away_team == "Karlsruher SC"
    assert (m.home_goals, m.away_goals) == (2, 1)


def test_continental_layout_puts_the_score_after_both_teams(continental):
    """`Home v Away  score`, not `Home score Away`.

    Parsing a continental line with the cup layout yields an empty away side and
    silently drops the match — which is how every UCL fixture went missing at
    first.
    """
    m = continental[0]
    assert m.home_team == "AC Milan"
    assert m.away_team == "Newcastle United FC"
    assert (m.home_goals, m.away_goals) == (0, 0)


def test_country_suffixes_are_captured_not_left_in_the_name(continental):
    m = continental[0]
    assert m.home_country == "ITA"
    assert m.away_country == "ENG"
    assert "(" not in m.home_team


def test_team_names_containing_digits_survive(cup):
    names = {m.home_team for m in cup}
    assert "FC Teutonia 05" in names          # digits, no hyphen
    assert "1. FC Kaiserslautern" in names    # leading number


def test_hyphenated_team_names_are_not_mistaken_for_scores(cup):
    assert any(m.home_team == "Rot-Weiss Essen" for m in cup)


# --- extra time and penalties (FR-8) ------------------------------------------

def test_regulation_extra_time_and_shootout_are_kept_apart(cup):
    """`4-2 pen. 3-3 a.e.t. (3-3, 1-2)`.

    A goals model must train on the 90-minute score, while qualification is
    decided by the shootout. Collapsing these into one score column would teach
    the model that Sandhausen scored four.
    """
    m = next(x for x in cup if x.home_team == "SV Sandhausen")
    assert (m.home_goals, m.away_goals) == (3, 3)          # regulation
    assert (m.home_goals_aet, m.away_goals_aet) == (3, 3)  # after extra time
    assert (m.home_pens, m.away_pens) == (4, 2)            # shootout
    assert (m.home_goals_ht, m.away_goals_ht) == (1, 2)
    assert m.went_to_extra_time and m.went_to_penalties


def test_extra_time_without_a_shootout(cup):
    m = next(x for x in cup if x.home_team == "Rot-Weiss Essen")
    assert (m.home_goals, m.away_goals) == (3, 3)          # after 90
    assert (m.home_goals_aet, m.away_goals_aet) == (3, 4)  # decided in ET
    assert m.went_to_extra_time
    assert not m.went_to_penalties
    assert m.home_pens is None


def test_a_normal_match_carries_no_extra_time_flags(cup):
    m = cup[0]
    assert not m.went_to_extra_time
    assert not m.went_to_penalties
    assert m.home_goals_aet is None


def test_penalty_form_is_tried_before_the_plain_form():
    """Otherwise `4-2 pen. 3-3` reads as a 4-2 final score."""
    m = of.parse_football_txt(
        "▪ Final\nSat May 25 2024\n  A FC  4-2 pen. 1-1 a.e.t. (1-1, 0-0)  B FC\n", 2023
    )[0]
    assert (m.home_goals, m.away_goals) == (1, 1)
    assert (m.home_pens, m.away_pens) == (4, 2)


# --- dates --------------------------------------------------------------------

def test_year_is_inferred_across_the_season_boundary(cup):
    first = cup[0]
    final = next(x for x in cup if x.stage == "final")
    assert first.date == pd.Timestamp("2023-08-11")   # inferred, no year on line
    assert final.date == pd.Timestamp("2024-05-25")   # explicit year


def test_dates_carry_forward_to_following_match_lines(cup):
    same_day = [m for m in cup if m.date == pd.Timestamp("2023-08-12")]
    assert len(same_day) == 2


# --- stage normalisation ------------------------------------------------------

@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Group, Matchday 1", "group_stage"),
        ("League, Matchday 7", "league_phase"),
        ("Finals, Round of 16", "round_of_16"),
        ("Finals, Quarterfinals", "quarter_final"),
        ("Semifinals", "semi_final"),
        ("Final", "final"),
        ("Playoffs, Matchday 2", "knockout_playoff"),
        ("Preliminary round", "preliminary_round"),
        ("Round 3", "round_3"),
    ],
)
def test_stage_labels_normalise(label, expected):
    assert of.normalise_stage(label) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [("Gruppe G", "group_stage"), ("10. Runde", "round_10"),
     ("Sechzehntelfinale", "round_of_32"), ("Achtelfinale", "round_of_16")],
)
def test_german_labels_normalise(label, expected):
    """Single files really do mix languages — "Gruppe G" beside "Group F"."""
    assert of.normalise_stage(label) == expected


def test_sechzehntelfinale_is_the_round_of_32_not_a_league_phase():
    """It appears untranslated in UEL 2020-21.

    Unmapped, it fell through to the UCL/UEL default (`swiss_league_phase`) and
    recorded 32 two-legged ties as Swiss league fixtures.
    """
    assert of.normalise_stage("Sechzehntelfinale") == "round_of_32"
    assert taxonomy.get("UEFA.UEL").resolve_format(
        stage="round_of_32", season="2020-2021"
    ) == "two_leg_knockout"


# --- the dangerous fallback ---------------------------------------------------

def test_unknown_stage_is_flagged_for_continental_competitions():
    """The UCL default is `swiss_league_phase`, so a silent fallback is a bug."""
    ucl = taxonomy.get("UEFA.UCL")
    assert not ucl.knows_stage("some_unmapped_round")
    assert ucl.knows_stage("round_of_16")


def test_unknown_stage_is_not_flagged_where_the_default_is_safe():
    """Every FA Cup round really is a single-leg tie, so no warning is warranted."""
    fa = taxonomy.get("ENG.FA_CUP")
    assert fa.knows_stage("round_7")
    assert not fa.default_format_is_league_like


def test_leagues_never_flag_stages():
    assert taxonomy.get("ENG.PL").knows_stage("anything_at_all")


# --- season helpers -----------------------------------------------------------

@pytest.mark.parametrize(("year", "directory"), [(2023, "2023-24"), (2009, "2009-10")])
def test_season_dir_matches_the_repo_layout(year, directory):
    assert of.season_dir(year) == directory


def test_season_label_is_the_canonical_project_form():
    assert of.season_label("2023-24") == "2023-2024"


# --- parser hygiene -----------------------------------------------------------

def test_headers_and_comments_are_skipped(cup):
    assert all(m.home_team not in ("=", "#") for m in cup)
    assert len(cup) == 6


def test_a_line_without_a_score_is_ignored():
    assert of.parse_football_txt("▪ Round 1\nFri Aug 11\n  A FC  v  B FC\n", 2023) == []


def test_every_mapped_competition_exists_in_the_taxonomy():
    for competition_id in of.SOURCES:
        assert competition_id in taxonomy.registry()


def test_no_league_is_mapped_to_the_cup_ingestion():
    """Leagues come from football-data.co.uk, which also carries their odds."""
    for competition_id in of.SOURCES:
        assert taxonomy.get(competition_id).competition_type != "league"


# --- schedule freshness -------------------------------------------------------

def test_schedules_are_fetched_with_an_expiry(monkeypatch):
    """A schedule is a claim about the future and must not be served stale.

    `build_fixtures` read an eleven-day-old cached copy on every local run,
    producing an artifact that listed already-played matches while reporting
    `generated_at` as now. The results archive keeps its unbounded cache; only
    the schedule path carries an expiry.
    """
    seen: list[float | None] = []

    def fake_fetch(repo, path, *, session=None, force=False, max_age=None):
        seen.append(max_age)
        return None

    monkeypatch.setattr(of, "fetch_file", fake_fetch)
    of.build_schedule("ENG.PL", [2026])

    assert seen and all(age == of.SCHEDULE_MAX_AGE_SECONDS for age in seen)


def test_the_schedule_expiry_is_shorter_than_a_day():
    """The scheduled refresh runs daily; a longer expiry would never fire."""
    assert 0 < of.SCHEDULE_MAX_AGE_SECONDS < 24 * 3600


def test_results_ingestion_keeps_its_unbounded_cache(monkeypatch):
    """Season files are immutable once played — re-downloading them buys nothing."""
    seen: list[float | None] = []

    def fake_fetch(repo, path, *, session=None, force=False, max_age=None):
        seen.append(max_age)
        return None

    monkeypatch.setattr(of, "fetch_file", fake_fetch)
    of.build_competition("ENG.FA_CUP", [2023])

    assert seen and all(age is None for age in seen)
