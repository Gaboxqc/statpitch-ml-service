"""Rating resolution at serving time (FR-9, Design §7).

This exists because the failure it guards against was live and silent: ratings
were keyed on `source_name`, which is null for every club fetched only as a cup
entrant, so 187 of 428 clubs fell through to a flat default. Two fourth-tier
sides came back as equals of each other and of the club hosting them, with no
error and no missing field — just a confident, wrong number.

The tests therefore assert on the *source* of a rating, not only its value. A
rating that is right by accident is not the property worth protecting.
"""

from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest

from statpitch import paths, taxonomy
from statpitch.serving import predictor as pr


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A miniature processed tree with the two key spaces and an alias."""
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            # A league club: known under both names.
            {"clubelo_name": "Arsenal", "source_name": "Arsenal", "elo": 2000.0,
             "valid_from": "2026-01-01"},
            # A cup-only club: fetched by clubelo_name, never seen by the league
            # ingestion, so source_name is null. This is the 187-club case.
            {"clubelo_name": "Saarbruecken", "source_name": None, "elo": 1500.0,
             "valid_from": "2026-01-01"},
            # An earlier interval that must lose to the later one.
            {"clubelo_name": "Arsenal", "source_name": "Arsenal", "elo": 1200.0,
             "valid_from": "2020-01-01"},
        ]
    ).to_parquet(processed / "elo_ratings_all.parquet")

    (processed / "cup_club_elo_map.json").write_text(
        json.dumps({"matched": {"1. FC Saarbrücken": "Saarbruecken"}}),
        encoding="utf-8",
    )
    (tmp_path / "entrant_prior.json").write_text(
        json.dumps({
            "home_advantage_elo": 24.58,
            "pooled_elo": 1547.55,
            "buckets": [
                {"competition_id": "ENG.FA_CUP", "entry_stage": "round_1",
                 "elo": 1330.5, "reliable": True},
                {"competition_id": "ENG.FA_CUP", "entry_stage": "round_3",
                 "elo": 1790.0, "reliable": True},
                {"competition_id": "ITA.COPPA_ITALIA", "entry_stage": "round_1",
                 "elo": 1220.0, "reliable": False},
            ],
        }),
        encoding="utf-8",
    )
    # The taxonomy also resolves through STATPITCH_DATA, so it travels with the
    # redirect rather than being left pointing at the real tree.
    shutil.copy(paths.REPO_ROOT / "data" / "competitions.json", tmp_path)
    monkeypatch.setenv("STATPITCH_DATA", str(tmp_path))
    taxonomy.reset_cache()
    yield tmp_path
    taxonomy.reset_cache()


@pytest.fixture
def artifacts(tree):
    return pr.Artifacts.load(tree / "processed")


# --- the bug this file exists for ---------------------------------------------

def test_a_cup_only_club_is_rated_rather_than_defaulted(artifacts):
    """The 187-club hole. `source_name` is null here; `clubelo_name` is not."""
    rating = artifacts.rate("Saarbruecken")
    assert rating.source == "club_elo"
    assert rating.elo == 1500.0


def test_both_name_spaces_resolve_to_the_same_club(artifacts):
    assert artifacts.rate("Arsenal").elo == 2000.0
    assert artifacts.rate("Arsenal").is_measured


def test_the_latest_interval_wins(artifacts):
    """Elo history is a series of intervals; a stale one is a wrong answer."""
    assert artifacts.rate("Arsenal").elo == 2000.0


def test_a_formal_cup_name_resolves_through_the_alias_map(artifacts):
    """Openfootball writes '1. FC Saarbrücken'; Club Elo writes 'Saarbruecken'."""
    rating = artifacts.rate("1. FC Saarbrücken")
    assert rating.source == "club_elo"
    assert rating.elo == 1500.0


# --- the prior is keyed on ENTRY stage, not match stage -----------------------

def test_an_unknown_club_falls_back_to_the_pooled_level_not_a_bare_default(artifacts):
    """A cup entrant is not an average club, and it is not a 1400 club either."""
    rating = artifacts.rate("Unknown FC", competition_id="ENG.FA_CUP")
    assert rating.source == "pooled_prior"
    assert rating.elo == pytest.approx(1547.55)


def test_the_entry_bucket_applies_only_when_the_caller_supplies_it(artifacts):
    stated = artifacts.rate(
        "Unknown FC", competition_id="ENG.FA_CUP", entry_stage="round_1"
    )
    assert stated.source == "entry_prior"
    assert stated.elo == pytest.approx(1330.5)


def test_the_match_stage_is_never_read_as_an_entry_stage(artifacts):
    """The conflation this project already documented once.

    A club entering the FA Cup in round 1 is a National League side and is still
    one in round 3. Reading the bucket off the round being played would rate it
    at 1790 — a Premier League entrant — for the offence of winning two ties.
    """
    predictor = pr.Predictor(artifacts)
    played_in_round_3 = predictor.predict(
        "ENG.FA_CUP", "Unknown FC", "Arsenal", stage="round_3"
    )
    assert played_in_round_3.home_rating.source == "pooled_prior"
    assert played_in_round_3.home_rating.elo != pytest.approx(1790.0)


def test_stating_the_entry_round_changes_the_prediction(artifacts):
    """If it made no difference the parameter would be decoration."""
    predictor = pr.Predictor(artifacts)
    pooled = predictor.predict("ENG.FA_CUP", "Unknown FC", "Arsenal", stage="round_3")
    entered_low = predictor.predict(
        "ENG.FA_CUP", "Unknown FC", "Arsenal", stage="round_3",
        home_entry_stage="round_1",
    )
    assert entered_low.one_x_two[0] < pooled.one_x_two[0]


def test_an_unreliable_bucket_is_not_used(artifacts):
    """Buckets below the sample threshold reproduce single results, not levels."""
    rating = artifacts.rate(
        "Unknown FC", competition_id="ITA.COPPA_ITALIA", entry_stage="round_1"
    )
    assert rating.source == "pooled_prior"


def test_a_club_with_no_competition_context_gets_the_bare_default(artifacts):
    rating = artifacts.rate("Unknown FC")
    assert rating.source == "default"
    assert rating.elo == pr.DEFAULT_ELO


# --- home advantage differs by competition type -------------------------------

def test_cup_home_advantage_is_the_measured_cup_figure(artifacts):
    """54.4 Elo is the league number; a cup tie must not borrow it."""
    assert artifacts.home_advantage("round_robin") == pytest.approx(54.4)
    assert artifacts.home_advantage("single_leg_knockout") == pytest.approx(24.58)


def test_the_swiss_league_phase_counts_as_a_league(artifacts):
    """It is a table, not a tie, so the league figure is the right one."""
    assert artifacts.home_advantage("swiss_league_phase") == pytest.approx(54.4)


def test_a_cup_tie_favours_the_host_less_than_a_league_match(artifacts):
    """The ~30 Elo that a league constant would wrongly hand the host.

    Domestic cups seed the weaker club at home, which is exactly why the two
    figures differ — and exactly the fixture where getting it wrong matters.
    """
    predictor = pr.Predictor(artifacts)
    league = predictor.rates("ENG.PL", "Arsenal", "Saarbruecken")
    cup = predictor.rates(
        "ENG.FA_CUP", "Arsenal", "Saarbruecken", resolved_format="single_leg_knockout"
    )
    assert cup[0] < league[0]


# --- the source travels with the answer ---------------------------------------

def test_the_response_reports_which_tier_of_evidence_was_used(artifacts):
    body = pr.Predictor(artifacts).predict("ENG.PL", "Arsenal", "Nobody").as_dict()
    assert body["ratings"]["home"]["source"] == "club_elo"
    assert body["ratings"]["away"]["source"] != "club_elo"
    assert body["fully_rated"] is False


def test_a_fully_rated_fixture_says_so(artifacts):
    body = pr.Predictor(artifacts).predict(
        "ENG.PL", "Arsenal", "Saarbruecken"
    ).as_dict()
    assert body["fully_rated"] is True


# --- degradation, not failure (NFR-7) -----------------------------------------

def test_an_empty_tree_still_serves_a_prediction(tmp_path):
    artifacts = pr.Artifacts.load(tmp_path)
    body = pr.Predictor(artifacts).predict("ENG.PL", "A", "B").as_dict()
    assert body["probabilities"]["home"] > 0
    assert body["fully_rated"] is False
