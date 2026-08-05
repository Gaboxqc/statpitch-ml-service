"""Club Elo tests (FR-9, FR-11). Offline — no test hits the API."""

from __future__ import annotations

import pandas as pd
import pytest

from statpitch.data import club_elo as ce


@pytest.fixture
def roster():
    rows = [
        ("Arsenal", "ENG", 1), ("Forest", "ENG", 1), ("Man United", "ENG", 1),
        ("Atletico", "ESP", 1), ("Real Madrid", "ESP", 1), ("Bilbao", "ESP", 1),
        ("Espanyol", "ESP", 1), ("Deportivo", "ESP", 1), ("Villarreal", "ESP", 1),
        ("Rayo Vallecano", "ESP", 1), ("Bayern", "GER", 1), ("Gladbach", "GER", 1),
        ("Koeln", "GER", 1), ("RB Leipzig", "GER", 1), ("Nuernberg", "GER", 2),
        ("Saint-Etienne", "FRA", 1), ("Malaga", "ESP", 2),
    ]
    df = pd.DataFrame(rows, columns=["clubelo_name", "country", "best_tier"])
    df["norm"] = df["clubelo_name"].map(ce.normalise)
    return df


@pytest.fixture
def history():
    """Two clubs, with rating changes over time."""
    return pd.DataFrame(
        {
            "source_name": ["Arsenal"] * 3 + ["Ath Madrid"] * 2,
            "clubelo_name": ["Arsenal"] * 3 + ["Atletico"] * 2,
            "country": ["ENG"] * 3 + ["ESP"] * 2,
            "tier": pd.array([1, 1, 1, 1, 2], dtype="Int64"),
            "elo": pd.array([1800.0, 1850.0, 1900.0, 1950.0, 1600.0], dtype="Float64"),
            "valid_from": pd.to_datetime(
                ["2024-01-01", "2024-03-01", "2024-05-01", "2024-01-01", "2024-06-01"]
            ),
            "valid_to": pd.to_datetime(
                ["2024-02-29", "2024-04-30", "2024-12-31", "2024-05-31", "2024-12-31"]
            ),
        }
    )


# --- normalisation ------------------------------------------------------------

@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Atlético", "Atletico"),
        ("M'Gladbach", "MGladbach"),
        ("Saint-Etienne", "Saint Etienne"),
        ("Nürnberg", "Nurnberg"),
        ("  Real  Madrid ", "RealMadrid"),
    ],
)
def test_normalisation_is_accent_and_punctuation_insensitive(a, b):
    assert ce.normalise(a) == ce.normalise(b)


def test_normalisation_keeps_genuinely_different_clubs_apart():
    assert ce.normalise("Real Madrid") != ce.normalise("Atletico")
    assert ce.normalise("Bilbao") != ce.normalise("Bilbao B")


def test_club_slug_strips_spaces_for_the_per_club_endpoint():
    assert ce.club_slug("Man United") == "ManUnited"
    assert ce.club_slug("RB Leipzig") == "RBLeipzig"


# --- name resolution ----------------------------------------------------------

def test_names_matching_after_normalisation_resolve_directly(roster):
    r = ce.resolve_names(["Arsenal", "Bayern"], roster)
    assert r.mapping["Arsenal"] == "Arsenal"
    assert not r.unmatched


def test_curated_aliases_resolve(roster):
    r = ce.resolve_names(["Nott'm Forest", "Bayern Munich", "Villareal"], roster)
    assert r.mapping["Nott'm Forest"] == "Forest"
    assert r.mapping["Bayern Munich"] == "Bayern"
    assert r.mapping["Villareal"] == "Villarreal"   # football-data's misspelling
    assert not r.unmatched


def test_atletico_does_not_resolve_to_real_madrid(roster):
    """The alias table's whole reason for existing.

    "Ath Madrid" fuzzy-matches to "Real Madrid" more closely than to "Atletico".
    Accepting that would attach the wrong club's strength to every Atlético
    fixture — wrong in a way that looks entirely plausible in the output.
    """
    r = ce.resolve_names(["Ath Madrid"], roster)
    assert r.mapping["Ath Madrid"] == "Atletico"
    assert r.mapping["Ath Madrid"] != "Real Madrid"


def test_unknown_names_are_reported_never_guessed(roster):
    r = ce.resolve_names(["Totally Fake FC", "Arsenal"], roster)
    assert "Totally Fake FC" in r.unmatched
    assert "Totally Fake FC" not in r.mapping
    assert r.coverage == 0.5


def test_unmatched_names_carry_suggestions_for_a_human(roster):
    r = ce.resolve_names(["Malagas"], roster)
    assert "Malaga" in r.suggestions["Malagas"]


def test_an_alias_pointing_outside_the_roster_is_attempted_but_flagged(roster):
    """Snapshots can miss clubs with brief top-flight spells, so the alias is tried.

    But it is reported as unverified rather than counted as a success. Four wrong
    aliases once reported as 100% coverage precisely because this path was silent,
    and Club Elo answers an unknown club with an empty CSV rather than a 404 — so
    nothing downstream raises either.
    """
    r = ce.resolve_names(["Arles"], roster, aliases={"Arles": "Arles-Avignon"})
    assert r.mapping["Arles"] == "Arles-Avignon"
    assert not r.unmatched
    assert "Arles" in r.unverified
    assert r.coverage < 1.0


def test_verified_aliases_do_not_count_as_unverified(roster):
    r = ce.resolve_names(["Ath Madrid"], roster)
    assert not r.unverified
    assert r.coverage == 1.0


def test_every_alias_target_exists_in_the_committed_roster():
    """The guard that would have caught four wrong aliases before they shipped.

    "Evian Thonon Gaillard" pointed at "Evian" (the club is "Evian TG"),
    "Gimnastic" at "Nastic" ("Tarragona"), "Ajaccio GFCO" at itself ("Gazelec"),
    and "La Coruna" at "Deportivo" ("Depor"). Each returned an empty CSV and was
    dropped with nothing louder than a log line.
    """
    import pandas as pd

    from statpitch import paths

    roster_path = paths.processed_dir() / "clubelo_roster.parquet"
    if not roster_path.exists():
        pytest.skip("roster snapshot not built yet — run the Club Elo ingestion")

    known = set(pd.read_parquet(roster_path)["clubelo_name"])
    missing = {k: v for k, v in ce.NAME_ALIASES.items() if v not in known}
    assert not missing, f"alias targets absent from the Club Elo roster: {missing}"


def test_alias_table_has_no_self_contradictions():
    # A value that is itself a key would mean a two-hop mapping, which resolve_names
    # does not perform.
    keys = set(ce.NAME_ALIASES)
    hops = {k: v for k, v in ce.NAME_ALIASES.items() if v in keys and v != k}
    assert not hops, f"alias table needs multi-hop resolution for {hops}"


# --- as-of lookups ------------------------------------------------------------

def test_elo_as_of_returns_the_interval_in_force(history):
    assert ce.elo_as_of(history, "Arsenal", "2024-04-01") == 1850.0
    assert ce.elo_as_of(history, "Arsenal", "2024-02-01") == 1800.0


def test_elo_as_of_is_strictly_before_the_date(history):
    """A Club Elo interval starting on match day already contains that result.

    Using it as a pre-match feature leaks the outcome into the model (NFR-10), so
    the boundary is strict: asking on 2024-05-01 must return the *previous*
    rating, not the one that starts that day.
    """
    assert ce.elo_as_of(history, "Arsenal", "2024-05-01") == 1850.0
    assert ce.elo_as_of(history, "Arsenal", "2024-05-02") == 1900.0


def test_elo_as_of_returns_none_before_a_clubs_first_rating(history):
    assert ce.elo_as_of(history, "Arsenal", "2020-01-01") is None


def test_elo_as_of_returns_none_for_an_unknown_club(history):
    assert ce.elo_as_of(history, "Nobody FC", "2024-05-01") is None


def test_tier_as_of_tracks_relegation(history):
    # Atlético fixture is synthetic: tier 1 until June, tier 2 after.
    assert ce.tier_as_of(history, "Ath Madrid", "2024-03-01") == 1
    assert ce.tier_as_of(history, "Ath Madrid", "2024-07-01") == 2


def test_tier_as_of_is_the_lower_division_prior_input(history):
    """FR-9: a sub-top-flight cup entrant must be identifiable as such."""
    assert ce.tier_as_of(history, "Ath Madrid", "2024-07-01") > 1


# --- roster probing -----------------------------------------------------------

def test_snapshot_dates_probe_twice_per_season():
    dates = ce.snapshot_dates(2020, 2022)
    assert len(dates) == 6
    assert "2020-10-15" in dates and "2021-03-15" in dates


def test_snapshot_dates_span_autumn_and_spring():
    """One probe per season misses clubs promoted and relegated inside it."""
    months = {d.split("-")[1] for d in ce.snapshot_dates(2020, 2021)}
    assert months == {"10", "03"}
