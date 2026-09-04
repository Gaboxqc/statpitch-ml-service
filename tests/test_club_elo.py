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

    Checked against `clubelo_roster_full.parquet`, not `clubelo_roster.parquet`.
    The latter holds the Big 5 only and has no consumer left in the codebase;
    the full roster is the one `map_fixture_clubs` and `fetch_league_club_elo`
    actually resolve against, so it is the set an alias has to be valid in. It
    is a superset, which does not weaken the guard: a mistyped club name is
    absent from 1,402 entries exactly as it was from 427.
    """
    import pandas as pd

    from statpitch import paths

    roster_path = paths.processed_dir() / "clubelo_roster_full.parquet"
    if not roster_path.exists():
        pytest.skip("roster snapshot not built yet — run the Club Elo ingestion")

    known = set(pd.read_parquet(roster_path)["clubelo_name"])
    missing = {k: v for k, v in ce.NAME_ALIASES.items() if v not in known}
    assert not missing, f"alias targets absent from the Club Elo roster: {missing}"

    # The same hazard, in the table that reconciles the other source's spelling.
    missing_of = {
        k: v for k, v in ce.OPENFOOTBALL_ALIASES.items() if v not in known
    }
    assert not missing_of, f"openfootball alias targets absent: {missing_of}"


def test_alias_table_has_no_self_contradictions():
    # A value that is itself a key would mean a two-hop mapping, which resolve_names
    # does not perform.
    keys = set(ce.NAME_ALIASES)
    hops = {k: v for k, v in ce.NAME_ALIASES.items() if v in keys and v != k}
    assert not hops, f"alias table needs multi-hop resolution for {hops}"


# --- cup-entrant matching -----------------------------------------------------

@pytest.fixture
def cup_roster():
    """A roster with the collisions that actually caused trouble."""
    rows = [
        ("Atletico", "ESP", 1), ("Atletico B", "ESP", 2), ("Real Madrid", "ESP", 1),
        ("Bilbao", "ESP", 1), ("Bilbao B", "ESP", 2),
        ("Sociedad", "ESP", 1), ("Sociedad B", "ESP", 2),
        ("Paris SG", "FRA", 1), ("Paris FC", "FRA", 2),
        ("Dortmund", "GER", 1), ("Koeln", "GER", 1), ("Fortuna Koeln", "GER", 2),
        ("Brugge", "BEL", 1), ("Cercle Brugge", "BEL", 1),
        ("Milan", "ITA", 1), ("Man City", "ENG", 1),
    ]
    df = pd.DataFrame(rows, columns=["clubelo_name", "country", "best_tier"])
    df["norm"] = df["clubelo_name"].map(ce.normalise)
    return df


def _resolve(names, roster):
    return ce.resolve_cup_clubs(names, roster)


def test_formal_names_resolve_to_club_elo_short_names(cup_roster):
    r = _resolve({"AC Milan": "ITA", "Borussia Dortmund": "GER"}, cup_roster)
    assert r.mapping["AC Milan"] == "Milan"
    assert r.mapping["Borussia Dortmund"] == "Dortmund"


def test_umlaut_transliteration_matches_club_elos_spelling(cup_roster):
    """Club Elo writes "Koeln"; NFKD alone gives "koln" and never matches."""
    r = _resolve({"1. FC Köln": "GER"}, cup_roster)
    assert r.mapping["1. FC Köln"] == "Koeln"


def test_psg_is_never_resolved_to_paris_fc(cup_roster):
    """The headline near-miss: both names reduce to "paris".

    Paris FC is a different, lower-division club. Accepting the match would put
    its rating on every PSG fixture — wrong in a way that looks entirely
    plausible in the output.
    """
    r = _resolve({"Paris Saint-Germain": "FRA"}, cup_roster)
    assert r.mapping.get("Paris Saint-Germain") != "Paris FC"
    assert "Paris Saint-Germain" not in r.mapping


def test_reserve_teams_do_not_block_their_first_team(cup_roster):
    """Club Elo marks reserves with a trailing "B".

    Dropping that single character makes "Atletico B" indistinguishable from
    "Atletico", so the first team resolves to nothing.
    """
    r = _resolve(
        {"Atlético Madrid": "ESP", "Athletic Bilbao": "ESP", "Real Sociedad": "ESP"},
        cup_roster,
    )
    assert r.mapping["Atlético Madrid"] == "Atletico"
    assert r.mapping["Athletic Bilbao"] == "Bilbao"
    assert r.mapping["Real Sociedad"] == "Sociedad"


def test_atletico_never_resolves_to_real_madrid(cup_roster):
    r = _resolve({"Atlético Madrid": "ESP"}, cup_roster)
    assert r.mapping["Atlético Madrid"] != "Real Madrid"


def test_token_subset_prefers_the_right_club_over_the_near_miss(cup_roster):
    """"Brugge" is a subset of "Club Brugge KV"; "Cercle Brugge" is not.

    Fuzzy matching ranks Cercle Brugge first, which is why it is not used.
    """
    r = _resolve({"Club Brugge KV": "BEL"}, cup_roster)
    assert r.mapping["Club Brugge KV"] == "Brugge"


def test_longer_roster_name_does_not_steal_a_shorter_query(cup_roster):
    r = _resolve({"1. FC Köln": "GER"}, cup_roster)
    assert r.mapping["1. FC Köln"] != "Fortuna Koeln"


def test_country_constraint_prevents_cross_border_collisions(cup_roster):
    """Without it, short names collide across half a dozen countries."""
    r = _resolve({"Milan": "ENG"}, cup_roster)
    assert "Milan" not in r.mapping


def test_nan_country_is_treated_as_unknown_not_as_a_key(cup_roster):
    """NaN is truthy, so `row.country or fallback` yields NaN, not the fallback.

    Left unhandled every continental club was keyed on NaN and matched nothing.
    """
    r = _resolve({"AC Milan": float("nan")}, cup_roster)
    assert r.mapping["AC Milan"] == "Milan"


def test_an_exact_key_match_wins_before_the_subset_fallback(cup_roster):
    r = _resolve({"Milan": "ITA"}, cup_roster)
    assert r.mapping["Milan"] == "Milan"


def test_ambiguity_is_reported_never_resolved(cup_roster):
    """Both "Lazio" and "Roma" are subsets of "Lazio Roma".

    Nothing in the name distinguishes which club is meant, so the pair is handed
    back for a human rather than decided by ordering.
    """
    roster = pd.concat([
        cup_roster,
        pd.DataFrame([
            {"clubelo_name": "Lazio", "country": "ITA", "best_tier": 1, "norm": "lazio"},
            {"clubelo_name": "Roma", "country": "ITA", "best_tier": 1, "norm": "roma"},
        ]),
    ])
    r = _resolve({"Lazio Roma": "ITA"}, roster)
    assert "Lazio Roma" not in r.mapping
    assert set(r.ambiguous["Lazio Roma"]) == {"Lazio", "Roma"}


def test_unmatched_clubs_are_reported(cup_roster):
    r = _resolve({"AFC Sudbury": "ENG"}, cup_roster)
    assert "AFC Sudbury" in r.unmatched
    assert r.coverage == 0.0


def test_club_elo_tier_limit_is_documented():
    """Verified against the API: tier 1-2 return history, tier 3+ return empty.

    This qualifies FR-9's claim that Club Elo covers "the full pyramid". Its own
    example — a Segunda División club in the Copa del Rey — is tier 2 and works.
    """
    assert ce.CLUB_ELO_TIER_LIMIT == 2


def test_legal_form_tokens_exclude_load_bearing_words():
    """Stripping "Real" from "Real Madrid" leaves "madrid" and starts colliding."""
    for word in ("real", "atletico", "athletic", "sporting", "union", "deportivo"):
        assert word not in ce._CLUB_TYPE_TOKENS


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
