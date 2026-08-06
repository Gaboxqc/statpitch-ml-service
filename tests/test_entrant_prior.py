"""Entry-round entrant prior tests (FR-9). Offline, synthetic data only.

The fit is checked against data with a known answer: clubs are generated with
true ratings, results are simulated from the Elo curve, and the estimator has to
recover the ordering and roughly the level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statpitch.models import entrant_prior as ep

# --- expected score -----------------------------------------------------------

def test_equal_ratings_give_an_even_match():
    assert ep.expected_score(1500, 1500) == pytest.approx(0.5)


def test_four_hundred_points_is_one_order_of_magnitude():
    assert ep.expected_score(1900, 1500) == pytest.approx(10 / 11, abs=1e-6)


def test_expected_score_is_monotonic_in_rating():
    assert ep.expected_score(1600, 1500) > ep.expected_score(1500, 1500)
    assert ep.expected_score(1400, 1500) < ep.expected_score(1500, 1500)


# --- entry stages -------------------------------------------------------------

def _match(comp, season, date, stage, home, away, hg=1, ag=0, neutral=False):
    return {
        "competition_id": comp, "season": season, "date": pd.Timestamp(date),
        "stage": stage, "home_team": home, "away_team": away,
        "home_goals": hg, "away_goals": ag, "neutral_venue": neutral,
    }


def test_entry_stage_is_the_first_appearance_not_the_last():
    """A round-1 entrant that reaches round 4 still entered in round 1."""
    matches = pd.DataFrame([
        _match("C", "2024-2025", "2024-08-01", "round_1", "Minnows", "A"),
        _match("C", "2024-2025", "2024-09-01", "round_2", "Minnows", "B"),
        _match("C", "2024-2025", "2024-10-01", "round_3", "Minnows", "C"),
    ])
    entries = ep.entry_stages(matches)
    row = entries[entries.club == "Minnows"].iloc[0]
    assert row["entry_stage"] == "round_1"


def test_entry_stage_is_tracked_per_season():
    matches = pd.DataFrame([
        _match("C", "2023-2024", "2023-08-01", "round_1", "Club", "A"),
        _match("C", "2024-2025", "2024-11-01", "round_3", "Club", "B"),
    ])
    entries = ep.entry_stages(matches)
    club = entries[entries.club == "Club"].set_index("season")
    assert club.loc["2023-2024", "entry_stage"] == "round_1"
    assert club.loc["2024-2025", "entry_stage"] == "round_3"


def test_entry_stage_covers_both_home_and_away_appearances():
    matches = pd.DataFrame([
        _match("C", "2024-2025", "2024-08-01", "round_1", "Home Club", "Away Club"),
    ])
    assert set(ep.entry_stages(matches)["club"]) == {"Home Club", "Away Club"}


def test_entry_stage_uses_dates_not_round_name_parsing():
    """Sources spell rounds inconsistently; date ordering sidesteps that."""
    matches = pd.DataFrame([
        _match("C", "2024-2025", "2024-08-01", "preliminary_round", "Club", "A"),
        _match("C", "2024-2025", "2024-09-01", "round_1", "Club", "B"),
    ])
    assert ep.entry_stages(matches).iloc[0]["entry_stage"] == "preliminary_round"


# --- fitting ------------------------------------------------------------------

def _simulate(seed=0, n_seasons=8):
    """Cup seasons where the true bucket ratings are known.

    Round 1 admits weak clubs, round 3 admits strong ones, and rated clubs enter
    in round 3 — the real shape of an FA Cup draw.
    """
    rng = np.random.default_rng(seed)
    true = {"round_1": 1300.0, "round_3": 1700.0}
    rated = {f"Rated {i}": 1800.0 + 40 * i for i in range(8)}
    home_advantage = 70.0

    rows = []
    for season_start in range(2016, 2016 + n_seasons):
        season = f"{season_start}-{season_start + 1}"
        weak = [f"Weak {season_start}-{i}" for i in range(16)]
        strong = [f"Strong {season_start}-{i}" for i in range(8)]

        def play(home, away, stage, date, h_elo, a_elo, season=season):
            p = ep.expected_score(h_elo + home_advantage, a_elo)
            hg, ag = (1, 0) if rng.random() < p else (0, 1)
            rows.append(_match("CUP", season, date, stage, home, away, hg, ag))

        # Round 1: weak v weak
        for i in range(0, len(weak), 2):
            play(weak[i], weak[i + 1], "round_1",
                 f"{season_start}-08-10", true["round_1"], true["round_1"])
        # Round 3: weak survivors and strong entrants meet rated clubs
        for i, club in enumerate(strong):
            opponent = list(rated)[i % len(rated)]
            play(club, opponent, "round_3",
                 f"{season_start}-11-10", true["round_3"], rated[opponent])
        for i in range(0, len(weak), 4):
            opponent = list(rated)[i % len(rated)]
            play(weak[i], opponent, "round_3",
                 f"{season_start}-11-11", true["round_1"], rated[opponent])
        # Rated v rated, which is what identifies home advantage
        names = list(rated)
        for i in range(0, len(names) - 1, 2):
            play(names[i], names[i + 1], "round_3",
                 f"{season_start}-11-12", rated[names[i]], rated[names[i + 1]])

    matches = pd.DataFrame(rows)

    elo_rows = []
    for name, value in rated.items():
        elo_rows.append({
            "clubelo_name": name, "source_name": name, "country": "ENG", "tier": 1,
            "elo": value, "valid_from": pd.Timestamp("2000-01-01"),
            "valid_to": pd.Timestamp("2030-01-01"),
        })
    elo = pd.DataFrame(elo_rows)
    mapping = {name: name for name in rated}
    return matches, elo, mapping, true, home_advantage


@pytest.fixture(scope="module")
def fitted():
    # 30 seasons, not 8. Home advantage is now estimated from rated-vs-rated play
    # alone, and 8 seasons yields only 32 such matches — below the threshold, so
    # it falls back to a default and the bucket ratings absorb the difference.
    matches, elo, mapping, true, home_advantage = _simulate(n_seasons=30)
    prior = ep.fit(matches, elo, mapping, bootstrap=40, seed=1)
    return prior, true, home_advantage


def test_fit_recovers_the_ordering_of_entry_rounds(fitted):
    """The whole point: a round-3 entrant is stronger than a round-1 entrant."""
    prior, _, _ = fitted
    assert prior.rating_for("CUP", "round_3") > prior.rating_for("CUP", "round_1")


def test_fit_recovers_the_true_ratings_within_tolerance(fitted):
    prior, true, _ = fitted
    assert prior.rating_for("CUP", "round_1") == pytest.approx(true["round_1"], abs=90)
    assert prior.rating_for("CUP", "round_3") == pytest.approx(true["round_3"], abs=90)


def test_home_advantage_is_positive_and_roughly_right():
    """Estimated from rated-vs-rated matches only, then held fixed.

    Fitting it jointly with the bucket ratings gave -27 Elo, then -3 Elo, against
    data that plainly shows a positive home effect: cups seed the weaker club at
    home, so within a bucket home is confounded with weakness, and a single bucket
    rating cannot express that.

    Averaged over seeds rather than asserted on one. Home advantage is inferred
    from win rates and 70 Elo is only a ~10pp shift, so a single simulated run is
    genuinely noisy — one seed drew exactly 50% home wins. Averaging tests the
    estimator instead of the draw.
    """
    estimates = []
    for seed in range(5):
        matches, elo, mapping, _, home_advantage = _simulate(seed=seed, n_seasons=30)
        prior = ep.fit(matches, elo, mapping, bootstrap=5, seed=1)
        estimates.append(prior.home_advantage)

    assert np.mean(estimates) > 0
    assert np.mean(estimates) == pytest.approx(home_advantage, abs=30)


def test_home_advantage_survives_the_seeding_confound():
    """The failure mode that produced a negative venue effect on real data.

    Every unrated club is placed at home against a stronger rated club, which is
    how domestic cups actually seed. A joint fit reads the resulting home deficit
    as negative home advantage; estimating from rated-vs-rated play does not.
    """
    matches, elo, mapping, _, home_advantage = _simulate(seed=3, n_seasons=30)
    rated = set(mapping)
    seeded = matches.copy()
    swap = seeded.home_team.isin(rated) & ~seeded.away_team.isin(rated)
    seeded.loc[swap, ["home_team", "away_team"]] = seeded.loc[
        swap, ["away_team", "home_team"]
    ].to_numpy()
    seeded.loc[swap, ["home_goals", "away_goals"]] = seeded.loc[
        swap, ["away_goals", "home_goals"]
    ].to_numpy()

    prior = ep.fit(seeded, elo, mapping, bootstrap=5, seed=1)
    assert prior.home_advantage > 0
    assert prior.home_advantage == pytest.approx(home_advantage, abs=45)


def test_home_advantage_falls_back_when_rated_play_is_too_thin():
    matches, elo, mapping, _, _ = _simulate(seed=0, n_seasons=2)
    prior = ep.fit(matches, elo, mapping, bootstrap=5, seed=1)
    assert prior.home_advantage == ep.DEFAULT_HOME_ADVANTAGE
    assert prior.diagnostics["home_advantage_matches"] < ep.MIN_HOME_ADVANTAGE_MATCHES


def test_every_bucket_reports_a_confidence_interval(fitted):
    prior, _, _ = fitted
    for bucket in prior.buckets.values():
        assert bucket.ci_low < bucket.elo < bucket.ci_high


def test_bucket_counts_are_reported(fitted):
    prior, _, _ = fitted
    bucket = prior.buckets[("CUP", "round_1")]
    assert bucket.n_matches > 0
    assert bucket.n_clubs > 0


def test_thin_buckets_are_pooled_rather_than_fitted_individually():
    """A one-match bucket fitted alone just reproduces that single result."""
    matches, elo, mapping, _, _ = _simulate(n_seasons=8)
    extra = pd.DataFrame([
        _match("CUP", "2016-2017", "2016-12-01", "round_9", "Oddity", "Rated 0", 0, 5)
    ])
    prior = ep.fit(
        pd.concat([matches, extra], ignore_index=True), elo, mapping, bootstrap=20, seed=1
    )
    assert ("CUP", "round_9") not in prior.buckets
    # ...and it falls back to the pooled estimate rather than vanishing.
    assert prior.rating_for("CUP", "round_9") == prior.pooled_elo


def test_unknown_bucket_falls_back_to_pooled(fitted):
    prior, _, _ = fitted
    assert prior.rating_for("CUP", "never_seen") == prior.pooled_elo
    assert prior.rating_for("OTHER.CUP", "round_1") == prior.pooled_elo


def test_fit_converges(fitted):
    prior, _, _ = fitted
    assert prior.diagnostics["converged"] == 1.0
    assert prior.n_matches_used > 0


def test_to_frame_is_sorted_and_labelled(fitted):
    prior, _, _ = fitted
    frame = prior.to_frame()
    assert set(frame.columns) >= {"competition_id", "entry_stage", "elo", "ci_low",
                                  "ci_high", "n_matches", "n_clubs", "reliable"}


def test_fit_rejects_data_with_nothing_to_learn_from():
    matches = pd.DataFrame([
        _match("CUP", "2024-2025", "2024-08-01", "round_1", "Rated 0", "Rated 1"),
    ])
    elo = pd.DataFrame([
        {"clubelo_name": n, "source_name": n, "country": "ENG", "tier": 1, "elo": 1800.0,
         "valid_from": pd.Timestamp("2000-01-01"), "valid_to": pd.Timestamp("2030-01-01")}
        for n in ("Rated 0", "Rated 1")
    ])
    # Rated-vs-rated matches are kept (they identify home advantage) but there is
    # no unrated population to estimate, which must be an explicit error rather
    # than an empty result.
    with pytest.raises(ValueError, match="no unrated entrants"):
        ep.fit(matches, elo, {"Rated 0": "Rated 0", "Rated 1": "Rated 1"}, bootstrap=5)


def test_ratings_are_looked_up_strictly_before_the_match():
    """A rating interval starting on match day already reflects that result."""
    matches = pd.DataFrame([
        _match("CUP", "2024-2025", "2024-08-10", "round_1", "Minnow", "Rated"),
    ])
    elo = pd.DataFrame([
        {"clubelo_name": "Rated", "source_name": "Rated", "country": "ENG", "tier": 1,
         "elo": 1800.0, "valid_from": pd.Timestamp("2024-08-10"),
         "valid_to": pd.Timestamp("2030-01-01")},
    ])
    # The only interval starts on match day. A same-day read would leak the result
    # into its own predictor, so the club is treated as unrated: the fit succeeds
    # with an entrant population rather than raising "no unrated entrants", which
    # is what it would do had the same-day rating been picked up.
    prior = ep.fit(matches, elo, {"Rated": "Rated"}, bootstrap=5)
    assert prior.n_matches_used == 1
    # Two clubs in one match is far below the pooling threshold, so it lands on
    # the pooled estimate rather than getting its own bucket.
    assert prior.rating_for("CUP", "round_1") == prior.pooled_elo
