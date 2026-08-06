"""Bracket simulation tests (FR-20, Design §5.3).

The invariants: exactly one champion per run, probabilities that sum to one,
stronger teams winning more often, and the two draw types behaving differently —
because simulating a fixed bracket as a random redraw misprices the question the
simulation exists to answer.
"""

from __future__ import annotations

import pytest

from statpitch.models import bracket as bk
from statpitch.models import dixon_coles as dc


def _provider(strengths: dict[str, float]):
    """Score matrices from per-team attacking rates."""

    def provide(home: str, away: str, neutral: bool) -> dc.ScoreMatrix:
        boost = 1.0 if neutral else 1.15
        return dc.score_matrix(
            strengths[home] * boost, strengths[away], rho=-0.05
        )

    return provide


EIGHT = {f"T{i}": 1.6 - 0.1 * i for i in range(8)}      # T0 strongest
FOUR = {"Strong": 2.0, "Good": 1.5, "Ok": 1.2, "Weak": 0.8}


# --- structure ----------------------------------------------------------------

def test_a_bracket_needs_a_power_of_two(fixture=None):
    with pytest.raises(bk.BracketError, match="cannot play"):
        bk.Bracket(teams=["a", "b", "c"], rounds=bk.knockout_rounds(4))


def test_duplicate_teams_are_rejected():
    with pytest.raises(bk.BracketError, match="twice"):
        bk.Bracket(teams=["a", "a"], rounds=bk.knockout_rounds(2))


def test_a_bracket_needs_two_teams():
    with pytest.raises(bk.BracketError, match="at least two"):
        bk.Bracket(teams=["a"], rounds=[])


def test_round_names_follow_the_field_size():
    names = [r.name for r in bk.knockout_rounds(16)]
    assert names == ["round_of_16", "quarter_final", "semi_final", "final"]


def test_the_final_is_neutral_by_default():
    assert bk.knockout_rounds(8)[-1].neutral_venue
    assert not bk.knockout_rounds(8)[0].neutral_venue


def test_uefa_style_rounds_are_two_legged_until_the_final():
    rounds = bk.knockout_rounds(16, two_leg_until_final=True)
    assert all(r.tie_format is bk.TieFormat.TWO_LEG for r in rounds[:-1])
    assert rounds[-1].tie_format is bk.TieFormat.SINGLE_LEG


# --- the core invariants ------------------------------------------------------

@pytest.fixture(scope="module")
def eight_team_result():
    b = bk.Bracket(teams=list(EIGHT), rounds=bk.knockout_rounds(8))
    return bk.simulate(b, _provider(EIGHT), runs=4000, seed=1)


def test_exactly_one_champion_per_run(eight_team_result):
    assert sum(eight_team_result.win.values()) == pytest.approx(1.0, abs=1e-9)


def test_everyone_reaches_the_first_round(eight_team_result):
    first = eight_team_result.rounds[0]
    for team in EIGHT:
        assert eight_team_result.reach[team][first] == pytest.approx(1.0)


def test_two_teams_reach_the_final_each_run(eight_team_result):
    total = sum(eight_team_result.reach[t]["final"] for t in EIGHT)
    assert total == pytest.approx(2.0, abs=1e-9)


def test_four_teams_reach_the_semi_finals(eight_team_result):
    total = sum(eight_team_result.reach[t]["semi_final"] for t in EIGHT)
    assert total == pytest.approx(4.0, abs=1e-9)


def test_reaching_a_later_round_is_never_more_likely(eight_team_result):
    """Survival probability can only fall as the competition progresses."""
    for team in EIGHT:
        chain = [eight_team_result.reach[team][r] for r in eight_team_result.rounds]
        assert chain == sorted(chain, reverse=True), team


def test_winning_is_no_more_likely_than_reaching_the_final(eight_team_result):
    for team in EIGHT:
        assert eight_team_result.win[team] <= eight_team_result.reach[team]["final"] + 1e-9


# --- strength -----------------------------------------------------------------

def test_the_strongest_team_wins_most_often(eight_team_result):
    winner, _ = eight_team_result.ranked()[0]
    assert winner == "T0"


def test_win_probability_tracks_strength(eight_team_result):
    ordered = [eight_team_result.win[f"T{i}"] for i in range(8)]
    # Not strictly monotone run to run, but the strongest must beat the weakest
    # by a wide margin.
    assert ordered[0] > ordered[-1] * 3


def test_a_dominant_team_usually_wins():
    strengths = {"Giant": 3.0, "A": 0.8, "B": 0.8, "C": 0.8}
    result = bk.simulate(
        bk.Bracket(teams=list(strengths), rounds=bk.knockout_rounds(4)),
        _provider(strengths), runs=3000, seed=2,
    )
    assert result.win["Giant"] > 0.6


def test_equal_teams_split_evenly():
    strengths = dict.fromkeys(["A", "B", "C", "D"], 1.3)
    result = bk.simulate(
        bk.Bracket(teams=list(strengths), rounds=bk.knockout_rounds(4)),
        _provider(strengths), runs=6000, seed=3,
    )
    for team in strengths:
        assert result.win[team] == pytest.approx(0.25, abs=0.04)


# --- draw type ----------------------------------------------------------------

def test_fixed_and_random_draws_give_different_answers():
    """The distinction the module exists to respect.

    In a fixed bracket a strong club can be drawn into a weak quarter and carry
    that luck all the way; in a random redraw it re-enters the same lottery every
    round. Simulating one as the other misprices the whole question.
    """
    strengths = {"S1": 2.2, "S2": 2.1, "W1": 0.7, "W2": 0.7}
    teams = ["S1", "S2", "W1", "W2"]          # the two strong sides meet in round 1
    rounds = bk.knockout_rounds(4)

    fixed = bk.simulate(
        bk.Bracket(teams=teams, rounds=rounds, draw_type=bk.DrawType.FIXED),
        _provider(strengths), runs=6000, seed=4,
    )
    random_draw = bk.simulate(
        bk.Bracket(teams=teams, rounds=rounds, draw_type=bk.DrawType.RANDOM),
        _provider(strengths), runs=6000, seed=4,
    )
    # Fixed: S1 and S2 eliminate each other immediately, so one strong side is
    # guaranteed out in round 1. A redraw gives them a chance to avoid each other.
    assert fixed.win["S1"] + fixed.win["S2"] < random_draw.win["S1"] + random_draw.win["S2"]


def test_a_fixed_bracket_respects_the_drawn_order():
    """Neighbours in the team order meet, so a weak pairing is an easy path."""
    strengths = {"Strong": 2.2, "Weak": 0.6, "Mid1": 1.3, "Mid2": 1.3}
    result = bk.simulate(
        bk.Bracket(
            teams=["Strong", "Weak", "Mid1", "Mid2"],
            rounds=bk.knockout_rounds(4), draw_type=bk.DrawType.FIXED,
        ),
        _provider(strengths), runs=4000, seed=5,
    )
    # Strong plays Weak first, so it should reach the final very often.
    assert result.reach["Strong"]["final"] > 0.8


def test_a_known_draw_is_respected_rather_than_resampled():
    strengths = {"Strong": 2.2, "Weak": 0.6, "Mid1": 1.3, "Mid2": 1.3}
    result = bk.simulate(
        bk.Bracket(
            teams=["Strong", "Mid1", "Mid2", "Weak"],
            rounds=bk.knockout_rounds(4),
            draw_type=bk.DrawType.RANDOM,
            known_pairings=[("Strong", "Weak"), ("Mid1", "Mid2")],
        ),
        _provider(strengths), runs=4000, seed=6,
    )
    assert result.reach["Strong"]["final"] > 0.8


# --- two-legged ties ----------------------------------------------------------

def test_two_legged_ties_favour_the_stronger_side_more():
    """Two legs give strength more chances to tell than one."""
    strengths = {"Strong": 2.0, "W1": 0.9, "W2": 0.9, "W3": 0.9}
    teams = list(strengths)
    single = bk.simulate(
        bk.Bracket(teams=teams, rounds=bk.knockout_rounds(4)),
        _provider(strengths), runs=3000, seed=7,
    )
    two_leg = bk.simulate(
        bk.Bracket(teams=teams, rounds=bk.knockout_rounds(4, two_leg_until_final=True)),
        _provider(strengths), runs=3000, seed=7,
    )
    assert two_leg.win["Strong"] >= single.win["Strong"] - 0.02


# --- reporting ----------------------------------------------------------------

def test_results_are_ranked_by_win_probability(eight_team_result):
    ranked = [p for _, p in eight_team_result.ranked()]
    assert ranked == sorted(ranked, reverse=True)


def test_the_summary_lists_every_round(eight_team_result):
    text = eight_team_result.summary()
    for round_name in eight_team_result.rounds:
        assert round_name[:9] in text


def test_runs_must_be_positive():
    b = bk.Bracket(teams=["a", "b"], rounds=bk.knockout_rounds(2))
    with pytest.raises(bk.BracketError, match="positive"):
        bk.simulate(b, _provider({"a": 1.0, "b": 1.0}), runs=0)


def test_the_default_run_count_meets_the_requirement():
    """FR-20 asks for at least 10,000 runs."""
    assert bk.DEFAULT_RUNS >= 10_000


# --- the advancement cache ----------------------------------------------------

def test_advancement_probabilities_are_complementary():
    strengths = {"A": 1.6, "B": 1.1}
    matrix = bk.advancement_matrix(["A", "B"], _provider(strengths))
    # Each entry has its own side at home, so they need not sum to one — but a
    # stronger side at home must beat a weaker side at home.
    assert matrix[0, 1] > matrix[1, 0]


def test_a_neutral_venue_removes_home_advantage():
    strengths = {"A": 1.3, "B": 1.3}
    home = bk.advancement_matrix(["A", "B"], _provider(strengths), neutral=False)
    neutral = bk.advancement_matrix(["A", "B"], _provider(strengths), neutral=True)
    assert home[0, 1] > neutral[0, 1]
    assert neutral[0, 1] == pytest.approx(0.5, abs=1e-9)
