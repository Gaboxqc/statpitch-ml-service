"""Cup and continental bracket simulation (FR-20, Design §5.3).

Simulates a competition from wherever it currently stands to the final, and
reports each club's probability of reaching every remaining round and of winning
it.

Fixed brackets and random redraws are genuinely different competitions
======================================================================

The UEFA knockout rounds run a **fixed bracket**: once the round of 16 is drawn,
a club's whole path to the final is determined, so it can be lucky or unlucky in
who sits on its side. The FA Cup and DFB-Pokal **redraw at random every round**,
so a club faces a fresh sample from the survivors each time.

That distinction is not cosmetic and both are implemented. In a fixed bracket a
strong club drawn into a weak quarter is materially more likely to reach the
final than the same club in a random-redraw competition, where it re-enters the
same lottery every round. Simulating one as the other misprices exactly the
question a bracket simulation exists to answer.

Cost
====

Ties are resolved through `knockout.resolve_single_leg` / `resolve_two_leg`,
which are the expensive part. A pairwise advancement matrix is computed once for
every club pair that could possibly meet, and the Monte Carlo then draws against
those cached numbers — so 10,000 runs cost thousands of random draws rather than
thousands of score-matrix builds.

This is the same Monte Carlo apparatus the bankroll simulation and the
simultaneous-Kelly solve use, kept generic so a third simulator never has to be
written.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from statpitch.models.dixon_coles import ScoreMatrix
from statpitch.models.knockout import resolve_single_leg, resolve_two_leg

log = logging.getLogger(__name__)

#: FR-20 requires at least this many runs.
DEFAULT_RUNS = 10_000

#: Matrix provider: (home, away, neutral) -> score matrix for that pairing.
MatrixProvider = Callable[[str, str, bool], ScoreMatrix]


class DrawType(StrEnum):
    FIXED = "fixed"          # UEFA knockout: the path is set when the round is drawn
    RANDOM = "random"        # FA Cup, DFB-Pokal: redrawn every round


class TieFormat(StrEnum):
    SINGLE_LEG = "single_leg_knockout"
    TWO_LEG = "two_leg_knockout"


class BracketError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RoundSpec:
    """One remaining round of a competition."""

    name: str
    tie_format: TieFormat = TieFormat.SINGLE_LEG
    neutral_venue: bool = False


@dataclass
class Bracket:
    """A competition from its current state to the final."""

    teams: list[str]
    rounds: list[RoundSpec]
    draw_type: DrawType = DrawType.RANDOM
    #: Ties already drawn for the next round, as (home, away) pairs. Used when a
    #: real draw has been made and should be respected rather than resampled.
    known_pairings: list[tuple[str, str]] | None = None
    #: First-leg results for two-legged ties already part-played.
    first_leg_scores: dict[tuple[str, str], tuple[int, int]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if len(self.teams) < 2:
            raise BracketError("a bracket needs at least two teams")
        if len(set(self.teams)) != len(self.teams):
            raise BracketError("a team cannot appear twice in a bracket")
        expected = 2 ** len(self.rounds)
        if len(self.teams) != expected:
            raise BracketError(
                f"{len(self.teams)} teams cannot play {len(self.rounds)} rounds; "
                f"expected {expected}. Byes must be resolved before simulating."
            )


@dataclass(frozen=True, slots=True)
class BracketResult:
    rounds: list[str]
    #: team -> round name -> probability of reaching it
    reach: dict[str, dict[str, float]]
    #: team -> probability of winning the competition
    win: dict[str, float]
    runs: int

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.win.items(), key=lambda kv: -kv[1])

    def summary(self, top: int = 10) -> str:
        header = f"{'team':24}" + "".join(f"{r[:9]:>10}" for r in self.rounds) + f"{'WIN':>10}"
        lines = [header]
        for team, probability in self.ranked()[:top]:
            row = f"{team:24}"
            row += "".join(f"{self.reach[team].get(r, 0.0):10.3f}" for r in self.rounds)
            lines.append(row + f"{probability:10.3f}")
        return "\n".join(lines)


def advancement_matrix(
    teams: list[str],
    provider: MatrixProvider,
    tie_format: TieFormat = TieFormat.SINGLE_LEG,
    *,
    neutral: bool = False,
) -> np.ndarray:
    """P[i][j] = probability team i advances past team j, with i at home.

    Computed once and reused across every Monte Carlo run. Resolving ties inside
    the simulation loop instead would rebuild the same score matrices tens of
    thousands of times for no additional information.
    """
    n = len(teams)
    out = np.full((n, n), 0.5)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            matrix = provider(teams[i], teams[j], neutral)
            if tie_format is TieFormat.TWO_LEG:
                second = provider(teams[j], teams[i], False)
                resolution = resolve_two_leg(matrix, second)
            else:
                resolution = resolve_single_leg(matrix)
            out[i, j] = resolution.home_advances
    return out


def _pair_up(order: np.ndarray) -> list[tuple[int, int]]:
    return [(int(order[k]), int(order[k + 1])) for k in range(0, len(order), 2)]


def simulate(
    bracket: Bracket,
    provider: MatrixProvider,
    *,
    runs: int = DEFAULT_RUNS,
    seed: int = 0,
) -> BracketResult:
    """Monte Carlo the bracket to a champion (FR-20)."""
    if runs < 1:
        raise BracketError("runs must be positive")

    teams = list(bracket.teams)
    index = {team: i for i, team in enumerate(teams)}
    rng = np.random.default_rng(seed)

    # One advancement matrix per distinct (format, venue) combination, since a
    # neutral-venue final and a two-legged semi resolve differently.
    cache: dict[tuple[TieFormat, bool], np.ndarray] = {}
    for spec in bracket.rounds:
        key = (spec.tie_format, spec.neutral_venue)
        if key not in cache:
            cache[key] = advancement_matrix(
                teams, provider, spec.tie_format, neutral=spec.neutral_venue
            )

    reach_counts = {r.name: np.zeros(len(teams)) for r in bracket.rounds}
    win_counts = np.zeros(len(teams))

    # A drawn first round is respected rather than resampled.
    opening: list[tuple[int, int]] | None = None
    if bracket.known_pairings:
        opening = [(index[h], index[a]) for h, a in bracket.known_pairings]

    for _ in range(runs):
        alive = np.arange(len(teams))

        for depth, spec in enumerate(bracket.rounds):
            for team_index in alive:
                reach_counts[spec.name][team_index] += 1

            if depth == 0 and opening is not None:
                pairings = opening
            elif bracket.draw_type is DrawType.RANDOM:
                # Redrawn every round: shuffle the survivors.
                pairings = _pair_up(rng.permutation(alive))
            else:
                # Fixed bracket: neighbours in the original order meet.
                pairings = _pair_up(alive)

            advance = cache[(spec.tie_format, spec.neutral_venue)]
            survivors = []
            for home, away in pairings:
                if rng.random() < advance[home, away]:
                    survivors.append(home)
                else:
                    survivors.append(away)
            alive = np.array(survivors)

        for team_index in alive:
            win_counts[team_index] += 1

    return BracketResult(
        rounds=[r.name for r in bracket.rounds],
        reach={
            team: {
                r.name: float(reach_counts[r.name][i] / runs) for r in bracket.rounds
            }
            for team, i in index.items()
        },
        win={team: float(win_counts[i] / runs) for team, i in index.items()},
        runs=runs,
    )


def knockout_rounds(n_teams: int, *, neutral_final: bool = True,
                    two_leg_until_final: bool = False) -> list[RoundSpec]:
    """Standard round names for a bracket of `n_teams`.

    UEFA plays every knockout round over two legs except the final, which is a
    single leg at a neutral venue; domestic cups are single-leg throughout.
    """
    names = {2: "final", 4: "semi_final", 8: "quarter_final",
             16: "round_of_16", 32: "round_of_32", 64: "round_of_64"}
    rounds = []
    remaining = n_teams
    while remaining >= 2:
        name = names.get(remaining, f"round_of_{remaining}")
        is_final = remaining == 2
        rounds.append(
            RoundSpec(
                name=name,
                tie_format=(
                    TieFormat.TWO_LEG
                    if two_leg_until_final and not is_final
                    else TieFormat.SINGLE_LEG
                ),
                neutral_venue=is_final and neutral_final,
            )
        )
        remaining //= 2
    return rounds
