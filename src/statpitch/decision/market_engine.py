"""Score matrix to every priced selection (FR-23, Design §6.1).

One Dixon-Coles matrix per fixture becomes ~60 selections here: 1X2, Double
Chance, Draw No Bet, Over/Under at every line, team totals, both-teams-to-score,
Asian Handicap at every line including quarters, and correct score. No selection
gets its own model — they are all summations over the same grid, which is what
keeps them mutually consistent. If the matrix says the home side wins 46% of the
time, the Asian Handicap at -0.5 says exactly 46% too, because it is the same
sum.

Payoff distributions, not win probabilities
===========================================

Every selection returns the full distribution over outcomes — win, half-win,
push, half-loss, loss — rather than a single probability. That is not
bookkeeping. Design §6.5 sizes stakes by log-growth, and log-growth on a
quarter-line Asian Handicap cannot be computed from a win probability: half the
stake can come back while the other half loses, and a two-outcome formula has
nowhere to put that. Anything staking a quarter line from a win probability alone
is silently mispricing it.

Correct score is generated and tagged `stakeable=False` (Requirements §3.2). Book
margins there run 15-25%, so model error dominates any apparent edge. It exists
to be displayed under FR-4 and to be structurally excluded from every staking
path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from statpitch.models.dixon_coles import ScoreMatrix

log = logging.getLogger(__name__)


class MarketFamily(StrEnum):
    ONE_X_TWO = "1x2"
    DOUBLE_CHANCE = "double_chance"
    DRAW_NO_BET = "draw_no_bet"
    TOTALS = "totals"
    TEAM_TOTALS = "team_totals"
    BTTS = "btts"
    ASIAN_HANDICAP = "asian_handicap"
    CORRECT_SCORE = "correct_score"


#: Families whose book margin is too wide for model error to survive
#: (Requirements §3.2). Structurally excluded from staking, not merely discouraged.
NON_STAKEABLE: frozenset[MarketFamily] = frozenset({MarketFamily.CORRECT_SCORE})


@dataclass(frozen=True, slots=True)
class Payoff:
    """Outcome distribution for one selection, as stake multipliers.

    `win` returns (odds - 1) per unit staked, `push` returns 0, `loss` returns -1,
    and the half variants return half of each. They must sum to 1.
    """

    win: float = 0.0
    half_win: float = 0.0
    push: float = 0.0
    half_loss: float = 0.0
    loss: float = 0.0

    @property
    def total(self) -> float:
        return self.win + self.half_win + self.push + self.half_loss + self.loss

    @property
    def win_probability(self) -> float:
        """Probability of a full or partial win, for display only.

        Deliberately not what staking consumes: a selection that half-wins 60% of
        the time is not the same bet as one that wins outright 60% of the time,
        and only the full distribution distinguishes them.
        """
        return self.win + self.half_win

    def expected_return(self, odds: float) -> float:
        """Expected profit per unit staked at `odds`."""
        return (
            self.win * (odds - 1.0)
            + self.half_win * (odds - 1.0) / 2.0
            + self.push * 0.0
            + self.half_loss * -0.5
            + self.loss * -1.0
        )

    def outcomes(self, odds: float) -> list[tuple[float, float]]:
        """(probability, return) pairs — the input log-growth actually needs."""
        return [
            (self.win, odds - 1.0),
            (self.half_win, (odds - 1.0) / 2.0),
            (self.push, 0.0),
            (self.half_loss, -0.5),
            (self.loss, -1.0),
        ]


@dataclass(frozen=True, slots=True)
class Selection:
    key: str
    family: MarketFamily
    payoff: Payoff
    line: float | None = None
    description: str = ""

    @property
    def stakeable(self) -> bool:
        return self.family not in NON_STAKEABLE

    @property
    def probability(self) -> float:
        return self.payoff.win_probability


def _binary(probability: float) -> Payoff:
    return Payoff(win=probability, loss=1.0 - probability)


def _asian(matrix: ScoreMatrix, line: float) -> Payoff:
    """Asian Handicap payoff, quarter lines included.

    A quarter line splits the stake across its two neighbours: the harder line at
    `line - 0.25` and the easier one at `line + 0.25`. Because both halves settle
    on the same goal margin, the joint outcome collapses to four cases.

    Writing `m` for the margin after the handicap, and noting the two neighbours
    differ by exactly 0.5:

      * the harder line wins  -> the easier one wins too      -> full win
      * the harder line pushes -> the easier one is 0.5 clear -> half win
      * the easier line pushes -> the harder one is 0.5 short -> half loss
      * the easier line loses -> the harder one loses too     -> full loss

    So the distribution reads straight off the neighbours, and no term needs the
    two to be combined probabilistically — they are perfectly dependent. Both
    cannot push at once, so a quarter line never returns a full push.

    `ScoreMatrix.asian_handicap` averages the neighbours for a quarter line,
    which gives the expected fraction won rather than the payoff distribution;
    that is the right answer for a probability and the wrong one for staking, so
    the neighbours are read directly here.
    """
    if abs(line * 4) % 2 != 1:
        win, push, loss = matrix.asian_handicap(line)
        return Payoff(win=win, push=push, loss=loss)

    harder_win, harder_push, _ = matrix.asian_handicap(line - 0.25)
    _, easier_push, easier_loss = matrix.asian_handicap(line + 0.25)

    return Payoff(
        win=harder_win,
        half_win=harder_push,
        push=0.0,
        half_loss=easier_push,
        loss=easier_loss,
    )


def derive(
    matrix: ScoreMatrix,
    *,
    totals_lines: tuple[float, ...] = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5),
    handicap_lines: tuple[float, ...] = (
        -3.0, -2.5, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
        0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0,
    ),
    correct_score_top_n: int = 10,
) -> list[Selection]:
    """Every selection derivable from one score matrix."""
    home, draw, away = matrix.one_x_two()
    out: list[Selection] = []

    def add(key, family, payoff, line=None, description=""):
        out.append(Selection(key, family, payoff, line, description))

    # 1X2
    add("1x2_home", MarketFamily.ONE_X_TWO, _binary(home), description="Home win")
    add("1x2_draw", MarketFamily.ONE_X_TWO, _binary(draw), description="Draw")
    add("1x2_away", MarketFamily.ONE_X_TWO, _binary(away), description="Away win")

    # Double chance
    add("dc_home_draw", MarketFamily.DOUBLE_CHANCE, _binary(home + draw), description="1X")
    add("dc_draw_away", MarketFamily.DOUBLE_CHANCE, _binary(draw + away), description="X2")
    add("dc_home_away", MarketFamily.DOUBLE_CHANCE, _binary(home + away), description="12")

    # Draw no bet — the draw refunds, so it is a push rather than a loss.
    add("dnb_home", MarketFamily.DRAW_NO_BET,
        Payoff(win=home, push=draw, loss=away), description="Home DNB")
    add("dnb_away", MarketFamily.DRAW_NO_BET,
        Payoff(win=away, push=draw, loss=home), description="Away DNB")

    # Totals
    for line in totals_lines:
        over = matrix.over(line)
        under = matrix.under(line)
        exact = max(0.0, 1.0 - over - under)   # non-zero only on whole lines
        add(f"over_{line}", MarketFamily.TOTALS,
            Payoff(win=over, push=exact, loss=under), line, f"Over {line}")
        add(f"under_{line}", MarketFamily.TOTALS,
            Payoff(win=under, push=exact, loss=over), line, f"Under {line}")

    # Team totals
    home_goals = matrix.matrix.sum(axis=1)
    away_goals = matrix.matrix.sum(axis=0)
    for side, marginal in (("home", home_goals), ("away", away_goals)):
        for line in (0.5, 1.5, 2.5):
            over = float(marginal[[i for i in range(len(marginal)) if i > line]].sum())
            add(f"{side}_over_{line}", MarketFamily.TEAM_TOTALS, _binary(over), line,
                f"{side.title()} over {line}")
            add(f"{side}_under_{line}", MarketFamily.TEAM_TOTALS, _binary(1.0 - over),
                line, f"{side.title()} under {line}")

    # Both teams to score
    btts = matrix.both_teams_to_score()
    add("btts_yes", MarketFamily.BTTS, _binary(btts), description="BTTS yes")
    add("btts_no", MarketFamily.BTTS, _binary(1.0 - btts), description="BTTS no")

    # Asian handicap, both sides.
    #
    # The away side of a handicap is the exact complement of the home side of the
    # SAME handicap, quoted at the opposite line: home -1.0 and away +1.0 settle on
    # one match with opposed outcomes. So the away payoff is the home payoff with
    # wins and losses swapped — not a fresh derivation at the opposite line, which
    # would price the away team as though it were giving the start rather than
    # receiving it.
    for line in handicap_lines:
        home_payoff = _asian(matrix, line)
        add(f"ah_home_{line}", MarketFamily.ASIAN_HANDICAP, home_payoff, line,
            f"Home {line:+g}")
        add(f"ah_away_{-line}", MarketFamily.ASIAN_HANDICAP,
            Payoff(win=home_payoff.loss, half_win=home_payoff.half_loss,
                   push=home_payoff.push, half_loss=home_payoff.half_win,
                   loss=home_payoff.win),
            -line, f"Away {-line:+g}")

    # Correct score — displayed under FR-4, never staked.
    for h, a, probability in matrix.top_scores(correct_score_top_n):
        add(f"cs_{h}_{a}", MarketFamily.CORRECT_SCORE, _binary(probability),
            description=f"{h}-{a}")

    return out


@dataclass
class MarketBook:
    """All selections for one fixture."""

    selections: list[Selection] = field(default_factory=list)

    @classmethod
    def from_matrix(cls, matrix: ScoreMatrix, **kwargs) -> MarketBook:
        return cls(derive(matrix, **kwargs))

    def stakeable(self) -> list[Selection]:
        return [s for s in self.selections if s.stakeable]

    def by_family(self, family: MarketFamily) -> list[Selection]:
        return [s for s in self.selections if s.family == family]

    def get(self, key: str) -> Selection | None:
        return next((s for s in self.selections if s.key == key), None)

    def __len__(self) -> int:
        return len(self.selections)
