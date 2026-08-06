"""Extra time and penalty shootouts (FR-8, Design §5.3).

In a single-leg knockout a draw is not a final result, so the score matrix alone
cannot answer "who advances". This resolves the rest: if regulation ends level,
thirty minutes of extra time are played, and if that is still level the tie goes
to a shootout.

What the data says, all measured on 365 extra-time and 222 shootout observations
=================================================================================

**Extra time is more open than a pro-rata extrapolation, not less.** The
conventional picture of cagey, defensive extra time does not hold: 1.101 goals in
thirty minutes against 0.927 expected from the league rate, a multiplier of 1.19.
Measured against the matches that actually reach extra time — which are level at
90 and therefore lower-scoring than average, at 2.371 goals — the multiplier is
1.39. The like-for-like figure is the one used, because the model is extrapolating
from *this* fixture's rate rather than from the league's.

**Extra time is decided by strength.** Splitting on the Elo difference, the home
side advances 23.4% of the time when much weaker, 56.7% when evenly matched and
80.0% when much stronger. So extra time is modelled as more football, at a
rescaled rate, and the score matrix does the work.

**A shootout is close to a coin flip.** Across the same strength split the home
side wins 52.4%, 53.6% and 63.6% — a far flatter gradient — and overall 55.6%,
which a binomial test cannot distinguish from even (p=0.315). Design §5.3 assumes
exactly this, and it is worth stating that the assumption was checked rather than
inherited. A shootout destroys most of the strength signal that extra time
preserves.

Consequence for a cup tie: the further a match goes, the less the model knows.
That is a property of the competition, not a defect of the model, and it belongs
in the reported uncertainty rather than being smoothed away.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from statpitch.models.dixon_coles import ScoreMatrix, score_matrix

log = logging.getLogger(__name__)

#: Extra time is 30 of the 90 regulation minutes.
EXTRA_TIME_FRACTION = 30.0 / 90.0

#: Goals arrive faster in extra time than a pro-rata extrapolation implies.
#: Measured against the matches that actually reach it, which are level at 90 and
#: therefore lower-scoring than average: 1.101 observed against 0.790 expected.
EXTRA_TIME_RATE_MULTIPLIER = 1.39

#: Home advantage in a shootout, measured at 0.556 over 222 observations and not
#: distinguishable from even (p=0.315). Held at 0.5 deliberately: an effect that
#: a binomial test cannot separate from a coin flip should not be encoded as one.
SHOOTOUT_HOME_ADVANTAGE = 0.5

#: How far a shootout is allowed to reflect team strength. Measured gradient is
#: shallow — 52.4% to 63.6% across the whole Elo range against 23.4% to 80.0% in
#: extra time — so strength is admitted only weakly.
SHOOTOUT_STRENGTH_WEIGHT = 0.15


@dataclass(frozen=True, slots=True)
class TieResolution:
    """Probabilities for how a single-leg knockout resolves (FR-8)."""

    home_in_regulation: float
    away_in_regulation: float
    home_in_extra_time: float
    away_in_extra_time: float
    home_on_penalties: float
    away_on_penalties: float

    @property
    def home_advances(self) -> float:
        return (
            self.home_in_regulation + self.home_in_extra_time + self.home_on_penalties
        )

    @property
    def away_advances(self) -> float:
        return (
            self.away_in_regulation + self.away_in_extra_time + self.away_on_penalties
        )

    @property
    def reaches_extra_time(self) -> float:
        return (
            self.home_in_extra_time + self.away_in_extra_time
            + self.home_on_penalties + self.away_on_penalties
        )

    @property
    def reaches_penalties(self) -> float:
        return self.home_on_penalties + self.away_on_penalties

    def as_dict(self) -> dict[str, float]:
        return {
            "home_advances": self.home_advances,
            "away_advances": self.away_advances,
            "decided_in_regulation": self.home_in_regulation + self.away_in_regulation,
            "reaches_extra_time": self.reaches_extra_time,
            "reaches_penalties": self.reaches_penalties,
        }


def extra_time_matrix(
    lambda_home: float,
    lambda_away: float,
    rho: float = 0.0,
    *,
    multiplier: float = EXTRA_TIME_RATE_MULTIPLIER,
    max_goals: int = 6,
) -> ScoreMatrix:
    """Score matrix for the extra-time period alone.

    Truncated lower than the regulation matrix because thirty minutes at these
    rates puts negligible mass beyond six goals, and the smaller grid keeps a
    bracket simulation cheap.
    """
    scale = EXTRA_TIME_FRACTION * multiplier
    return score_matrix(
        lambda_home * scale, lambda_away * scale, rho=rho, max_goals=max_goals
    )


def shootout_probability(
    strength_edge: float = 0.0,
    *,
    base: float = SHOOTOUT_HOME_ADVANTAGE,
    weight: float = SHOOTOUT_STRENGTH_WEIGHT,
) -> float:
    """Probability the home side wins a shootout.

    `strength_edge` is the home side's regulation win probability minus the away
    side's, so it sits in [-1, 1]. It is admitted at a low weight because the
    measured gradient is shallow: a shootout discards most of what the model knows
    about the two teams, and pretending otherwise would report false confidence on
    exactly the ties where the model is least informative.
    """
    return float(min(0.99, max(0.01, base + weight * strength_edge * 0.5)))


def resolve_single_leg(
    matrix: ScoreMatrix,
    *,
    extra_time: bool = True,
    multiplier: float = EXTRA_TIME_RATE_MULTIPLIER,
) -> TieResolution:
    """Resolve a single-leg knockout to who advances (FR-8).

    `extra_time=False` covers competitions that go straight to penalties.
    """
    home, draw, away = matrix.one_x_two()

    if not extra_time:
        p_home = shootout_probability(home - away)
        return TieResolution(home, away, 0.0, 0.0, draw * p_home, draw * (1 - p_home))

    et = extra_time_matrix(
        matrix.lambda_home, matrix.lambda_away, matrix.rho, multiplier=multiplier
    )
    et_home, et_draw, et_away = et.one_x_two()

    # The strength edge carried into a shootout comes from regulation, where the
    # model actually has information, not from the extra-time period alone.
    p_home_pens = shootout_probability(home - away)

    return TieResolution(
        home_in_regulation=home,
        away_in_regulation=away,
        home_in_extra_time=draw * et_home,
        away_in_extra_time=draw * et_away,
        home_on_penalties=draw * et_draw * p_home_pens,
        away_on_penalties=draw * et_draw * (1.0 - p_home_pens),
    )


def resolve_two_leg(
    leg_one: ScoreMatrix,
    leg_two: ScoreMatrix,
    *,
    first_leg_score: tuple[int, int] | None = None,
    multiplier: float = EXTRA_TIME_RATE_MULTIPLIER,
) -> TieResolution:
    """Aggregate qualification over two legs (FR-7).

    `leg_two` is oriented from the second leg's home side, which is the FIRST
    leg's away side — so the aggregate is accumulated with that flip applied.
    Getting the orientation wrong silently reverses every tie.

    Once the first leg has been played, `first_leg_score` conditions on it and
    only the second leg is simulated. The away-goals rule is not applied: UEFA
    abolished it from 2021-2022 and the taxonomy carries the cutoff for the
    historical seasons where it still applied.
    """
    resolution_home = 0.0
    resolution_away = 0.0
    level = 0.0

    first_leg = (
        None if first_leg_score is None else first_leg_score
    )

    for i in range(leg_one.matrix.shape[0]):
        for j in range(leg_one.matrix.shape[1]):
            if first_leg is not None:
                if (i, j) != first_leg:
                    continue
                weight_one = 1.0
            else:
                weight_one = float(leg_one.matrix[i, j])
                if weight_one <= 0:
                    continue

            for k in range(leg_two.matrix.shape[0]):
                for m in range(leg_two.matrix.shape[1]):
                    weight = weight_one * float(leg_two.matrix[k, m])
                    if weight <= 0:
                        continue
                    # Tie's home side scored i in leg one and m in leg two,
                    # since it is the away side of leg two.
                    aggregate_home = i + m
                    aggregate_away = j + k
                    if aggregate_home > aggregate_away:
                        resolution_home += weight
                    elif aggregate_home < aggregate_away:
                        resolution_away += weight
                    else:
                        level += weight

    # A level aggregate goes to extra time in the second leg, then penalties.
    et = extra_time_matrix(
        leg_two.lambda_home, leg_two.lambda_away, leg_two.rho, multiplier=multiplier
    )
    et_second_home, et_draw, et_second_away = et.one_x_two()

    total = resolution_home + resolution_away + level
    edge = (resolution_home - resolution_away) / total if total > 0 else 0.0
    p_home_pens = shootout_probability(edge)

    return TieResolution(
        home_in_regulation=resolution_home,
        away_in_regulation=resolution_away,
        # Extra time is played at the second leg's venue, so its home side is the
        # tie's AWAY side.
        home_in_extra_time=level * et_second_away,
        away_in_extra_time=level * et_second_home,
        home_on_penalties=level * et_draw * p_home_pens,
        away_on_penalties=level * et_draw * (1.0 - p_home_pens),
    )
