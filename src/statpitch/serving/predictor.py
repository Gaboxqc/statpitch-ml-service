"""Format-aware match prediction (Design §5.3, §7).

Inference branches on the competition's format, resolved for the specific stage
and season rather than read off the competition:

* `round_robin` / `swiss_league_phase` — 1X2 plus the score matrix
* `single_leg_knockout` — the same, plus extra time and shootouts (FR-8), because
  a draw is not a final result
* `two_leg_knockout` — aggregate qualification across both legs (FR-7)

Getting the branch wrong does not raise. It returns a confident, well-formed
prediction that answers a different question — a draw probability for a tie that
cannot be drawn — so the branch is resolved through the taxonomy and the resulting
format is echoed in every response.

Artifacts load once at startup (NFR-2). Nothing here trains, reads a CSV or hits
a network, so a prediction is a matrix build and some summation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date as Date

import pandas as pd

from statpitch import taxonomy
from statpitch.decision.market_engine import MarketBook
from statpitch.models import knockout
from statpitch.models.dixon_coles import ScoreMatrix, score_matrix
from statpitch.models.entrant_prior import ELO_SCALE

log = logging.getLogger(__name__)

#: Fallback when a club has no rating at all. Deliberately below the rated
#: population rather than at its mean — an unknown club is more often small.
DEFAULT_ELO = 1400.0

#: League-average goal rates, used when a competition has no fitted environment.
DEFAULT_HOME_RATE = 1.45
DEFAULT_AWAY_RATE = 1.20

#: Elo-to-goals conversion. A 400-point edge roughly doubles the goal ratio.
ELO_GOAL_SENSITIVITY = 0.55


class PredictionError(ValueError):
    pass


@dataclass
class Artifacts:
    """Everything inference needs, loaded once (Design §7)."""

    elo: dict[str, float] = field(default_factory=dict)
    goal_environment: dict[str, tuple[float, float]] = field(default_factory=dict)
    rho: dict[str, float] = field(default_factory=dict)
    entrant_prior: dict[str, float] = field(default_factory=dict)
    home_advantage_elo: float = 54.0

    @classmethod
    def load(cls, data_dir=None) -> Artifacts:
        """Load from the processed data tree, tolerating absent artifacts.

        A missing artifact degrades the prediction rather than failing the
        request (NFR-7); the response reports what was available.
        """
        from statpitch import paths

        root = data_dir or paths.processed_dir()
        artifacts = cls()

        elo_path = root / "elo_ratings_all.parquet"
        if not elo_path.exists():
            elo_path = root / "elo_ratings.parquet"
        if elo_path.exists():
            frame = pd.read_parquet(elo_path)
            latest = frame.sort_values("valid_from").groupby("source_name").last()
            artifacts.elo = {str(k): float(v) for k, v in latest["elo"].items()}
            log.info("artifacts: %d club ratings", len(artifacts.elo))

        return artifacts

    def rating(self, club: str) -> float:
        return self.elo.get(club, DEFAULT_ELO)

    def environment(self, competition_id: str) -> tuple[float, float]:
        return self.goal_environment.get(
            competition_id, (DEFAULT_HOME_RATE, DEFAULT_AWAY_RATE)
        )


@dataclass(frozen=True, slots=True)
class Prediction:
    competition_id: str
    home_team: str
    away_team: str
    format: str
    stage: str | None
    neutral_venue: bool
    matrix: ScoreMatrix
    #: Present only for knockout formats, where a draw is not a final result.
    tie: knockout.TieResolution | None = None
    odds_coverage: bool = True

    @property
    def one_x_two(self) -> tuple[float, float, float]:
        return self.matrix.one_x_two()

    @property
    def expected_goals(self) -> tuple[float, float]:
        return self.matrix.expected_goals()

    def markets(self) -> MarketBook:
        return MarketBook.from_matrix(self.matrix)

    def as_dict(self) -> dict:
        home, draw, away = self.one_x_two
        xg_home, xg_away = self.expected_goals
        out = {
            "competition_id": self.competition_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "format": self.format,
            "stage": self.stage,
            "neutral_venue": self.neutral_venue,
            "probabilities": {"home": home, "draw": draw, "away": away},
            "expected_goals": {"home": xg_home, "away": xg_away},
            "over_under": {
                f"over_{line}": self.matrix.over(line) for line in (1.5, 2.5, 3.5)
            },
            "btts": self.matrix.both_teams_to_score(),
            "correct_scores": [
                {"home": h, "away": a, "probability": p}
                for h, a, p in self.matrix.top_scores(10)
            ],
            "odds_coverage": self.odds_coverage,
        }
        if self.tie is not None:
            out["tie"] = self.tie.as_dict()
        return out


class Predictor:
    """Serves predictions from loaded artifacts."""

    def __init__(self, artifacts: Artifacts | None = None):
        self.artifacts = artifacts or Artifacts()

    # --- goal rates ------------------------------------------------------

    def rates(
        self, competition_id: str, home: str, away: str, *, neutral: bool = False
    ) -> tuple[float, float]:
        """Convert a rating difference into a pair of goal rates.

        Home advantage is applied on the Elo scale and only at a real venue. A
        neutral final removes it entirely, which is what makes a neutral-venue
        prediction differ from the same fixture at home.
        """
        base_home, base_away = self.artifacts.environment(competition_id)
        edge = self.artifacts.rating(home) - self.artifacts.rating(away)
        if not neutral:
            edge += self.artifacts.home_advantage_elo

        # Split the edge symmetrically: a stronger side both scores more and
        # concedes less, which keeps total goals roughly stable as the edge grows.
        shift = ELO_GOAL_SENSITIVITY * edge / ELO_SCALE
        return base_home * (10 ** (shift / 2)), base_away * (10 ** (-shift / 2))

    # --- prediction ------------------------------------------------------

    def predict(
        self,
        competition_id: str,
        home: str,
        away: str,
        *,
        stage: str | None = None,
        season: str | int | None = None,
        neutral: bool | None = None,
        first_leg_score: tuple[int, int] | None = None,
    ) -> Prediction:
        """Predict one fixture, branching on its resolved format (Design §5.3)."""
        competition = taxonomy.get(competition_id)
        resolved = competition.resolve_format(stage=stage, season=season)
        at_neutral = (
            competition.is_neutral_venue(stage) if neutral is None else bool(neutral)
        )

        lambda_home, lambda_away = self.rates(
            competition_id, home, away, neutral=at_neutral
        )
        rho = self.artifacts.rho.get(competition_id, 0.0)
        matrix = score_matrix(
            lambda_home, lambda_away,
            rho=_safe_rho(rho, lambda_home, lambda_away),
        )

        tie = None
        if resolved == "single_leg_knockout":
            tie = knockout.resolve_single_leg(matrix, extra_time=competition.extra_time)
        elif resolved == "two_leg_knockout":
            # Leg two is played at the other ground, so the rates invert.
            second_home, second_away = self.rates(
                competition_id, away, home, neutral=False
            )
            leg_two = score_matrix(
                second_home, second_away,
                rho=_safe_rho(rho, second_home, second_away),
            )
            tie = knockout.resolve_two_leg(
                matrix, leg_two, first_leg_score=first_leg_score
            )

        return Prediction(
            competition_id=competition_id,
            home_team=home,
            away_team=away,
            format=resolved,
            stage=stage,
            neutral_venue=at_neutral,
            matrix=matrix,
            tie=tie,
            odds_coverage=competition.odds_coverage,
        )

    def predict_tie(
        self,
        competition_id: str,
        team_a: str,
        team_b: str,
        *,
        season: str | int | None = None,
        first_leg_score: tuple[int, int] | None = None,
    ) -> Prediction:
        """Two-legged aggregate prediction (FR-7), with `team_a` at home first."""
        return self.predict(
            competition_id, team_a, team_b,
            stage="semi_final", season=season,
            first_leg_score=first_leg_score,
        )

    def rank_teams(self, competition_id: str, teams: list[str]) -> list[dict]:
        return sorted(
            (
                {"team": t, "elo": self.artifacts.rating(t)}
                for t in teams
            ),
            key=lambda row: -row["elo"],
        )


def _safe_rho(rho: float, lambda_home: float, lambda_away: float) -> float:
    """Clamp rho into the range these rates allow.

    A rho fitted on a competition's average rates can be out of range for one
    unusually high-scoring fixture, and the matrix would then raise mid-request.
    """
    from statpitch.models.dixon_coles import rho_bounds

    low, high = rho_bounds(lambda_home, lambda_away)
    return float(min(max(rho, low), high))


def today_fixtures(matches: pd.DataFrame, on: Date | str | None = None) -> pd.DataFrame:
    """Fixtures scheduled for a given date, across every in-scope competition."""
    day = pd.Timestamp(on or pd.Timestamp.today()).normalize()
    return matches[matches["date"].dt.normalize() == day]
