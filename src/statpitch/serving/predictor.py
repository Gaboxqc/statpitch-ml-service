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

import json
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

#: Last-resort fallback when a club has no rating and no entrant prior applies.
#: Deliberately below the rated population rather than at its mean — an unknown
#: club is more often small. Reaching this is reported, never silent.
DEFAULT_ELO = 1400.0

#: League-average goal rates, used when a competition has no fitted environment.
DEFAULT_HOME_RATE = 1.45
DEFAULT_AWAY_RATE = 1.20

#: Elo-to-goals conversion. A 400-point edge roughly doubles the goal ratio.
ELO_GOAL_SENSITIVITY = 0.55

#: Home advantage differs by more than a factor of two between competition types,
#: and this was measured rather than assumed: 54.4 Elo over 19,763 league matches
#: against 24.6 over 982 rated-vs-rated cup matches. Domestic cups seed the weaker
#: club at home, so a league constant applied to a cup tie over-favours the host by
#: ~30 Elo — precisely in the lower-division-hosts-a-big-club ties that most need
#: getting right.
LEAGUE_HOME_ADVANTAGE_ELO = 54.4
CUP_HOME_ADVANTAGE_ELO = 24.6

#: Formats played as a league table. Everything else is a knockout tie.
LEAGUE_FORMATS = frozenset({"round_robin", "swiss_league_phase"})


class PredictionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Rating:
    """A club's strength and, as importantly, where it came from."""

    elo: float
    #: `club_elo` | `entry_prior` | `pooled_prior` | `default`
    source: str

    @property
    def is_measured(self) -> bool:
        return self.source == "club_elo"


@dataclass
class Artifacts:
    """Everything inference needs, loaded once (Design §7)."""

    elo: dict[str, float] = field(default_factory=dict)
    #: Openfootball's formal club names ("1. FC Köln") to Club Elo's short ones.
    aliases: dict[str, str] = field(default_factory=dict)
    goal_environment: dict[str, tuple[float, float]] = field(default_factory=dict)
    rho: dict[str, float] = field(default_factory=dict)
    #: (competition_id, entry_stage) -> fitted Elo for clubs entering there (FR-9).
    entrant_prior: dict[tuple[str, str], float] = field(default_factory=dict)
    #: Fallback for an unrated cup entrant whose entry stage has no fitted bucket.
    pooled_entrant_elo: float | None = None
    home_advantage_elo: float = LEAGUE_HOME_ADVANTAGE_ELO
    cup_home_advantage_elo: float = CUP_HOME_ADVANTAGE_ELO
    #: Upcoming fixtures, built offline by `scripts/build_fixtures.py`. None when
    #: the artifact is absent, which the API reports as a refusal rather than as
    #: an empty slate — "no source" and "nothing on today" are different answers.
    fixtures: pd.DataFrame | None = None
    #: When that artifact was built. A fixture list is a claim about the future
    #: and kickoff times move, so its age is part of the answer.
    fixtures_generated_at: str | None = None

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
            artifacts.elo = _latest_ratings(pd.read_parquet(elo_path))

        # Two alias maps, kept separate at rest because they were built for
        # different sources — cup rosters and league fixture lists — and merged
        # here because a lookup does not care which file a name came from.
        for alias_name in ("cup_club_elo_map.json", "fixture_club_elo_map.json"):
            alias_path = root / alias_name
            if not alias_path.exists():
                continue
            matched = json.loads(alias_path.read_text(encoding="utf-8")).get("matched", {})
            artifacts.aliases.update({str(k): str(v) for k, v in matched.items()})

        fixtures_path = root / "fixtures.parquet"
        if fixtures_path.exists():
            frame = pd.read_parquet(fixtures_path)
            artifacts.fixtures = frame
            if "generated_at" in frame.columns and not frame.empty:
                artifacts.fixtures_generated_at = str(frame["generated_at"].iloc[0])

        prior_path = paths.data_root() / "entrant_prior.json"
        if prior_path.exists():
            artifacts._load_entrant_prior(
                json.loads(prior_path.read_text(encoding="utf-8"))
            )

        log.info(
            "artifacts: %d ratings, %d aliases, %d entrant buckets, %s fixtures",
            len(artifacts.elo), len(artifacts.aliases), len(artifacts.entrant_prior),
            "no" if artifacts.fixtures is None else len(artifacts.fixtures),
        )
        return artifacts

    def _load_entrant_prior(self, raw: dict) -> None:
        self.entrant_prior = {
            (str(b["competition_id"]), str(b["entry_stage"])): float(b["elo"])
            for b in raw.get("buckets", [])
            if b.get("reliable", True)
        }
        if raw.get("pooled_elo") is not None:
            self.pooled_entrant_elo = float(raw["pooled_elo"])
        # The cup figure is the one this artifact measures; the league control it
        # was validated against stays at its own constant.
        if raw.get("home_advantage_elo") is not None:
            self.cup_home_advantage_elo = float(raw["home_advantage_elo"])

    # --- ratings ---------------------------------------------------------

    def rate(
        self,
        club: str,
        *,
        competition_id: str | None = None,
        entry_stage: str | None = None,
    ) -> Rating:
        """Resolve a club to a rating, and say which tier of evidence supplied it.

        Order: a measured Club Elo rating, then the fitted entry-round prior, then
        the pooled entrant level, then a bare default. Anything past the first is a
        materially weaker claim, so the response carries the source rather than
        presenting all four as equal.

        `entry_stage` is the round the club ENTERED the competition, never the
        round this fixture is played in. A club that enters the FA Cup in round 1
        is a National League side and remains one in round 4; reading the bucket
        off the match stage would rate it as a Premier League entrant for winning
        three ties. Callers that do not know the entry round get the pooled level,
        which is the honest answer rather than a confident wrong one.
        """
        for key in (club, self.aliases.get(club)):
            if key is not None and key in self.elo:
                return Rating(self.elo[key], "club_elo")

        if competition_id is not None and entry_stage is not None:
            bucket = self.entrant_prior.get((competition_id, entry_stage))
            if bucket is not None:
                return Rating(bucket, "entry_prior")

        if competition_id is not None and self.pooled_entrant_elo is not None:
            return Rating(self.pooled_entrant_elo, "pooled_prior")

        return Rating(DEFAULT_ELO, "default")

    def rating(self, club: str) -> float:
        return self.rate(club).elo

    def environment(self, competition_id: str) -> tuple[float, float]:
        return self.goal_environment.get(
            competition_id, (DEFAULT_HOME_RATE, DEFAULT_AWAY_RATE)
        )

    def home_advantage(self, resolved_format: str) -> float:
        """League or cup home advantage, chosen by how the fixture is played."""
        if resolved_format in LEAGUE_FORMATS:
            return self.home_advantage_elo
        return self.cup_home_advantage_elo


def _latest_ratings(frame: pd.DataFrame) -> dict[str, float]:
    """Latest Elo per club, keyed by every name the club is known under.

    Both key spaces are needed. `source_name` is football-data.co.uk's name and is
    null for the 187 clubs fetched only as cup entrants; `clubelo_name` covers all
    428 but is not what league fixtures arrive under. Keying on `source_name`
    alone silently drops every cup-only club to the default rating, which is how a
    fourth-tier side ends up modelled as an equal of the club hosting it.
    """
    ordered = frame.sort_values("valid_from")
    ratings: dict[str, float] = {}
    for column in ("clubelo_name", "source_name"):
        if column not in ordered.columns:
            continue
        latest = ordered.dropna(subset=[column]).groupby(column)["elo"].last()
        ratings.update({str(k): float(v) for k, v in latest.items()})
    return ratings


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
    home_rating: Rating | None = None
    away_rating: Rating | None = None

    @property
    def fully_rated(self) -> bool:
        """Both clubs carry a measured rating rather than a prior."""
        return bool(
            self.home_rating and self.home_rating.is_measured
            and self.away_rating and self.away_rating.is_measured
        )

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
            # Which tier of evidence backs each club. A prediction built on two
            # priors is a much weaker claim than one built on two measured
            # ratings, and the probabilities alone cannot express the difference.
            "ratings": {
                side: None if r is None else {"elo": r.elo, "source": r.source}
                for side, r in (("home", self.home_rating), ("away", self.away_rating))
            },
            "fully_rated": self.fully_rated,
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
        self,
        competition_id: str,
        home: str,
        away: str,
        *,
        neutral: bool = False,
        resolved_format: str = "round_robin",
        home_entry_stage: str | None = None,
        away_entry_stage: str | None = None,
    ) -> tuple[float, float]:
        """Convert a rating difference into a pair of goal rates.

        Home advantage is applied on the Elo scale, only at a real venue, and at
        the rate measured for this kind of fixture — a cup tie gets the cup
        figure, not the league one. A neutral final removes it entirely, which is
        what makes a neutral-venue prediction differ from the same fixture at home.
        """
        base_home, base_away = self.artifacts.environment(competition_id)
        home_rating = self.artifacts.rate(
            home, competition_id=competition_id, entry_stage=home_entry_stage
        )
        away_rating = self.artifacts.rate(
            away, competition_id=competition_id, entry_stage=away_entry_stage
        )
        edge = home_rating.elo - away_rating.elo
        if not neutral:
            edge += self.artifacts.home_advantage(resolved_format)

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
        home_entry_stage: str | None = None,
        away_entry_stage: str | None = None,
    ) -> Prediction:
        """Predict one fixture, branching on its resolved format (Design §5.3).

        `home_entry_stage` / `away_entry_stage` are the rounds each club entered
        the competition, and are only consulted for a club with no measured
        rating. They are deliberately not defaulted from `stage`.
        """
        competition = taxonomy.get(competition_id)
        resolved = competition.resolve_format(stage=stage, season=season)
        at_neutral = (
            competition.is_neutral_venue(stage) if neutral is None else bool(neutral)
        )

        lambda_home, lambda_away = self.rates(
            competition_id, home, away,
            neutral=at_neutral, resolved_format=resolved,
            home_entry_stage=home_entry_stage, away_entry_stage=away_entry_stage,
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
                competition_id, away, home,
                neutral=False, resolved_format=resolved,
                home_entry_stage=away_entry_stage, away_entry_stage=home_entry_stage,
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
            home_rating=self.artifacts.rate(
                home, competition_id=competition_id, entry_stage=home_entry_stage
            ),
            away_rating=self.artifacts.rate(
                away, competition_id=competition_id, entry_stage=away_entry_stage
            ),
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
