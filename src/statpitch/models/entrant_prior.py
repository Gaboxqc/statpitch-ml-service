"""Strength prior for cup entrants Club Elo does not rate (FR-9).

Club Elo rates only the top two tiers (`club_elo.CLUB_ELO_TIER_LIMIT`), and
domestic cups admit clubs from far deeper. Those entrants need *some* strength
estimate or every tie involving them is unpredictable by construction.

The conditioning variable is the round at which a club **enters** the
competition, not the round a given match belongs to. Cups seed by division, so
entry round is the sharpest tier proxy the data contains: an FA Cup side entering
in round 1 is a National League or League Two club, while one entering in round 3
is a Premier League or Championship club. Using match round instead would be
badly wrong — a round-1 entrant that wins three ties is still a round-1-calibre
club when it reaches round 4.

How the fit works
=================

Each bucket (competition x entry round) gets one Elo-scale rating. Ratings are
estimated **jointly**, because the alternative does not work: early cup rounds are
overwhelmingly unrated-vs-unrated, so a bucket cannot be anchored on its own.
Matches against rated opponents pin the overall level, and unrated-vs-unrated
matches carry the buckets' positions relative to each other. Solving them
together uses both.

Home advantage is estimated **first and separately**, from matches where both
ratings are known, then held fixed. Fitting it jointly with the bucket ratings
fails for a structural reason, not a numerical one: domestic cups seed the
lower-tier club at home, so within a bucket the home side is systematically the
weaker club. A bucket carries only one rating and cannot express that, so the
optimiser charges the deficit to the venue term — which came out at -27 Elo, then
-3 Elo, against data that plainly shows a positive home effect. Where both
ratings are known the confound disappears and the venue term is the only thing
left to explain the result.

Measured that way, cup home advantage is **~25 Elo against ~54 in the leagues**
(19,763 league matches as a control, which also validates the estimator against a
well-known quantity). Applying a league home-advantage constant to cup fixtures
would over-favour the host by roughly 30 Elo, and it would do so exactly in the
lower-division-hosts-a-big-club ties this prior exists to handle.

Every bucket is reported with its sample count and a bootstrap confidence
interval — an estimate without a dispersion figure does not ship (NFR-10).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

log = logging.getLogger(__name__)

#: Elo scale constant. 400 points is one order of magnitude in expected score.
ELO_SCALE = 400.0

#: Starting guess for an unrated entrant, and the fallback when a bucket has too
#: little data to fit. Roughly a mid-tier-2 club, deliberately below the rated
#: population rather than at its mean.
DEFAULT_ENTRANT_ELO = 1300.0

#: Below this many matches a bucket is not fitted; it inherits the pooled
#: estimate instead, and says so.
MIN_BUCKET_MATCHES = 20

#: Rated-vs-rated matches needed before home advantage is estimated from data.
MIN_HOME_ADVANTAGE_MATCHES = 100

#: Fallback venue effect when there is too little rated-vs-rated play to measure
#: one. A conventional league figure, deliberately not tuned.
DEFAULT_HOME_ADVANTAGE = 60.0


def expected_score(rating: np.ndarray | float, opponent: np.ndarray | float) -> np.ndarray:
    """Standard Elo expectation for the rating difference."""
    return 1.0 / (1.0 + np.power(10.0, (np.asarray(opponent) - np.asarray(rating)) / ELO_SCALE))


def _fit_home_advantage(
    home_elo: np.ndarray, away_elo: np.ndarray, score: np.ndarray
) -> float:
    """Venue effect in Elo points, from matches where both ratings are known."""

    def nll(params: np.ndarray) -> float:
        expected = expected_score(home_elo + params[0], away_elo)
        expected = np.clip(expected, 1e-9, 1 - 1e-9)
        return float(-np.sum(score * np.log(expected) + (1 - score) * np.log(1 - expected)))

    return float(minimize(nll, np.array([DEFAULT_HOME_ADVANTAGE]), method="L-BFGS-B").x[0])


@dataclass(frozen=True, slots=True)
class BucketEstimate:
    competition_id: str
    entry_stage: str
    elo: float
    n_matches: int
    n_clubs: int
    ci_low: float
    ci_high: float
    fitted: bool

    @property
    def is_reliable(self) -> bool:
        return self.fitted and self.n_matches >= MIN_BUCKET_MATCHES


@dataclass(frozen=True, slots=True)
class EntrantPrior:
    buckets: dict[tuple[str, str], BucketEstimate]
    home_advantage: float
    pooled_elo: float
    n_matches_used: int
    diagnostics: dict[str, float] = field(default_factory=dict)

    def rating_for(self, competition_id: str, entry_stage: str) -> float:
        """Prior rating for an unrated club, by where it entered the competition."""
        hit = self.buckets.get((competition_id, entry_stage))
        if hit is not None and hit.is_reliable:
            return hit.elo
        return self.pooled_elo

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "competition_id": b.competition_id,
                "entry_stage": b.entry_stage,
                "elo": round(b.elo, 1),
                "ci_low": round(b.ci_low, 1),
                "ci_high": round(b.ci_high, 1),
                "n_matches": b.n_matches,
                "n_clubs": b.n_clubs,
                "reliable": b.is_reliable,
            }
            for b in self.buckets.values()
        ]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values(["competition_id", "elo"]).reset_index(drop=True)


# --- entry rounds -------------------------------------------------------------

def entry_stages(matches: pd.DataFrame) -> pd.DataFrame:
    """The stage at which each club first appears in a competition-season.

    Ordering is by date rather than by parsing round names, which keeps it robust
    to the label variants the sources actually use ("Round 1", "1. Runde",
    "1. Round") and to competitions that skip rounds entirely.
    """
    sides = []
    for side in ("home", "away"):
        part = matches[["competition_id", "season", "date", "stage", f"{side}_team"]].copy()
        part = part.rename(columns={f"{side}_team": "club"})
        sides.append(part)

    appearances = pd.concat(sides, ignore_index=True)
    appearances = appearances[appearances["date"].notna()]
    appearances = appearances.sort_values("date")

    first = appearances.groupby(["competition_id", "season", "club"], as_index=False).first()
    return first.rename(columns={"stage": "entry_stage"})[
        ["competition_id", "season", "club", "entry_stage"]
    ]


# --- fitting ------------------------------------------------------------------

def _build_design(
    matches: pd.DataFrame,
    rated_elo: dict[tuple[str, pd.Timestamp], float],
    entry_lookup: dict[tuple[str, str, str], str],
) -> pd.DataFrame:
    """One row per usable match, with each side's rating or bucket index."""
    rows = []
    for m in matches.itertuples():
        if pd.isna(m.date) or pd.isna(m.home_goals) or pd.isna(m.away_goals):
            continue

        record = {"competition_id": m.competition_id}
        for side in ("home", "away"):
            club = getattr(m, f"{side}_team")
            elo = rated_elo.get((club, m.date))
            if elo is not None:
                record[f"{side}_elo"] = elo
                record[f"{side}_bucket"] = None
            else:
                stage = entry_lookup.get((m.competition_id, m.season, club))
                if stage is None:
                    record = None
                    break
                record[f"{side}_elo"] = np.nan
                record[f"{side}_bucket"] = (m.competition_id, stage)
        if record is None:
            continue

        # Matches with both sides rated are KEPT, even though they say nothing
        # about the unrated population. They are what identifies home advantage:
        # both ratings are known, so the only free parameter explaining the result
        # is the venue term.
        #
        # Dropping them fitted home advantage at -27 Elo. Domestic cups seed the
        # lower-tier club at home, so weak-and-home is confounded with the venue
        # effect; without rated-vs-rated matches to pin it, the venue parameter
        # simply absorbs "unrated clubs are weak" and comes out negative.

        home_goals, away_goals = int(m.home_goals), int(m.away_goals)
        record["score"] = 1.0 if home_goals > away_goals else (
            0.5 if home_goals == away_goals else 0.0
        )
        record["neutral"] = bool(getattr(m, "neutral_venue", False))
        rows.append(record)

    return pd.DataFrame(rows)


def fit(
    matches: pd.DataFrame,
    elo_table: pd.DataFrame,
    club_mapping: dict[str, str],
    *,
    bootstrap: int = 200,
    seed: int = 0,
) -> EntrantPrior:
    """Fit entry-round priors from cup results.

    `club_mapping` maps a source club name to its Club Elo name; clubs absent from
    it are the unrated population this prior exists for.
    """
    entries = entry_stages(matches)
    entry_lookup = {
        (r.competition_id, r.season, r.club): r.entry_stage for r in entries.itertuples()
    }

    # Rated clubs: look their Elo up as of the day before each match (never after,
    # or the result leaks into its own predictor).
    rated_elo: dict[tuple[str, pd.Timestamp], float] = {}
    elo_by_club: dict[str, pd.DataFrame] = {
        name: group.sort_values("valid_from")
        for name, group in elo_table.groupby("clubelo_name")
    }
    needed = set()
    for m in matches.itertuples():
        if pd.isna(m.date):
            continue
        for side in ("home", "away"):
            club = getattr(m, f"{side}_team")
            if club in club_mapping:
                needed.add((club, m.date))

    for club, date in needed:
        history = elo_by_club.get(club_mapping[club])
        if history is None:
            continue
        prior_rows = history[history["valid_from"] < date]
        if not prior_rows.empty:
            rated_elo[(club, date)] = float(prior_rows.iloc[-1]["elo"])

    design = _build_design(matches, rated_elo, entry_lookup)
    if design.empty:
        raise ValueError("no usable matches: every fixture had two rated or two unknown sides")

    # Buckets seen too rarely get folded into a shared pooled parameter rather
    # than each receiving their own. Fitted individually, a one-match bucket
    # produces a rating that merely reproduces that single result — the first run
    # returned -147 and 677 Elo for buckets of one match. Those are not estimates.
    appearances: dict[tuple[str, str], int] = {}
    for side in ("home_bucket", "away_bucket"):
        for bucket in design[side]:
            if bucket is not None:
                appearances[bucket] = appearances.get(bucket, 0) + 1

    POOLED = ("__pooled__", "__pooled__")
    thin = {b for b, n in appearances.items() if n < MIN_BUCKET_MATCHES}
    if thin:
        log.info(
            "entrant prior: pooling %d bucket(s) below %d matches: %s",
            len(thin), MIN_BUCKET_MATCHES, sorted(thin),
        )

    def bucket_of(raw: tuple[str, str] | None) -> tuple[str, str] | None:
        if raw is None:
            return None
        return POOLED if raw in thin else raw

    design = design.assign(
        home_bucket=[bucket_of(b) for b in design["home_bucket"]],
        away_bucket=[bucket_of(b) for b in design["away_bucket"]],
    )

    bucket_keys = sorted(
        {b for b in pd.concat([design["home_bucket"], design["away_bucket"]]) if b is not None}
    )
    if not bucket_keys:
        raise ValueError(
            "no unrated entrants found: every match had two rated sides, so there is "
            "no entrant population to estimate a prior for"
        )
    index = {key: i for i, key in enumerate(bucket_keys)}

    home_idx = np.array([index[b] if b is not None else -1 for b in design["home_bucket"]])
    away_idx = np.array([index[b] if b is not None else -1 for b in design["away_bucket"]])
    home_elo = design["home_elo"].to_numpy(dtype=float)
    away_elo = design["away_elo"].to_numpy(dtype=float)
    score = design["score"].to_numpy(dtype=float)
    neutral = design["neutral"].to_numpy(dtype=bool)

    # --- stage 1: home advantage, from matches where both ratings are known ---
    #
    # Fitting it jointly with the bucket ratings does not work, and the reason is
    # structural rather than numerical. Domestic cups seed the lower-tier club at
    # home (always in the DFB-Pokal's early rounds, by rank in the Copa del Rey and
    # Coupe de France). Within a bucket the home side is therefore systematically
    # the weaker club, but a bucket has only one rating and cannot express that —
    # so the optimiser charges the deficit to the venue term and home advantage
    # comes out at roughly zero or negative.
    #
    # Where both ratings are known the confound disappears: strength is measured,
    # so the venue term is the only thing left to explain the result. Home
    # advantage is estimated there and held fixed while the buckets are fitted.
    both_rated = (home_idx < 0) & (away_idx < 0) & ~neutral
    if both_rated.sum() >= MIN_HOME_ADVANTAGE_MATCHES:
        home_advantage = _fit_home_advantage(
            home_elo[both_rated], away_elo[both_rated], score[both_rated]
        )
        home_advantage_matches = int(both_rated.sum())
    else:
        log.warning(
            "entrant prior: only %d rated-vs-rated matches, too few to identify home "
            "advantage; falling back to %.0f Elo",
            int(both_rated.sum()), DEFAULT_HOME_ADVANTAGE,
        )
        home_advantage = DEFAULT_HOME_ADVANTAGE
        home_advantage_matches = int(both_rated.sum())

    def negative_log_likelihood(params: np.ndarray) -> float:
        ratings = params
        advantage = home_advantage
        h = np.where(home_idx >= 0, ratings[np.maximum(home_idx, 0)], home_elo)
        a = np.where(away_idx >= 0, ratings[np.maximum(away_idx, 0)], away_elo)
        expected = expected_score(h + np.where(neutral, 0.0, advantage), a)
        expected = np.clip(expected, 1e-9, 1 - 1e-9)
        # Draws enter as half a win and half a loss, the usual Elo convention.
        return float(-np.sum(score * np.log(expected) + (1 - score) * np.log(1 - expected)))

    start = np.full(len(bucket_keys), DEFAULT_ENTRANT_ELO)
    result = minimize(negative_log_likelihood, start, method="L-BFGS-B")
    ratings = result.x

    # Bootstrap over matches for the per-bucket interval.
    rng = np.random.default_rng(seed)
    samples = np.empty((bootstrap, len(bucket_keys)))
    n = len(design)
    for b in range(bootstrap):
        pick = rng.integers(0, n, n)

        def nll_resampled(params: np.ndarray, pick: np.ndarray = pick) -> float:
            r = params
            h = np.where(home_idx[pick] >= 0, r[np.maximum(home_idx[pick], 0)], home_elo[pick])
            a = np.where(away_idx[pick] >= 0, r[np.maximum(away_idx[pick], 0)], away_elo[pick])
            expected = expected_score(h + np.where(neutral[pick], 0.0, home_advantage), a)
            expected = np.clip(expected, 1e-9, 1 - 1e-9)
            s = score[pick]
            return float(-np.sum(s * np.log(expected) + (1 - s) * np.log(1 - expected)))

        samples[b] = minimize(nll_resampled, result.x, method="L-BFGS-B").x

    counts = _bucket_counts(design, matches, entry_lookup, club_mapping)

    buckets: dict[tuple[str, str], BucketEstimate] = {}
    pooled_from_fit = None
    for key, i in index.items():
        competition_id, stage = key
        if key == POOLED:
            pooled_from_fit = float(ratings[i])
            continue
        n_matches, n_clubs = counts.get(key, (0, 0))
        buckets[key] = BucketEstimate(
            competition_id=competition_id,
            entry_stage=stage,
            elo=float(ratings[i]),
            n_matches=n_matches,
            n_clubs=n_clubs,
            ci_low=float(np.percentile(samples[:, i], 2.5)),
            ci_high=float(np.percentile(samples[:, i], 97.5)),
            fitted=bool(result.success),
        )

    # Prefer the pooled parameter the fit actually estimated over an average of
    # the per-bucket results: it is estimated from the thin buckets themselves,
    # which is what an unseen or sparse bucket most resembles.
    reliable = [b for b in buckets.values() if b.is_reliable]
    if pooled_from_fit is not None:
        pooled = pooled_from_fit
    elif reliable:
        pooled = float(
            np.average([b.elo for b in reliable], weights=[b.n_matches for b in reliable])
        )
    else:
        pooled = DEFAULT_ENTRANT_ELO

    return EntrantPrior(
        buckets=buckets,
        home_advantage=home_advantage,
        pooled_elo=pooled,
        n_matches_used=len(design),
        diagnostics={
            "converged": float(result.success),
            "log_likelihood": -float(result.fun),
            "n_buckets": float(len(buckets)),
            "home_advantage_matches": float(home_advantage_matches),
        },
    )


def _bucket_counts(
    design: pd.DataFrame,
    matches: pd.DataFrame,
    entry_lookup: dict[tuple[str, str, str], str],
    club_mapping: dict[str, str],
) -> dict[tuple[str, str], tuple[int, int]]:
    counts: dict[tuple[str, str], int] = {}
    for side in ("home_bucket", "away_bucket"):
        for bucket in design[side]:
            if bucket is not None:
                counts[bucket] = counts.get(bucket, 0) + 1

    clubs: dict[tuple[str, str], set[str]] = {}
    for m in matches.itertuples():
        for side in ("home", "away"):
            club = getattr(m, f"{side}_team")
            if club in club_mapping:
                continue
            stage = entry_lookup.get((m.competition_id, m.season, club))
            if stage is not None:
                clubs.setdefault((m.competition_id, stage), set()).add(club)

    return {k: (counts.get(k, 0), len(v)) for k, v in clubs.items()}
