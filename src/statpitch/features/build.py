"""Feature construction for `model_pure` (Design §4, Phase 2).

Every feature here is computed in a **single chronological pass** over the merged
match log, with per-club state carried forward. That is a deliberate structural
choice rather than a stylistic one.

Why a chronological pass instead of groupby/shift
=================================================

NFR-10 forbids any feature computed from post-match information. The usual pandas
idiom — group by club, roll, then `shift(1)` — gets this right only if every
window, every sort and every merge is correct, and a single missed shift produces
a feature that quietly contains the result it is meant to predict. The failure is
invisible: the model simply looks brilliant, and stays brilliant right up until it
meets the closing line.

Walking forward in time and updating state *after* emitting each row makes the
guarantee structural. At the moment a match's features are written, the loop has
only ever seen earlier matches. There is no window to misalign.

The competition-crossing rule
=============================

Form, congestion and rest are computed **per club across all competitions**, not
per competition (Design §4, FR-17). A club that played a UEFA tie on Wednesday is
tired on Saturday whichever table it appears in, and a club in poor league form
does not become fresh because the next match is a cup tie. This is what makes the
merged league + cup match log from Phase 1 worth having.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Rolling windows, in matches. Design §4 carries last-5 and last-10 from v1.
FORM_WINDOWS: tuple[int, ...] = (5, 10)

#: Congestion lookback (FR-17).
CONGESTION_DAYS = 14

#: Rest days are capped so a club returning from a summer break does not emit a
#: 90-day outlier that dominates a tree split.
MAX_REST_DAYS = 30

#: How many prior meetings the head-to-head features look back over.
H2H_WINDOW = 10

#: Points for a win/draw under the modern three-point system.
WIN_POINTS, DRAW_POINTS = 3.0, 1.0


@dataclass
class _ClubState:
    """Rolling history for one club, across every competition it plays in."""

    results: deque = field(default_factory=lambda: deque(maxlen=max(FORM_WINDOWS)))
    goals_for: deque = field(default_factory=lambda: deque(maxlen=max(FORM_WINDOWS)))
    goals_against: deque = field(default_factory=lambda: deque(maxlen=max(FORM_WINDOWS)))
    #: xG is tracked separately from goals because it is only available for the
    #: Big 5 from 2014/15 — a club can have twenty matches of form history and
    #: none of xG, and conflating the two would silently shorten the xG window.
    xg_for: deque = field(default_factory=lambda: deque(maxlen=max(FORM_WINDOWS)))
    xg_against: deque = field(default_factory=lambda: deque(maxlen=max(FORM_WINDOWS)))
    dates: deque = field(default_factory=lambda: deque(maxlen=60))
    last_date: pd.Timestamp | None = None

    #: Venue-split history. Pooled form averages a club's home and away records
    #: together, which hides a real and large split: most clubs are markedly
    #: better at home, and the two records answer different questions about a
    #: fixture. The home side's HOME record and the away side's AWAY record are
    #: what actually bear on it.
    venue_results: dict[str, deque] = field(
        default_factory=lambda: {
            "home": deque(maxlen=max(FORM_WINDOWS)),
            "away": deque(maxlen=max(FORM_WINDOWS)),
        }
    )
    venue_scored: dict[str, deque] = field(
        default_factory=lambda: {
            "home": deque(maxlen=max(FORM_WINDOWS)),
            "away": deque(maxlen=max(FORM_WINDOWS)),
        }
    )
    venue_conceded: dict[str, deque] = field(
        default_factory=lambda: {
            "home": deque(maxlen=max(FORM_WINDOWS)),
            "away": deque(maxlen=max(FORM_WINDOWS)),
        }
    )
    #: Consecutive matches, at this venue, in which the club scored / kept a
    #: clean sheet. Streaks are what a pooled mean cannot express.
    scoring_streak: dict[str, int] = field(
        default_factory=lambda: {"home": 0, "away": 0}
    )
    clean_sheet_streak: dict[str, int] = field(
        default_factory=lambda: {"home": 0, "away": 0}
    )

    def form(self, window: int) -> float:
        """Points per game over the last `window` matches, NaN if never played.

        NaN rather than None throughout: None makes an all-missing column infer
        `object` dtype, so the same feature typed differently depending on which
        rows were in the frame. That surfaced as a spurious failure of the
        leakage test, which compares a truncated build against the full one.
        """
        if not self.results:
            return float("nan")
        recent = list(self.results)[-window:]
        return float(np.mean(recent))

    def mean_goals(self, series: deque, window: int) -> float:
        if not series:
            return float("nan")
        return float(np.mean(list(series)[-window:]))

    def matches_within(self, on: pd.Timestamp, days: int) -> int:
        cutoff = on - pd.Timedelta(days=days)
        return sum(1 for d in self.dates if cutoff <= d < on)

    def venue_form(self, venue: str, window: int) -> float:
        series = self.venue_results[venue]
        if not series:
            return float("nan")
        return float(np.mean(list(series)[-window:]))

    def venue_mean(self, series: deque, window: int) -> float:
        if not series:
            return float("nan")
        return float(np.mean(list(series)[-window:]))

    def goal_volatility(self, window: int) -> float | None:
        """Spread of goals scored, which a rolling mean cannot express.

        Two clubs averaging 1.5 goals are different propositions if one scores
        1,2,1,2,1 and the other 0,0,5,1,1 — most obviously for totals markets.
        """
        if len(self.goals_for) < 3:
            return float("nan")
        return float(np.std(list(self.goals_for)[-window:]))

    def rest_days(self, on: pd.Timestamp) -> float:
        if self.last_date is None:
            return float("nan")
        return float(min((on - self.last_date).days, MAX_REST_DAYS))

    def record(
        self,
        date: pd.Timestamp,
        points: float,
        scored: int,
        conceded: int,
        xg_for: float | None = None,
        xg_against: float | None = None,
        venue: str | None = None,
    ) -> None:
        self.results.append(points)
        self.goals_for.append(scored)
        self.goals_against.append(conceded)
        self.dates.append(date)
        self.last_date = date
        # Only appended when measured. Pushing a zero for an unmeasured match
        # would read as "created no chances" and drag every rolling xG down.
        if xg_for is not None and xg_against is not None:
            self.xg_for.append(xg_for)
            self.xg_against.append(xg_against)

        if venue is not None:
            self.venue_results[venue].append(points)
            self.venue_scored[venue].append(scored)
            self.venue_conceded[venue].append(conceded)
            self.scoring_streak[venue] = (
                self.scoring_streak[venue] + 1 if scored > 0 else 0
            )
            self.clean_sheet_streak[venue] = (
                self.clean_sheet_streak[venue] + 1 if conceded == 0 else 0
            )


def _points(scored: int, conceded: int) -> float:
    if scored > conceded:
        return WIN_POINTS
    return DRAW_POINTS if scored == conceded else 0.0


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def merge_match_log(
    league_matches: pd.DataFrame, cup_matches: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Combine league and cup matches into one chronological log.

    The merge is what makes cross-competition form and congestion possible; the
    columns are reduced to the intersection both sources can supply.
    """
    columns = [
        "match_id", "competition_id", "season", "date",
        "home_team", "away_team", "home_goals", "away_goals",
    ]
    frames = [league_matches[columns].copy()]

    if cup_matches is not None and not cup_matches.empty:
        cups = cup_matches.copy()
        # Cup rows carry the 90-minute score; extra time and shootouts are
        # separate columns and must not leak into a goals feature.
        frames.append(cups[columns])

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["date"].notna()]
    merged = merged[merged["home_goals"].notna() & merged["away_goals"].notna()]
    merged = merged.drop_duplicates(subset="match_id", keep="first")
    # Sort by date, then match_id so the order is deterministic within a day.
    return merged.sort_values(["date", "match_id"]).reset_index(drop=True)


def build_features(
    matches: pd.DataFrame,
    elo_lookup: dict[tuple[str, pd.Timestamp], float] | None = None,
    xg_lookup: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Emit one feature row per match, using only earlier matches.

    `elo_lookup` maps (club, date) to the club's rating strictly before that date;
    see `statpitch.data.club_elo.elo_as_of`.

    `xg_lookup` maps match_id to that match's (home_xg, away_xg). It is consumed
    only when updating state *after* a row is emitted, so a match's own xG never
    reaches its own features.
    """
    if matches.empty:
        return pd.DataFrame()

    ordered = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    clubs: dict[str, _ClubState] = defaultdict(_ClubState)
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))

    rows: list[dict] = []
    for m in ordered.itertuples():
        home, away, date = m.home_team, m.away_team, m.date
        home_state, away_state = clubs[home], clubs[away]

        row: dict[str, object] = {
            "match_id": m.match_id,
            "competition_id": m.competition_id,
            "season": m.season,
            "date": date,
            "home_team": home,
            "away_team": away,
        }

        for side, state in (("home", home_state), ("away", away_state)):
            for window in FORM_WINDOWS:
                row[f"{side}_form_{window}"] = state.form(window)
                row[f"{side}_goals_for_{window}"] = state.mean_goals(state.goals_for, window)
                row[f"{side}_goals_against_{window}"] = state.mean_goals(
                    state.goals_against, window
                )
            for window in FORM_WINDOWS:
                row[f"{side}_xg_for_{window}"] = state.mean_goals(state.xg_for, window)
                row[f"{side}_xg_against_{window}"] = state.mean_goals(
                    state.xg_against, window
                )
            # Positive means the club has scored more than its chances deserved,
            # which historically regresses — the signal rolling goals cannot see.
            row[f"{side}_xg_overperformance"] = (
                float("nan")
                if not state.xg_for or not state.goals_for
                else float(np.mean(list(state.goals_for)[-len(state.xg_for):]))
                - float(np.mean(list(state.xg_for)))
            )
            row[f"{side}_xg_matches"] = len(state.xg_for)
            # Venue-split: the home club's home record, the away club's away one.
            for window in FORM_WINDOWS:
                row[f"{side}_venue_form_{window}"] = state.venue_form(side, window)
                row[f"{side}_venue_scored_{window}"] = state.venue_mean(
                    state.venue_scored[side], window
                )
                row[f"{side}_venue_conceded_{window}"] = state.venue_mean(
                    state.venue_conceded[side], window
                )
            row[f"{side}_venue_matches"] = len(state.venue_results[side])
            row[f"{side}_scoring_streak"] = state.scoring_streak[side]
            row[f"{side}_clean_sheet_streak"] = state.clean_sheet_streak[side]
            row[f"{side}_goal_volatility"] = state.goal_volatility(max(FORM_WINDOWS))
            row[f"{side}_rest_days"] = state.rest_days(date)
            row[f"{side}_matches_14d"] = state.matches_within(date, CONGESTION_DAYS)
            row[f"{side}_matches_played"] = len(state.dates)

        # Head-to-head, from the home side's perspective, prior meetings only.
        history = h2h[_pair_key(home, away)]
        if history:
            points = [p for p, h in history if h == home]
            row["h2h_matches"] = len(history)
            row["h2h_home_ppg"] = float(np.mean(points)) if points else float("nan")
        else:
            row["h2h_matches"] = 0
            row["h2h_home_ppg"] = float("nan")

        if elo_lookup is not None:
            home_elo = elo_lookup.get((home, date))
            away_elo = elo_lookup.get((away, date))
            row["home_elo"] = home_elo
            row["away_elo"] = away_elo
            row["home_elo"] = float("nan") if home_elo is None else home_elo
            row["away_elo"] = float("nan") if away_elo is None else away_elo
            row["elo_diff"] = (
                float("nan")
                if home_elo is None or away_elo is None
                else home_elo - away_elo
            )

        # Derived differentials, which trees split on far more readily than the
        # raw pair.
        for window in FORM_WINDOWS:
            h, a = row[f"home_form_{window}"], row[f"away_form_{window}"]
            row[f"form_diff_{window}"] = h - a
        for window in FORM_WINDOWS:
            h, a = row[f"home_xg_for_{window}"], row[f"away_xg_for_{window}"]
            row[f"xg_diff_{window}"] = h - a
        h_rest, a_rest = row["home_rest_days"], row["away_rest_days"]
        row["rest_diff"] = h_rest - a_rest
        row["congestion_diff"] = row["home_matches_14d"] - row["away_matches_14d"]
        for window in FORM_WINDOWS:
            h = row[f"home_venue_form_{window}"]
            a = row[f"away_venue_form_{window}"]
            row[f"venue_form_diff_{window}"] = h - a
        row["scoring_streak_diff"] = (
            row["home_scoring_streak"] - row["away_scoring_streak"]
        )

        rows.append(row)

        # --- state update happens AFTER the row is emitted -------------------
        # This ordering is the leakage guarantee. Moving it above the append
        # would let each match contribute to its own features.
        scored, conceded = int(m.home_goals), int(m.away_goals)
        home_points, away_points = _points(scored, conceded), _points(conceded, scored)
        home_xg, away_xg = (xg_lookup or {}).get(m.match_id, (None, None))
        home_state.record(date, home_points, scored, conceded, home_xg, away_xg, "home")
        away_state.record(date, away_points, conceded, scored, away_xg, home_xg, "away")
        history.append((home_points, home))

    frame = pd.DataFrame(rows)
    log.info("features: %d rows, %d columns", len(frame), len(frame.columns))
    return frame


def attach_outcomes(features: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Join the target columns on after features are built.

    Deliberately a separate step: outcomes never exist in the same frame as the
    feature construction, so they cannot be reached by it.
    """
    targets = matches[["match_id", "home_goals", "away_goals"]].copy()
    targets["result"] = np.where(
        targets["home_goals"] > targets["away_goals"], "H",
        np.where(targets["home_goals"] == targets["away_goals"], "D", "A"),
    )
    targets["total_goals"] = targets["home_goals"] + targets["away_goals"]
    return features.merge(targets, on="match_id", how="left")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model input columns — everything that is neither an identifier nor a target."""
    excluded = {
        "match_id", "competition_id", "season", "date", "home_team", "away_team",
        "home_goals", "away_goals", "result", "total_goals",
    }
    return [c for c in frame.columns if c not in excluded]


def drop_burn_in(frame: pd.DataFrame, min_matches: int = 5) -> pd.DataFrame:
    """Drop rows where either club has too little history for form to mean anything.

    The first matches of a club's presence in the log carry null or one-match form.
    Keeping them trains the model on noise labelled as signal.
    """
    return frame[
        (frame["home_matches_played"] >= min_matches)
        & (frame["away_matches_played"] >= min_matches)
    ].reset_index(drop=True)
