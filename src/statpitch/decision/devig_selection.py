"""Empirical de-vig method selection per competition (FR-28, Phase 5 / Notebook 12).

Design §6.2's selection procedure: de-vig every historical closing price with each
method, then compare the resulting fair probabilities against realised outcomes by
log-loss and calibration error. The winner is persisted per competition.

Two rules this module exists to keep
====================================

**The holdout season is never touched.** Selection runs on the training window
only (`decision_config.benchmark.training_seasons()`). Choosing a de-vig method on
the holdout would quietly consume the one untouched season NFR-10 reserves for the
final Phase 8 report — and would do so before a single model had been trained.

**Fair probability comes from the consensus, never the maximum.** `odds_avg` is
the input here. De-vigging `Max*` produces fair probabilities that look like free
money and are not, because the maximum of N noisy prices sits above consensus by
construction (FR-16a).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from statpitch.decision import devig as dv

log = logging.getLogger(__name__)

#: 1X2 selections in the fixed order the scoring assumes.
SELECTIONS = ("home", "draw", "away")

#: football-data result letters, mapped to an index into SELECTIONS.
RESULT_INDEX = {"H": 0, "D": 1, "A": 2}

#: Probability bins for the calibration error.
CALIBRATION_BINS = 10


@dataclass(frozen=True, slots=True)
class MethodScore:
    competition_id: str
    method: dv.Method
    log_loss: float
    ece: float
    brier: float
    n_matches: int
    mean_margin: float

    def as_row(self) -> dict[str, object]:
        return {
            "competition_id": self.competition_id,
            "method": self.method,
            "log_loss": round(self.log_loss, 6),
            "ece": round(self.ece, 6),
            "brier": round(self.brier, 6),
            "n_matches": self.n_matches,
            "mean_margin": round(self.mean_margin, 5),
        }


def build_market_frame(
    odds: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    seasons: list[str] | None = None,
    odds_regime: str | None = None,
    price_column: str = "odds_avg",
    snapshot: str = "close",
) -> pd.DataFrame:
    """One row per match with the three 1X2 prices and the realised outcome."""
    if price_column in ("odds_max", "odds_panel_max"):
        raise ValueError(
            f"refusing to de-vig {price_column!r}: the maximum of N book prices sits "
            "above consensus by construction, so de-vigging it fabricates edge "
            "(FR-16a). Fair probability comes from the average."
        )

    wanted = odds[(odds["snapshot"] == snapshot) & (odds["market"] == "1x2")].copy()
    wanted = wanted[wanted[price_column].notna()]
    if seasons is not None:
        wanted = wanted[wanted["season"].isin(seasons)]
    if odds_regime is not None:
        wanted = wanted[wanted["odds_regime"] == odds_regime]
    if wanted.empty:
        return pd.DataFrame()

    wide = wanted.pivot_table(
        index=["match_id", "competition_id", "season"],
        columns="selection",
        values=price_column,
        aggfunc="first",
    ).reset_index()

    missing = [s for s in SELECTIONS if s not in wide.columns]
    if missing:
        raise ValueError(f"odds frame is missing 1X2 selections: {missing}")
    wide = wide.dropna(subset=list(SELECTIONS))

    results = matches[["match_id", "result"]].dropna(subset=["result"])
    frame = wide.merge(results, on="match_id", how="inner")
    frame = frame[frame["result"].isin(RESULT_INDEX)]

    # A price at or below 1.0 is a data error, and de-vigging rejects it outright.
    for selection in SELECTIONS:
        frame = frame[frame[selection] > 1.0]

    return frame.reset_index(drop=True)


def _expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int = CALIBRATION_BINS
) -> float:
    """Average gap between predicted probability and observed frequency.

    Computed over the flattened selection-level probabilities, which is what
    FR-16b's per-decile calibration curve is built from.
    """
    flat_p = probabilities.ravel()
    flat_y = outcomes.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(flat_p, edges[1:-1]), 0, bins - 1)

    total = 0.0
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        gap = abs(float(flat_p[mask].mean()) - float(flat_y[mask].mean()))
        total += gap * float(mask.sum()) / flat_p.size
    return total


def score_method(
    frame: pd.DataFrame, competition_id: str, method: dv.Method
) -> MethodScore | None:
    """Score one method on one competition's matches."""
    subset = frame[frame["competition_id"] == competition_id]
    if subset.empty:
        return None

    odds_matrix = subset[list(SELECTIONS)].to_numpy(dtype=float)
    probabilities = dv.devig_many(odds_matrix, method)
    overrounds = (1.0 / odds_matrix).sum(axis=1)

    outcomes = np.zeros_like(probabilities)
    rows = np.arange(len(subset))
    outcomes[rows, [RESULT_INDEX[r] for r in subset["result"]]] = 1.0

    clipped = np.clip(probabilities, 1e-12, 1.0)
    log_loss = float(-np.mean(np.sum(outcomes * np.log(clipped), axis=1)))
    brier = float(np.mean(np.sum((probabilities - outcomes) ** 2, axis=1)))
    ece = _expected_calibration_error(probabilities, outcomes)

    return MethodScore(
        competition_id=competition_id,
        method=method,
        log_loss=log_loss,
        ece=ece,
        brier=brier,
        n_matches=len(subset),
        mean_margin=float(np.mean(overrounds) - 1.0),
    )


def compare(frame: pd.DataFrame) -> pd.DataFrame:
    """Score every method on every competition present."""
    rows = []
    for competition_id in sorted(frame["competition_id"].unique()):
        for method in dv.METHODS:
            score = score_method(frame, competition_id, method)
            if score is not None:
                rows.append(score.as_row())
    return pd.DataFrame(rows)


def select(comparison: pd.DataFrame, *, criterion: str = "log_loss") -> dict[str, str]:
    """Lowest-scoring method per competition, ignoring whether the gap is real.

    Log-loss is the default criterion because it is a strictly proper scoring
    rule: it is minimised only by the true probabilities, so it cannot be gamed by
    a method that is well calibrated on average while being wrong per match.

    **This is a ranking, not a decision.** Use `select_significant` for anything
    that gets persisted — on real data the three methods finish within 0.05% of
    each other, and the "winner" here is usually noise.
    """
    if criterion not in ("log_loss", "ece", "brier"):
        raise ValueError(f"unknown selection criterion {criterion!r}")

    winners: dict[str, str] = {}
    for competition_id, group in comparison.groupby("competition_id"):
        best = group.sort_values(criterion).iloc[0]
        winners[str(competition_id)] = str(best["method"])
    return winners


@dataclass(frozen=True, slots=True)
class PairedTest:
    competition_id: str
    challenger: dv.Method
    baseline: dv.Method
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    n_matches: int

    @property
    def is_significant(self) -> bool:
        """Whether the interval excludes zero."""
        return (self.ci_low > 0) or (self.ci_high < 0)

    @property
    def challenger_wins(self) -> bool:
        return self.is_significant and self.mean_difference < 0


def paired_test(
    frame: pd.DataFrame,
    competition_id: str,
    challenger: dv.Method,
    baseline: dv.Method,
) -> PairedTest:
    """Per-match log-loss difference between two methods on the same fixtures.

    Paired rather than compared on aggregates, because both methods see identical
    matches: the pairing removes fixture difficulty entirely and leaves only the
    method difference, which is a far more sensitive test than two separate means.
    """
    from scipy import stats

    subset = frame[frame["competition_id"] == competition_id]
    if subset.empty:
        raise ValueError(f"no matches for {competition_id!r}")

    odds_matrix = subset[list(SELECTIONS)].to_numpy(dtype=float)
    outcomes = np.zeros((len(subset), len(SELECTIONS)))
    outcomes[np.arange(len(subset)), [RESULT_INDEX[r] for r in subset["result"]]] = 1.0

    def per_match_loss(method: dv.Method) -> np.ndarray:
        probabilities = np.clip(dv.devig_many(odds_matrix, method), 1e-12, 1.0)
        return -np.sum(outcomes * np.log(probabilities), axis=1)

    difference = per_match_loss(challenger) - per_match_loss(baseline)
    n = len(difference)
    standard_error = float(difference.std(ddof=1) / np.sqrt(n))
    _, p_value = stats.ttest_1samp(difference, 0.0)

    return PairedTest(
        competition_id=competition_id,
        challenger=challenger,
        baseline=baseline,
        mean_difference=float(difference.mean()),
        ci_low=float(difference.mean() - 1.96 * standard_error),
        ci_high=float(difference.mean() + 1.96 * standard_error),
        p_value=float(p_value),
        n_matches=n,
    )


def select_significant(
    frame: pd.DataFrame, *, default: dv.Method = "shin"
) -> tuple[dict[str, str], list[PairedTest]]:
    """Select a method per competition, but only where the gap is real.

    A competition keeps `default` unless some method beats it by a margin whose
    95% interval excludes zero. This exists because ranking alone would persist
    noise: measured on the Big-5 training window the three methods finish within
    0.05% of each other on log-loss, every interval spans zero, and the nominal
    "winner" flips between competitions with no pattern. Writing those into the
    config would make an unmeasured choice look like a measured one, and would
    make a later reader believe Bundesliga genuinely warranted proportional
    de-vigging.
    """
    winners: dict[str, str] = {}
    tests: list[PairedTest] = []

    for competition_id in sorted(frame["competition_id"].unique()):
        chosen = default
        best_difference = 0.0
        for challenger in dv.METHODS:
            if challenger == default:
                continue
            test = paired_test(frame, competition_id, challenger, default)
            tests.append(test)
            if test.challenger_wins and test.mean_difference < best_difference:
                chosen = challenger
                best_difference = test.mean_difference
        winners[competition_id] = chosen

    return winners, tests


def summarise(comparison: pd.DataFrame, criterion: str = "log_loss") -> str:
    """Human-readable table, with the margin over the runner-up made explicit."""
    lines = []
    for competition_id, group in comparison.groupby("competition_id"):
        ordered = group.sort_values(criterion).reset_index(drop=True)
        best, second = ordered.iloc[0], ordered.iloc[1]
        gap = float(second[criterion]) - float(best[criterion])
        lines.append(
            f"{competition_id:16} winner={best['method']:12} "
            f"{criterion}={best[criterion]:.5f} "
            f"(next: {second['method']}, +{gap:.5f})  n={int(best['n_matches'])}"
        )
    return "\n".join(lines)
