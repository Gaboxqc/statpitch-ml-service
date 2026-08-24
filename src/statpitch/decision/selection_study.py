"""Which sharp reference, if any, can be traded forward (Plan §4 Phase C).

MODEL_CARD §5 records the project's only positive finding: Friday-to-close CLV
of +0.51% (t=3.47) on selections identified by reference to **Pinnacle**. Phase A
then established that Pinnacle is not in the live fixture feed — the feed carries
B365, BFD, BV, BW, PP, SKB and BFE. So the rule that was measured cannot be
traded, and this module exists to ask whether anything that *can* be traded
reproduces it.

The test that matters, and the one that does not
================================================

A selection rule is measured by whether the **consensus** moves toward it by
kickoff (`avg -> avg`), not by whether the best available price does
(`best -> best`).

That distinction is not pedantry; it is the difference between a signal and an
artifact. Every rule here selects on `odds_max`, so scoring it on how `odds_max`
subsequently moves is scoring a variable on itself. A rule that picks selections
where the best quote sits unusually far above the consensus will show large
positive `best -> best` CLV purely because that spread reverts — measured here at
+11.64% narrowing to +10.23% on selected rows, while the consensus moved 0.28%
*against* the bet. Regression to the mean, wearing the costume of edge.

`avg -> avg` cannot be gamed that way. It asks whether the whole market agreed,
after the fact, that the price was wrong.

Clustering
==========

Three selections on one match settle from a single scoreline, so they are not
independent observations. Standard errors are clustered on `match_id`. In
practice the correction is small here — a selective rule picks barely more than
one selection per match — but an uncorrected t on a rule that fired on all three
would be overstated by roughly sqrt(3), and that is not a correction anyone
should have to remember to apply by hand.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from statpitch.data import football_data_live as live
from statpitch.decision import devig

log = logging.getLogger(__name__)

ORDER = ("home", "draw", "away")

#: Reference price column -> the book prefix it comes from in the raw files.
#: `None` means it is an aggregate rather than a single book.
REFERENCE_BOOKS: dict[str, str | None] = {
    "odds_pinnacle": "PS",
    "odds_b365": "B365",
    "odds_bfe": "BFE",
    "odds_avg": None,
}

#: Aggregates that are published in the live fixture feed even though they are
#: not one book. `Avg` and `Max` are columns of `fixtures.csv`, verified in
#: Phase A against the live header.
LIVE_AGGREGATES = frozenset({"odds_avg", "odds_max"})


def in_live_feed(column: str) -> bool:
    """Whether a reference column can actually be read at bet time.

    Checked against `football_data_live.LIVE_BOOK_PREFIXES` rather than restated,
    so that a change to what the feed publishes cannot leave this out of date.
    """
    if column in LIVE_AGGREGATES:
        return True
    book = REFERENCE_BOOKS.get(column)
    return book is not None and book in live.LIVE_BOOK_PREFIXES


@dataclass(frozen=True, slots=True)
class CLVResult:
    """CLV for one selection rule, with a match-clustered standard error."""

    reference: str
    in_live_feed: bool
    threshold: float
    n_selections: int
    n_matches: int
    mean_clv: float
    clustered_t: float
    naive_t: float
    positive_rate: float

    @property
    def is_significant(self) -> bool:
        """Two-sided 5%, on the clustered statistic rather than the naive one."""
        return abs(self.clustered_t) >= 1.96

    def as_dict(self) -> dict:
        return {**asdict(self), "is_significant": self.is_significant}


def wide_frame(odds: pd.DataFrame, *, regime: str) -> pd.DataFrame:
    """One row per match x selection with preclose and close prices side by side.

    Restricted to 1X2 in the modern schema era: consensus closing columns do not
    exist before 2019/20, and CLV needs both ends.
    """
    frame = odds[
        (odds["market"] == "1x2")
        & (odds["odds_schema_era"] == "modern")
        & (odds["odds_regime"] == regime)
    ]
    if frame.empty:
        return pd.DataFrame()

    price_columns = [c for c in frame.columns if c.startswith("odds_")]
    pre = frame[frame["snapshot"] == "preclose"].set_index(["match_id", "selection"])
    close = frame[frame["snapshot"] == "close"].set_index(["match_id", "selection"])
    joined = pre[[*price_columns, "season", "competition_id"]].join(
        close[price_columns], rsuffix="_close", how="inner"
    )
    return joined.rename(
        columns={c: c + "_pre" for c in price_columns}
    ).reset_index()


def devigged(frame: pd.DataFrame, reference: str, method: str = "shin") -> pd.Series:
    """Fair probabilities for one reference, de-vigged per match.

    Keyed on the *reference* (`odds_bfe`) rather than the wide frame's column
    (`odds_bfe_pre`), and named `odds_bfe_q` to match what `evaluate` looks for.
    An earlier version named its output after the input column, so composing the
    two silently selected nothing unless the caller renamed in between — which
    `study` did and no one else would have.

    A match missing any leg of the triplet is dropped: a book can only be
    de-vigged as a complete set.
    """
    column = reference + "_pre"
    if column not in frame.columns:
        return pd.Series(dtype=float)
    grid = frame.pivot_table(
        index="match_id", columns="selection", values=column, aggfunc="first"
    )
    missing = [s for s in ORDER if s not in grid.columns]
    if missing:
        return pd.Series(dtype=float)
    grid = grid[list(ORDER)].dropna()
    if grid.empty:
        return pd.Series(dtype=float)
    probabilities = devig.devig_many(grid.to_numpy(), method)
    stacked = pd.DataFrame(
        probabilities, index=grid.index, columns=list(ORDER)
    ).stack()
    stacked.index.names = ["match_id", "selection"]
    return stacked.rename(reference + "_q")


def clustered_t(values: np.ndarray, groups: np.ndarray) -> float:
    """t for a mean, with the standard error clustered on `groups`."""
    if len(values) < 2:
        return float("nan")
    mean = float(values.mean())
    centred = pd.DataFrame({"v": values - mean, "g": groups})
    sums = centred.groupby("g")["v"].sum().to_numpy()
    n, n_groups = len(values), len(sums)
    if n_groups < 2:
        return float("nan")
    variance = (sums**2).sum() / (n**2) * (n_groups / (n_groups - 1))
    if variance <= 0:
        return float("nan")
    return mean / float(np.sqrt(variance))


def evaluate(
    frame: pd.DataFrame,
    reference: str,
    threshold: float,
    *,
    price: str = "odds_max_pre",
    taken: str = "odds_avg_pre",
    closed: str = "odds_avg_close",
    min_selections: int = 50,
) -> CLVResult | None:
    """CLV for "back it when the best quote beats `reference`'s fair value".

    Scored `avg -> avg` by default. See the module docstring for why scoring it
    on the column the rule selects with would measure mean reversion instead.
    """
    column = reference + "_q"
    if column not in frame.columns or price not in frame.columns:
        return None
    selected = frame[(frame[price] * frame[column] - 1.0) > threshold]
    selected = selected.dropna(subset=[taken, closed])
    if len(selected) < min_selections:
        return None

    clv = (selected[taken] / selected[closed] - 1.0).to_numpy(dtype=float)
    groups = selected["match_id"].to_numpy()
    naive = (
        float(clv.mean() / (clv.std(ddof=1) / np.sqrt(len(clv))))
        if clv.std(ddof=1) > 0 else float("nan")
    )
    return CLVResult(
        reference=reference,
        in_live_feed=in_live_feed(reference),
        threshold=threshold,
        n_selections=len(selected),
        n_matches=int(pd.unique(groups).size),
        mean_clv=float(clv.mean()),
        clustered_t=clustered_t(clv, groups),
        naive_t=naive,
        positive_rate=float((clv > 0).mean()),
    )


def baseline(
    frame: pd.DataFrame,
    *,
    taken: str = "odds_avg_pre",
    closed: str = "odds_avg_close",
) -> CLVResult | None:
    """CLV over the whole book with no selection rule at all.

    The control that decides whether a rule selected anything. If the unselected
    book already drifts in the same direction, every "rule" measured against it
    inherits that drift and none of them are evidence of skill. Pre-break the
    baseline is -0.09%; post-break it is -0.03%, so the positive numbers in that
    regime are not a tide lifting all boats.
    """
    usable = frame.dropna(subset=[taken, closed])
    if usable.empty:
        return None
    clv = (usable[taken] / usable[closed] - 1.0).to_numpy(dtype=float)
    groups = usable["match_id"].to_numpy()
    naive = (
        float(clv.mean() / (clv.std(ddof=1) / np.sqrt(len(clv))))
        if clv.std(ddof=1) > 0 else float("nan")
    )
    return CLVResult(
        reference="none (whole book)",
        in_live_feed=True,
        threshold=float("nan"),
        n_selections=len(usable),
        n_matches=int(pd.unique(groups).size),
        mean_clv=float(clv.mean()),
        clustered_t=clustered_t(clv, groups),
        naive_t=naive,
        positive_rate=float((clv > 0).mean()),
    )


def study(
    odds: pd.DataFrame,
    *,
    regime: str,
    exclude_seasons: tuple[str, ...] = (),
    references: tuple[str, ...] = ("odds_pinnacle", "odds_b365", "odds_bfe", "odds_avg"),
    thresholds: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03),
    method: str = "shin",
) -> tuple[list[CLVResult], dict]:
    """Every reference at every threshold, on one odds regime.

    `exclude_seasons` is how the NFR-10 holdout stays untouched. Regimes are never
    pooled: Pinnacle was dropped from the published Max/Avg aggregates on
    2025-07-23, which changes what "the best quote" even means.
    """
    frame = wide_frame(odds, regime=regime)
    if frame.empty:
        return [], {"regime": regime, "matches": 0, "note": "no rows for this regime"}
    if exclude_seasons:
        frame = frame[~frame["season"].isin(exclude_seasons)]

    for reference in references:
        probabilities = devigged(frame, reference, method)
        if probabilities.empty:
            continue
        frame = frame.merge(
            probabilities,
            left_on=["match_id", "selection"],
            right_index=True,
            how="left",
        )

    control = baseline(frame)
    results = [] if control is None else [control]
    results += [
        result
        for reference in references
        for threshold in thresholds
        if (result := evaluate(frame, reference, threshold)) is not None
    ]
    meta = {
        "regime": regime,
        "seasons": sorted(set(frame["season"])),
        "excluded_seasons": list(exclude_seasons),
        "matches": int(frame["match_id"].nunique()),
        "selections": int(len(frame)),
        "devig_method": method,
    }
    return results, meta
