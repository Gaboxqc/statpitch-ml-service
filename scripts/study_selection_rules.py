"""Measure whether any tradeable sharp reference reproduces the CLV finding.

    python scripts/study_selection_rules.py

Writes `data/selection_rule_study.json`, which is the evidence behind whatever
`decision_config.selection_rule` says. Reproducible from committed artifacts
alone: `closing_odds.parquet` now carries `odds_bfe`, so this needs no raw
downloads and no network.

The question
============

MODEL_CARD §5's +0.51% CLV was measured on **Pinnacle**-referenced selections.
Phase A found Pinnacle absent from the live fixture feed. A rule that can only be
measured backwards is not a strategy, so this asks the obvious follow-up: does
any reference the feed *does* carry behave the same way?

Both odds regimes are reported and never pooled. Pinnacle was dropped from the
published Max/Avg aggregates on 2025-07-23, which changes what "the best quote"
means, and the live feed operates in the post-break regime.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import pandas as pd

from statpitch import decision_config, paths
from statpitch.decision import selection_study as study

log = logging.getLogger("selection_study")

OUTPUT_NAME = "selection_rule_study.json"


def _render(results, title: str) -> None:
    log.info("")
    log.info("%s", title)
    log.info(
        "  %-14s %-9s %6s | %6s %8s %9s %9s %8s",
        "reference", "in feed", "thr", "n", "matches", "CLV", "clustered t", "pos",
    )
    for r in results:
        log.info(
            "  %-14s %-9s %6.0f%% | %6d %8d %8.2f%% %+9.2f %7.1f%%",
            r.reference.replace("odds_", ""),
            "YES" if r.in_live_feed else "no",
            r.threshold * 100, r.n_selections, r.n_matches,
            r.mean_clv * 100, r.clustered_t, r.positive_rate * 100,
        )


#: A competition needs all three to be treated as evidenced, matching the bar
#: Requirements line 250 sets for the rule as a whole.
MIN_SEASONS = 2
MIN_T = 2.0


def per_competition(
    odds: pd.DataFrame,
    *,
    reference: str,
    exclude_seasons: tuple[str, ...],
    regime: str = "pre_2025_07_23",
    threshold: float = 0.0,
) -> dict:
    """The same rule, measured separately in each competition.

    The pooled number is what MODEL_CARD §5 reports, and pooling is exactly what
    hides a competition where the rule does not work: five leagues at t>3 will
    carry one at t<0 to a comfortable aggregate. Adding three leagues at once
    made that concrete rather than theoretical, so the breakdown is now recorded
    as evidence alongside the pooled figure rather than derived ad hoc.

    A competition failing here has not lost its odds coverage — its closing
    prices exist and were used to produce these very numbers. What it lacks is a
    reason to believe the selection rule earns anything there.
    """
    frame = study.wide_frame(odds, regime=regime)
    if frame.empty:
        return {}
    frame = frame[~frame["season"].isin(exclude_seasons)]
    probabilities = study.devigged(frame, reference)
    if probabilities.empty:
        return {}
    frame = frame.merge(
        probabilities, left_on=["match_id", "selection"],
        right_index=True, how="left",
    )

    out: dict = {
        "reference": reference,
        "regime": regime,
        "threshold": threshold,
        "excluded_seasons": list(exclude_seasons),
        "bar": {"min_seasons": MIN_SEASONS, "min_clustered_t": MIN_T,
                "mean_clv": "> 0"},
        "competitions": {},
    }
    for competition_id, group in frame.groupby("competition_id"):
        result = study.evaluate(group, reference, threshold)
        seasons = int(group["season"].nunique())
        if result is None:
            out["competitions"][str(competition_id)] = {
                "seasons": seasons, "clears": False,
                "why": "too few selections to measure",
            }
            continue
        clears = (
            seasons >= MIN_SEASONS
            and result.clustered_t > MIN_T
            and result.mean_clv > 0
        )
        out["competitions"][str(competition_id)] = {
            **result.as_dict(), "seasons": seasons, "clears": clears,
        }
    out["clears"] = sorted(
        c for c, r in out["competitions"].items() if r.get("clears")
    )
    out["does_not_clear"] = sorted(
        c for c, r in out["competitions"].items() if not r.get("clears")
    )
    return out


def _render_per_competition(block: dict) -> None:
    if not block:
        return
    log.info("")
    log.info(
        "per competition — %s, threshold %.0f%%, %s",
        block["reference"].replace("odds_", ""),
        block["threshold"] * 100, block["regime"],
    )
    log.info(
        "  %-16s %6s %8s %5s %9s %9s  %s",
        "competition", "n", "matches", "szns", "CLV", "clustered t", "verdict",
    )
    for competition_id, r in sorted(block["competitions"].items()):
        if "mean_clv" not in r:
            log.info("  %-16s %6s %8s %5d  %s", competition_id, "-", "-",
                     r["seasons"], r["why"])
            continue
        log.info(
            "  %-16s %6d %8d %5d %8.2f%% %+9.2f  %s",
            competition_id, r["n_selections"], r["n_matches"], r["seasons"],
            r["mean_clv"] * 100, r["clustered_t"],
            "clears" if r["clears"] else "DOES NOT CLEAR",
        )
    log.info("")
    log.info("  clears        : %s", ", ".join(block["clears"]) or "none")
    log.info("  does not clear: %s", ", ".join(block["does_not_clear"]) or "none")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = decision_config.config()
    holdout = (config.benchmark.holdout_season,)

    odds_path = paths.odds_file()
    if not odds_path.exists():
        log.error("no odds archive at %s", odds_path)
        return 1
    odds = pd.read_parquet(odds_path)

    payload: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": (
            "MODEL_CARD §5's +0.51% CLV was measured on Pinnacle-referenced "
            "selections. Pinnacle is not published in the live fixture feed. Does "
            "any reference the feed does carry reproduce it?"
        ),
        "scored_on": (
            "avg -> avg. Every rule selects using odds_max, so scoring it on how "
            "odds_max then moves would score a variable on itself; the consensus "
            "moving toward the bet is the only test the selection cannot game."
        ),
        "holdout_excluded": list(holdout),
        "regimes": {},
    }

    for regime, note in (
        ("pre_2025_07_23", "the window MODEL_CARD §5 measured; holdout excluded"),
        ("post_2025_07_23", "the regime the live feed operates in today"),
    ):
        results, meta = study.study(
            odds, regime=regime, exclude_seasons=holdout,
        )
        _render(results, f"{regime} — {note}")
        payload["regimes"][regime] = {
            **meta,
            "note": note,
            "results": [r.as_dict() for r in results],
        }

    tradeable = [
        r
        for regime in payload["regimes"].values()
        for r in regime["results"]
        if r["in_live_feed"] and r["is_significant"] and r["mean_clv"] > 0
    ]
    payload["tradeable_significant_positive"] = len(tradeable)

    payload["per_competition"] = per_competition(
        odds, reference=config.selection_rule.reference or "odds_pinnacle",
        exclude_seasons=holdout,
    )
    _render_per_competition(payload["per_competition"])

    destination = paths.data_root() / OUTPUT_NAME
    destination.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    log.info("")
    log.info("wrote %s", destination)
    log.info(
        "tradeable references with significant positive CLV: %d", len(tradeable)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
