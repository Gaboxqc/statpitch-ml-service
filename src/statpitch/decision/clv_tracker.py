"""Bet ledger and closing-line value (FR-26, FR-29, Design §6.6).

CLV is this project's headline metric, and not by preference — by measurement.
Over the same selections, ROI on 38,763 settled bets could not resolve whether an
edge existed (+1.29%, t=1.19), while CLV on 4,929 priced bets resolved it clearly
(+0.51%, t=3.47). CLV strips the outcome out entirely and asks only whether the
price moved the right way, so it converges on roughly an eighth of the sample.

Like must be compared with like
===============================

The single most dangerous mistake available here, and one this project made
before catching it: comparing a price taken at the BEST available quote against a
CLOSING CONSENSUS. That is a max-versus-mean spread, not line movement, and it
manufactured an apparent +5.4% CLV on *every selection in the book* — including
ones chosen at random. The tell was that the baseline showed the same figure as
the selected bets, and that CLV measured in probability points sat at 0.0000
while the percentage read +5%.

So a ledger entry records which price source it took, and settlement refuses to
compare against a different one. The invariant is enforced rather than described.

The reporting rule
==================

Design §6.6 is explicit and `CLVReport.verdict` implements it: positive ROI with
negative CLV is reported as the ABSENCE of demonstrated edge, not as success. A
few hundred bets of positive ROI is routinely produced by luck; CLV is what says
whether the prices were right.

Everything here is labelled "Friday-to-close CLV". football-data.co.uk's base
snapshot is Friday afternoon rather than a true opening line, so measured movement
understates what an early bettor could capture. Valid signal, accurate name.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: Mandatory label (Design §6.6). The Friday snapshot is not an opening line.
CLV_LABEL = "Friday-to-close CLV"


class PriceSource(StrEnum):
    """Which quote a price came from. Mixing these invalidates CLV."""

    BEST = "best"            # maximum across books
    CONSENSUS = "consensus"  # average across books
    SHARP = "sharp"          # a single sharp reference book


class Result(StrEnum):
    WON = "won"
    LOST = "lost"
    PUSH = "push"
    HALF_WON = "half_won"
    HALF_LOST = "half_lost"
    VOID = "void"

    @property
    def stake_multiplier(self) -> float:
        return {
            Result.WON: 1.0, Result.HALF_WON: 0.5, Result.PUSH: 0.0,
            Result.HALF_LOST: -0.5, Result.LOST: -1.0, Result.VOID: 0.0,
        }[self]


class LedgerError(RuntimeError):
    pass


@dataclass
class LedgerEntry:
    """One graded recommendation, appended at flag time (Design §6.6)."""

    ts_flagged: str
    fixture_id: str
    competition_id: str
    selection: str
    market_family: str
    price_source: str
    odds_taken: float
    p_model: float
    q_fair: float
    edge_prob: float
    grade: str
    stake_fraction: float
    kelly_lambda: float
    w: float
    config_version: str
    p_std: float | None = None

    # --- filled by settlement -------------------------------------------
    odds_closing: float | None = None
    closing_price_source: str | None = None
    result: str | None = None
    clv_pct: float | None = None
    clv_prob: float | None = None
    profit: float | None = None

    @property
    def is_settled(self) -> bool:
        return self.result is not None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> LedgerEntry:
        return cls(**json.loads(line))


def flag(
    *,
    fixture_id: str,
    competition_id: str,
    selection: str,
    market_family: str,
    odds_taken: float,
    price_source: PriceSource,
    p_model: float,
    q_fair: float,
    grade: str,
    stake_fraction: float,
    kelly_lambda: float,
    w: float,
    config_version: str,
    p_std: float | None = None,
    now: datetime | None = None,
) -> LedgerEntry:
    """Build a ledger entry at flag time."""
    if odds_taken <= 1.0:
        raise LedgerError(f"odds_taken must exceed 1.0, got {odds_taken}")
    return LedgerEntry(
        ts_flagged=(now or datetime.now(UTC)).isoformat(),
        fixture_id=fixture_id,
        competition_id=competition_id,
        selection=selection,
        market_family=market_family,
        price_source=str(price_source),
        odds_taken=float(odds_taken),
        p_model=float(p_model),
        q_fair=float(q_fair),
        edge_prob=float(p_model - q_fair),
        grade=grade,
        stake_fraction=float(stake_fraction),
        kelly_lambda=float(kelly_lambda),
        w=float(w),
        config_version=config_version,
        p_std=p_std,
    )


def settle(
    entry: LedgerEntry,
    *,
    odds_closing: float,
    closing_price_source: PriceSource,
    result: Result,
    q_fair_closing: float | None = None,
) -> LedgerEntry:
    """Fill closing price, result and CLV (FR-26).

    Refuses to settle against a different price source than the one taken. A price
    taken at the best quote and closed against the consensus measures the spread
    between those two sources, not the movement of the line — and produces a large
    positive CLV on every bet regardless of selection.
    """
    if str(closing_price_source) != entry.price_source:
        raise LedgerError(
            f"{entry.selection}: took a {entry.price_source} price and tried to settle "
            f"against a {closing_price_source} one. CLV must compare like with like — "
            "mixing sources measures a max-versus-mean spread rather than line movement"
        )
    if odds_closing <= 1.0:
        raise LedgerError(f"odds_closing must exceed 1.0, got {odds_closing}")

    entry.odds_closing = float(odds_closing)
    entry.closing_price_source = str(closing_price_source)
    entry.result = str(result)
    entry.clv_pct = entry.odds_taken / float(odds_closing) - 1.0
    if q_fair_closing is not None:
        # CLV in probability points (FR-26). Reported alongside the percentage
        # because the two disagreeing is the signature of a unit mismatch.
        entry.clv_prob = float(q_fair_closing) - entry.q_fair
    entry.profit = entry.stake_fraction * _profit_multiplier(result, entry.odds_taken)
    return entry


def _profit_multiplier(result: Result, odds: float) -> float:
    multiplier = result.stake_multiplier
    return multiplier * (odds - 1.0) if multiplier > 0 else multiplier


@dataclass
class BetLedger:
    """Append-only JSONL ledger (FR-29).

    Append-only on purpose: the value of a track record is that earlier entries
    cannot be revised once results are known.
    """

    path: Path
    _entries: list[LedgerEntry] = field(default_factory=list)
    #: Refuse writes. If None, defaults from STATPITCH_READ_ONLY so a deployment
    #: can set it once in its environment rather than at every call site.
    read_only: bool | None = field(default=None)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.read_only is None:
            self.read_only = os.environ.get("STATPITCH_READ_ONLY", "") not in ("", "0")
        if not self.read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._entries = [
                LedgerEntry.from_json(line)
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def _require_writable(self, action: str) -> None:
        """Guard the one failure mode a read-only host makes silent.

        On a free-tier host the filesystem is ephemeral: a write succeeds, the
        instance spins down, and the entry is gone. The ledger's whole value is
        that it is a record, so a lost append is worse than a refused one. The
        ledger is owned by the scheduled job that commits it to the repository,
        and the deployed API only ever reads it.
        """
        if self.read_only:
            raise LedgerError(
                f"refusing to {action}: this ledger is read-only "
                "(STATPITCH_READ_ONLY). The deployed API serves the ledger the "
                "scheduled job commits; writing here would be discarded when the "
                "instance restarts, leaving a gap nothing can detect."
            )

    def append(self, entry: LedgerEntry) -> None:
        self._require_writable("append to the ledger")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(entry.to_json() + "\n")
        self._entries.append(entry)

    def rewrite(self) -> None:
        """Persist settlements. Rewrites the file from the in-memory entries."""
        self._require_writable("rewrite the ledger")
        self.path.write_text(
            "".join(e.to_json() + "\n" for e in self._entries), encoding="utf-8"
        )

    @property
    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def settled(self) -> list[LedgerEntry]:
        return [e for e in self._entries if e.is_settled]

    def pending(self) -> list[LedgerEntry]:
        return [e for e in self._entries if not e.is_settled]

    def __len__(self) -> int:
        return len(self._entries)


# --- reporting ----------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CLVReport:
    label: str
    n: int
    mean_clv: float
    clv_se: float
    positive_rate: float
    mean_roi: float
    roi_se: float

    @property
    def clv_t(self) -> float:
        return self.mean_clv / self.clv_se if self.clv_se > 0 else 0.0

    @property
    def roi_t(self) -> float:
        return self.mean_roi / self.roi_se if self.roi_se > 0 else 0.0

    @property
    def clv_is_significant(self) -> bool:
        return abs(self.clv_t) >= 2.0

    def verdict(self) -> str:
        """Design §6.6's reporting rule, applied rather than described."""
        if self.n == 0:
            return "no settled bets"
        if self.mean_clv < 0 and self.mean_roi > 0:
            return (
                f"ROI is positive ({self.mean_roi:+.2%}) but {self.label} is negative "
                f"({self.mean_clv:+.2%}). Reported as ABSENCE of demonstrated edge: "
                "over a few hundred bets a positive ROI is routinely produced by "
                "luck, and the prices say the selections were wrong."
            )
        if self.clv_is_significant and self.mean_clv > 0:
            return (
                f"{self.label} {self.mean_clv:+.2%} (t={self.clv_t:+.2f}, "
                f"{self.positive_rate:.1%} positive) over {self.n} bets — "
                "evidence of genuine edge"
            )
        if self.mean_clv > 0:
            return (
                f"{self.label} {self.mean_clv:+.2%} but t={self.clv_t:+.2f} — "
                "directionally positive, not yet distinguishable from zero"
            )
        return f"{self.label} {self.mean_clv:+.2%} (t={self.clv_t:+.2f}) — no edge shown"


def report(entries: list[LedgerEntry], label: str = CLV_LABEL) -> CLVReport:
    """Aggregate CLV and ROI over settled bets."""
    settled = [e for e in entries if e.is_settled and e.clv_pct is not None]
    if not settled:
        return CLVReport(label, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    clv = np.array([e.clv_pct for e in settled], dtype=float)
    staked = np.array([e.stake_fraction for e in settled], dtype=float)
    profit = np.array([e.profit or 0.0 for e in settled], dtype=float)

    # ROI per unit staked; bets flagged but not staked still carry CLV.
    active = staked > 0
    roi = profit[active] / staked[active] if active.any() else np.zeros(1)

    return CLVReport(
        label=label,
        n=len(settled),
        mean_clv=float(clv.mean()),
        clv_se=float(clv.std(ddof=1) / np.sqrt(len(clv))) if len(clv) > 1 else 0.0,
        positive_rate=float((clv > 0).mean()),
        mean_roi=float(roi.mean()),
        roi_se=float(roi.std(ddof=1) / np.sqrt(len(roi))) if len(roi) > 1 else 0.0,
    )


def report_by(
    entries: list[LedgerEntry], attribute: str, label: str = CLV_LABEL
) -> dict[str, CLVReport]:
    """CLV broken down by competition, market family or grade (FR-26)."""
    groups: dict[str, list[LedgerEntry]] = {}
    for entry in entries:
        groups.setdefault(str(getattr(entry, attribute)), []).append(entry)
    return {key: report(group, label) for key, group in sorted(groups.items())}


def summarise(entries: list[LedgerEntry], label: str = CLV_LABEL) -> str:
    overall = report(entries, label)
    lines = [
        f"{label}: {overall.mean_clv:+.4f} (SE {overall.clv_se:.4f}, "
        f"t={overall.clv_t:+.2f}, {overall.positive_rate:.1%} positive, n={overall.n})",
        f"ROI: {overall.mean_roi:+.4f} (SE {overall.roi_se:.4f}, t={overall.roi_t:+.2f})",
        "",
        overall.verdict(),
    ]
    return "\n".join(lines)
