"""Typed access to `decision_config.json` (NFR-12).

Two rules this module exists to enforce, both of which are easy to violate by
accident and expensive to discover later:

1. **No Decision Layer parameter is ever hardcoded.** Every λ, w, threshold and
   cutoff is read from the versioned config, so a backtest result can be
   reproduced exactly from its `config_version`.
2. **Placeholder parameters cannot produce live recommendations.** Until Phase 5
   fits `w` and Notebook 12 selects a de-vig method per competition, the config is
   marked `placeholder` and `require_fitted()` refuses to hand it to the staking
   path. A stake sized from unfitted defaults looks exactly like a real one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from statpitch import paths


class DecisionConfigError(ValueError):
    """Malformed decision config, or an attempt to use placeholder parameters live."""


@dataclass(frozen=True, slots=True)
class Grading:
    e_peak: float
    sigma: float
    e_ceiling: float
    cutoffs: dict[str, float]
    stake_multiplier: dict[str, float]
    subscore_weights: dict[str, float]


@dataclass(frozen=True, slots=True)
class Guardrails:
    max_p_std: float
    max_book_margin: float
    odds_ceiling: float
    other_competition_fixture_within_hours: int
    suppress_dead_rubbers: bool
    suppress_unconfirmed_lineup_with_key_doubt: bool


@dataclass(frozen=True, slots=True)
class Staking:
    kelly_lambda: float
    lambda_frontier: tuple[float, ...]
    cap_per_bet: float
    cap_per_matchday: float
    min_stake_fraction: float


@dataclass(frozen=True, slots=True)
class MarketEngine:
    matrix_max_goals: int
    asian_handicap_lines: tuple[float, ...]
    total_goals_lines: tuple[float, ...]
    correct_score_top_n: int
    non_stakeable_markets: frozenset[str]


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    config_version: str
    status: str
    w: float | None
    w_fitted: bool
    devig_default_method: str
    devig_method_per_competition: dict[str, str | None]
    grading: Grading
    guardrails: Guardrails
    staking: Staking
    market_engine: MarketEngine
    clv_label: str
    min_cell_sample_size: int
    bootstrap_resamples: int
    pinnacle_break_date: str
    allow_pooling_across_regimes: bool
    raw: dict[str, Any]

    @property
    def is_placeholder(self) -> bool:
        return self.status == "placeholder" or not self.w_fitted

    def devig_method(self, competition_id: str) -> str:
        """Selected method for a competition, falling back to the default (FR-28)."""
        return (
            self.devig_method_per_competition.get(competition_id)
            or self.devig_default_method
        )

    def require_fitted(self, action: str = "produce staking recommendations") -> None:
        """Gate for any code path that sizes a real stake."""
        if self.is_placeholder:
            raise DecisionConfigError(
                f"refusing to {action}: decision_config '{self.config_version}' is a "
                f"placeholder (status={self.status!r}, w_fitted={self.w_fitted}). "
                "Fit w at the Phase 5 checkpoint and select a de-vig method per "
                "competition in Notebook 12 first — a stake sized from unfitted "
                "defaults is indistinguishable from a real one."
            )


def load(path: Path | None = None) -> DecisionConfig:
    path = path or paths.decision_config_file()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise DecisionConfigError(f"decision config not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise DecisionConfigError(f"decision config is not valid JSON: {path}: {exc}") from None

    version = raw.get("config_version")
    if not version:
        raise DecisionConfigError("decision config must carry a config_version (NFR-12)")

    shrink = raw.get("market_shrinkage", {})
    w = shrink.get("w")
    if w is not None and not 0.0 <= float(w) <= 1.0:
        raise DecisionConfigError(f"market shrinkage w must lie in [0, 1], got {w}")

    devig = raw.get("devig", {})
    default_method = devig.get("default_method", "shin")
    implemented = set(devig.get("methods_implemented") or ("proportional", "power", "shin"))
    if default_method not in implemented:
        raise DecisionConfigError(
            f"devig default_method {default_method!r} is not in methods_implemented"
        )
    per_comp = dict(devig.get("method_per_competition") or {})
    for comp, method in per_comp.items():
        if method is not None and method not in implemented:
            raise DecisionConfigError(f"{comp}: unknown de-vig method {method!r}")

    g = raw.get("grading", {})
    grading = Grading(
        e_peak=float(g.get("e_peak", 0.04)),
        sigma=float(g.get("sigma", 0.05)),
        e_ceiling=float(g.get("e_ceiling", 0.12)),
        cutoffs=dict(g.get("cutoffs") or {}),
        stake_multiplier=dict(g.get("stake_multiplier") or {}),
        subscore_weights=dict(g.get("subscore_weights") or {}),
    )
    _validate_grading(grading)

    gr = raw.get("guardrails", {})
    guardrails = Guardrails(
        max_p_std=float(gr.get("max_p_std", 0.06)),
        max_book_margin=float(gr.get("max_book_margin", 0.08)),
        odds_ceiling=float(gr.get("odds_ceiling", 8.0)),
        other_competition_fixture_within_hours=int(
            gr.get("other_competition_fixture_within_hours", 72)
        ),
        suppress_dead_rubbers=bool(gr.get("suppress_dead_rubbers", True)),
        suppress_unconfirmed_lineup_with_key_doubt=bool(
            gr.get("suppress_unconfirmed_lineup_with_key_doubt", True)
        ),
    )

    s = raw.get("staking", {})
    staking = Staking(
        kelly_lambda=float(s.get("kelly_lambda", 0.25)),
        lambda_frontier=tuple(float(x) for x in s.get("lambda_frontier") or (0.25,)),
        cap_per_bet=float(s.get("cap_per_bet", 0.02)),
        cap_per_matchday=float(s.get("cap_per_matchday", 0.10)),
        min_stake_fraction=float(s.get("min_stake_fraction", 0.0005)),
    )
    if not 0.0 < staking.kelly_lambda <= 1.0:
        raise DecisionConfigError(f"kelly_lambda must lie in (0, 1], got {staking.kelly_lambda}")
    if staking.cap_per_bet > staking.cap_per_matchday:
        raise DecisionConfigError(
            "cap_per_bet exceeds cap_per_matchday — a single bet could breach the "
            "matchday exposure limit"
        )

    me = raw.get("market_engine", {})
    market_engine = MarketEngine(
        matrix_max_goals=int(me.get("matrix_max_goals", 10)),
        asian_handicap_lines=tuple(float(x) for x in me.get("asian_handicap_lines") or ()),
        total_goals_lines=tuple(float(x) for x in me.get("total_goals_lines") or ()),
        correct_score_top_n=int(me.get("correct_score_top_n", 10)),
        # `or` would be wrong here: an explicit empty list is a deliberate attempt to
        # make every market stakeable and must reach the check below, not silently
        # fall back to the default that contains correct_score.
        non_stakeable_markets=frozenset(
            me.get("non_stakeable_markets", ("correct_score",))
        ),
    )
    if "correct_score" not in market_engine.non_stakeable_markets:
        raise DecisionConfigError(
            "correct_score must remain non-stakeable (Requirements §3.2): its ~15-25% "
            "book margin means model error dominates any apparent edge"
        )

    rep = raw.get("reporting", {})
    reg = raw.get("odds_regime", {})

    return DecisionConfig(
        config_version=version,
        status=raw.get("status", "unknown"),
        w=None if w is None else float(w),
        w_fitted=bool(shrink.get("w_fitted", False)) and w is not None,
        devig_default_method=default_method,
        devig_method_per_competition=per_comp,
        grading=grading,
        guardrails=guardrails,
        staking=staking,
        market_engine=market_engine,
        clv_label=rep.get("clv_label", "Friday-to-close CLV"),
        min_cell_sample_size=int(rep.get("min_cell_sample_size", 50)),
        bootstrap_resamples=int(rep.get("bootstrap_resamples", 10000)),
        pinnacle_break_date=reg.get("pinnacle_break_date", "2025-07-23"),
        allow_pooling_across_regimes=bool(reg.get("allow_pooling_across_regimes", False)),
        raw=raw,
    )


def _validate_grading(g: Grading) -> None:
    if g.sigma <= 0:
        raise DecisionConfigError("grading sigma must be positive")
    if g.e_ceiling <= g.e_peak:
        raise DecisionConfigError(
            f"e_ceiling ({g.e_ceiling}) must exceed e_peak ({g.e_peak}); otherwise every "
            "bet at peak confidence is immediately graded F"
        )
    order = ["A", "B", "C", "D"]
    missing = [k for k in order if k not in g.cutoffs]
    if missing:
        raise DecisionConfigError(f"grading cutoffs missing grades: {missing}")
    values = [g.cutoffs[k] for k in order]
    if values != sorted(values, reverse=True):
        raise DecisionConfigError(f"grading cutoffs must decrease A>B>C>D, got {g.cutoffs}")
    for grade in ("D", "F"):
        if g.stake_multiplier.get(grade, 0.0) != 0.0:
            raise DecisionConfigError(
                f"grade {grade} must carry a zero stake multiplier (Design §6.4)"
            )


@lru_cache(maxsize=1)
def _cached() -> DecisionConfig:
    return load()


def config() -> DecisionConfig:
    """Process-wide cached config (loaded once at API startup)."""
    return _cached()


def reset_cache() -> None:
    _cached.cache_clear()
