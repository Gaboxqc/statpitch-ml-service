"""The integration contract: provenance and machine-readable refusals.

StatPitch is deployed as a **stateless** service. It owns no database. A separate
API consumes it, persists the results, and serves a frontend from its own store
(see `docs/04_ML_Roadmap.md`). That architecture makes the response shape the
product: a field renamed here is a migration in someone else's database.

Two problems this module exists to solve, both of which are invisible until a
consumer has already stored a season of rows.

Provenance
==========

A stored prediction is only as useful as its traceability. Once the weekly retrain
of Roadmap §11 lands, predictions under the same fixture key will come from
different models, and a consumer holding rows with no version stamp cannot tell
which — so it cannot compare them, expire them, or report a track record honestly.
Every response therefore carries `model_version`, `config_version`,
`schema_version` and `generated_at`.

`model_version` names **the path that actually produced the number**, not the
package version. Today that path is the Elo-to-goal-rate mapping in `predictor`
feeding a Dixon-Coles matrix — *not* the fitted XGBoost goal model the model card
evaluates, which is Roadmap §2's open gap. When a fitted artifact starts serving,
this string changes, and that change is precisely the signal a downstream store
needs to segregate the rows.

Refusals that survive the hop
=============================

This project refuses to answer rather than answering badly, and every refusal
cites the measurement behind it — `/best-bet` quotes w=0.000 over 5,306 matches,
`/card/today` quotes the unfitted config. That is the most honest thing the API
does, and until now it existed only as English prose in a `note` or `reason`
string.

Prose does not survive a hop into a database. A consumer storing
`"No selection is recommended. The fitted market-shrinkage weight w is 0.000 …"`
has stored a blob it can render and nothing else: it cannot group by refusal
cause, alert when a cause changes, or show the number next to the sentence. So a
refusal is emitted as a *structured* object as well — a stable `reason_code`, the
unchanged human sentence, and the measurement as data.

The prose is not replaced. NFR-13 forbids renaming, removing or retyping an
existing field, and consumers already read `reason`. The structure is additive:
new keys, alongside the old ones, which a client that ignores unknown keys will
never see.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from statpitch import __version__, decision_config

#: Bumped only on a BREAKING change to a response shape. Additive keys — which is
#: all NFR-13 permits anyway — do not move it. A consumer pins on this to know
#: whether its stored rows are still interpretable.
SCHEMA_VERSION = 1

#: The inference path currently serving predictions. Not the package version: two
#: releases can serve identical numbers, and one release could serve different
#: ones.
#:
#: "poisson", not "dixon-coles", and the correction is the point. `Artifacts.rho`
#: is never populated, so the served matrix applies no low-score correction at
#: all — it is independent Poisson wearing a Dixon-Coles code path. Roadmap §2
#: measured this and found the obvious repair makes things worse; see the note on
#: `Artifacts.goal_environment`. Naming it accurately is the whole function of
#: this string, and a consumer keying on it deserves the version that says what
#: it will actually get.
SERVED_MODEL = "elo-poisson"


def model_version() -> str:
    return f"{SERVED_MODEL}-{__version__}"


class ReasonCode(StrEnum):
    """Stable identifiers for every way this API declines to answer.

    These are part of the contract in the same way a path is. A consumer will
    branch on them and store them; renaming one silently changes the meaning of
    historical rows, so they are added to and never edited.
    """

    #: No free odds source covers this competition, so no market benchmark exists.
    #: A data-availability limit, not a modelling choice (Requirements §9).
    NO_ODDS_COVERAGE = "NO_ODDS_COVERAGE"

    #: `decision_config` is a placeholder; `StakingEngine.require_fitted()` refuses
    #: to size a stake from unfitted parameters (NFR-12).
    DECISION_CONFIG_UNFITTED = "DECISION_CONFIG_UNFITTED"

    #: The fitted market-shrinkage weight is zero — the model adds nothing over the
    #: closing line, so no selection is justified (Requirements §9).
    SHRINKAGE_WEIGHT_ZERO = "SHRINKAGE_WEIGHT_ZERO"

    #: Best-bet-per-match selection measured worse than committing to one market.
    MAX_EDGE_SELECTION_HARMFUL = "MAX_EDGE_SELECTION_HARMFUL"

    #: No settled bets to resample. Simulating an empty track record is a
    #: simulation of nothing.
    EMPTY_LEDGER = "EMPTY_LEDGER"

    #: Live fixture listing is not wired yet (Roadmap §7). Distinct from "no
    #: fixtures today", which is an empty list rather than a refusal.
    NO_FIXTURE_SOURCE = "NO_FIXTURE_SOURCE"

    #: The card was computed and every selection graded below the staking cutoff.
    #: Distinct from DECISION_CONFIG_UNFITTED, which stops the card being sized at
    #: all, and from NO_CARD_SOURCE, which means it was never built. A consumer
    #: charting "why is the slate empty" needs the three separated.
    NO_QUALIFYING_SELECTION = "NO_QUALIFYING_SELECTION"

    #: No card artifact is present. `scripts/build_card.py` has not run.
    NO_CARD_SOURCE = "NO_CARD_SOURCE"

    #: Selections ARE being recommended, under a rule whose evidence does not yet
    #: cover the price panel it runs on. Not a refusal — the payload carries bets
    #: — but it travels with them so a consumer stores the caveat alongside the
    #: rows rather than discovering it later.
    SELECTION_RULE_EXPERIMENTAL = "SELECTION_RULE_EXPERIMENTAL"


class OpenModel(BaseModel):
    """Base for every response model.

    `extra="allow"` is load-bearing, not laziness. A `response_model` normally
    *filters* the payload to its declared fields, which would silently drop keys
    the moment a route returns more than the model names — exactly the field
    removal NFR-13 forbids, delivered as a side effect of adding documentation.

    Allowing extras inverts that: declared fields become the guaranteed,
    schema-documented core that a consumer can codegen against, and everything
    else passes through untouched. The alternative — exhaustively declaring every
    nested key of every response — buys stricter typing at the cost of a contract
    that breaks whenever the payload grows.

    `protected_namespaces` is cleared for a related reason: `model_version` sits
    inside pydantic's reserved `model_` prefix and would warn on every import. The
    field name is fixed by the contract, so the namespace protection gives way.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class Provenance(OpenModel):
    """Which model, which config, which shape, and when."""

    model_version: str
    config_version: str
    schema_version: int
    generated_at: str


class Refusal(OpenModel):
    """A declined answer, as data.

    `reason` keeps the exact human sentence the route already returned;
    `measurement` carries the numbers that sentence quotes, so a consumer can
    render the prose and still chart the cause.
    """

    available: bool = False
    reason_code: ReasonCode
    reason: str
    measurement: dict[str, Any] = {}


def provenance() -> dict[str, Any]:
    """Provenance keys, for merging into any response."""
    return {
        "model_version": model_version(),
        "config_version": decision_config.config().config_version,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def refusal_object(
    code: ReasonCode, reason: str, /, **measurement: Any
) -> dict[str, Any]:
    """The structured refusal itself, for nesting under a caller-chosen key."""
    return {
        "available": False,
        "reason_code": str(code),
        "reason": reason,
        "measurement": measurement,
    }


def refusal(
    code: ReasonCode, reason: str, /, **measurement: Any
) -> dict[str, Any]:
    """A refusal under the conventional top-level `refusal` key.

    Callers merge this alongside the prose field they already return, never in
    place of it.
    """
    return {"refusal": refusal_object(code, reason, **measurement)}


# --- the measurements behind each refusal -------------------------------------
#
# Kept here rather than inline at each route so the same finding cannot be quoted
# with two different numbers, which is how a measurement stops being one.

#: MODEL_CARD §1, re-fitted after the xG features landed.
W_MEASUREMENT: dict[str, Any] = {
    "w": 0.0,
    "w_confidence_interval": [0.0, 0.100],
    "n_validation_matches": 5306,
    "model_log_loss": 0.9845,
    "market_log_loss": 0.9698,
}

#: Plan §4 Phase C — why staking stays disabled even though the card now computes.
#:
#: Two independent gates, and a consumer charting "why is the slate empty" needs
#: both. `w`=0 removes the model's contribution; this removes the price rule's.
SELECTION_RULE_MEASUREMENT: dict[str, Any] = {
    "selection_rule_status": "candidate",
    "candidate_reference": "betfair_exchange",
    "pinnacle_clv_pre_break": 0.0051,
    "pinnacle_clv_pre_break_t": 7.53,
    "pinnacle_in_live_feed": False,
    "betfair_clv_post_break": 0.0251,
    "betfair_clv_post_break_t": 7.86,
    "unselected_baseline_post_break": -0.0003,
    "seasons_available_post_break": 1,
    "seasons_required": 2,
    "evidence": "data/selection_rule_study.json",
}

#: MODEL_CARD §4 — why ranking a fixture's markets is worse than not.
MAX_EDGE_MEASUREMENT: dict[str, Any] = {
    "best_bet_per_match_roi": -0.0212,
    "single_market_roi": 0.0013,
    "growth_ranked_roi": -0.0327,
    "share_of_picks_in_1x2": 0.545,
}
