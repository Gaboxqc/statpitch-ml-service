"""Edge and expected value (FR-16a, Design §6.3).

Three numbers go in and they must never be confused with one another:

    q_fair    de-vigged CONSENSUS price   — the best estimate of true probability
    o_avail   the MAXIMUM quoted price    — what can actually be taken
    p_model   the model's own probability — from model_pure, never model_informed

The separation is the point. De-vigging `o_avail` to get a fair probability
produces numbers that look like free money and are not: the maximum of N noisy
quotes sits above consensus by construction, so its implied probability is
systematically too low and every selection looks underpriced. This module takes
the two as separate arguments and offers no path that derives one from the other.

Model edge and price edge are separated too
===========================================

Total expected value decomposes exactly:

    EV = p_model * o - 1
       = (q_fair * o - 1)          <- price edge: the price beats consensus
       + (p_model - q_fair) * o    <- model edge: the model beats consensus

Design §6.3 asks for these to be reported separately because they behave
differently. Price edge needs no model skill at all — it is one book being off
the market — and in this project's own measurements it is the component with
evidence behind it. Model edge needs the model to know something the consensus
does not, which the fitted `w` of zero says it does not.

Keeping them apart means a bet driven entirely by a stale price is not reported
as a model insight.

De-vigging happens upstream. This module takes probabilities and prices, so it is
indifferent to whether `q_fair` came from a consensus average or from a sharp
reference book — a distinction that matters, since the sharp reference is the one
that produced measurable closing-line value here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from statpitch.decision.market_engine import Payoff, Selection

log = logging.getLogger(__name__)

#: A quoted price at or below this returns less than the stake and is invalid.
MIN_ODDS = 1.0


class ValueError_(ValueError):
    """Raised for malformed value inputs."""


@dataclass(frozen=True, slots=True)
class ValueAssessment:
    """Edge decomposition for one selection at one price."""

    key: str
    p_model: float
    q_fair: float
    o_avail: float
    payoff: Payoff

    # --- the two market numbers, kept apart ------------------------------

    @property
    def fair_odds(self) -> float:
        """The price consensus implies. NOT the price on offer."""
        return 1.0 / self.q_fair if self.q_fair > 0 else float("inf")

    @property
    def price_advantage(self) -> float:
        """How much better the available price is than the consensus price."""
        return self.o_avail / self.fair_odds - 1.0

    # --- edges -----------------------------------------------------------

    @property
    def edge_prob(self) -> float:
        """Model probability minus consensus, in probability points (FR-16a)."""
        return self.p_model - self.q_fair

    @property
    def expected_value(self) -> float:
        """Total EV per unit staked, over the full payoff distribution.

        Uses the payoff distribution rather than `p * o - 1` so that pushes and
        quarter lines are priced correctly — a selection that refunds half the
        stake is not the same bet as one that loses it.
        """
        return _ev_at(self.payoff, self.p_model, self.o_avail)

    @property
    def price_edge(self) -> float:
        """EV attributable to the price alone, with no model skill assumed.

        This is what a bettor earns by taking the best quote while believing
        exactly what the consensus believes.
        """
        return _ev_at(self.payoff, self.q_fair, self.o_avail)

    @property
    def model_edge(self) -> float:
        """EV attributable to the model disagreeing with consensus.

        By construction `price_edge + model_edge == expected_value`, so a bet can
        be attributed to its actual source rather than credited to the model by
        default.
        """
        return self.expected_value - self.price_edge

    @property
    def is_value(self) -> bool:
        return self.expected_value > 0.0

    @property
    def driven_by_price(self) -> bool:
        """Whether the price, not the model, is doing the work."""
        return self.price_edge > self.model_edge


def _rescale(payoff: Payoff, probability: float) -> Payoff:
    """Re-weight a payoff distribution to a different win probability.

    The engine's payoff comes from the model's own matrix, so its outcome
    probabilities already embody `p_model`. To evaluate the same selection under a
    different probability — the consensus, say — the winning and losing mass is
    rescaled while the push structure is preserved, since a push is a property of
    the bet's settlement rules rather than of who is likely to win.
    """
    winning = payoff.win + payoff.half_win
    losing = payoff.half_loss + payoff.loss
    if winning <= 0 or losing <= 0:
        return payoff

    target_win = max(min(probability, 1.0 - payoff.push), 0.0)
    target_loss = max(1.0 - payoff.push - target_win, 0.0)

    win_scale = target_win / winning
    loss_scale = target_loss / losing
    return Payoff(
        win=payoff.win * win_scale,
        half_win=payoff.half_win * win_scale,
        push=payoff.push,
        half_loss=payoff.half_loss * loss_scale,
        loss=payoff.loss * loss_scale,
    )


def _ev_at(payoff: Payoff, probability: float, odds: float) -> float:
    return _rescale(payoff, probability).expected_return(odds)


def assess(
    selection: Selection,
    q_fair: float,
    o_avail: float,
    p_model: float | None = None,
) -> ValueAssessment:
    """Assess one selection at one available price.

    `p_model` defaults to the selection's own probability, which is where it comes
    from in normal use; it is overridable so a shrunk probability (Design §6.5's
    `p_used`) can be assessed without rebuilding the market book.
    """
    if not 0.0 <= q_fair <= 1.0:
        raise ValueError_(f"{selection.key}: q_fair must be a probability, got {q_fair}")
    if o_avail <= MIN_ODDS:
        raise ValueError_(
            f"{selection.key}: available odds must exceed {MIN_ODDS}, got {o_avail} — "
            "a price never returns less than the stake"
        )

    probability = selection.probability if p_model is None else p_model
    if not 0.0 <= probability <= 1.0:
        raise ValueError_(f"{selection.key}: p_model must be a probability, got {probability}")

    return ValueAssessment(
        key=selection.key,
        p_model=probability,
        q_fair=q_fair,
        o_avail=o_avail,
        payoff=selection.payoff,
    )


def assess_book(
    selections: list[Selection],
    fair: dict[str, float],
    available: dict[str, float],
    model: dict[str, float] | None = None,
) -> list[ValueAssessment]:
    """Assess every selection that has both a fair probability and a price.

    Selections without a quoted price are skipped rather than defaulted. A missing
    market is not a free bet, and inventing a price for one is how a backtest
    quietly starts trading markets that were never available.
    """
    out = []
    for selection in selections:
        if not selection.stakeable:
            continue
        q = fair.get(selection.key)
        o = available.get(selection.key)
        if q is None or o is None:
            continue
        try:
            out.append(
                assess(selection, q, o, (model or {}).get(selection.key))
            )
        except ValueError_ as exc:
            log.debug("value: skipping %s (%s)", selection.key, exc)
    return out


def summarise(assessments: list[ValueAssessment]) -> str:
    lines = [
        f"{'selection':22} {'p_model':>8} {'q_fair':>8} {'price':>7} "
        f"{'EV':>8} {'price_ed':>9} {'model_ed':>9}"
    ]
    for a in sorted(assessments, key=lambda x: -x.expected_value):
        lines.append(
            f"{a.key:22} {a.p_model:8.4f} {a.q_fair:8.4f} {a.o_avail:7.2f} "
            f"{a.expected_value:+8.4f} {a.price_edge:+9.4f} {a.model_edge:+9.4f}"
        )
    return "\n".join(lines)
