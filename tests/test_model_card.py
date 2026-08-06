"""The model card is a claim about the code, so it is tested like one.

A card that drifts from what the system actually does is worse than no card:
it launders a stale number into something that reads as verified. These tests
pin the figures the card states against the values the code serves, so the two
cannot diverge silently.

The headline claim under test is the uncomfortable one — `w` = 0, the model does
not beat the closing line. Nothing may quietly soften that.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from statpitch import decision_config, taxonomy
from statpitch.decision import staking
from statpitch.serving.app import DISCLAIMER, edge_map

ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "docs" / "MODEL_CARD.md"


@pytest.fixture(scope="module")
def card() -> str:
    return CARD_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def findings() -> dict:
    return edge_map()["findings"]


# --- the card exists and is reachable -----------------------------------------

def test_the_card_is_present(card):
    assert card.strip()


def test_the_readme_points_at_it():
    """A card nobody is sent to does not do its job."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/MODEL_CARD.md" in readme


# --- the headline number ------------------------------------------------------

def test_the_card_states_w_as_zero(card, findings):
    """Requirements §8.4 makes reporting `w` mandatory and §9 makes it the verdict."""
    assert findings["market_shrinkage_w"] == 0.0
    assert re.search(r"\*\*0\.000\*\*", card), "w must be stated, in bold, as 0.000"


def test_the_card_does_not_claim_the_model_beats_the_market(card):
    lowered = card.lower()
    assert "does not beat the closing line" in lowered
    for overclaim in ("beats the market", "beats the closing line", "proven edge"):
        assert overclaim not in lowered


def test_the_reported_confidence_interval_matches_the_served_one(card, findings):
    low, high = findings["w_confidence_interval"]
    assert f"[{low:.3f}, {high:.3f}]" in card


def test_the_log_loss_gap_is_reported_as_served(card, findings):
    assert f"{findings['model_vs_market_log_loss_gap']:.4f}" in card


# --- the one significant result ----------------------------------------------

def test_the_clv_result_matches_the_served_figures(card, findings):
    """The single statistically significant finding in the project."""
    clv = findings["measured"]["sharp_reference_clv"]
    assert f"{clv['t']:.2f}" in card
    assert f"{clv['clv'] * 100:.2f}%" in card
    assert str(clv["n"]) in card.replace(",", "")


def test_the_card_qualifies_what_the_clv_result_is_not(card):
    """It is evidence about prices, not about this model.

    Reported without that qualification it reads as a working strategy, which is
    the specific misreading most likely to cost someone money.
    """
    assert "What this result is not" in card


# --- limitations that are structural, not editorial ---------------------------

def test_the_odds_coverage_gap_is_stated_at_its_true_size(card):
    covered = len(taxonomy.registry().with_odds_coverage())
    total = len(taxonomy.registry())
    assert f"{covered} of {total} competitions" in card


def test_the_card_reports_the_config_as_unfitted(card):
    config = decision_config.config()
    assert config.is_placeholder
    assert config.config_version in card


def test_staking_really_is_disabled(card):
    """The card says the engine refuses; assert it actually does."""
    assert "the engine refuses" in card
    with pytest.raises(decision_config.DecisionConfigError):
        staking.StakingEngine(decision_config.config())


def test_the_card_carries_the_advisory_only_designation(card):
    """NFR-11 applies to the documentation as much as to the responses."""
    assert "Advisory only (NFR-11)" in card
    for claim in ("place wagers", "hold funds"):
        assert claim in card
        assert claim in DISCLAIMER


# --- the failed attempts stay in the record -----------------------------------

@pytest.mark.parametrize(
    "attempt",
    ["Understat xG", "Venue-split form", "Best market per match", "22 divisions"],
)
def test_every_attempt_against_the_headline_is_listed(card, attempt):
    """A card listing only what worked is a sales sheet.

    These four are the attempts that failed to overturn `w`=0, and they bound how
    much of the search space was actually covered.
    """
    assert attempt in card


def test_the_max_edge_result_is_recorded_with_its_sign(card):
    """-2.12% is why the grader distrusts large edges; losing it loses the reason."""
    assert "−2.12%" in card or "-2.12%" in card


def test_the_spec_corrections_are_listed(card):
    """Eight places where the spec did not survive contact with the sources."""
    corrections = card.split("## 8.")[1].split("## 9.")[0]
    assert corrections.count("| FR-") + corrections.count("| §") + corrections.count(
        "| NFR-"
    ) >= 6
