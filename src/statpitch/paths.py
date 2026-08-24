"""Canonical filesystem layout (Design §8).

Every module resolves paths through here so that tests can redirect the whole
tree with one environment variable instead of monkeypatching call sites.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/statpitch/paths.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Root of the data tree. Override with STATPITCH_DATA for tests/Colab."""
    return Path(os.environ.get("STATPITCH_DATA", REPO_ROOT / "data"))


def raw_dir() -> Path:
    """Unmodified downloads, exactly as fetched. Never committed, always reproducible."""
    return data_root() / "raw"


def cache_dir() -> Path:
    """HTTP/scrape cache — politeness layer for Understat/Transfermarkt (NFR-5)."""
    return data_root() / "cache"


def processed_dir() -> Path:
    """Cleaned, schema-conformant datasets built from raw/."""
    return data_root() / "processed"


def models_dir() -> Path:
    return Path(os.environ.get("STATPITCH_MODELS", REPO_ROOT / "models"))


# --- Design §8 named artifacts -------------------------------------------------

def competitions_file() -> Path:
    return data_root() / "competitions.json"


def decision_config_file() -> Path:
    return data_root() / "decision_config.json"


def bet_ledger_file() -> Path:
    return data_root() / "bet_ledger.jsonl"


def matches_file() -> Path:
    return processed_dir() / "matches_clean.parquet"


def odds_file() -> Path:
    return processed_dir() / "closing_odds.parquet"


def elo_file() -> Path:
    return processed_dir() / "elo_ratings.parquet"


def fixtures_file() -> Path:
    """Upcoming fixtures, built offline by `scripts/build_fixtures.py`.

    Serving reads this at startup and never fetches it: NFR-2 forbids a network
    call on a request path, so a fixture list that is live at request time is not
    available at any latency this project accepts.
    """
    return processed_dir() / "fixtures.parquet"


def live_odds_file() -> Path:
    """Append-only log of captured pre-match prices (Plan §4 Phase A).

    One row per capture x fixture x market x selection. Append-only because a
    CLV measurement is the difference between two captures of the same
    selection: overwriting the earlier one does not refresh the data, it deletes
    the half that cannot be re-fetched.
    """
    return processed_dir() / "live_odds.parquet"


def card_file() -> Path:
    """Today's graded selections, built offline by `scripts/build_card.py`.

    Serving reads this and never computes it: deriving 86 selections per fixture
    and solving a joint Kelly allocation is far outside NFR-2's ~200 ms budget,
    and the market engine has no business on a request path.
    """
    return processed_dir() / "card.parquet"


def ensure_dirs() -> None:
    for d in (raw_dir(), cache_dir(), processed_dir(), models_dir()):
        d.mkdir(parents=True, exist_ok=True)
