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


def ensure_dirs() -> None:
    for d in (raw_dir(), cache_dir(), processed_dir(), models_dir()):
        d.mkdir(parents=True, exist_ok=True)
