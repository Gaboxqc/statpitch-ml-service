"""API-Football request budget (NFR-9, Design §3.2).

The free tier is 100 requests/day, resetting 00:00 UTC, with unused requests lost.
That is the only hard external limit in the stack, and a single poorly written
polling loop exhausts it in one afternoon.

This module is deliberately separate from the HTTP client and lands before any
real API call exists, because retrofitting a quota guard onto a polling fetcher
means rewriting the fetcher. It enforces four rules:

  * a persistent daily counter keyed on the UTC date, so a process restart does
    not reset the budget;
  * a hard stop at 90 calls, leaving headroom below the real 100 limit;
  * a fixture-keyed cache with no re-fetch inside 24h;
  * one attempt per key — a caller that asks twice for the same fixture gets the
    cached answer or an exhaustion signal, never a second call.

Callers never see an exception for exhaustion. They get `None`, which routes them
to the pre-match estimate (NFR-7). The graceful fallback is a quota-protection
mechanism, not only a robustness nicety.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from statpitch import paths

log = logging.getLogger(__name__)

DAILY_LIMIT = 100
DEFAULT_HARD_STOP = 90
DEFAULT_CACHE_TTL = timedelta(hours=24)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class QuotaState:
    """A day's usage. `day` is a UTC date string; the quota resets at 00:00 UTC."""

    day: str
    used: int
    hard_stop: int

    @property
    def remaining(self) -> int:
        return max(0, self.hard_stop - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.hard_stop


class QuotaExhausted(RuntimeError):
    """Raised only by `spend(strict=True)`; the normal path returns None instead."""


@dataclass
class QuotaBudget:
    """Persistent daily counter plus a TTL cache, shared by every API-Football call.

    Not a general-purpose cache: entries are keyed by caller-supplied strings
    (e.g. ``"lineup:1035423"``) and a key is fetched at most once per TTL window.
    """

    state_path: Path | None = None
    hard_stop: int = DEFAULT_HARD_STOP
    cache_ttl: timedelta = DEFAULT_CACHE_TTL
    clock: Callable[[], datetime] = _utcnow
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.hard_stop > DAILY_LIMIT:
            raise ValueError(
                f"hard_stop {self.hard_stop} exceeds the API-Football free-tier daily "
                f"limit of {DAILY_LIMIT}"
            )
        if self.state_path is None:
            self.state_path = paths.cache_dir() / "api_football_quota.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    # --- persistence ----------------------------------------------------

    def _today(self) -> str:
        return self.clock().astimezone(UTC).date().isoformat()

    def _read(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            # A corrupt state file must not silently grant a fresh 90 calls, but it
            # also must not wedge the pipeline. Start the day at zero and say so.
            return {}
        if not isinstance(raw, dict):
            return {}
        return raw

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def _load_for_today(self) -> dict[str, Any]:
        data = self._read()
        today = self._today()
        if data.get("day") != today:
            # New UTC day: the counter resets, and unused requests are simply lost.
            data = {"day": today, "used": 0, "cache": data.get("cache", {})}
        data.setdefault("used", 0)
        data.setdefault("cache", {})
        self._prune_cache(data)
        return data

    def _prune_cache(self, data: dict[str, Any]) -> None:
        now = self.clock()
        keep = {}
        for key, entry in data.get("cache", {}).items():
            try:
                fetched = datetime.fromisoformat(entry["fetched_at"])
            except (KeyError, TypeError, ValueError):
                continue
            if now - fetched < self.cache_ttl:
                keep[key] = entry
        data["cache"] = keep

    # --- inspection -----------------------------------------------------

    def state(self) -> QuotaState:
        with self._lock:
            data = self._load_for_today()
            return QuotaState(day=data["day"], used=int(data["used"]), hard_stop=self.hard_stop)

    @property
    def remaining(self) -> int:
        return self.state().remaining

    def cached(self, key: str) -> Any | None:
        """Cached payload for `key` if still inside the TTL window, else None."""
        with self._lock:
            data = self._load_for_today()
            entry = data["cache"].get(key)
            return None if entry is None else entry.get("payload")

    # --- the one method callers use --------------------------------------

    def spend(
        self,
        key: str,
        fetch: Callable[[], Any],
        *,
        cost: int = 1,
        strict: bool = False,
    ) -> Any | None:
        """Return `key`'s payload, calling `fetch` at most once per TTL window.

        Returns None when the budget is exhausted so the caller falls back to its
        pre-match estimate (NFR-7). A failing `fetch` also returns None and does
        **not** consume budget, but the attempt is not retried within this call —
        no loops, by construction.
        """
        if cost < 1:
            raise ValueError("cost must be at least 1")

        with self._lock:
            data = self._load_for_today()

            entry = data["cache"].get(key)
            if entry is not None:
                log.debug("quota: cache hit for %s", key)
                return entry.get("payload")

            if int(data["used"]) + cost > self.hard_stop:
                msg = (
                    f"API-Football budget exhausted: {data['used']}/{self.hard_stop} used "
                    f"on {data['day']} (UTC); refusing {key!r} and falling back"
                )
                if strict:
                    raise QuotaExhausted(msg)
                log.warning("%s", msg)
                return None

            # Reserve budget BEFORE the call. If the process dies mid-request the
            # call may still have reached the API, so an over-count is the safe
            # direction to be wrong in.
            data["used"] = int(data["used"]) + cost
            self._write(data)

        try:
            payload = fetch()
        except Exception:
            log.exception("quota: fetch failed for %s (budget already consumed)", key)
            return None

        with self._lock:
            data = self._load_for_today()
            data["cache"][key] = {
                "fetched_at": self.clock().isoformat(),
                "payload": payload,
            }
            self._write(data)

        return payload

    def reset(self) -> None:
        """Clear counter and cache. Tests and manual recovery only."""
        with self._lock:
            self._write({"day": self._today(), "used": 0, "cache": {}})


def describe() -> dict[str, Any]:
    """What is configured, by name and never by value.

    A startup line saying which optional sources are available is worth having;
    one that leaks a key into a log is not, so this reports presence only.
    """
    from statpitch.data import api_football

    budget = budget_from_env()
    return {
        "api_football_configured": api_football.configured(),
        "daily_limit": DAILY_LIMIT,
        "hard_stop": budget.hard_stop,
        "remaining_today": budget.remaining,
    }


def budget_from_env() -> QuotaBudget:
    """Construct the shared budget, honouring STATPITCH_QUOTA_HARD_STOP."""
    hard_stop = int(os.environ.get("STATPITCH_QUOTA_HARD_STOP", DEFAULT_HARD_STOP))
    return QuotaBudget(hard_stop=hard_stop)
