"""Polite, cached HTTP fetching (NFR-5).

Every scraped or downloaded source in the project goes through here so the
politeness policy lives in one place: an identifying User-Agent, a minimum delay
between hits on the same host, bounded retries with backoff, and an on-disk cache
so a re-run of a notebook does not re-hit the origin at all.

Note the division of responsibility: this module is about being a good citizen on
*unmetered* sources. API-Football's metered quota is enforced separately in
`statpitch.quota`, because a cache alone cannot protect a hard daily budget.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests

from statpitch import paths

log = logging.getLogger(__name__)

USER_AGENT = (
    "StatPitch/2.0 (personal football-analytics project; "
    "polite scraper; contact via project repository)"
)


class FetchError(RuntimeError):
    """A download failed after exhausting retries."""


@dataclass
class PoliteSession:
    """Rate-limited, retrying, disk-cached HTTP GET."""

    min_interval: float = 1.0          # seconds between requests to the same host
    timeout: float = 30.0
    max_retries: int = 3
    backoff: float = 2.0
    cache_root: Path | None = None
    #: Extra headers sent with every request from this session. Some endpoints
    #: are only reachable with them — Understat's JSON routes return 404 without
    #: `X-Requested-With: XMLHttpRequest`.
    headers: dict[str, str] = field(default_factory=dict)
    _last_hit: dict[str, float] = field(default_factory=dict, repr=False)
    _session: requests.Session | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.cache_root is None:
            self.cache_root = paths.cache_dir() / "http"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        if self._session is None:
            self._session = requests.Session()
        # An injected transport keeps the politeness policy — delay, retries,
        # cache — while letting a caller supply something that can reach a host
        # a bare requests.Session cannot. Transfermarkt sits behind a challenge
        # and needs cloudscraper; putting that exception here rather than in the
        # scraper means it still goes through the one rate limiter (NFR-5).
        self._session.headers.update({"User-Agent": USER_AGENT})
        if self.headers:
            self._session.headers.update(self.headers)

    # --- caching --------------------------------------------------------

    def _cache_path(self, url: str, suffix: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        host = urlparse(url).netloc.replace(":", "_")
        return self.cache_root / host / f"{digest}{suffix}"

    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        last = self._last_hit.get(host)
        if last is not None:
            wait = self.min_interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
        self._last_hit[host] = time.monotonic()

    # --- the public API -------------------------------------------------

    def get_bytes(
        self,
        url: str,
        *,
        cache: bool = True,
        force: bool = False,
        suffix: str = ".bin",
        max_age: float | None = None,
    ) -> bytes:
        """GET `url`, returning the body. Served from disk cache unless `force`.

        `max_age` is how many seconds old a cached copy may be before it is
        re-fetched. The default of None keeps a cached body forever, which is
        right for the results archive — a 2015/16 season file does not change —
        and wrong for anything describing the future.

        It was wrong for fixture schedules, silently. `build_fixtures` read an
        eleven-day-old openfootball snapshot on every local run, so the artifact
        it produced listed matches that had already been played while stamping
        `generated_at` with the current time. CI, which has no cache, produced a
        different and correct answer from identical code. Staleness that only
        appears on one machine is the expensive kind.

        When a re-fetch fails and a stale copy exists, the stale copy is served
        with a warning rather than raising. Being unable to reach the origin is
        a reason to fall back loudly, not a reason to have no data — but a 404 is
        excluded, because that means the file is genuinely gone upstream and
        serving a cached copy would resurrect it indefinitely.
        """
        path = self._cache_path(url, suffix)
        cached = cache and not force and path.exists()

        if cached and max_age is not None:
            age = time.time() - path.stat().st_mtime
            if age > max_age:
                log.debug(
                    "http: cached copy of %s is %.1fh old, past the %.1fh limit",
                    url, age / 3600, max_age / 3600,
                )
                cached = False

        if cached:
            log.debug("http: cache hit %s", url)
            return path.read_bytes()

        try:
            body = self._get_with_retries(url)
        except FetchError as exc:
            if path.exists() and not force and "404" not in str(exc):
                age = (time.time() - path.stat().st_mtime) / 3600
                log.warning(
                    "http: %s unreachable (%s) — serving a cached copy %.1fh old. "
                    "Anything derived from it is that stale.",
                    url, exc, age,
                )
                return path.read_bytes()
            raise

        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(body)
            tmp.replace(path)
        return body

    def download_to(self, url: str, dest: Path, *, force: bool = False) -> Path:
        """Fetch `url` into `dest`, skipping the request entirely if it already exists.

        Raw downloads are kept verbatim on disk: reruns are free, the origin is not
        re-hit, and the cleaning step can be re-derived without network access.
        """
        if dest.exists() and not force:
            log.debug("http: %s already downloaded", dest.name)
            return dest
        body = self._get_with_retries(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(body)
        tmp.replace(dest)
        return dest

    def _get_with_retries(self, url: str) -> bytes:
        assert self._session is not None
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle(url)
            try:
                resp = self._session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
            else:
                if resp.status_code == 404:
                    # A missing season/division file is normal (a league may not
                    # have existed, or the season has not started). Do not retry.
                    raise FetchError(f"404 Not Found: {url}")
                if resp.ok:
                    return resp.content
                last_exc = FetchError(f"HTTP {resp.status_code} for {url}")

            if attempt < self.max_retries:
                delay = self.backoff ** attempt
                log.warning("http: %s failed (attempt %d), retrying in %.1fs", url, attempt, delay)
                time.sleep(delay)

        raise FetchError(f"failed after {self.max_retries} attempts: {url}") from last_exc
