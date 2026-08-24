"""Cache behaviour of the polite HTTP layer (NFR-5).

Offline: the session's transport is replaced with a fake, so these assert the
caching policy rather than anyone's network.

The policy is the point. An unbounded cache is correct for the results archive —
a 2015/16 season file is immutable — and silently wrong for anything describing
the future. `build_fixtures` read an eleven-day-old openfootball schedule on
every local run while stamping the artifact `generated_at` now, so it listed
matches that had already been played. CI, with no cache, produced a different
and correct answer from the same code.
"""

from __future__ import annotations

import os
import time

import pytest
import requests

from statpitch.data.http import FetchError, PoliteSession

URL = "https://example.invalid/schedule.txt"


class FakeTransport:
    """Stands in for `requests.Session`, counting calls."""

    def __init__(self, *, body=b"fresh", status=200, raises=False,
                 response_headers=None):
        self.headers: dict[str, str] = {}
        self.body = body
        self.status = status
        self.raises = raises
        self.response_headers = response_headers or {}
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        if self.raises:
            raise requests.RequestException("network down")
        return FakeResponse(self.body, self.status, self.response_headers)


class FakeResponse:
    def __init__(self, content, status, headers=None):
        self.content = content
        self.status_code = status
        self.headers = headers or {}

    @property
    def ok(self):
        return 200 <= self.status_code < 300


def _session(tmp_path, transport):
    return PoliteSession(
        min_interval=0.0,
        max_retries=1,          # no backoff sleeps in the failure paths
        cache_root=tmp_path / "cache",
        _session=transport,
    )


def _age(path, seconds):
    """Backdate a cached file so it reads as `seconds` old."""
    past = time.time() - seconds
    os.utime(path, (past, past))


def _cached_file(session, url=URL):
    return session._cache_path(url, ".txt")


# --- the default: cache forever -----------------------------------------------

def test_a_cached_body_is_served_without_a_second_request(tmp_path):
    transport = FakeTransport()
    session = _session(tmp_path, transport)

    assert session.get_bytes(URL, suffix=".txt") == b"fresh"
    assert session.get_bytes(URL, suffix=".txt") == b"fresh"
    assert transport.calls == 1


def test_without_max_age_even_an_ancient_copy_is_served(tmp_path):
    """Correct for the results archive, and the bug for schedules."""
    transport = FakeTransport()
    session = _session(tmp_path, transport)
    session.get_bytes(URL, suffix=".txt")
    _age(_cached_file(session), 400 * 24 * 3600)

    assert session.get_bytes(URL, suffix=".txt") == b"fresh"
    assert transport.calls == 1


# --- expiry -------------------------------------------------------------------

def test_a_copy_older_than_max_age_is_refetched(tmp_path):
    transport = FakeTransport()
    session = _session(tmp_path, transport)
    session.get_bytes(URL, suffix=".txt")
    _age(_cached_file(session), 11 * 24 * 3600)     # the observed staleness

    transport.body = b"updated"
    assert session.get_bytes(URL, suffix=".txt", max_age=6 * 3600) == b"updated"
    assert transport.calls == 2


def test_a_copy_within_max_age_is_still_served_from_disk(tmp_path):
    transport = FakeTransport()
    session = _session(tmp_path, transport)
    session.get_bytes(URL, suffix=".txt")
    _age(_cached_file(session), 60)

    assert session.get_bytes(URL, suffix=".txt", max_age=6 * 3600) == b"fresh"
    assert transport.calls == 1


def test_the_refreshed_body_replaces_the_cached_one(tmp_path):
    """A re-fetch must update the cache, or every later read re-downloads."""
    transport = FakeTransport()
    session = _session(tmp_path, transport)
    session.get_bytes(URL, suffix=".txt")
    _age(_cached_file(session), 11 * 24 * 3600)
    transport.body = b"updated"
    session.get_bytes(URL, suffix=".txt", max_age=6 * 3600)

    assert session.get_bytes(URL, suffix=".txt", max_age=6 * 3600) == b"updated"
    assert transport.calls == 2


def test_force_refetches_regardless_of_age(tmp_path):
    transport = FakeTransport()
    session = _session(tmp_path, transport)
    session.get_bytes(URL, suffix=".txt")
    transport.body = b"updated"

    assert session.get_bytes(URL, suffix=".txt", force=True) == b"updated"
    assert transport.calls == 2


# --- falling back when the origin is unreachable ------------------------------

def test_an_unreachable_origin_falls_back_to_the_stale_copy(tmp_path, caplog):
    """No data is worse than stale data — but it has to say so."""
    transport = FakeTransport()
    session = _session(tmp_path, transport)
    session.get_bytes(URL, suffix=".txt")
    _age(_cached_file(session), 11 * 24 * 3600)

    transport.raises = True
    with caplog.at_level("WARNING"):
        assert session.get_bytes(URL, suffix=".txt", max_age=6 * 3600) == b"fresh"
    assert "serving a cached copy" in caplog.text


def test_a_404_does_not_resurrect_a_cached_copy(tmp_path):
    """404 means the file is gone upstream, not that the network failed.

    Falling back here would keep serving a deleted file forever — and
    `openfootball.fetch_file` turns a 404 into None, which is how an unplayed
    cup with no published draw is represented.
    """
    transport = FakeTransport()
    session = _session(tmp_path, transport)
    session.get_bytes(URL, suffix=".txt")
    _age(_cached_file(session), 11 * 24 * 3600)

    transport.status = 404
    with pytest.raises(FetchError, match="404"):
        session.get_bytes(URL, suffix=".txt", max_age=6 * 3600)


def test_a_failure_with_no_cached_copy_still_raises(tmp_path):
    transport = FakeTransport(raises=True)
    session = _session(tmp_path, transport)

    with pytest.raises(FetchError):
        session.get_bytes(URL, suffix=".txt", max_age=6 * 3600)


def test_force_does_not_fall_back_to_the_copy_it_was_told_to_bypass(tmp_path):
    """`force` is how the live-odds capture guarantees a genuinely new read."""
    transport = FakeTransport()
    session = _session(tmp_path, transport)
    session.get_bytes(URL, suffix=".txt")

    transport.raises = True
    with pytest.raises(FetchError):
        session.get_bytes(URL, suffix=".txt", force=True)


# --- response headers, for metered APIs that report their own budget ----------

def test_headers_are_returned_alongside_the_body(tmp_path):
    """A metered API reports its remaining budget in a header, and that number is
    more trustworthy than anything counted locally."""
    transport = FakeTransport(response_headers={"X-Requests-Remaining": "487"})
    session = _session(tmp_path, transport)

    body, headers = session.get_with_headers(URL, suffix=".txt")
    assert body == b"fresh"
    assert headers["x-requests-remaining"] == "487"


def test_header_names_are_lowercased(tmp_path):
    """HTTP header names are case-insensitive; callers should not have to guess."""
    transport = FakeTransport(response_headers={"X-Requests-Used": "13"})
    session = _session(tmp_path, transport)
    _, headers = session.get_with_headers(URL, suffix=".txt")
    assert "x-requests-used" in headers


def test_a_cache_hit_reports_no_headers_rather_than_stale_ones(tmp_path):
    """There was no response to read them from.

    Empty must mean "unknown" to the caller, never "zero remaining" — a budget
    guard that read a cache hit as an exhausted quota would refuse to work.
    """
    transport = FakeTransport(response_headers={"X-Requests-Remaining": "487"})
    session = _session(tmp_path, transport)
    session.get_with_headers(URL, suffix=".txt")

    _, headers = session.get_with_headers(URL, suffix=".txt")
    assert headers == {}
    assert transport.calls == 1


def test_get_bytes_still_returns_only_the_body(tmp_path):
    """The older signature is unchanged; every existing caller reads bytes."""
    session = _session(tmp_path, FakeTransport())
    assert session.get_bytes(URL, suffix=".txt") == b"fresh"
