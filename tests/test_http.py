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
        self.reason = {
            200: "OK", 300: "Multiple Choices", 404: "Not Found",
            500: "Internal Server Error",
        }.get(status, "Unknown")

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
    suffix = ".csv" if url.endswith(".csv") else ".txt"
    return session._cache_path(url, suffix)


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


# --- credentials must not reach a log ----------------------------------------

def test_a_credential_query_value_is_redacted():
    """The Odds API authenticates with `?apiKey=`, so its key would otherwise be
    written verbatim into every retry warning and from there into CI logs."""
    from statpitch.data.http import redact_url

    masked = redact_url("https://x.test/v4/e?apiKey=super-secret&regions=eu")
    assert "super-secret" not in masked
    assert "apiKey=***" in masked
    # The parameter NAME survives: which credential was missing is the thing a
    # reader needs.
    assert "regions=eu" in masked


def test_a_url_without_a_query_is_unchanged():
    from statpitch.data.http import redact_url

    assert redact_url("https://x.test/a/b") == "https://x.test/a/b"


def test_redaction_is_case_insensitive_on_the_parameter_name():
    from statpitch.data.http import redact_url

    for name in ("apiKey", "APIKEY", "api_key", "token", "secret"):
        assert "s3cr3t" not in redact_url(f"https://x.test/e?{name}=s3cr3t")


def test_a_retry_warning_does_not_leak_the_key(tmp_path, caplog):
    transport = FakeTransport(raises=True)
    session = PoliteSession(
        min_interval=0.0, max_retries=1,
        cache_root=tmp_path / "cache", _session=transport,
    )
    url = "https://x.test/v4/events?apiKey=super-secret"
    with caplog.at_level("WARNING"), pytest.raises(FetchError) as excinfo:
        session.get_bytes(url, suffix=".json")

    assert "super-secret" not in caplog.text
    # And not through the exception message either, which callers log.
    assert "super-secret" not in str(excinfo.value)


def test_a_404_message_does_not_leak_the_key(tmp_path):
    transport = FakeTransport(status=404)
    session = PoliteSession(
        min_interval=0.0, max_retries=1,
        cache_root=tmp_path / "cache", _session=transport,
    )
    with pytest.raises(FetchError) as excinfo:
        session.get_bytes("https://x.test/e?apiKey=super-secret", suffix=".json")
    assert "super-secret" not in str(excinfo.value)


# --- "absent upstream" is not the same as "the request failed" ---------------

def test_a_soft_404_is_not_treated_as_success(tmp_path):
    """football-data.co.uk answers an unpublished season with 300, not 404.

    `requests.Response.ok` is true for anything under 400, so the HTML body of a
    300 was saved as a season CSV — and `download_to` skips a path that exists,
    so the poisoned file was never re-fetched. The 2026/27 Bundesliga file sat on
    disk as a 1,134-byte error page named D1.csv, and would have blocked that
    season from ever ingesting.
    """
    transport = FakeTransport(body=b"<!DOCTYPE HTML><html>300</html>", status=300)
    session = _session(tmp_path, transport)
    with pytest.raises(FetchError, match="HTTP 300"):
        session.get_bytes("https://x.test/2627/D1.csv", suffix=".csv")


def test_a_soft_404_is_not_cached(tmp_path):
    """The poisoning is what made it permanent rather than transient."""
    transport = FakeTransport(body=b"<!DOCTYPE HTML>", status=300)
    session = _session(tmp_path, transport)
    url = "https://x.test/2627/D1.csv"
    with pytest.raises(FetchError):
        session.get_bytes(url, suffix=".csv")
    assert not session._cache_path(url, ".csv").exists()


def test_a_3xx_is_not_retried(tmp_path):
    """It is a stable answer, so retrying only wastes politeness budget."""
    transport = FakeTransport(body=b"<html>", status=300)
    session = PoliteSession(
        min_interval=0.0, max_retries=3,
        cache_root=tmp_path / "cache", _session=transport,
    )
    with pytest.raises(FetchError):
        session.get_bytes("https://x.test/a.csv", suffix=".csv")
    assert transport.calls == 1


def test_absence_covers_both_the_404_and_the_300_forms():
    from statpitch.data.http import is_absent

    assert is_absent(FetchError("404 Not Found: https://x.test/a"))
    assert is_absent(FetchError("HTTP 300 Multiple Choices: https://x.test/a"))
    assert not is_absent(FetchError("failed after 3 attempts: https://x.test/a"))
    assert not is_absent(FetchError("HTTP 500 Server Error: https://x.test/a"))


def test_a_server_error_is_still_retried_and_still_falls_back(tmp_path, caplog):
    """A 500 is not absence — it is a failure, and a cached copy still serves."""
    transport = FakeTransport()
    session = _session(tmp_path, transport)
    url = "https://x.test/a.csv"
    session.get_bytes(url, suffix=".csv")

    # max_age forces a re-fetch; without it the cached copy is served and the
    # 500 path is never reached, which is how this test first passed vacuously.
    transport.status = 500
    _age(_cached_file(session, url), 10 * 3600)
    with caplog.at_level("WARNING"):
        assert session.get_bytes(url, suffix=".csv", max_age=3600) == b"fresh"
    assert "serving a cached copy" in caplog.text
