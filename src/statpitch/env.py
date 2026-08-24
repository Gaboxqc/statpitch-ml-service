"""Read `.env`, because `.env.example` has always said it would.

Its first line reads "Copy to .env for local development", and nothing in the
project ever opened the file. A credential put there was silently ignored: every
collector reported itself as unconfigured and skipped, which is indistinguishable
from having no key at all — the failure mode those collectors were carefully
written to make visible, arriving through the front door.

No dependency
=============

`python-dotenv` would do this, and would also have to be kept out of
`requirements-serving.txt`, which `tests/test_deployment.py` polices. The format
is four lines of parsing, so it is four lines of parsing.

The real environment always wins
================================

A value already present in `os.environ` is never overwritten. CI and Render set
their secrets as real environment variables, and a stale `.env` left in a working
copy must not quietly shadow them — that is how a deployment ends up using a
developer's expired key. `override=True` exists for tests and says what it does.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from statpitch import paths

log = logging.getLogger(__name__)

DEFAULT_NAME = ".env"


def parse(text: str) -> dict[str, str]:
    """Parse `KEY=VALUE` lines, skipping blanks and comments.

    Values may be quoted; surrounding single or double quotes are stripped so a
    key pasted with them still works. Anything without an `=` is skipped rather
    than raising: a half-edited file should cost one variable, not the run.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        # `export FOO=bar` is what people paste out of shell instructions.
        if name.startswith("export "):
            name = name[len("export "):].strip()
        if not name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[name] = value
    return out


def load_dotenv(
    path: Path | None = None, *, override: bool = False
) -> dict[str, str]:
    """Load `.env` into the environment. Returns the names it set.

    Absent, this is a no-op — the file is optional by design, and every consumer
    already treats a missing credential as a capability it does not have rather
    than an error.
    """
    target = Path(path) if path is not None else paths.REPO_ROOT / DEFAULT_NAME
    if not target.exists():
        log.debug("env: no %s", target)
        return {}

    try:
        parsed = parse(target.read_text(encoding="utf-8"))
    except OSError as exc:
        log.warning("env: could not read %s — %s", target, exc)
        return {}

    applied: dict[str, str] = {}
    for name, value in parsed.items():
        if not override and name in os.environ:
            continue
        os.environ[name] = value
        applied[name] = value

    if applied:
        # Names only. A credential must never reach a log, and the names are what
        # a reader needs to answer "why did the collector still skip".
        log.info(
            "env: loaded %d value(s) from %s: %s",
            len(applied), target.name, ", ".join(sorted(applied)),
        )
    skipped = sorted(set(parsed) - set(applied))
    if skipped:
        log.info(
            "env: %s already set in the environment, not overridden by %s",
            ", ".join(skipped), target.name,
        )
    return applied
