"""The model registry (Roadmap §1.2).

Until now training happened once, in a test, and left nothing behind. `GoalModel.fit`
was called only from `tests/test_goals.py`; no script trained it, no artifact was
written, and the deployed API served a different inference path entirely. The
registry is what turns "a model was fitted at some point" into "this model,
trained on this data, by this code, scoring this".

What a registry entry has to carry, and why each field is not optional
=====================================================================

**Training window.** Two artifacts that differ only in which seasons they saw
are not comparable, and the difference is invisible in the boosters.

**Feature list, in order.** The single most dangerous mismatch available here.
`GoalModel.predict` selects columns by name, so a reordered frame is safe — but a
frame missing a column raises `KeyError` mid-request, and a frame with an *extra*
column silently trains a different model than the one that was evaluated.
Recording the list means the mismatch is caught at load, by `verify_features`,
rather than in whatever the caller was doing.

**Git SHA and data checksums.** "Reproduces from a clean checkout" is a claim
about code *and* inputs. A feature file rebuilt from a re-scraped source can
differ without any commit, and then a metric moves for reasons no diff explains.

**Every headline metric, per fold.** A single number from a single split cannot
distinguish a better model from a luckier one, which is exactly what a promotion
gate (Roadmap §11.2) has to decide.

Promotion is separate from registration
=======================================

Registering an artifact records that it exists and how it scored. Promoting it
says it is the one to serve. They are kept apart deliberately: automatic
retraining that promotes whatever it just built is a mechanism for shipping a
regression quietly, and the gate that prevents that needs somewhere to record a
model it declined to promote.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

REGISTRY_SCHEMA = 1
REGISTRY_NAME = "registry.json"


class RegistryError(RuntimeError):
    pass


@dataclass
class Entry:
    """One trained artifact, and everything needed to interpret its numbers."""

    version: str
    created_at: str
    git_sha: str
    git_dirty: bool
    #: Seasons the boosters were fitted on, and the seasons scored per fold.
    train_seasons: list[str]
    validation_seasons: list[str]
    #: NFR-10. Recorded so an entry that touched it is self-incriminating rather
    #: than merely absent from someone's memory.
    holdout_season: str
    holdout_touched: bool
    feature_columns: list[str]
    n_features: int
    n_train_rows: int
    params: dict[str, Any]
    #: sha256 of each input file, because a metric can move without a commit.
    input_checksums: dict[str, str]
    #: Per-fold and aggregate scores.
    metrics: dict[str, Any]
    promoted: bool = False
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def git_sha() -> tuple[str, bool]:
    """Current commit and whether the tree is dirty.

    A dirty tree is recorded rather than refused: a run that produced numbers is
    worth keeping even when it cannot be reproduced exactly, provided nobody can
    later mistake it for one that can.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        log.warning("registry: git unavailable; provenance will be incomplete")
        return "unknown", True
    return sha, bool(status)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def version_for(sha: str, *, now: datetime | None = None) -> str:
    """A version that sorts by date and points at its own source."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d")
    return f"goals-{stamp}-{sha[:8]}"


@dataclass
class Registry:
    path: Path
    entries: list[Entry] = field(default_factory=list)

    @classmethod
    def load(cls, directory: Path) -> Registry:
        path = Path(directory) / REGISTRY_NAME
        if not path.exists():
            return cls(path=path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema") != REGISTRY_SCHEMA:
            raise RegistryError(
                f"registry schema {raw.get('schema')!r} is not {REGISTRY_SCHEMA}"
            )
        return cls(
            path=path,
            entries=[Entry(**entry) for entry in raw.get("entries", [])],
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "schema": REGISTRY_SCHEMA,
                    "entries": [e.as_dict() for e in self.entries],
                },
                indent=2,
                sort_keys=False,
            )
            + "\n",
            encoding="utf-8",
        )

    def add(self, entry: Entry) -> None:
        if any(e.version == entry.version for e in self.entries):
            raise RegistryError(
                f"version {entry.version!r} is already registered; a version "
                "identifies one artifact and must not be reused"
            )
        self.entries.append(entry)

    def get(self, version: str) -> Entry:
        for entry in self.entries:
            if entry.version == version:
                return entry
        raise RegistryError(f"no registered model {version!r}")

    @property
    def promoted(self) -> Entry | None:
        """The artifact currently marked for serving, if any."""
        for entry in reversed(self.entries):
            if entry.promoted:
                return entry
        return None

    def promote(self, version: str) -> Entry:
        """Mark one artifact as the one to serve, demoting any other.

        Exactly one promoted entry at a time: two would make "which model
        produced this number" unanswerable, which is the question the whole
        registry exists to answer.
        """
        target = self.get(version)
        for entry in self.entries:
            entry.promoted = entry.version == version
        return target


def verify_features(expected: list[str], actual: list[str]) -> None:
    """Fail loudly when a frame does not match what the artifact was trained on.

    Both directions matter and they fail differently. Missing columns raise a
    `KeyError` deep inside a prediction; extra ones do not raise at all, and the
    caller quietly gets a model evaluated on a different feature set than the one
    it is now being fed.
    """
    if list(expected) == list(actual):
        return
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    if not missing and not extra:
        raise RegistryError(
            "feature columns match by name but not by order; the artifact was "
            f"trained on {expected[:3]}... and received {actual[:3]}..."
        )
    raise RegistryError(
        f"feature mismatch — missing {missing or 'none'}, unexpected {extra or 'none'}"
    )
