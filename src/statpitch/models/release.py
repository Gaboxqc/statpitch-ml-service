"""Model artifacts as GitHub Release assets (Roadmap §10.1).

Trained boosters are gitignored at ~1.7 MB a run, because a weekly retrain would
add that to the repository every week. The registry entry — window, metrics,
feature list, checksums, commit — is the durable record and *is* committed.

The consequence, until now, was that a fresh checkout had a registry describing
artifacts nobody had. The weekly refresh worked around it by **retraining from
scratch every run**, ~2.5 minutes of runner time spent rebuilding a model that
had not changed, purely to get the file back. This replaces that: the artifact is
published once and downloaded when needed.

Why `gh` rather than an HTTP client
===================================

The CLI is already present on GitHub's runners, already authenticated there via
`GITHUB_TOKEN`, and handles private repositories without this module holding a
credential. Reaching for `requests` would mean building auth, pagination and
asset lookup by hand, and putting a token where a subprocess boundary currently
is. The cost is a hard dependency on an external binary, which is why every
failure path here says plainly what is missing and what to run instead.

Nothing on a serving path imports this. `requirements-serving.txt` excludes the
training stack and `tests/test_deployment.py` enforces that; artifacts reach the
deployed image at build time, not at request time.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from statpitch import paths

log = logging.getLogger(__name__)

#: One release per artifact, tagged by its version. A version identifies exactly
#: one model (`Registry.add` refuses to reuse one), so the tag inherits that
#: guarantee rather than needing its own.
TAG_PREFIX = "model-"

#: How long to wait on the network before giving up, in seconds.
TIMEOUT = 300


class ReleaseError(RuntimeError):
    pass


def tag_for(version: str) -> str:
    return f"{TAG_PREFIX}{version}"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, check=True, timeout=TIMEOUT
        )
    except FileNotFoundError as exc:
        raise ReleaseError(
            "the GitHub CLI (`gh`) is not installed, so model artifacts cannot be "
            "published or fetched. Install it, or rebuild locally with "
            "`python scripts/train.py`."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ReleaseError(f"`{' '.join(args[:2])}` timed out after {TIMEOUT}s") from exc
    except subprocess.CalledProcessError as exc:
        raise ReleaseError(
            f"`{' '.join(args[:2])}` failed ({exc.returncode}): "
            f"{(exc.stderr or '').strip()}"
        ) from exc


def archive_name(version: str) -> str:
    return f"{version}.tar.gz"


def pack(version: str, destination: Path) -> Path:
    """Tar a model directory for upload."""
    source = paths.models_dir() / version
    if not source.is_dir():
        raise ReleaseError(
            f"no artifact at {source}. Train it first with `python scripts/train.py`."
        )
    archive = destination / archive_name(version)
    with tarfile.open(archive, "w:gz") as tar:
        # `arcname=version` keeps the version directory inside the archive, so
        # extraction lands at models/<version>/ rather than scattering three JSON
        # files into models/.
        tar.add(source, arcname=version)
    return archive


def publish(version: str, *, notes: str = "") -> str:
    """Upload a trained artifact as a release asset, returning the tag."""
    tag = tag_for(version)
    with tempfile.TemporaryDirectory() as workspace:
        archive = pack(version, Path(workspace))
        existing = subprocess.run(
            ["gh", "release", "view", tag], capture_output=True, text=True
        )
        if existing.returncode == 0:
            # A version identifies one artifact, so re-uploading means the same
            # model rebuilt. `--clobber` keeps that idempotent instead of failing
            # a rerun; it cannot silently replace a *different* model, because a
            # different model would have a different version.
            _run(["gh", "release", "upload", tag, str(archive), "--clobber"])
            log.info("replaced the asset on existing release %s", tag)
        else:
            _run([
                "gh", "release", "create", tag, str(archive),
                "--title", f"Goal model {version}",
                "--notes", notes or f"Trained artifact {version}.",
            ])
            log.info("created release %s", tag)
    return tag


def fetch(version: str) -> Path:
    """Download and unpack an artifact, returning its local directory."""
    target = paths.models_dir() / version
    tag = tag_for(version)
    with tempfile.TemporaryDirectory() as workspace:
        _run([
            "gh", "release", "download", tag,
            "--pattern", archive_name(version), "--dir", workspace,
        ])
        archive = Path(workspace) / archive_name(version)
        if not archive.exists():
            raise ReleaseError(f"release {tag} has no asset {archive_name(version)}")

        # Extract to a staging directory and move into place, so an interrupted
        # download cannot leave a half-written model that later looks complete.
        staging = Path(workspace) / "unpacked"
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(staging, filter="data")
        unpacked = staging / version
        if not unpacked.is_dir():
            raise ReleaseError(
                f"{archive_name(version)} does not contain a {version}/ directory"
            )
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(unpacked), str(target))
    log.info("fetched %s from release %s", version, tag)
    return target


def ensure_local(version: str) -> Path:
    """The artifact's directory, downloading it only if it is not already here.

    Local first on purpose: a developer who has just trained a model should not
    have their copy replaced by whatever was last published, and a run that needs
    no network should not require one.
    """
    target = paths.models_dir() / version
    if (target / "model.json").exists():
        return target
    log.info("%s is not present locally; fetching from its release", version)
    return fetch(version)
