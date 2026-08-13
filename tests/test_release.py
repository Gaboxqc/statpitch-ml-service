"""Model artifacts as release assets (Roadmap §10.1).

Boosters are gitignored, so a fresh checkout has a registry describing artifacts
nobody has. The weekly refresh used to work around that by retraining from
scratch every run — 2.5 minutes rebuilding a model that had not changed, purely
to get the file back.

`gh` is stubbed throughout. These tests are about the decisions around the
subprocess, not about GitHub: which version gets fetched, whether a local copy
wins, and whether a partial download can be mistaken for a complete model.
"""

from __future__ import annotations

import subprocess
import tarfile

import pytest

from statpitch.models import release


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("STATPITCH_MODELS", str(tmp_path / "models"))
    (tmp_path / "models").mkdir(parents=True)
    return tmp_path / "models"


def _artifact(models_dir, version: str):
    directory = models_dir / version
    directory.mkdir(parents=True)
    (directory / "model.json").write_text('{"schema": 1}', encoding="utf-8")
    (directory / "home.json").write_text("{}", encoding="utf-8")
    (directory / "away.json").write_text("{}", encoding="utf-8")
    return directory


def test_the_tag_is_derived_from_the_version():
    """A version identifies one artifact, so the tag inherits that guarantee."""
    assert release.tag_for("goals-20260813-abc") == "model-goals-20260813-abc"


def test_packing_keeps_the_version_directory(models_dir, tmp_path):
    """Otherwise extraction scatters three JSON files into models/."""
    _artifact(models_dir, "goals-1")
    archive = release.pack("goals-1", tmp_path)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "goals-1/model.json" in names


def test_packing_an_artifact_that_does_not_exist_says_what_to_run(models_dir, tmp_path):
    with pytest.raises(release.ReleaseError, match="scripts/train.py"):
        release.pack("goals-missing", tmp_path)


def test_a_local_artifact_is_not_replaced_by_a_download(models_dir, monkeypatch):
    """A developer who has just trained should not be overwritten."""
    _artifact(models_dir, "goals-1")

    def fail(*args, **kwargs):
        raise AssertionError("ensure_local reached the network with a local copy")

    monkeypatch.setattr(release, "fetch", fail)
    assert release.ensure_local("goals-1") == models_dir / "goals-1"


def test_a_missing_artifact_is_fetched(models_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(release, "fetch", lambda v: calls.append(v) or models_dir / v)
    release.ensure_local("goals-2")
    assert calls == ["goals-2"]


def test_a_directory_without_model_json_counts_as_missing(models_dir, monkeypatch):
    """A half-written download must not read as a complete model."""
    (models_dir / "goals-3").mkdir()
    calls = []
    monkeypatch.setattr(release, "fetch", lambda v: calls.append(v) or models_dir / v)
    release.ensure_local("goals-3")
    assert calls == ["goals-3"]


def test_a_missing_gh_binary_says_how_to_proceed(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(release.ReleaseError, match="scripts/train.py"):
        release._run(["gh", "release", "view", "model-x"])


def test_a_failing_gh_call_surfaces_its_stderr(monkeypatch):
    def failure(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="release not found")

    monkeypatch.setattr(subprocess, "run", failure)
    with pytest.raises(release.ReleaseError, match="release not found"):
        release._run(["gh", "release", "view", "model-x"])


def test_fetch_unpacks_into_place(models_dir, tmp_path, monkeypatch):
    source = _artifact(models_dir, "goals-4")
    archive = release.pack("goals-4", tmp_path)
    # Remove the local copy so fetch has something to restore.
    for path in source.iterdir():
        path.unlink()
    source.rmdir()

    def fake_run(args):
        if args[1:3] == ["release", "download"]:
            destination = args[args.index("--dir") + 1]
            (tmp_path / archive.name).replace(f"{destination}/{archive.name}")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(release, "_run", fake_run)
    result = release.fetch("goals-4")
    assert (result / "model.json").exists()


def test_fetch_rejects_an_archive_without_the_version_directory(
    models_dir, tmp_path, monkeypatch
):
    stray = tmp_path / "stray.tar.gz"
    with tarfile.open(stray, "w:gz") as tar:
        loose = tmp_path / "model.json"
        loose.write_text("{}", encoding="utf-8")
        tar.add(loose, arcname="model.json")

    def fake_run(args):
        destination = args[args.index("--dir") + 1]
        stray.replace(f"{destination}/{release.archive_name('goals-5')}")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(release, "_run", fake_run)
    with pytest.raises(release.ReleaseError, match="does not contain"):
        release.fetch("goals-5")
