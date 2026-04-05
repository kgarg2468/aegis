from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from backend.app import main


def test_iter_run_dirs_includes_episode_prefix(tmp_path: Path, monkeypatch) -> None:
    runs = tmp_path / "runs"
    (runs / "run001").mkdir(parents=True)
    (runs / "ep001").mkdir()
    (runs / "notes").mkdir()

    monkeypatch.chdir(tmp_path)

    discovered = [entry.name for entry in main._iter_run_dirs()]
    assert discovered == ["ep001", "run001"]


def test_find_replay_bundle_resolves_episode_runs(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "runs" / "ep002" / "replays" / "replay_hero_01"
    bundle.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)

    found = main._find_replay_bundle("replay_hero_01")
    assert found == bundle
