from pathlib import Path

from backend.app.replay.run_manager import create_run_dirs, next_run_id


def test_next_run_id_increments(tmp_path: Path):
    (tmp_path / "run001").mkdir()
    (tmp_path / "run003").mkdir()
    assert next_run_id(tmp_path) == "run004"


def test_create_run_dirs_layout(tmp_path: Path):
    out = create_run_dirs(tmp_path, run_id="run007")
    assert out["run_id"] == "run007"
    assert (tmp_path / "run007" / "train").exists()
    assert (tmp_path / "run007" / "eval").exists()
    assert (tmp_path / "run007" / "replays").exists()
    assert (tmp_path / "run007" / "logs").exists()
