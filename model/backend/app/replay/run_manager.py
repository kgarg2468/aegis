from __future__ import annotations

from pathlib import Path


def next_run_id(runs_root: str | Path) -> str:
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)

    existing_nums: list[int] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("run") and name[3:].isdigit():
            existing_nums.append(int(name[3:]))

    next_num = max(existing_nums, default=0) + 1
    return f"run{next_num:03d}"


def create_run_dirs(runs_root: str | Path, run_id: str | None = None) -> dict[str, Path | str]:
    root = Path(runs_root)
    if run_id is None:
        run_id = next_run_id(root)

    run_dir = root / run_id
    train_dir = run_dir / "train"
    eval_dir = run_dir / "eval"
    replay_dir = run_dir / "replays"
    logs_dir = run_dir / "logs"

    for path in [run_dir, train_dir, eval_dir, replay_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "train_dir": train_dir,
        "eval_dir": eval_dir,
        "replay_dir": replay_dir,
        "logs_dir": logs_dir,
    }
