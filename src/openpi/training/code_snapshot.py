"""Save the exact training code alongside an experiment's checkpoints."""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any

_FALLBACK_SOURCE_DIRS = ("docs", "examples", "packages", "scripts", "src")
_FALLBACK_ROOT_FILES = (
    ".gitignore",
    ".gitmodules",
    ".pre-commit-config.yaml",
    ".python-version",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "LICENSE_GEMMA.txt",
    "README.md",
    "TRAINING_CHECKPOINT_REGISTRY.md",
    "pyproject.toml",
    "uv.lock",
)
_IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "node_modules",
    "wandb",
}


def _run_git(repository_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _git_files(repository_root: Path) -> list[Path] | None:
    try:
        output = subprocess.run(
            ["git", "-C", str(repository_root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return [Path(item.decode()) for item in output.split(b"\0") if item]


def _copy_repository(repository_root: Path, destination: Path) -> int:
    git_files = _git_files(repository_root)
    if git_files is None:
        candidates = [repository_root / name for name in (*_FALLBACK_SOURCE_DIRS, *_FALLBACK_ROOT_FILES)]
        files = [
            path
            for candidate in candidates
            if candidate.exists()
            for path in ([candidate] if candidate.is_file() else candidate.rglob("*"))
        ]
        relative_files = [path.relative_to(repository_root) for path in files if path.is_file()]
    else:
        relative_files = git_files

    copied = 0
    for relative_path in relative_files:
        if any(part in _IGNORED_PARTS for part in relative_path.parts):
            continue
        source = repository_root / relative_path
        if not source.is_file() and not source.is_symlink():
            # Git submodules appear as directory entries. Their commit IDs are
            # captured separately in git_submodules.txt.
            continue
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, target)
        copied += 1
    return copied


def _write_git_metadata(repository_root: Path, destination: Path) -> None:
    commands = {
        "git_head.txt": ("rev-parse", "HEAD"),
        "git_status.txt": ("status", "--short", "--branch"),
        "git_diff.patch": ("diff", "--binary", "HEAD"),
        "git_submodules.txt": ("submodule", "status", "--recursive"),
    }
    for filename, args in commands.items():
        try:
            content = _run_git(repository_root, *args)
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            content = f"Unavailable: {error!r}\n"
        (destination / filename).write_text(content, encoding="utf-8")


def save_code_snapshot(
    checkpoint_dir: Path | str,
    *,
    config: Any,
    launch_kind: str,
    repository_root: Path | str | None = None,
) -> Path:
    """Atomically save source, config, and Git metadata for one training launch."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    repository_root = (
        Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[3]
    )
    snapshots_dir = checkpoint_dir / "code_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now(datetime.UTC)
    snapshot_name = f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}_{os.getpid()}"
    final_dir = snapshots_dir / snapshot_name
    temporary_dir = Path(tempfile.mkdtemp(prefix=".tmp_", dir=snapshots_dir))
    try:
        source_dir = temporary_dir / "repository"
        source_dir.mkdir()
        copied_files = _copy_repository(repository_root, source_dir)
        if copied_files == 0:
            raise RuntimeError(f"No source files found under repository root {repository_root}")

        config_data = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else config
        (temporary_dir / "train_config.json").write_text(
            json.dumps(config_data, indent=2, sort_keys=True, default=repr) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "checkpoint_dir": str(checkpoint_dir),
            "copied_file_count": copied_files,
            "created_at_utc": now.isoformat(),
            "launch_kind": launch_kind,
            "pid": os.getpid(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "repository_root": str(repository_root),
        }
        (temporary_dir / "snapshot_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_git_metadata(repository_root, temporary_dir)
        temporary_dir.rename(final_dir)
        (snapshots_dir / "LATEST").write_text(snapshot_name + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    logging.info("Saved training code snapshot with %d files to %s", copied_files, final_dir)
    return final_dir
