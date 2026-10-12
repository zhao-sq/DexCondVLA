from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from openpi.training import code_snapshot


@dataclasses.dataclass
class _TestConfig:
    name: str = "test_config"
    batch_size: int = 4


def test_save_code_snapshot(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    (repository / "src").mkdir(parents=True)
    (repository / "scripts").mkdir()
    (repository / "src" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "scripts" / "train.py").write_text("print('train')\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    (repository / ".pytest_cache").mkdir()
    (repository / ".pytest_cache" / "ignored").write_text("ignored", encoding="utf-8")

    checkpoint_dir = tmp_path / "checkpoints" / "experiment"
    snapshot = code_snapshot.save_code_snapshot(
        checkpoint_dir,
        config=_TestConfig(),
        launch_kind="new",
        repository_root=repository,
    )

    assert (snapshot / "repository" / "src" / "model.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert (snapshot / "repository" / "scripts" / "train.py").is_file()
    assert not (snapshot / "repository" / ".pytest_cache").exists()
    assert json.loads((snapshot / "train_config.json").read_text(encoding="utf-8"))["batch_size"] == 4
    metadata = json.loads((snapshot / "snapshot_metadata.json").read_text(encoding="utf-8"))
    assert metadata["launch_kind"] == "new"
    assert metadata["python_executable"]
    assert (checkpoint_dir / "code_snapshots" / "LATEST").read_text(encoding="utf-8").strip() == snapshot.name
