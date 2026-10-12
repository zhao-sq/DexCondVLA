"""Append ``_success`` to videos that pass the relaxed Task26 rescore."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def _atomic_write(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        handle.write(content)
    os.replace(temporary, path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rollout_dir = args.result_root / "rollouts"
    rescore_path = args.result_root / "rescore_without_object_upright" / "per_episode.jsonl"
    manifest_json = rollout_dir / "rollout_manifest.json"
    manifest_csv = rollout_dir / "rollout_manifest.csv"

    rescore_rows = _load_jsonl(rescore_path)
    manifest: list[dict[str, Any]] = json.loads(manifest_json.read_text(encoding="utf-8"))
    if len(rescore_rows) != len(manifest):
        raise RuntimeError(f"Rescore/manifest length mismatch: {len(rescore_rows)} != {len(manifest)}")

    relaxed_by_episode = {
        (row["channel"], int(row["episode_index"])): bool(row["success_without_object_upright"]) for row in rescore_rows
    }
    renames: list[dict[str, Any]] = []
    for row in manifest:
        key = (row["channel"], int(row["episode"]))
        if key not in relaxed_by_episode:
            raise RuntimeError(f"Manifest entry has no relaxed rescore: {key}")
        relaxed_success = relaxed_by_episode[key]
        row["relaxed_success_without_object_upright"] = relaxed_success
        if not relaxed_success:
            continue

        original_name = row["video"]
        original = rollout_dir / original_name
        if original.stem.endswith("_success"):
            renamed = original
        else:
            renamed = original.with_name(f"{original.stem}_success{original.suffix}")

        if original != renamed:
            if original.is_file() and renamed.exists():
                raise FileExistsError(f"Both original and renamed video exist: {original}, {renamed}")
            if not original.exists() and not renamed.is_file():
                raise FileNotFoundError(f"Neither original nor renamed video exists: {original}, {renamed}")
            renames.append(
                {
                    "rollout": int(row["rollout"]),
                    "channel": row["channel"],
                    "episode": int(row["episode"]),
                    "old_video": original.name,
                    "new_video": renamed.name,
                }
            )
            if not args.dry_run and original.exists():
                original.rename(renamed)
        row["video"] = renamed.name

    for item in renames:
        print(
            f"rollout {item['rollout']:03d} {item['channel']} episode {item['episode']:03d}: "
            f"{item['old_video']} -> {item['new_video']}"
        )
    print(f"relaxed successes: {sum(relaxed_by_episode.values())}/{len(relaxed_by_episode)}")

    if args.dry_run:
        return

    _atomic_write(manifest_json, json.dumps(manifest, indent=2) + "\n")
    fieldnames = list(manifest[0])
    with tempfile.NamedTemporaryFile("w", dir=rollout_dir, delete=False, newline="", encoding="utf-8") as handle:
        temporary_csv = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)
    os.replace(temporary_csv, manifest_csv)
    _atomic_write(
        rollout_dir / "relaxed_success_video_renames.json",
        json.dumps(renames, indent=2) + "\n",
    )


if __name__ == "__main__":
    main()
