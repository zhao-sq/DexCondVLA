"""Create a Bench2Dex collect config containing only requested cameras."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import yaml


def resolve_relative_paths(config: dict, source_path: Path) -> dict:
    scene_background = config.get("scene_background")
    if isinstance(scene_background, dict):
        uri = scene_background.get("uri")
        if isinstance(uri, str) and uri and "://" not in uri:
            uri_path = Path(uri)
            if not uri_path.is_absolute():
                scene_background["uri"] = str((source_path.parent / uri_path).resolve())
    return config


def filter_cameras(config: dict, camera_ids: list[str]) -> dict:
    requested = tuple(dict.fromkeys(camera_ids))
    if not requested:
        raise ValueError("At least one camera id is required")

    camera_list = config.get("camera_list")
    if not isinstance(camera_list, list):
        raise ValueError("Bench2Dex collect config is missing camera_list")

    cameras_by_id = {
        str(camera.get("id")): camera for camera in camera_list if isinstance(camera, dict) and camera.get("id")
    }
    missing = [camera_id for camera_id in requested if camera_id not in cameras_by_id]
    if missing:
        raise ValueError(f"Requested cameras are absent from collect config: {missing}")

    config["camera_list"] = [cameras_by_id[camera_id] for camera_id in requested]
    camera_pixel = config.get("camera_pixel")
    if isinstance(camera_pixel, dict):
        config["camera_pixel"] = {
            camera_id: camera_pixel[camera_id] for camera_id in requested if camera_id in camera_pixel
        }
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", action="append", dest="cameras", required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Bench2Dex collect config must be a mapping: {args.input}")

    config = resolve_relative_paths(config, args.input.resolve())
    filtered = filter_cameras(config, args.cameras)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        yaml.safe_dump(filtered, handle, sort_keys=False)
        temporary_path = Path(handle.name)
    temporary_path.replace(args.output)
    selected = [camera["id"] for camera in filtered["camera_list"]]
    print(f"Wrote {args.output} with cameras: {selected}")


if __name__ == "__main__":
    main()
