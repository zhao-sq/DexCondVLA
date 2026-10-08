from pathlib import Path

from scripts.filter_bench2dex_collect_config import filter_cameras
from scripts.filter_bench2dex_collect_config import resolve_relative_paths


def test_filter_cameras_preserves_requested_order_and_other_settings() -> None:
    config = {
        "capture": {"fps": 20},
        "camera_pixel": {"a": [1, 1], "b": [2, 2], "c": [3, 3]},
        "camera_list": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
    }

    result = filter_cameras(config, ["c", "a"])

    assert result["capture"] == {"fps": 20}
    assert result["camera_list"] == [{"id": "c"}, {"id": "a"}]
    assert result["camera_pixel"] == {"c": [3, 3], "a": [1, 1]}


def test_resolve_relative_scene_background_uri_uses_source_directory() -> None:
    config = {"scene_background": {"uri": "../../assets/scene.usda"}}

    result = resolve_relative_paths(config, Path("/repo/configs/collect/default.yaml"))

    assert result["scene_background"]["uri"] == "/repo/assets/scene.usda"
