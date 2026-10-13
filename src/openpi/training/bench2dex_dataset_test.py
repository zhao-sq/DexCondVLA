import cv2
import h5py
import numpy as np

from openpi.policies import bench2dex_policy
from openpi.training import bench2dex_dataset


def _write_episode(path):
    frames = 8
    names = np.asarray([f"joint_{i}".encode() for i in range(58)])
    image = np.full((32, 48, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    jpeg = encoded.tobytes()
    with h5py.File(path, "w") as f:
        robot = f.create_group("robot")
        robot.create_dataset("qpos", data=np.arange(frames * 58, dtype=np.float32).reshape(frames, 58))
        robot.create_dataset("joint_names", data=names)
        action = f.create_group("action")
        action.create_dataset("commanded", data=np.ones((frames, 58), dtype=np.float32))
        action.create_dataset("action_valid", data=np.asarray([False, True, True, True, True, True, True, True]))
        action.create_dataset("action_names", data=names)
        f.create_dataset("frame_valid", data=np.ones(frames, dtype=bool))
        time = f.create_group("time")
        time.create_dataset("sim_step", data=np.arange(frames) * 10)
        meta = f.create_group("meta")
        meta.create_dataset("homing_start_sim_step", data=70)
        meta.create_dataset("instruction", data="place the cans on the tray")
        labels = f.create_group("labels")
        left_active = np.asarray([False, False, True, True, True, False, False, False])
        right_active = np.asarray([False, False, False, False, True, True, False, False])
        labels.create_dataset("left_gripper_active", data=left_active)
        labels.create_dataset("right_gripper_active", data=right_active)
        labels.create_dataset("gripper_active", data=np.logical_or(left_active, right_active))
        cameras = f.create_group("cameras")
        dtype = h5py.vlen_dtype(np.dtype("uint8"))
        for camera in bench2dex_dataset.DEFAULT_CAMERA_NAMES:
            rgb = cameras.create_group(camera).create_dataset("rgb", (frames,), dtype=dtype)
            for index in range(frames):
                rgb[index] = np.frombuffer(jpeg, dtype=np.uint8)


def test_bench2dex_hdf5_and_policy_transform(tmp_path):
    episode = tmp_path / "episode_000000.hdf5"
    _write_episode(episode)
    dataset = bench2dex_dataset.Bench2DexHdf5Dataset([str(episode)], action_horizon=3)

    assert len(dataset) == 4
    raw = dataset[0]
    assert raw["state"].shape == (58,)
    assert raw["actions"].shape == (3, 58)
    assert set(raw["images"]) == set(bench2dex_dataset.DEFAULT_CAMERA_NAMES)

    processed = bench2dex_policy.Bench2DexInputs()(raw)
    assert processed["state"].shape == (58,)
    assert processed["actions"].shape == (3, 58)
    assert set(processed["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(image.shape == (32, 48, 3) for image in processed["image"].values())


def test_bench2dex_processor_owns_source_adaptation(tmp_path):
    episode = tmp_path / "episode_000000.hdf5"
    _write_episode(episode)
    processor = bench2dex_dataset.Bench2DexDatasetProcessor(source=str(tmp_path))

    dataset = processor.create_dataset(action_horizon=3, action_dim=58)

    assert isinstance(dataset, bench2dex_dataset.Bench2DexHdf5Dataset)
    assert len(dataset) == 4


def test_bench2dex_loads_gripper_activity_chunk(tmp_path):
    episode = tmp_path / "episode_000000.hdf5"
    _write_episode(episode)
    dataset = bench2dex_dataset.Bench2DexHdf5Dataset([str(episode)], action_horizon=3, include_gripper_activity=True)

    raw = dataset[0]

    np.testing.assert_array_equal(raw["left_gripper_active_actions"], np.asarray([0.0, 1.0, 1.0]))
    np.testing.assert_array_equal(raw["right_gripper_active_actions"], np.asarray([0.0, 0.0, 0.0]))


def test_bench2dex_aligned_action_states(tmp_path):
    episode = tmp_path / "episode_000000.hdf5"
    _write_episode(episode)
    dataset = bench2dex_dataset.Bench2DexHdf5Dataset([str(episode)], action_horizon=3, include_action_states=True)

    raw = dataset[0]
    assert raw["action_states"].shape == (3, 58)
    np.testing.assert_array_equal(raw["action_states"][0], raw["state"])

    processed = bench2dex_policy.Bench2DexInputs()(raw)
    assert processed["action_states"].shape == processed["actions"].shape
