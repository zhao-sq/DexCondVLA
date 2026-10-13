import numpy as np

from openpi.training import config


def test_task26_dual_expert_arm_config_and_projection(tmp_path):
    train_config = config.get_config("pi05_bench2dex_task26_dual_expert_arm")

    assert train_config.model.action_dim == 32
    assert train_config.model.action_horizon == 50
    assert train_config.data.arm_only is True
    assert train_config.data.assets.asset_id == "bench2dex_task26_dual_expert_arm_activity"
    data_config = train_config.data.create(tmp_path, train_config.model)
    assert data_config.dataset_processor_options["source_action_dim"] == 58

    state = np.arange(58, dtype=np.float32)
    actions = np.stack((state, state + 100.0))
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    sample = {
        "state": state,
        "actions": actions,
        "left_gripper_active_actions": np.asarray([0.0, 1.0], dtype=np.float32),
        "right_gripper_active_actions": np.asarray([1.0, 0.0], dtype=np.float32),
        "prompt": "move both arms",
        "images": {
            "cam_overhead": image,
            "cam_wrist_left": image,
            "cam_wrist_right": image,
        },
    }
    for transform in data_config.data_transforms.inputs:
        sample = transform(sample)

    arm_indices = (0, 1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12, 13)
    np.testing.assert_array_equal(sample["state"], state[list(arm_indices)])
    expected_actions = np.column_stack(
        (
            actions[:, list(arm_indices[:7])],
            np.asarray([0.0, 1.0], dtype=np.float32),
            actions[:, list(arm_indices[7:])],
            np.asarray([1.0, 0.0], dtype=np.float32),
        )
    )
    np.testing.assert_array_equal(sample["actions"], expected_actions)

    for transform in data_config.model_transforms.inputs:
        sample = transform(sample)
    assert sample["state"].shape == (32,)
    assert sample["actions"].shape == (2, 32)
    np.testing.assert_array_equal(sample["state"][:14], state[list(arm_indices)])
    np.testing.assert_array_equal(sample["actions"][:, :16], expected_actions)
    np.testing.assert_array_equal(sample["state"][14:], np.zeros(18, dtype=np.float32))
    np.testing.assert_array_equal(sample["actions"][:, 16:], np.zeros((2, 16), dtype=np.float32))

    output = {"actions": np.arange(2 * 32, dtype=np.float32).reshape(2, 32)}
    for transform in data_config.data_transforms.outputs:
        output = transform(output)
    assert output["actions"].shape == (2, 16)
    expected_output = np.arange(2 * 32, dtype=np.float32).reshape(2, 32)[:, :16]
    expected_output[:, (7, 15)] = 1.0
    np.testing.assert_array_equal(output["actions"], expected_output)


def test_task26_action_expert_config():
    train_config = config.get_config("pi05_bench2dex_task26_action_expert")

    assert train_config.model.action_dim == 58
    assert train_config.model.action_horizon == 50
    assert train_config.batch_size == 64
    assert train_config.num_train_steps == 20_000
    assert train_config.data.dataset_processor == "bench2dex_hdf5"
    assert train_config.data.camera_names == (
        "cam_overhead",
        "cam_wrist_left",
        "cam_wrist_right",
    )
    assert train_config.data.action_representation == "absolute"
    assert str(train_config.checkpoint_base_dir) == "/mnt/ssd1/szhao/DexTool/openpi_checkpoints"
    assert str(train_config.assets_base_dir) == "/mnt/ssd1/szhao/DexTool/openpi_assets"


def test_task26_action_expert_25k_config():
    train_config = config.get_config("pi05_bench2dex_task26_action_expert_25k")

    assert train_config.model.action_dim == 58
    assert train_config.model.action_horizon == 50
    assert train_config.batch_size == 64
    assert train_config.num_train_steps == 25_000
    assert train_config.lr_schedule.decay_steps == 25_000
    assert train_config.data.action_representation == "absolute"
    assert train_config.data.assets.assets_dir == (
        "/mnt/ssd1/szhao/DexTool/openpi_assets/pi05_bench2dex_task26_action_expert"
    )
    assert train_config.data.assets.asset_id == "bench2dex_task26"
    assert str(train_config.checkpoint_base_dir) == "/mnt/ssd1/szhao/DexTool/ckp"


def test_task26_delta_action_expert_config(tmp_path):
    train_config = config.get_config("pi05_bench2dex_task26_action_expert_state_delta")

    assert train_config.model.action_dim == 58
    assert train_config.model.action_horizon == 50
    assert train_config.data.action_representation == "state_delta"
    assert train_config.data.assets.asset_id == "bench2dex_task26_state_delta"
    data_config = train_config.data.create(tmp_path, train_config.model)
    assert data_config.dataset_processor_options["include_action_states"] is False


def test_task26_state_delta_output_stays_in_delta_space(tmp_path):
    train_config = config.get_config("pi05_bench2dex_task26_action_expert_state_delta")
    data_config = train_config.data.create(tmp_path, train_config.model)
    state = np.arange(58, dtype=np.float32)
    absolute_actions = np.stack((state + 1.0, state - 2.0))
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    sample = {
        "state": state.copy(),
        "actions": absolute_actions.copy(),
        "images": {
            "cam_overhead": image,
            "cam_wrist_left": image,
            "cam_wrist_right": image,
        },
    }

    for transform in data_config.data_transforms.inputs:
        sample = transform(sample)

    np.testing.assert_allclose(sample["actions"][0], np.ones(58, dtype=np.float32))
    np.testing.assert_allclose(sample["actions"][1], np.full(58, -2.0, dtype=np.float32))

    output = {"state": state.copy(), "actions": sample["actions"].copy()}
    for transform in data_config.data_transforms.outputs:
        output = transform(output)

    np.testing.assert_allclose(output["actions"], sample["actions"])


def test_task26_aligned_delta_output_stays_in_delta_space(tmp_path):
    train_config = config.get_config("pi05_bench2dex_task26_action_expert_aligned_delta")
    assert train_config.data.action_representation == "aligned_delta"
    assert train_config.data.assets.asset_id == "bench2dex_task26_aligned_delta"
    data_config = train_config.data.create(tmp_path, train_config.model)
    assert data_config.dataset_processor_options["include_action_states"] is True
    state = np.arange(58, dtype=np.float32)
    absolute_actions = np.stack((state + 1.0, state + 3.0, state - 2.0))
    action_states = np.stack((state, state + 1.0, state + 3.0))
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    sample = {
        "state": state.copy(),
        "actions": absolute_actions.copy(),
        "action_states": action_states,
        "images": {
            "cam_overhead": image,
            "cam_wrist_left": image,
            "cam_wrist_right": image,
        },
    }

    for transform in data_config.data_transforms.inputs:
        sample = transform(sample)

    np.testing.assert_allclose(sample["actions"][0], np.full(58, 1.0, dtype=np.float32))
    np.testing.assert_allclose(sample["actions"][1], np.full(58, 2.0, dtype=np.float32))
    np.testing.assert_allclose(sample["actions"][2], np.full(58, -5.0, dtype=np.float32))
    assert "action_states" not in sample

    output = {"state": state.copy(), "actions": sample["actions"].copy()}
    for transform in data_config.data_transforms.outputs:
        output = transform(output)

    np.testing.assert_allclose(output["actions"], sample["actions"])
