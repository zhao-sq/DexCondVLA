from openpi.training import config


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
    assert str(train_config.checkpoint_base_dir) == "/mnt/ssd1/szhao/DexTool/openpi_checkpoints"
    assert str(train_config.assets_base_dir) == "/mnt/ssd1/szhao/DexTool/openpi_assets"
