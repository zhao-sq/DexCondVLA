"""Train the first shared left/right Task26 Hand DiT with frozen DINOv3 features."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch

from openpi.models_pytorch.dinov3_wrist_encoder import Dinov3WristEncoder
from openpi.models_pytorch.hand_dit import HandDiT
from openpi.models_pytorch.hand_dit import HandDiTConfig
from openpi.policies.dexterous_action_layout import TASK26_BIMANUAL_LAYOUT
from openpi.policies.dual_expert_hand_policy import DualExpertNormalization
from openpi.policies.dual_expert_hand_policy import load_hand_dit_checkpoint
from openpi.policies.dual_expert_hand_policy import save_hand_dit_checkpoint
from openpi.training import bench2dex_dataset
from openpi.training import code_snapshot


def _repeat_loader(loader: torch.utils.data.DataLoader):
    while True:
        yield from loader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--storage-root",
        type=pathlib.Path,
        default=pathlib.Path("/mnt/ssd1/szhao/DexTool"),
        help="Allowed root for training outputs; set this explicitly on remote hosts.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=pathlib.Path,
        default=pathlib.Path("/mnt/ssd1/szhao/DexTool/bench2dex/task26/replay-generalization"),
    )
    parser.add_argument(
        "--dinov3-repo",
        type=pathlib.Path,
        default=pathlib.Path("/mnt/ssd1/szhao/DexTool/dinov3"),
    )
    parser.add_argument(
        "--dinov3-checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("/mnt/ssd1/szhao/LeveFD/dinov3_vitb16.pth"),
    )
    parser.add_argument(
        "--norm-stats",
        type=pathlib.Path,
        default=pathlib.Path(
            "/mnt/ssd1/szhao/DexTool/openpi_assets/pi05_bench2dex_task26_action_expert/bench2dex_task26/norm_stats.json"
        ),
    )
    parser.add_argument("--registry", type=pathlib.Path, default=pathlib.Path("TRAINING_CHECKPOINT_REGISTRY.md"))
    parser.add_argument("--resume", type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--max-vision-tokens", type=int, default=64)
    parser.add_argument("--disable-arm-condition", action="store_true")
    parser.add_argument("--disable-vision-condition", action="store_true")
    parser.add_argument("--disable-state-condition", action="store_true")
    return parser.parse_args()


def _validate_launch(args: argparse.Namespace) -> None:
    if min(args.steps, args.batch_size, args.log_every, args.save_every) <= 0:
        raise ValueError("--steps, --batch-size, --log-every, and --save-every must be positive")
    output = args.output_dir.expanduser().resolve()
    storage_root = args.storage_root.expanduser().resolve()
    if not output.is_relative_to(storage_root):
        raise ValueError(f"Training output must stay under {storage_root}, got {output}")
    registry_text = args.registry.read_text(encoding="utf-8")
    if f"## `{args.experiment_name}`" not in registry_text:
        raise RuntimeError(f"Register {args.experiment_name!r} in {args.registry} before starting this training run")
    if args.resume is None and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty new-run directory: {output}")


def _normalize_training_batch(
    state: torch.Tensor,
    actions: torch.Tensor,
    left_gripper_active_actions: torch.Tensor,
    right_gripper_active_actions: torch.Tensor,
    normalization: DualExpertNormalization,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layout = TASK26_BIMANUAL_LAYOUT
    left_arm = normalization.normalize(layout.gather(actions, "left", "arm"), kind="action", indices=layout.left_arm)
    right_arm = normalization.normalize(layout.gather(actions, "right", "arm"), kind="action", indices=layout.right_arm)
    left_hand = normalization.normalize(layout.gather(actions, "left", "hand"), kind="action", indices=layout.left_hand)
    right_hand = normalization.normalize(
        layout.gather(actions, "right", "hand"), kind="action", indices=layout.right_hand
    )
    left_state = normalization.normalize(layout.gather(state, "left", "hand"), kind="state", indices=layout.left_hand)
    right_state = normalization.normalize(
        layout.gather(state, "right", "hand"), kind="state", indices=layout.right_hand
    )
    left_activity = left_gripper_active_actions.to(dtype=actions.dtype).unsqueeze(-1)
    right_activity = right_gripper_active_actions.to(dtype=actions.dtype).unsqueeze(-1)
    left_condition = torch.cat((left_arm, left_activity), dim=-1)
    right_condition = torch.cat((right_arm, right_activity), dim=-1)
    arm = torch.cat((left_condition, right_condition), dim=0)
    hand = torch.cat((left_hand, right_hand), dim=0)
    hand_state = torch.cat((left_state, right_state), dim=0)
    side_ids = torch.cat(
        (
            torch.zeros(state.shape[0], device=state.device, dtype=torch.long),
            torch.ones(state.shape[0], device=state.device, dtype=torch.long),
        )
    )
    return arm, hand, hand_state, side_ids


def main() -> None:
    args = _parse_args()
    _validate_launch(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    launch_config = {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(args).items()}
    (output_dir / "launch_config.json").write_text(
        json.dumps(launch_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Python interpreter: {sys.executable}", flush=True)
    print(f"Device: {device}", flush=True)
    code_snapshot.save_code_snapshot(
        output_dir,
        config=launch_config,
        launch_kind="resume" if args.resume else "new",
    )

    normalization = DualExpertNormalization.from_openpi_json(args.norm_stats, use_quantiles=True)
    if args.resume:
        model, restored_normalization, checkpoint = load_hand_dit_checkpoint(args.resume, device=device)
        if restored_normalization != normalization:
            raise ValueError("Resume checkpoint normalization does not match --norm-stats")
        start_step = int(checkpoint["step"])
    else:
        config = HandDiTConfig(
            action_horizon=50,
            arm_dim=TASK26_BIMANUAL_LAYOUT.arm_dim + 1,
            hand_dim=TASK26_BIMANUAL_LAYOUT.hand_dim,
            hand_state_dim=TASK26_BIMANUAL_LAYOUT.hand_dim,
            vision_feature_dim=Dinov3WristEncoder.feature_dim,
            model_dim=args.model_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_vision_tokens=args.max_vision_tokens,
            condition_on_arm=not args.disable_arm_condition,
            condition_on_vision=not args.disable_vision_condition,
            condition_on_state=not args.disable_state_condition,
        )
        model = HandDiT(config).to(device)
        checkpoint = {}
        start_step = 0
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    wrist_encoder = (
        Dinov3WristEncoder(args.dinov3_repo, args.dinov3_checkpoint).to(device)
        if model.config.condition_on_vision
        else None
    )
    dataset = bench2dex_dataset.Bench2DexHdf5Dataset(
        bench2dex_dataset.list_episode_files(str(args.dataset_dir)),
        action_horizon=model.config.action_horizon,
        action_dim=TASK26_BIMANUAL_LAYOUT.action_dim,
        include_gripper_activity=True,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        generator=generator,
    )
    batches = _repeat_loader(loader)
    metadata = {
        "experiment_name": args.experiment_name,
        "layout": TASK26_BIMANUAL_LAYOUT.name,
        "arm_condition_source": "ground_truth",
        "dinov3_repo": str(args.dinov3_repo.resolve()),
        "dinov3_checkpoint": str(args.dinov3_checkpoint.resolve()),
        "dinov3_frozen": True,
        "vision_condition_enabled": model.config.condition_on_vision,
    }

    running_loss = 0.0
    last_log = time.monotonic()
    for step in range(start_step, args.steps):
        batch = next(batches)
        state = batch["state"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        arm, target_hand, hand_state, side_ids = _normalize_training_batch(
            state,
            actions,
            batch["left_gripper_active_actions"].to(device, non_blocking=True),
            batch["right_gripper_active_actions"].to(device, non_blocking=True),
            normalization,
        )
        if wrist_encoder is not None:
            wrist_images = torch.cat(
                (
                    batch["images"]["cam_wrist_left"],
                    batch["images"]["cam_wrist_right"],
                ),
                dim=0,
            ).to(device, non_blocking=True)
            with torch.inference_mode():
                wrist_tokens = wrist_encoder(wrist_images)
        else:
            wrist_tokens = None

        optimizer.zero_grad(set_to_none=True)
        loss = model.flow_matching_loss(target_hand, arm, wrist_tokens, hand_state, side_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += float(loss.detach())

        completed_step = step + 1
        if completed_step % args.log_every == 0:
            elapsed = time.monotonic() - last_log
            print(
                f"step={completed_step} loss={running_loss / args.log_every:.6f} "
                f"steps_per_second={args.log_every / elapsed:.3f}",
                flush=True,
            )
            running_loss = 0.0
            last_log = time.monotonic()
        if completed_step % args.save_every == 0 or completed_step == args.steps:
            checkpoint_path = output_dir / f"step_{completed_step:08d}.pt"
            save_hand_dit_checkpoint(
                checkpoint_path,
                model,
                normalization,
                step=completed_step,
                optimizer_state=optimizer.state_dict(),
                metadata=metadata,
            )
            (output_dir / "LATEST").write_text(checkpoint_path.name + "\n", encoding="utf-8")
            print(f"checkpoint={checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
