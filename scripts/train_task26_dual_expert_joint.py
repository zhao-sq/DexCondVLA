"""Jointly train the Task26 π0.5 Action Expert and scratch Hand DiT."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
import pathlib
import random
import sys
import tempfile
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import train as vla_train

from openpi.models import model as model_lib
from openpi.models_pytorch.dinov3_wrist_encoder import Dinov3WristEncoder
from openpi.models_pytorch.hand_dit import HandDiT
from openpi.models_pytorch.hand_dit import HandDiTConfig
from openpi.models_pytorch.vlm_action_adapter import VLMActionAdapter
from openpi.models_pytorch.vlm_action_adapter import VLMActionAdapterConfig
from openpi.policies.dexterous_action_layout import TASK26_BIMANUAL_LAYOUT
from openpi.policies.dual_expert_hand_policy import DualExpertNormalization
from openpi.policies.dual_expert_hand_policy import save_hand_dit_checkpoint
from openpi.training import bench2dex_dataset
from openpi.training import checkpoints
from openpi.training import code_snapshot
from openpi.training import config as config_lib
from openpi.training import data_loader as data_loader_lib
from openpi.training import dual_expert_joint
from openpi.training import sharding


class _PairedDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dataset, transformed_dataset):
        self.raw_dataset = raw_dataset
        self.transformed_dataset = transformed_dataset

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        return self.transformed_dataset[index], self.raw_dataset[index]


def _collate_pairs(items):
    transformed, raw = zip(*items, strict=True)
    transformed_batch = jax.tree.map(
        lambda *values: np.stack([np.asarray(value) for value in values], axis=0), *transformed
    )
    return transformed_batch, torch.utils.data.default_collate(raw)


def _repeat(loader):
    while True:
        yield from loader


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--dataset-dir", type=pathlib.Path, required=True)
    parser.add_argument("--assets-base-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-base-dir", type=pathlib.Path, required=True)
    parser.add_argument("--hand-norm-stats", type=pathlib.Path, required=True)
    parser.add_argument("--dinov3-repo", type=pathlib.Path, required=True)
    parser.add_argument("--dinov3-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--registry", type=pathlib.Path, default=pathlib.Path("TRAINING_CHECKPOINT_REGISTRY.md"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--hand-learning-rate", type=float, default=1e-4)
    parser.add_argument("--adapter-learning-rate", type=float, default=2e-5)
    parser.add_argument("--adapter-vla-loss-weight", type=float, default=1.0)
    parser.add_argument("--vlm-conditioned-v2", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--validation-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Run the requested steps without writing checkpoints")
    return parser.parse_args()


def _joint_train_step(config, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(model, step_rng, observation, actions):
        chunk_loss, predicted, vlm_features = model.compute_loss_denoised_actions_and_vlm_features(
            step_rng, observation, actions, train=True
        )
        return jnp.mean(chunk_loss), (predicted, vlm_features)

    step_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, (predicted, vlm_features)), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, step_rng, observation, actions
    )
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )
    return new_state, {"vla_loss": loss, "vla_grad_norm": optax.global_norm(grads)}, predicted, vlm_features


def _vla_validation_step(state, rng, batch):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    loss, predicted, vlm_features = model.compute_loss_denoised_actions_and_vlm_features(rng, *batch, train=False)
    return jnp.mean(loss), predicted, vlm_features


def _put_vla_batch(batch, data_sharding):
    batch = jax.tree.map(lambda value: jax.make_array_from_process_local_data(data_sharding, value), batch)
    return model_lib.Observation.from_dict(batch), batch["actions"]


def _hand_batch(raw, predicted, vlm_features, vla_stats, hand_normalization, device, wrist_encoder, adapter=None):
    state = raw["state"].to(device, non_blocking=True)
    actions = raw["actions"].to(device, non_blocking=True)
    predicted_torch = torch.as_tensor(
        np.array(jax.device_get(predicted), dtype=np.float32, copy=True)[..., :16], device=device
    )
    adapter_loss = torch.zeros((), device=device)
    vlm_tokens = None
    if adapter is None:
        arm = dual_expert_joint.predicted_vla_actions_to_hand_condition(
            np.asarray(jax.device_get(predicted)),
            vla_action_q01=vla_stats.q01,
            vla_action_q99=vla_stats.q99,
            hand_normalization=hand_normalization,
            device=device,
        )
    else:
        features = torch.as_tensor(np.array(jax.device_get(vlm_features), dtype=np.float32, copy=True), device=device)
        corrected, vlm_tokens = adapter(predicted_torch, features)
        arm = dual_expert_joint.torch_vla_actions_to_hand_condition(
            corrected,
            vla_action_q01=vla_stats.q01,
            vla_action_q99=vla_stats.q99,
            hand_normalization=hand_normalization,
        )
        target_normalized = torch.as_tensor(raw["vla_actions"], device=device)[..., :16]
        adapter_loss = torch.mean(torch.square(corrected - target_normalized))
        vlm_tokens = torch.cat((vlm_tokens, vlm_tokens), dim=0)
    target, hand_state, side_ids = dual_expert_joint.hand_targets_and_state(state, actions, hand_normalization)
    wrist_images = torch.cat((raw["images"]["cam_wrist_left"], raw["images"]["cam_wrist_right"]), dim=0).to(
        device, non_blocking=True
    )
    with torch.inference_mode():
        wrist_tokens = wrist_encoder(wrist_images)
    return (
        arm if adapter is not None else arm.detach(),
        target,
        hand_state,
        side_ids,
        wrist_tokens,
        vlm_tokens,
        adapter_loss,
    )


def _validate(
    state,
    hand_model,
    validation_loader,
    validation_batches,
    data_sharding,
    replicated_sharding,
    vla_stats,
    hand_normalization,
    device,
    wrist_encoder,
    adapter=None,
):
    hand_model.eval()
    vla_losses, hand_losses, adapter_losses = [], [], []
    validation_rng = jax.random.key(9_271)
    validation_step = jax.jit(
        _vla_validation_step,
        in_shardings=(None, replicated_sharding, data_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
    )
    with torch.no_grad():
        for index, (transformed, raw) in enumerate(validation_loader):
            if index >= validation_batches:
                break
            vla_batch = _put_vla_batch(transformed, data_sharding)
            vla_loss, predicted, vlm_features = validation_step(
                state, jax.random.fold_in(validation_rng, index), vla_batch
            )
            raw["vla_actions"] = torch.as_tensor(np.asarray(transformed["actions"]))
            arm, target, hand_state, side_ids, wrist_tokens, vlm_tokens, adapter_loss = _hand_batch(
                raw, predicted, vlm_features, vla_stats, hand_normalization, device, wrist_encoder, adapter
            )
            hand_loss = hand_model.flow_matching_loss(
                target,
                arm,
                wrist_tokens,
                hand_state,
                side_ids,
                noise=torch.zeros_like(target),
                flow_time=torch.full((target.shape[0],), 0.5, device=device),
                vlm_tokens=vlm_tokens,
            )
            vla_losses.append(float(jax.device_get(vla_loss)))
            hand_losses.append(float(hand_loss))
            adapter_losses.append(float(adapter_loss))
    hand_model.train()
    return float(np.mean(vla_losses)), float(np.mean(hand_losses)), float(np.mean(adapter_losses))


def main():
    args = _parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5)")
    registry = args.registry.read_text(encoding="utf-8")
    if f"## `{args.experiment_name}`" not in registry:
        raise RuntimeError("Register the unique experiment name before launch")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"Python interpreter: {sys.executable}", flush=True)

    base = config_lib.get_config("pi05_bench2dex_task26_dual_expert_arm")
    data = dataclasses.replace(
        base.data,
        repo_id=str(args.dataset_dir.resolve()),
        assets=dataclasses.replace(base.data.assets, assets_dir=str(args.assets_base_dir.resolve())),
    )
    config = dataclasses.replace(
        base,
        exp_name=args.experiment_name,
        data=data,
        assets_base_dir=str(args.assets_base_dir.resolve()),
        checkpoint_base_dir=str(args.checkpoint_base_dir.resolve()),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.steps,
        save_interval=args.save_every,
        log_interval=args.log_every,
        keep_period=None,
        wandb_enabled=False,
    )
    checkpoint_root = (
        pathlib.Path(tempfile.mkdtemp(prefix="openpi_joint_smoke_")) / "checkpoints"
        if args.smoke
        else pathlib.Path(config.checkpoint_dir)
    )
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        checkpoint_root,
        keep_period=None,
        overwrite=False,
        resume=False,
        max_to_keep=None,
    )
    if resuming:
        raise AssertionError("New joint-run entry point unexpectedly resumed")
    code_snapshot.save_code_snapshot(checkpoint_root, config=config, launch_kind="new")

    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = bench2dex_dataset.Bench2DexHdf5Dataset(
        bench2dex_dataset.list_episode_files(str(args.dataset_dir)),
        action_horizon=config.model.action_horizon,
        action_dim=TASK26_BIMANUAL_LAYOUT.action_dim,
        include_gripper_activity=True,
    )
    transformed_dataset = data_loader_lib.transform_dataset(raw_dataset, data_config)
    paired = _PairedDataset(raw_dataset, transformed_dataset)
    generator = torch.Generator().manual_seed(args.seed)
    validation_size = max(args.batch_size, int(len(paired) * args.validation_fraction))
    training_size = len(paired) - validation_size
    training_set, validation_set = torch.utils.data.random_split(
        paired, (training_size, validation_size), generator=generator
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": _collate_pairs,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": args.num_workers > 0,
        "multiprocessing_context": "spawn" if args.num_workers > 0 else None,
    }
    training_loader = torch.utils.data.DataLoader(training_set, shuffle=True, generator=generator, **loader_kwargs)
    validation_loader = torch.utils.data.DataLoader(validation_set, shuffle=False, **loader_kwargs)
    training_batches = _repeat(training_loader)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    train_state, state_sharding = vla_train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(train_state)
    train_step = jax.jit(
        functools.partial(_joint_train_step, config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding, replicated_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    device = torch.device("cuda:0")
    hand_normalization = DualExpertNormalization.from_openpi_json(args.hand_norm_stats, use_quantiles=True)
    hand_model = HandDiT(
        HandDiTConfig(
            action_horizon=config.model.action_horizon,
            arm_dim=8,
            hand_dim=TASK26_BIMANUAL_LAYOUT.hand_dim,
            hand_state_dim=TASK26_BIMANUAL_LAYOUT.hand_dim,
            vision_feature_dim=Dinov3WristEncoder.feature_dim,
            condition_on_vlm=args.vlm_conditioned_v2,
            detach_arm_condition=not args.vlm_conditioned_v2,
        )
    ).to(device)
    adapter = VLMActionAdapter(VLMActionAdapterConfig()).to(device) if args.vlm_conditioned_v2 else None
    parameter_groups = [{"params": hand_model.parameters(), "lr": args.hand_learning_rate}]
    if adapter is not None:
        parameter_groups.append({"params": adapter.parameters(), "lr": args.adapter_learning_rate})
    hand_optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
    wrist_encoder = Dinov3WristEncoder(args.dinov3_repo, args.dinov3_checkpoint).to(device).eval()
    for parameter in wrist_encoder.parameters():
        parameter.requires_grad = False
    vla_stats = data_config.norm_stats["actions"]
    metadata = {
        "experiment_name": args.experiment_name,
        "training": "joint_same_batch",
        "arm_condition_source": "vla_single_forward_denoised_prediction",
        "hand_to_vla_gradient": "shared_post_vla_adapter" if adapter is not None else "stopped",
        "activity_condition": "hard_0_or_1",
        "vlm_final_layer_condition": bool(adapter is not None),
        "vlm_frozen": True,
        "dinov3_frozen": True,
        "vla_action_expert_initialization": "pi05_base",
        "hand_initialization": "scratch",
        "hand_learning_rate": args.hand_learning_rate,
        "vla_action_q01": np.asarray(vla_stats.q01).tolist(),
        "vla_action_q99": np.asarray(vla_stats.q99).tolist(),
    }

    running_vla = running_hand = running_adapter = 0.0
    last_log = time.monotonic()
    best_step, best_score = 0, float("inf")
    hand_model.train()
    for step in range(args.steps):
        transformed, raw = next(training_batches)
        vla_batch = _put_vla_batch(transformed, data_sharding)
        with sharding.set_mesh(mesh):
            train_state, info, predicted, vlm_features = train_step(train_rng, train_state, vla_batch)
        raw["vla_actions"] = torch.as_tensor(np.asarray(transformed["actions"]))
        arm, target, hand_state, side_ids, wrist_tokens, vlm_tokens, adapter_loss = _hand_batch(
            raw, predicted, vlm_features, vla_stats, hand_normalization, device, wrist_encoder, adapter
        )
        hand_optimizer.zero_grad(set_to_none=True)
        hand_loss = hand_model.flow_matching_loss(
            target, arm, wrist_tokens, hand_state, side_ids, vlm_tokens=vlm_tokens
        )
        torch_loss = hand_loss + args.adapter_vla_loss_weight * adapter_loss
        torch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(hand_model.parameters()) + ([] if adapter is None else list(adapter.parameters())), 1.0
        )
        hand_optimizer.step()
        running_vla += float(jax.device_get(info["vla_loss"]))
        running_hand += float(hand_loss.detach())
        running_adapter += float(adapter_loss.detach())
        completed = step + 1
        if completed % args.log_every == 0:
            elapsed = time.monotonic() - last_log
            print(
                f"step={completed} vla_loss={running_vla / args.log_every:.6f} "
                f"hand_loss={running_hand / args.log_every:.6f} "
                f"adapter_vla_loss={running_adapter / args.log_every:.6f} "
                f"steps_per_second={args.log_every / elapsed:.3f}",
                flush=True,
            )
            running_vla = running_hand = running_adapter = 0.0
            last_log = time.monotonic()
        should_save = completed % args.save_every == 0 or completed == args.steps
        if should_save and not args.smoke:
            val_vla, val_hand, val_adapter = _validate(
                train_state,
                hand_model,
                validation_loader,
                args.validation_batches,
                data_sharding,
                replicated_sharding,
                vla_stats,
                hand_normalization,
                device,
                wrist_encoder,
                adapter,
            )
            score = val_vla + val_hand + args.adapter_vla_loss_weight * val_adapter
            if score < best_score:
                best_score, best_step = score, completed
            checkpoints.save_state(
                manager, train_state, type("Loader", (), {"data_config": lambda _: data_config})(), completed
            )
            manager.wait_until_finished()
            step_dir = checkpoint_root / str(completed)
            save_hand_dit_checkpoint(
                step_dir / "hand_dit.pt",
                hand_model,
                hand_normalization,
                step=completed,
                optimizer_state=hand_optimizer.state_dict(),
                metadata=metadata,
                adapter=adapter,
            )
            (step_dir / "joint_metrics.json").write_text(
                json.dumps(
                    {
                        "vla_validation_loss": val_vla,
                        "hand_validation_loss": val_hand,
                        "adapter_vla_validation_loss": val_adapter,
                        "score": score,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            removed = dual_expert_joint.retain_latest_and_best(
                checkpoint_root, latest_step=completed, best_step=best_step
            )
            print(
                f"checkpoint={step_dir} validation_score={score:.6f} best_step={best_step} removed={removed}",
                flush=True,
            )
    manager.wait_until_finished()
    training_batches.close()


if __name__ == "__main__":
    main()
