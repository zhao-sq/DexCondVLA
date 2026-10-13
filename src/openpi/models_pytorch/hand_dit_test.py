import dataclasses

import torch

from openpi.models_pytorch.hand_dit import HandDiT
from openpi.models_pytorch.hand_dit import HandDiTConfig


def _small_config() -> HandDiTConfig:
    return HandDiTConfig(
        action_horizon=5,
        arm_dim=3,
        hand_dim=4,
        hand_state_dim=4,
        vision_feature_dim=12,
        model_dim=32,
        num_layers=2,
        num_heads=4,
        max_vision_tokens=6,
    )


def test_hand_dit_shared_side_batch_shapes_and_gradients() -> None:
    model = HandDiT(_small_config())
    batch_size = 4
    arm = torch.randn(batch_size, 5, 3)
    vision = torch.randn(batch_size, 17, 12)
    state = torch.randn(batch_size, 4)
    target = torch.randn(batch_size, 5, 4)
    side_ids = torch.tensor([0, 1, 0, 1])

    velocity = model(torch.randn_like(target), torch.rand(batch_size), arm, vision, state, side_ids)
    assert velocity.shape == target.shape

    loss = model.flow_matching_loss(target, arm, vision, state, side_ids)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.output_projection.weight.grad is not None


def test_hand_dit_sampling_and_condition_ablation() -> None:
    config = dataclasses.replace(
        _small_config(), condition_on_arm=False, condition_on_vision=True, condition_on_state=False
    )
    model = HandDiT(config)
    batch_size = 2
    arm = torch.randn(batch_size, 5, 3)
    vision = torch.randn(batch_size, 9, 12)
    state = torch.randn(batch_size, 4)
    side_ids = torch.tensor([0, 1])
    noise = torch.randn(batch_size, 5, 4)

    sampled = model.sample(arm, vision, state, side_ids, num_steps=3, noise=noise.clone())

    # DiT-Zero initialization begins as a zero velocity field.
    torch.testing.assert_close(sampled, noise)


def test_hand_dit_can_disable_vision_without_tokens() -> None:
    config = dataclasses.replace(
        _small_config(), condition_on_arm=True, condition_on_vision=False, condition_on_state=True
    )
    model = HandDiT(config)
    batch_size = 2
    arm = torch.randn(batch_size, 5, 3)
    state = torch.randn(batch_size, 4)
    side_ids = torch.tensor([0, 1])
    noise = torch.randn(batch_size, 5, 4)

    sampled = model.sample(arm, None, state, side_ids, num_steps=2, noise=noise.clone())

    torch.testing.assert_close(sampled, noise)


def test_hand_dit_masked_loss_ignores_missing_side() -> None:
    model = HandDiT(_small_config())
    batch_size = 2
    target = torch.randn(batch_size, 5, 4)
    arm = torch.randn(batch_size, 5, 3)
    vision = torch.randn(batch_size, 8, 12)
    state = torch.randn(batch_size, 4)
    side_ids = torch.tensor([0, 1])
    noise = torch.randn_like(target)
    flow_time = torch.rand(batch_size)

    masked = model.flow_matching_loss(
        target,
        arm,
        vision,
        state,
        side_ids,
        side_valid=torch.tensor([True, False]),
        noise=noise,
        flow_time=flow_time,
    )
    single = model.flow_matching_loss(
        target[:1],
        arm[:1],
        vision[:1],
        state[:1],
        side_ids[:1],
        noise=noise[:1],
        flow_time=flow_time[:1],
    )
    torch.testing.assert_close(masked, single)
