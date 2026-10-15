import torch

from openpi.models_pytorch.vlm_action_adapter import VLMActionAdapter
from openpi.models_pytorch.vlm_action_adapter import VLMActionAdapterConfig


def test_adapter_starts_with_zero_action_residual_and_receives_both_gradients():
    config = VLMActionAdapterConfig(vlm_feature_dim=12, model_dim=8, action_dim=4, num_vlm_tokens=2)
    model = VLMActionAdapter(config)
    actions = torch.randn(3, 5, 4)
    features = torch.randn(3, 12)

    corrected, tokens = model(actions, features)
    torch.testing.assert_close(corrected, actions)
    assert tokens.shape == (3, 2, 8)

    loss = corrected.square().mean() + tokens.square().mean()
    loss.backward()
    assert model.action_residual[-1].weight.grad is not None
    assert model.token_projection.weight.grad is not None
