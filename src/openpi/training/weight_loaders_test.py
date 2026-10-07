import jax
import numpy as np
import pytest

from openpi.training import weight_loaders


def test_shape_compatible_loader_keeps_allowed_new_shape(monkeypatch):
    checkpoint = {
        "backbone": {"kernel": np.ones((2, 2), dtype=np.float32)},
        "action_in_proj": {"kernel": np.ones((32, 4), dtype=np.float32)},
    }
    target = {
        "backbone": {"kernel": jax.ShapeDtypeStruct((2, 2), np.dtype("float32"))},
        "action_in_proj": {"kernel": jax.ShapeDtypeStruct((58, 4), np.dtype("float32"))},
    }
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *args, **kwargs: checkpoint)  # noqa: SLF001
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    loader = weight_loaders.ShapeCompatibleCheckpointWeightLoader(
        "unused", allowed_mismatch_regexes=(r".*action_in_proj.*",)
    )

    loaded = loader.load(target)

    np.testing.assert_array_equal(loaded["backbone"]["kernel"], checkpoint["backbone"]["kernel"])
    assert isinstance(loaded["action_in_proj"]["kernel"], jax.ShapeDtypeStruct)
    assert loaded["action_in_proj"]["kernel"].shape == (58, 4)


def test_shape_compatible_loader_rejects_unexpected_shape(monkeypatch):
    checkpoint = {"backbone": {"kernel": np.ones((3, 2), dtype=np.float32)}}
    target = {"backbone": {"kernel": jax.ShapeDtypeStruct((2, 2), np.dtype("float32"))}}
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *args, **kwargs: checkpoint)  # noqa: SLF001
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    loader = weight_loaders.ShapeCompatibleCheckpointWeightLoader(
        "unused", allowed_mismatch_regexes=(r".*action_in_proj.*",)
    )

    with pytest.raises(ValueError, match="Unexpected checkpoint shape mismatch"):
        loader.load(target)
