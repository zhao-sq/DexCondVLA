"""Frozen official DINOv3 patch-token encoder for wrist-camera images."""

from __future__ import annotations

import importlib
import pathlib
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

DINOV3_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINOV3_IMAGE_STD = (0.229, 0.224, 0.225)


def _import_official_backbone(repo_path: pathlib.Path):
    if not (repo_path / "dinov3" / "hub" / "backbones.py").is_file():
        raise FileNotFoundError(f"{repo_path} is not an official DINOv3 checkout; expected dinov3/hub/backbones.py")
    repo_string = str(repo_path)
    already_present = repo_string in sys.path
    if not already_present:
        sys.path.insert(0, repo_string)
    try:
        module = importlib.import_module("dinov3.hub.backbones")
        module_path = pathlib.Path(module.__file__).resolve()
        if not module_path.is_relative_to(repo_path):
            raise RuntimeError(f"Python already imported DINOv3 from {module_path}, not requested checkout {repo_path}")
        return module
    finally:
        if not already_present:
            sys.path.remove(repo_string)


class Dinov3WristEncoder(nn.Module):
    """Load a frozen ViT-B/16 from an official checkout and local checkpoint.

    This wrapper deliberately returns only ``x_norm_patchtokens``.  CLS and
    storage/register tokens are excluded so the hand policy retains spatial
    wrist-image structure without depending on DINOv3's global tokens.
    """

    feature_dim = 768

    def __init__(
        self,
        repo_path: pathlib.Path | str,
        checkpoint_path: pathlib.Path | str,
        *,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        self.repo_path = pathlib.Path(repo_path).expanduser().resolve()
        self.checkpoint_path = pathlib.Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {self.checkpoint_path}")
        if image_size <= 0 or image_size % 16:
            raise ValueError("DINOv3 image_size must be a positive multiple of patch size 16")
        self.image_size = image_size

        backbones = _import_official_backbone(self.repo_path)
        self.backbone = backbones.dinov3_vitb16(weights=str(self.checkpoint_path))
        self.backbone.requires_grad_(requires_grad=False)
        self.backbone.eval()
        self.register_buffer("image_mean", torch.tensor(DINOV3_IMAGE_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(DINOV3_IMAGE_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):  # noqa: FBT001, FBT002
        # The wrapper may live inside a training module, but DINOv3 is always frozen.
        super().train(mode=False)
        self.backbone.eval()
        return self

    def preprocess(self, images: np.ndarray | torch.Tensor) -> torch.Tensor:
        images = torch.as_tensor(images, device=self.image_mean.device)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"Expected wrist images [B,H,W,C] or [B,C,H,W], got {images.shape}")
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        elif images.shape[1] != 3:
            raise ValueError(f"Expected three image channels, got {images.shape}")
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.float()
            if images.numel() and (images.min() < 0.0 or images.max() > 1.0):
                raise ValueError("Floating wrist images must be in [0, 1]")
        images = F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return (images - self.image_mean) / self.image_std

    @torch.no_grad()
    def forward(self, images: np.ndarray | torch.Tensor) -> torch.Tensor:
        pixels = self.preprocess(images)
        features = self.backbone.forward_features(pixels)
        patch_tokens = features.get("x_norm_patchtokens")
        if patch_tokens is None or patch_tokens.ndim != 3 or patch_tokens.shape[-1] != self.feature_dim:
            raise RuntimeError(f"Unexpected DINOv3 patch-token output: {type(patch_tokens)}")
        return patch_tokens.float()
