"""Mask2Former-style foreground inference used by MatDeplot.

The architecture is reconstructed directly from the released checkpoint:
a MiT-B3 encoder, half-resolution FPN pixel decoder, 12 mask queries, and a
three-layer Transformer decoder. Public callers normally use
``get_line_prob_multiscale`` to obtain an original-resolution probability map.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_CKPT = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "mask2foremer_q12_d3_h256_best.pt"
)


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization matching the training checkpoint."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNormAct(nn.Module):
    """Convolution, channel normalization, and GELU activation."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, padding=padding, bias=False),
            LayerNorm2d(c_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HalfResFPNDecoder(nn.Module):
    """Top-down FPN with concatenation and half-resolution refinement."""

    def __init__(self, in_channels=(64, 128, 320, 512), hidden: int = 256) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, hidden, 1) for c in in_channels])
        self.output_convs = nn.ModuleList([ConvNormAct(hidden, hidden) for _ in in_channels])
        self.fuse = nn.Sequential(
            ConvNormAct(hidden * len(in_channels), hidden),
            ConvNormAct(hidden, hidden),
        )
        self.half_res_refine = nn.Sequential(
            ConvNormAct(hidden, hidden),
            ConvNormAct(hidden, hidden),
        )
        self.mask_proj = nn.Conv2d(hidden, hidden, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        laterals = [proj(f) for proj, f in zip(self.lateral, features)]
        pyramid: list[torch.Tensor] = [laterals[-1]]
        for idx in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(pyramid[0], size=laterals[idx].shape[-2:],
                               mode="bilinear", align_corners=False)
            pyramid.insert(0, laterals[idx] + up)
        pyramid = [conv(feat) for conv, feat in zip(self.output_convs, pyramid)]
        top_size = pyramid[0].shape[-2:]
        merged = torch.cat(
            [
                F.interpolate(
                    feature,
                    size=top_size,
                    mode="bilinear",
                    align_corners=False,
                )
                for feature in pyramid
            ],
            dim=1,
        )
        fused = self.fuse(merged)
        half_res = F.interpolate(
            fused, size=(top_size[0] * 2, top_size[1] * 2),
            mode="bilinear", align_corners=False,
        )
        half_res = self.half_res_refine(half_res)
        return self.mask_proj(half_res)


class MLP(nn.Module):
    """Three-layer MLP whose module keys match the released checkpoint."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 3) -> None:
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionEmbeddingSine(nn.Module):
    """Normalized two-dimensional sinusoidal position encoding."""

    def __init__(self, num_pos_feats: int = 128, temperature: float = 10000.0) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        y_embed = torch.arange(h, device=x.device, dtype=torch.float32).unsqueeze(1).repeat(1, w)
        x_embed = torch.arange(w, device=x.device, dtype=torch.float32).unsqueeze(0).repeat(h, 1)
        y_embed = y_embed / max(h - 1, 1) * 2.0 * math.pi
        x_embed = x_embed / max(w - 1, 1) * 2.0 * math.pi
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=3).flatten(2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=3).flatten(2)
        pos = torch.cat([pos_y, pos_x], dim=2).permute(2, 0, 1)
        return pos.unsqueeze(0).expand(b, -1, -1, -1).to(x.dtype)


class Mask2FormerStyleModel(nn.Module):
    """Architecture reconstructed for the released foreground checkpoint."""

    def __init__(
        self,
        encoder_name: str = "mit_b3",
        hidden_dim: int = 256,
        n_queries: int = 12,
        dec_layers: int = 3,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_queries = n_queries

        self.encoder = smp.encoders.get_encoder(encoder_name, weights=None)
        in_channels = tuple(self.encoder.out_channels[-4:])  # (64, 128, 320, 512)
        self.pixel_decoder = HalfResFPNDecoder(in_channels=in_channels, hidden=hidden_dim)

        self.position_embedding = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=0.0, batch_first=True, norm_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.query_embed = nn.Embedding(n_queries, hidden_dim)
        self.query_pos = nn.Embedding(n_queries, hidden_dim)

        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, num_layers=3)
        self.objectness_head = nn.Linear(hidden_dim, 1)
        self.aux_seg_head = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def _select_encoder_features(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        return list(feats[-4:])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encoder(x)
        feats = self._select_encoder_features(feats)
        mask_features = self.pixel_decoder(feats)
        pos = self.position_embedding(mask_features)
        b, c, _, _ = mask_features.shape
        memory = (mask_features + pos).flatten(2).transpose(1, 2)
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        q = q + self.query_pos.weight.unsqueeze(0)
        decoded = self.decoder(q, memory)
        decoded = self.decoder_norm(decoded)

        mask_embed = self.mask_embed(decoded)
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features) / math.sqrt(c)
        objectness_logits = self.objectness_head(decoded).squeeze(-1)
        aux_logits = self.aux_seg_head(mask_features)
        return {
            "mask_logits": mask_logits,
            "objectness_logits": objectness_logits,
            "aux_logits": aux_logits,
        }


_MODEL_CACHE: dict[tuple[str, str], Mask2FormerStyleModel] = {}


def load_model(
    ckpt_path: Union[str, Path] = DEFAULT_CKPT,
    device: Optional[Union[str, torch.device]] = None,
    cache: bool = True,
) -> Mask2FormerStyleModel:
    """Load the checkpoint and return a cached model in evaluation mode."""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    key = (str(ckpt_path), str(device))
    if cache and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    model = Mask2FormerStyleModel(
        encoder_name=cfg.get("encoder", "mit_b3"),
        hidden_dim=cfg.get("hidden_dim", 256),
        n_queries=cfg.get("n_queries", 12),
        dec_layers=cfg.get("dec_layers", 3),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        suffix = " ..." if len(missing) > 8 else ""
        print(
            f"[mask2former] missing keys ({len(missing)}): "
            f"{missing[:8]}{suffix}"
        )
    if unexpected:
        suffix = " ..." if len(unexpected) > 8 else ""
        print(
            f"[mask2former] unexpected keys ({len(unexpected)}): "
            f"{unexpected[:8]}{suffix}"
        )
    model.eval().to(device)
    if cache:
        _MODEL_CACHE[key] = model
    return model


def _preprocess(img_bgr: np.ndarray, infer_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (infer_size, infer_size), interpolation=cv2.INTER_LINEAR)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).contiguous().float()
    return tensor, (h, w)


@torch.inference_mode()
def get_line_binary_mask(
    image: Union[str, Path, np.ndarray],
    ckpt_path: Union[str, Path] = DEFAULT_CKPT,
    infer_size: int = 640,
    threshold: float = 0.5,
    obj_threshold: float = 0.5,
    device: Optional[Union[str, torch.device]] = None,
    return_prob: bool = False,
) -> np.ndarray:
    """Return the union probability or mask of objectness-filtered queries."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        img = image
    tensor, (src_h, src_w) = _preprocess(img, infer_size)

    model = load_model(ckpt_path=ckpt_path, device=device)
    dev = next(model.parameters()).device
    tensor = tensor.to(dev, non_blocking=True)

    out = model(tensor)
    mask_logits = out["mask_logits"]      # [1, Q, h, w]
    obj_logits = out["objectness_logits"]  # [1, Q]
    obj_prob = torch.sigmoid(obj_logits).squeeze(0)  # [Q]
    keep = obj_prob > obj_threshold
    if not keep.any():
        keep = obj_prob >= obj_prob.max()

    mask_prob = torch.sigmoid(mask_logits).squeeze(0)  # [Q, h, w]
    mask_prob = mask_prob[keep]                         # [K, h, w]
    union = mask_prob.max(dim=0).values.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
    union = F.interpolate(union, size=(src_h, src_w), mode="bilinear", align_corners=False)
    prob_np = union.squeeze(0).squeeze(0).float().cpu().numpy()

    if return_prob:
        return prob_np.astype(np.float32)
    return prob_np >= threshold


@torch.inference_mode()
def get_line_prob_multiscale(
    image: Union[str, Path, np.ndarray],
    ckpt_path: Union[str, Path] = DEFAULT_CKPT,
    scales: tuple = (640, 1024),
    obj_threshold: float = 0.5,
    device: Optional[Union[str, torch.device]] = None,
    flip_tta: bool = False,
) -> np.ndarray:
    """Fuse multi-scale and optional horizontal-flip predictions by max."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        img = image
    H, W = img.shape[:2]
    acc = np.zeros((H, W), dtype=np.float32)
    variants = [(img, False)]
    if flip_tta:
        variants.append((img[:, ::-1].copy(), True))
    for var_img, was_flipped in variants:
        for s in scales:
            p = get_line_binary_mask(var_img, ckpt_path=ckpt_path,
                                     infer_size=int(s),
                                     obj_threshold=obj_threshold,
                                     device=device,
                                     return_prob=True)
            if was_flipped:
                p = p[:, ::-1]
            acc = np.maximum(acc, p.astype(np.float32))
    return acc
