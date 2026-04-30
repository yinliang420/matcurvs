"""
Mask2Former-style 线条/散点分割模型本地推理。

架构重建自 checkpoint ``models/mask2foremer_q12_d3_h256_best.pt``
(原训练脚本 train_v10.Mask2FormerStyleModel, 位于集群)。

结构:
    encoder        = MiT-B3 (Segformer backbone, 取 4 个 stage)
    pixel_decoder  = FPN (lateral 1x1 + output_convs 3x3 + concat fuse + half_res_refine + mask_proj)
    decoder        = 标准 nn.TransformerDecoder, d_model=256, nhead=8, ff=1024, 3 layers
    heads          = query_embed(12,256)+query_pos, mask_embed (3-layer MLP),
                     objectness_head (256->1), aux_seg_head (二值分割, 输出 1/2 分辨率)

对外主函数 :func:`get_line_binary_mask` 接受图像路径/ndarray, 返回与原图同尺寸的
二值掩膜 (所有线条+散点像素为 True, 其余为 False)。
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

DEFAULT_CKPT = Path(__file__).parent / "models" / "mask2foremer_q12_d3_h256_best.pt"


class LayerNorm2d(nn.Module):
    """对 [B, C, H, W] 沿 C 归一化的自定义 LN (eps=1e-6), 与训练脚本精确对齐。"""

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
    """Conv (bias=False) + LayerNorm2d + GELU, padding=kernel_size//2。"""

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
    """Top-down FPN + concat 融合 + 上采到 1/2 精修。"""

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
            [F.interpolate(f, size=top_size, mode="bilinear", align_corners=False) for f in pyramid],
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
    """3-layer MLP, 对齐 checkpoint 的 `net.0 / net.2 / net.4` 键。"""

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
    """DETR 标准 sinusoidal 二维位置编码 (归一化坐标版)。"""

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
    """对齐 ``mask2foremer_q12_d3_h256_best.pt`` 的前向模型。"""

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
    """加载 checkpoint,返回 eval 模式的模型。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
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
        print(f"[mask2former] missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[mask2former] unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")
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
    """获取所有线条/散点像素的二值掩膜。

    采用 per-query mask 的并集作为最终二值预测: 对 objectness > obj_threshold
    的 query 取 sigmoid(mask_logits) 的逐像素 max, 再与 threshold 比较。
    (训练脚本里 ``aux_seg_head`` 的 LayerNorm2d 未被训练, 输出近乎常数, 不能作为
    可靠信号; Mask2Former 的主预测来自 query 分支。)

    Parameters
    ----------
    image : 路径或 BGR ndarray。
    ckpt_path : checkpoint 路径。
    infer_size : 推理方边长 (与训练一致 640)。
    threshold : 逐像素 sigmoid 阈值。
    obj_threshold : query objectness 阈值, 低于此的 query 被丢弃。
    return_prob : True 时返回概率图 (float32, 0..1), 否则返回 bool 掩膜。
    """
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
        keep = obj_prob >= obj_prob.max()  # 退回最强的 query 至少保留 1 个

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
    """iter11: 多尺度 TTA. 每个 scale 跑一遍 m2f, prob 上采到原分辨率,
    取逐像素 max. 召回细线 / 虚线 / 低对比线条.
    iter12 phase C: 可选 horizontal flip TTA, 翻转后再翻回, prob max 融合."""
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


@torch.inference_mode()
def get_line_per_query(
    image: Union[str, Path, np.ndarray],
    ckpt_path: Union[str, Path] = DEFAULT_CKPT,
    infer_size: int = 1024,
    obj_threshold: float = 0.3,
    device: Optional[Union[str, torch.device]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """iter12 phase D: 暴露 per-query 实例级 mask, 用于 K>=5 时的 query-NMS 分支.

    Returns
    -------
    probs : (K, H, W) float32 — sigmoid mask prob, 已上采到原图分辨率, 仅保留 obj_prob > obj_threshold 的 query.
    obj   : (K,) float32 — 对应每个 query 的 objectness 概率.
    """
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
    mask_logits = out["mask_logits"]
    obj_logits = out["objectness_logits"]
    obj_prob = torch.sigmoid(obj_logits).squeeze(0)
    keep = obj_prob > obj_threshold
    if not keep.any():
        keep = obj_prob >= obj_prob.max()

    mask_prob = torch.sigmoid(mask_logits).squeeze(0)[keep]
    obj_keep = obj_prob[keep]
    mask_prob = F.interpolate(mask_prob.unsqueeze(0),
                              size=(src_h, src_w),
                              mode="bilinear", align_corners=False).squeeze(0)
    return (mask_prob.float().cpu().numpy().astype(np.float32),
            obj_keep.float().cpu().numpy().astype(np.float32))


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test"
    p = Path(path)
    if p.is_dir():
        imgs = sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")))[:3]
    else:
        imgs = [p]
    for im in imgs:
        mask = get_line_binary_mask(im)
        print(f"{im.name}: shape={mask.shape} fg_frac={mask.mean():.4f}")
