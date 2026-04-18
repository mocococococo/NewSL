"""Transformer 版の損失関数。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def calculate_kld_loss(
    logits: torch.Tensor,
    target_distribution: torch.Tensor,
) -> torch.Tensor:
    """logits と教師分布の Kullback-Leibler Divergence を計算する。

    `target_distribution` は各行の合計が 1.0 の確率分布であることを想定する。
    0 確率の要素は KLD に寄与しないため、log(0) を直接計算しないように扱う。
    """

    if logits.shape != target_distribution.shape:
        raise ValueError(
            "logits and target_distribution must have the same shape: "
            f"{tuple(logits.shape)} != {tuple(target_distribution.shape)}"
        )
    if logits.dim() < 2:
        raise ValueError(f"logits must have at least 2 dims, got {logits.dim()}")

    target_distribution = target_distribution.to(dtype=logits.dtype, device=logits.device)
    if not torch.isfinite(target_distribution).all():
        raise ValueError("target_distribution contains non-finite values")
    if (target_distribution < 0).any():
        raise ValueError("target_distribution contains negative values")

    target_sum = target_distribution.sum(dim=-1)
    expected_sum = torch.ones_like(target_sum)
    if not torch.allclose(target_sum, expected_sum, atol=1e-4, rtol=1e-4):
        raise ValueError("target_distribution must sum to 1.0 along the last dimension")

    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, target_distribution, reduction="batchmean")


__all__ = [
    "calculate_kld_loss",
]
