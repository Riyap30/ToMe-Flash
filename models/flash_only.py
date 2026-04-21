"""
models/flash_only.py

FlashAttention-2-only model for Phase 3 of the ToMe-Flash benchmark.

Surgically replaces the scaled-dot-product attention kernel in every DeiT-B/16
transformer block with flash_attn_func while keeping all other weights (QKV
projection, output projection, MLP, norms, classification head) exactly as
loaded from the pretrained checkpoint.

No ToMe functions are imported or called here — this is the FA-only condition.
"""

import timm
import torch
import torch.nn as nn
from flash_attn import flash_attn_func


class FlashAttentionBlock(nn.Module):
    """Drop-in replacement for a timm ViT Attention module using FlashAttention-2.

    Reuses the existing qkv and proj weights from the original timm Attention
    instance — no weight reinitialization occurs.

    Args:
        original_attn: The timm Attention module whose weights are adopted.
    """

    def __init__(self, original_attn: nn.Module) -> None:
        super().__init__()
        self.qkv       = original_attn.qkv         # Linear(C, 3*C)
        self.proj      = original_attn.proj         # Linear(C, C)
        self.num_heads: int = original_attn.num_heads
        # head_dim derived from QKV weight: out_features = 3 * num_heads * head_dim
        self.head_dim: int = (
            original_attn.qkv.weight.shape[0] // (3 * self.num_heads)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FlashAttention-2 forward pass.

        Args:
            x: (B, N, C) input tensor.

        Returns:
            (B, N, C) output tensor, matching the original Attention.forward() contract.
        """
        B, N, C = x.shape

        # Project to Q, K, V and reshape to (B, N, 3, H, D)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        # flash_attn_func expects (B, N, H, D) — unbind along dim 2
        q, k, v = qkv.unbind(2)   # each: (B, N, num_heads, head_dim)

        # flash_attn_func handles 1/sqrt(d) scaling internally
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)  # (B, N, H, D)

        out = out.reshape(B, N, C)
        return self.proj(out)


def load_flash_model(device: str = "cuda") -> nn.Module:
    """Load DeiT-B/16 with all 12 attention blocks replaced by FlashAttentionBlock.

    Weights are loaded from the pretrained timm checkpoint and the QKV / proj
    parameters are shared with the new FlashAttentionBlock instances (no copy,
    no reinitialization).

    A forward-pass sanity check on a dummy input is run before returning.

    Args:
        device: Target device string (e.g. 'cuda').

    Returns:
        Model in eval mode on *device* in bfloat16.

    Raises:
        AssertionError: If the sanity-check output shape is not (1, 1000).
    """
    model: nn.Module = timm.create_model(
        "deit_base_patch16_224",
        pretrained=True,
    )
    model.eval()
    model.to(device)
    model.to(torch.bfloat16)

    # Replace every block's .attn module in-place — no tome calls, ever.
    for block in model.blocks:
        block.attn = FlashAttentionBlock(block.attn)

    print(f"FlashAttentionBlock patched into all {len(model.blocks)} transformer blocks.")

    # Sanity check: single forward pass on a dummy input
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (1, 1000), (
        f"Sanity check FAILED: expected output shape (1, 1000), got {tuple(out.shape)}"
    )
    print("Sanity check PASSED: output shape (1, 1000) confirmed.")

    return model
