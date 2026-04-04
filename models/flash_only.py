"""
FlashAttention-only model for Phase 2 of the ToMe-Flash benchmark.

Replaces the standard scaled-dot-product attention in each DeiT-B/16 transformer
block with the FlashAttention-2 kernel (flash_attn_func) while keeping all other
weights and the token sequence unchanged.

TODO (Phase 2): implement FlashAttentionBlock and load_flash_model().
"""

import torch
import torch.nn as nn


class FlashAttentionBlock(nn.Module):
    """Drop-in replacement for a timm ViT attention block using FlashAttention-2.

    The QKV projection weights are copied from the original timm attention
    module; only the attention computation is replaced with flash_attn_func.

    Args:
        original_attn: The timm Attention module to copy weights from.

    TODO (Phase 2):
        from flash_attn import flash_attn_func

        Forward logic:
          1. Project input x with self.qkv          → (B, N, 3, H, D)
          2. Reshape to (B, N, H, D) for q, k, v each
          3. out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
          4. Reshape out to (B, N, H*D) and apply self.proj
        Note: flash_attn_func expects float16 / bfloat16 inputs.
              Cast inside forward(), cast result back to original dtype.
    """

    def __init__(self, original_attn: nn.Module) -> None:
        super().__init__()
        # TODO (Phase 2): copy qkv and proj weights from original_attn
        raise NotImplementedError("Phase 2: implement FlashAttentionBlock.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, C).

        Returns:
            Output tensor of shape (B, N, C).

        TODO (Phase 2): implement FlashAttention-2 forward pass.
        """
        raise NotImplementedError("Phase 2: implement FlashAttentionBlock.forward()")


def load_flash_model(device: str = "cuda") -> nn.Module:
    """Load DeiT-B/16 with all attention blocks replaced by FlashAttentionBlock.

    Args:
        device: Target device string.

    Returns:
        Model in eval mode with FlashAttention-2 attention blocks.

    TODO (Phase 2):
        from models.baseline import load_baseline_model
        model = load_baseline_model(device=device)
        for block in model.blocks:
            block.attn = FlashAttentionBlock(block.attn)
        return model
    """
    raise NotImplementedError("Phase 2: implement load_flash_model()")
