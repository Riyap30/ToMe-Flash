"""
Combined ToMe + FlashAttention-2 model for Phase 3 of the ToMe-Flash benchmark.

Implements a CUSTOM attention class that performs bipartite token merging
followed by FlashAttention-2 on the reduced sequence — WITHOUT using
tome.patch.timm(), which would overwrite the FlashAttention module.

Required forward() order (see ToMeFlashAttentionBlock.forward):
  # Step 1: Bipartite partition of token sequence
  # Step 2: Cosine similarity → merge r most similar pairs (average features)
  # Step 3: QKV projection on merged (shorter) sequence
  # Step 4: flash_attn_func(q, k, v) on merged sequence
  # NOTE: Do NOT use tome.patch.timm() here — it will overwrite the FlashAttn module.

Token count note:
  DeiT-B/16 starts with 197 tokens (196 patch + 1 CLS) for 224×224 input.
  After merging r tokens per layer across 12 layers the final count is:
    197 - (r × 12)
  e.g. r=8 → 101 tokens at the last layer.
  FlashAttention-2 still produces correct results at non-multiples of 64.

TODO (Phase 3): implement ToMeFlashAttentionBlock and load_combined_model().
"""

import torch
import torch.nn as nn


class ToMeFlashAttentionBlock(nn.Module):
    """Custom attention block combining ToMe token merging and FlashAttention-2.

    This class must NOT rely on tome.patch.timm(), which patches attention
    modules globally and would overwrite the FlashAttention kernel inserted here.

    Args:
        original_attn: The timm Attention module to copy QKV/proj weights from.
        r:             Number of token pairs to merge per forward call.

    TODO (Phase 3): implement __init__ and forward.
    """

    def __init__(self, original_attn: nn.Module, r: int = 8) -> None:
        super().__init__()
        self.r = r
        # TODO (Phase 3): copy qkv and proj weights from original_attn
        raise NotImplementedError("Phase 3: implement ToMeFlashAttentionBlock.__init__()")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merged token + FlashAttention-2 forward pass.

        Args:
            x: Input tensor of shape (B, N, C) where N = current token count.

        Returns:
            Output tensor of shape (B, N - r, C) — r tokens fewer than input.

        TODO (Phase 3):
          # Step 1: Bipartite partition
          #   Split tokens into two sets A and B of size N//2 each
          #   (CLS token is kept separate and re-appended after merging)

          # Step 2: Cosine similarity + merge
          #   Compute pairwise cosine similarity between A and B
          #   For each of the r most similar (a_i, b_i) pairs:
          #     merged_token = (a_i + b_i) / 2  (average features)
          #   Replace the r pairs with their merged tokens
          #   → new sequence length: N - r

          # Step 3: QKV projection on merged sequence
          #   qkv = self.qkv(x_merged)  → (B, N-r, 3, H, D)

          # Step 4: FlashAttention-2
          #   from flash_attn import flash_attn_func
          #   out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
          #   → (B, N-r, H*D) after reshape
          #   Apply self.proj and return

          # NOTE: token counts are tracked per-layer; the caller (transformer
          #   block) must NOT assume a fixed sequence length after this module.
        """
        raise NotImplementedError("Phase 3: implement ToMeFlashAttentionBlock.forward()")


def load_combined_model(r: int = 8, device: str = "cuda") -> nn.Module:
    """Load DeiT-B/16 with all attention blocks replaced by ToMeFlashAttentionBlock.

    Args:
        r:      Tokens merged per layer. Sweep: r ∈ {4, 8, 16}.
        device: Target device string.

    Returns:
        Model in eval mode with combined ToMe + FlashAttention-2 blocks.

    TODO (Phase 3):
        from models.baseline import load_baseline_model
        model = load_baseline_model(device=device)
        for block in model.blocks:
            block.attn = ToMeFlashAttentionBlock(block.attn, r=r)
        # Do NOT call tome.patch.timm(model, r=r) — incompatible with above.
        return model
    """
    raise NotImplementedError("Phase 3: implement load_combined_model()")
