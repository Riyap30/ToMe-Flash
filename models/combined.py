"""
models/combined.py

Phase 4: ToMe bipartite soft matching + FlashAttention-2, applied together to
DeiT-B/16 in a single controlled pass.

Patching strategy:
  1. Load fresh timm DeiT-B/16 — no prior patching.
  2. Replace each block.attn with CombinedAttentionBlock (merge + FA2 in one module).
  3. Monkey-patch each block.forward to thread the ToMe size tensor through residuals.
  4. Wrap model.blocks in _BlocksWithSize so size state is managed across all 12 layers
     while model.blocks(x) still returns a plain Tensor (interface unchanged).

No tome.patch.timm() is called anywhere.  bipartite_soft_matching and merge_wavg
are imported directly from tome.merge and invoked as pure stateless functions.

Size-weighted attention bias (adding log(size) to attention logits, per the ToMe
paper's Eq. 2) is NOT applied: flash_attn_func does not accept per-token bias
tensors.  The size tensor is tracked and used only for weighted-mean token merging
in merge_wavg.
"""

from typing import Callable, Optional, Tuple

import timm
import torch
import torch.nn as nn
from flash_attn import flash_attn_func
from tome.merge import bipartite_soft_matching, merge_wavg


class CombinedAttentionBlock(nn.Module):
    """Single attention module combining ToMe token merging with FlashAttention-2.

    Tokens are merged BEFORE QKV projection, so the expensive linear projections
    and the attention kernel operate on N_merged < N tokens.

    The merge function produced in forward() is stored as self._merge_fn so the
    surrounding block wrapper can apply the same merge to the un-normed residual
    path (which must match attn_out's reduced sequence length before addition).

    Args:
        original_attn: Pretrained timm Attention module.  qkv and proj Linear
                       layers are reused by reference — no weight copy.
        r:             Token pairs to merge per forward pass.
    """

    def __init__(self, original_attn: nn.Module, r: int) -> None:
        super().__init__()
        self.qkv       = original_attn.qkv    # Linear(C, 3*C), bias included
        self.proj      = original_attn.proj   # Linear(C, C)
        self.num_heads: int = original_attn.num_heads
        self.head_dim:  int = (
            original_attn.qkv.weight.shape[0] // (3 * self.num_heads)
        )
        self.r = r
        self._merge_fn: Optional[Callable] = None  # set each forward; read by block wrapper

    def forward(
        self,
        x: torch.Tensor,
        size: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Merge → QKV → FlashAttention-2 → proj.

        Args:
            x:    (B, N, C) — already layer-normed by the surrounding block.
            size: (B, N, 1) — how many original tokens each merged token
                  represents.  None on the first block; merge_wavg initialises
                  it to all-ones in that case.

        Returns:
            output — (B, N_merged, C)
            size   — (B, N_merged, 1)  updated token sizes
        """
        B, N, C = x.shape

        # Bipartite soft matching on the normed input tokens.
        # class_token=True: CLS token (index 0) is never selected as a merge source.
        if self.r > 0:
            merge, _unmerge = bipartite_soft_matching(x, self.r, class_token=True)
            self._merge_fn = merge
            # merge_wavg: weighted-mean merge of x; sum-merge of size.
            # If size is None it is initialised to torch.ones_like(x[..., :1]).
            x, size = merge_wavg(merge, x, size)
        else:
            self._merge_fn = None

        N_merged = x.shape[1]

        # QKV on merged tokens → reshape to (B, N_merged, 3, H, D)
        qkv = self.qkv(x).reshape(B, N_merged, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)   # each (B, N_merged, H, D)

        # flash_attn_func handles 1/sqrt(d) scaling; expects (B, N, H, D)
        out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)

        out = out.reshape(B, N_merged, C)
        return self.proj(out), size


def _patch_block_forward(block: nn.Module) -> None:
    """Replace block.forward with a size-aware version.

    The new forward:
      1. Calls CombinedAttentionBlock (which merges then runs FA2).
      2. Applies the same merge function to the un-normed x so that the
         residual addition x + attn_out is shape-compatible.
      3. Returns (x, size) instead of just x.

    Handles both timm naming conventions:
      - New timm (0.6+): drop_path1 / drop_path2 + ls1 / ls2
      - Old timm:        drop_path (shared) + no LayerScale

    The closure captures the block instance via 'b' to avoid the classic
    loop-variable-capture bug when called inside a for-loop.
    """
    b = block

    # Detect which timm Block API is present once, at patch time.
    _new_api = hasattr(b, "drop_path1")

    if _new_api:
        def forward(
            x: torch.Tensor,
            size: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            attn_out, size = b.attn(b.norm1(x), size)
            merge_fn = b.attn._merge_fn
            if merge_fn is not None:
                x = merge_fn(x)
            x = x + b.drop_path1(b.ls1(attn_out))
            x = x + b.drop_path2(b.ls2(b.mlp(b.norm2(x))))
            return x, size
    else:
        def forward(
            x: torch.Tensor,
            size: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            attn_out, size = b.attn(b.norm1(x), size)
            merge_fn = b.attn._merge_fn
            if merge_fn is not None:
                x = merge_fn(x)
            x = x + b.drop_path(attn_out)
            x = x + b.drop_path(b.mlp(b.norm2(x)))
            return x, size

    block.forward = forward


class _BlocksWithSize(nn.Module):
    """Thin wrapper around the blocks Sequential that manages the size tensor.

    Calling model.blocks(x) returns a plain Tensor — unchanged interface for
    timm's forward_features — while size is threaded internally across all layers.
    """

    def __init__(self, blocks: nn.Sequential) -> None:
        super().__init__()
        # Keep original Sequential as a registered submodule so all parameters
        # remain accessible via model.parameters() / model.state_dict().
        self.blocks = blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size: Optional[torch.Tensor] = None
        for block in self.blocks:
            x, size = block(x, size)
        return x


def load_combined_model(device: str = "cuda", r: int = 8) -> nn.Module:
    """Load DeiT-B/16 with ToMe (r=8) + FlashAttention-2 in all 12 blocks.

    Always starts from a fresh timm checkpoint — never from an already-patched
    tome_only or flash_only model.  Both modifications are applied in a single
    controlled pass so there is no patching-order conflict.

    Patching order:
      1. block.attn   -> CombinedAttentionBlock   (merge + FA2, no tome.patch call)
      2. block.forward -> _patch_block_forward     (threads size through residuals)
      3. model.blocks  -> _BlocksWithSize           (manages size across layers)

    Args:
        device: CUDA device string (e.g. 'cuda' or 'cuda:0').
        r:      Token pairs merged per block. Default 8 -> 197 - 8x12 = 101 final tokens.

    Returns:
        Model in eval mode, bfloat16, on *device*.

    Raises:
        AssertionError: If the sanity-check forward pass does not produce (1, 1000).
    """
    model: nn.Module = timm.create_model("deit_base_patch16_224", pretrained=True)
    model.eval()
    model.to(device)
    model.to(torch.bfloat16)

    # Step 1 — replace every block's .attn with CombinedAttentionBlock
    for block in model.blocks:
        block.attn = CombinedAttentionBlock(block.attn, r=r)

    # Step 2 — patch each block's forward to accept / return size
    for block in model.blocks:
        _patch_block_forward(block)

    # Step 3 — wrap the blocks Sequential so size is threaded across layers
    #           while model.blocks(x) still returns a plain Tensor
    model.blocks = _BlocksWithSize(model.blocks)

    n_blocks = len(model.blocks.blocks)
    final_tokens = 197 - r * n_blocks
    print(
        f"CombinedAttentionBlock patched into all {n_blocks} transformer blocks "
        f"(r={r}, final tokens per image: {final_tokens})."
    )
    print(
        "Note: size-weighted attention bias is not applied (flash_attn_func does "
        "not support per-token logit biases); size tensor used for weighted-mean "
        "merging only."
    )

    # Sanity check: one forward pass on a dummy bfloat16 input
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (1, 1000), (
        f"Sanity check FAILED: expected output shape (1, 1000), got {tuple(out.shape)}"
    )
    print("Sanity check PASSED: output shape (1, 1000) confirmed.")

    return model
