"""
ToMe-only model for Phase 2 of the ToMe-Flash benchmark.

Applies Token Merging (ToMe) to DeiT-B/16 using tome.patch.timm().
The merging ratio r controls how many token pairs are merged per layer.

TODO (Phase 2): implement load_tome_model().
"""

import torch
import torch.nn as nn


def load_tome_model(r: int = 8, device: str = "cuda") -> nn.Module:
    """Load DeiT-B/16 patched with ToMe token merging.

    Args:
        r:      Number of token pairs to merge per transformer layer.
                Sweep values: r ∈ {4, 8, 16}.
        device: Target device string.

    Returns:
        ToMe-patched model in eval mode on the specified device.

    TODO (Phase 2):
        import tome
        from models.baseline import load_baseline_model
        model = load_baseline_model(device=device)
        tome.patch.timm(model, r=r)
        # NOTE: do NOT use tome.patch.timm() in combined.py — it will overwrite
        # the custom FlashAttention module inserted there.
        return model
    """
    raise NotImplementedError("Phase 2: implement load_tome_model()")
