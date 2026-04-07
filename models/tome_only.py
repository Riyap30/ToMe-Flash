"""
models/tome_only.py

Loads DeiT-B/16 with Token Merging (ToMe) applied via bipartite soft matching.
Used for the ToMe-only condition in the 2×2 factorial experiment.

ToMe reference: Bolya et al., "Token Merging: Your ViT But Faster", ICLR 2023.

IMPORTANT: tome.patch.timm() modifies the model in-place. Always load a fresh
model with timm.create_model() before patching — never re-patch an already-patched
model, as the merge hooks will stack incorrectly.
"""

from typing import Dict

import timm
import tome
import torch.nn as nn


def load_tome_model(r: int, device: str = "cuda") -> nn.Module:
    """Load a fresh DeiT-B/16 and apply ToMe bipartite soft matching with ratio r.

    Args:
        r:      Number of token pairs merged per transformer layer.
                Sweep values: r ∈ {4, 8, 16}.
        device: Target device string (e.g. 'cuda' or 'cpu').

    Returns:
        ToMe-patched model in eval mode on the specified device.
    """
    model: nn.Module = timm.create_model(
        "deit_base_patch16_224",
        pretrained=True,
    )
    model.eval()
    model.to(device)

    # Patch in-place; must be called on a fresh model each time.
    # r is set as a model attribute after patching — not a patch.timm() argument.
    tome.patch.timm(model)
    model.r = r

    final_tokens = 197 - r * 12
    print(f"ToMe applied with r={r}. Tokens per image at final layer: {final_tokens}")

    return model


def get_tome_info(_model: nn.Module, r: int) -> Dict:
    """Return and print a summary of ToMe token reduction statistics.

    Args:
        model: A ToMe-patched timm model (used only for context; info is derived
               from r and the fixed DeiT-B/16 architecture).
        r:     Merge ratio used when patching the model.

    Returns:
        dict with keys: r, initial_tokens, final_tokens, reduction_pct
    """
    initial_tokens = 197
    final_tokens   = initial_tokens - r * 12
    reduction_pct  = round((r * 12 / initial_tokens) * 100, 1)

    info = {
        "r":              r,
        "initial_tokens": initial_tokens,
        "final_tokens":   final_tokens,
        "reduction_pct":  reduction_pct,
    }

    print("=" * 50)
    print(f"ToMe configuration  (r={r})")
    print("=" * 50)
    print(f"  Initial tokens : {initial_tokens}  (196 patch + 1 CLS)")
    print(f"  Merged per layer: {r}  ×  12 layers  =  {r * 12} total")
    print(f"  Final tokens   : {final_tokens}")
    print(f"  Token reduction: {reduction_pct}%")
    print("=" * 50)

    return info
