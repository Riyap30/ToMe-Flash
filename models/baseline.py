"""
Baseline DeiT-B/16 model loader for Phase 1 of the ToMe-Flash benchmark.

Loads the pretrained DeiT-B/16 checkpoint via timm, moves it to the requested
device in eval mode, and exposes a helper that prints and asserts key architectural
parameters expected by subsequent phases (embed_dim=768, num_heads=12, head_dim=64).
"""

import timm
import torch
import torch.nn as nn


def load_baseline_model(device: str = "cuda") -> nn.Module:
    """Load pretrained DeiT-B/16 in evaluation mode.

    Args:
        device: Target device string, e.g. 'cuda' or 'cpu'.

    Returns:
        The model in eval mode on the specified device.
    """
    model: nn.Module = timm.create_model(
        "deit_base_patch16_224",
        pretrained=True,
    )
    model.eval()
    model.to(device)
    return model


def get_model_info(model: nn.Module) -> None:
    """Print key architectural stats and assert DeiT-B/16 parameter values.

    Checks:
      - embed_dim  == 768
      - num_heads  == 12
      - head_dim   == 64  (embed_dim / num_heads)

    Args:
        model: A timm DeiT-B/16 model instance.

    Raises:
        AssertionError: If any architectural parameter does not match the
                        expected DeiT-B/16 specification.
    """
    n_params: int = sum(p.numel() for p in model.parameters())

    # timm ViT / DeiT models expose these attributes on the model itself
    embed_dim: int = model.embed_dim          # type: ignore[attr-defined]
    num_heads: int = model.blocks[0].attn.num_heads  # type: ignore[attr-defined]
    head_dim: int  = embed_dim // num_heads

    assert embed_dim == 768, (
        f"Expected embed_dim=768, got {embed_dim}. "
        "Is this actually DeiT-B/16?"
    )
    assert num_heads == 12, (
        f"Expected num_heads=12, got {num_heads}."
    )
    assert head_dim == 64, (
        f"Expected head_dim=64, got {head_dim}. "
        "FlashAttention-2 requires head_dim in {{16,32,64,128}}."
    )

    print("=" * 50)
    print("Model: DeiT-B/16 (baseline)")
    print(f"  Parameters : {n_params:,}")
    print(f"  embed_dim  : {embed_dim}  ✓")
    print(f"  num_heads  : {num_heads}  ✓")
    print(f"  head_dim   : {head_dim}  ✓  (compatible with FlashAttention-2)")
    print("=" * 50)
