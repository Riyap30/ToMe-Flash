"""
Model definitions for the 2×2 factorial benchmark experiment.

  baseline   — DeiT-B/16, standard forward pass (Phase 1)
  tome_only  — ToMe-patched DeiT-B/16 (Phase 2)
  flash_only — DeiT-B/16 with FlashAttention-2 kernel (Phase 2)
  combined   — Custom ToMe + FlashAttention-2 attention class (Phase 3)
"""

from .baseline import load_baseline_model, get_model_info

__all__ = ["load_baseline_model", "get_model_info"]
