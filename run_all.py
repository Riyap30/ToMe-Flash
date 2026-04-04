"""
Full 2×2 factorial experiment orchestrator (Phases 1–3).

Runs all four conditions in sequence:
  A: Baseline         (no ToMe, no FlashAttention)
  B: ToMe only        (r ∈ {4, 8, 16})
  C: FlashAttn only
  D: Combined ToMe + FlashAttn (r ∈ {4, 8, 16})

TODO (Phase 4): implement end-to-end run_all() logic and statistical comparisons.
"""

# Phase 2/3 stub imports — not yet implemented
from models import tome_only    # noqa: F401  (stub)
from models import flash_only   # noqa: F401  (stub)
from models import combined     # noqa: F401  (stub)


def main() -> None:
    print("Phase 1 complete. Phases 2-3 not yet implemented.")


if __name__ == "__main__":
    main()
