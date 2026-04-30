#!/usr/bin/env python3
"""Generate presentation-ready evaluation figures for ToMe-Flash experiments.

Usage:
    python scripts/generate_eval_figures.py --results_dir "results latest"
"""

from eval_figures.pipeline import main


if __name__ == "__main__":
    main()
