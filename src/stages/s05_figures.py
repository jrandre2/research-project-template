#!/usr/bin/env python3
"""
Stage 05: Figure Generation

Purpose: Generate publication-quality figures.

Input Files
-----------
- data_work/panel.parquet
- data_work/diagnostics/*.csv

Output Files
------------
- figures/*.png
- figures/*.pdf

Usage
-----
    python src/pipeline.py make_figures
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.figure_style import apply_style

# Define paths
DATA_DIR = Path('data_work')
DIAG_DIR = DATA_DIR / 'diagnostics'
FIG_DIR = Path('figures')


def main():
    """Execute figure generation pipeline."""
    print("Stage 05: Figure Generation")
    print("-" * 40)

    # Ensure output directory exists
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Apply consistent styling
    apply_style()

    # TODO: Implement your figure generation here
    # Example:
    # import pandas as pd
    # import matplotlib.pyplot as plt
    #
    # df = pd.read_csv(DIAG_DIR / 'estimation_results.csv')
    # fig, ax = plt.subplots()
    # ax.plot(df['x'], df['y'])
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # fig.savefig(FIG_DIR / 'fig_main_result.png', dpi=300, bbox_inches='tight')

    print(f"Output: {FIG_DIR}/")
    print("Stage 05 complete.")


if __name__ == '__main__':
    main()
