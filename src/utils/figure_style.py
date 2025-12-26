#!/usr/bin/env python3
"""
Figure styling configuration for manuscript.

This module sets matplotlib defaults to match the LaTeX article document style
used by Quarto. Import this module at the start of any script that generates
figures for the manuscript.

Usage
-----
from src.utils.figure_style import apply_style
apply_style()
"""
import matplotlib.pyplot as plt

# Style parameters matching LaTeX article class
FONT_FAMILY = 'serif'
FONT_SIZE = 10
TITLE_SIZE = 11
LABEL_SIZE = 10
TICK_SIZE = 9
LEGEND_SIZE = 9

# Figure dimensions (inches) for single-column layout
FIG_WIDTH_SINGLE = 6.5
FIG_HEIGHT_SINGLE = 4.5

# For two-panel figures
FIG_WIDTH_DOUBLE = 6.5
FIG_HEIGHT_DOUBLE = 8.0

# DPI for saved figures
DPI = 300


def apply_style():
    """Apply consistent figure styling for manuscript figures."""
    plt.rcParams.update({
        # Font settings
        'font.family': FONT_FAMILY,
        'font.size': FONT_SIZE,

        # Title and labels
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': LABEL_SIZE,

        # Ticks
        'xtick.labelsize': TICK_SIZE,
        'ytick.labelsize': TICK_SIZE,

        # Legend
        'legend.fontsize': LEGEND_SIZE,
        'legend.framealpha': 0.9,

        # Figure size (default)
        'figure.figsize': (FIG_WIDTH_SINGLE, FIG_HEIGHT_SINGLE),
        'figure.dpi': 100,
        'savefig.dpi': DPI,

        # Grid
        'axes.grid': True,
        'grid.alpha': 0.3,

        # Spines
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Layout
        'figure.constrained_layout.use': True,
        'axes.xmargin': 0.0,
        'axes.autolimit_mode': 'data',
    })


def get_figure_single():
    """Create a single-panel figure with standard dimensions."""
    apply_style()
    return plt.figure(figsize=(FIG_WIDTH_SINGLE, FIG_HEIGHT_SINGLE))


def get_figure_double():
    """Create a two-panel figure with standard dimensions."""
    apply_style()
    return plt.figure(figsize=(FIG_WIDTH_DOUBLE, FIG_HEIGHT_DOUBLE))


if __name__ == '__main__':
    # Test the style
    apply_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9], 'o-', label='Test data')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_title('Test Figure with Manuscript Style')
    ax.legend()
    plt.savefig('test_style.png')
    print('Saved test_style.png')
