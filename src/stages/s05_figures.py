#!/usr/bin/env python3
"""
Stage 05: Figure Generation

Purpose: Generate publication-quality figures.

This stage handles:
- Event study plots
- Main results visualization
- Robustness check plots
- Panel data summary plots

Input Files
-----------
- data_work/panel.parquet
- data_work/diagnostics/*.csv

Output Files
------------
- manuscript_quarto/figures/*.png

Usage
-----
    python src/pipeline.py make_figures
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.helpers import (
    get_data_dir,
    get_figures_dir,
    load_data,
    load_diagnostic,
    ensure_dir,
)
from utils.figure_style import (
    apply_style,
    get_color_palette,
    save_figure,
    annotate_panel,
)
from stages._qa_utils import generate_qa_report, QAMetrics


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = 'panel.parquet'
OUTCOME_VAR = 'outcome'
TREATMENT_VAR = 'treatment'


# ============================================================
# FIGURE FUNCTIONS
# ============================================================

def plot_event_study(
    df: pd.DataFrame,
    output_path: Path,
    pre_periods: int = 12,
    post_periods: int = 12
) -> Path:
    """
    Create event study plot.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with event_time column
    output_path : Path
        Output path for figure
    pre_periods : int
        Number of pre-treatment periods to show
    post_periods : int
        Number of post-treatment periods to show

    Returns
    -------
    Path
        Path to saved figure
    """
    if 'event_time' not in df.columns:
        print("  Warning: No event_time column, skipping event study plot")
        return None

    # Calculate means by event time
    df_valid = df[df['event_time'].notna()].copy()
    event_means = df_valid.groupby('event_time')[OUTCOME_VAR].agg(['mean', 'std', 'count'])
    event_means['se'] = event_means['std'] / np.sqrt(event_means['count'])

    # Filter to desired range
    event_means = event_means[
        (event_means.index >= -pre_periods) &
        (event_means.index <= post_periods)
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = get_color_palette('treatment')

    # Plot means with confidence bands
    ax.fill_between(
        event_means.index,
        event_means['mean'] - 1.96 * event_means['se'],
        event_means['mean'] + 1.96 * event_means['se'],
        alpha=0.2,
        color=colors[0]
    )
    ax.plot(event_means.index, event_means['mean'], 'o-', color=colors[0], markersize=6)

    # Reference lines
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Treatment')

    ax.set_xlabel('Event Time (periods relative to treatment)')
    ax.set_ylabel(f'Mean {OUTCOME_VAR.title()}')
    ax.set_title('Event Study: Dynamic Treatment Effects')
    ax.legend(loc='upper left')

    plt.tight_layout()
    save_figure(fig, output_path, formats=['png'])
    plt.close(fig)

    return output_path


def plot_treatment_control_means(
    df: pd.DataFrame,
    output_path: Path
) -> Path:
    """
    Plot treatment vs control group means over time.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    output_path : Path
        Output path for figure

    Returns
    -------
    Path
        Path to saved figure
    """
    if 'period' not in df.columns or 'ever_treated' not in df.columns:
        print("  Warning: Missing columns for treatment/control plot")
        return None

    # Calculate group means
    group_means = df.groupby(['period', 'ever_treated'])[OUTCOME_VAR].mean().unstack()
    group_means.columns = ['Control', 'Treatment']

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = get_color_palette('treatment')

    ax.plot(group_means.index, group_means['Treatment'], 'o-',
            color=colors[0], label='Treatment Group', markersize=5)
    ax.plot(group_means.index, group_means['Control'], 's-',
            color=colors[1], label='Control Group', markersize=5)

    # Treatment period line
    treatment_period = df.loc[df[TREATMENT_VAR] == 1, 'period'].min()
    if pd.notna(treatment_period):
        ax.axvline(treatment_period, color='gray', linestyle='--',
                   linewidth=1, alpha=0.7, label='Treatment Start')

    ax.set_xlabel('Period')
    ax.set_ylabel(f'Mean {OUTCOME_VAR.title()}')
    ax.set_title('Treatment and Control Group Trends')
    ax.legend(loc='best')

    plt.tight_layout()
    save_figure(fig, output_path, formats=['png'])
    plt.close(fig)

    return output_path


def plot_coefficient_comparison(
    output_path: Path
) -> Path:
    """
    Plot coefficient estimates across specifications.

    Parameters
    ----------
    output_path : Path
        Output path for figure

    Returns
    -------
    Path
        Path to saved figure
    """
    try:
        results = load_diagnostic('estimation_results')
    except FileNotFoundError:
        print("  Warning: No estimation results found")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = get_color_palette('default')

    # Create coefficient plot
    y_positions = range(len(results))
    ax.errorbar(
        results['coefficient'],
        y_positions,
        xerr=1.96 * results['std_error'],
        fmt='o',
        color=colors[0],
        capsize=4,
        capthick=1.5,
        markersize=8
    )

    # Reference line at zero
    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(results['specification'])
    ax.set_xlabel('Coefficient Estimate (95% CI)')
    ax.set_title('Comparison of Coefficient Estimates Across Specifications')

    plt.tight_layout()
    save_figure(fig, output_path, formats=['png'])
    plt.close(fig)

    return output_path


def plot_robustness_summary(output_path: Path) -> Path:
    """
    Plot robustness check results.

    Parameters
    ----------
    output_path : Path
        Output path for figure

    Returns
    -------
    Path
        Path to saved figure
    """
    try:
        results = load_diagnostic('robustness_results')
    except FileNotFoundError:
        print("  Warning: No robustness results found")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = get_color_palette('default')

    # Group by test type
    test_types = results['test_type'].unique()
    y_offset = 0
    y_positions = []
    y_labels = []

    for test_type in test_types:
        subset = results[results['test_type'] == test_type]

        for _, row in subset.iterrows():
            y_positions.append(y_offset)
            y_labels.append(row['test_name'])

            # Color by significance
            color = colors[0] if row['p_value'] < 0.05 else colors[2]

            ax.errorbar(
                row['coefficient'],
                y_offset,
                xerr=1.96 * row['std_error'],
                fmt='o',
                color=color,
                capsize=3,
                markersize=6
            )
            y_offset += 1

        y_offset += 0.5  # Gap between groups

    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Coefficient Estimate (95% CI)')
    ax.set_title('Robustness Check Results')

    plt.tight_layout()
    save_figure(fig, output_path, formats=['png'])
    plt.close(fig)

    return output_path


def plot_outcome_distribution(
    df: pd.DataFrame,
    output_path: Path
) -> Path:
    """
    Plot distribution of outcome variable.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    output_path : Path
        Output path for figure

    Returns
    -------
    Path
        Path to saved figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = get_color_palette('default')

    # Histogram
    axes[0].hist(df[OUTCOME_VAR].dropna(), bins=50, color=colors[0], alpha=0.7, edgecolor='white')
    axes[0].set_xlabel(OUTCOME_VAR.title())
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Outcome')

    # By treatment status
    if TREATMENT_VAR in df.columns:
        for i, (treat, group) in enumerate(df.groupby(TREATMENT_VAR)):
            label = 'Treated' if treat == 1 else 'Control'
            axes[1].hist(group[OUTCOME_VAR].dropna(), bins=30, alpha=0.5,
                        label=label, color=colors[i])
        axes[1].set_xlabel(OUTCOME_VAR.title())
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Outcome by Treatment Status')
        axes[1].legend()

    plt.tight_layout()
    save_figure(fig, output_path, formats=['png'])
    plt.close(fig)

    return output_path


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main(
    figures: Optional[list[str]] = None,
    verbose: bool = True
):
    """
    Execute figure generation pipeline.

    Parameters
    ----------
    figures : list, optional
        Specific figures to generate (default: all)
    verbose : bool
        Print detailed output
    """
    print("=" * 60)
    print("Stage 05: Figure Generation")
    print("=" * 60)

    # Setup paths
    work_dir = get_data_dir('work')
    fig_dir = get_figures_dir()
    input_path = work_dir / INPUT_FILE

    ensure_dir(fig_dir)

    # Apply publication styling
    apply_style('publication')

    # Load data
    print(f"\n  Loading: {INPUT_FILE}")
    if not input_path.exists():
        print(f"  ERROR: Input file not found: {input_path}")
        print("  Run previous stages first.")
        sys.exit(1)

    df = load_data(input_path)
    print(f"    -> {len(df):,} rows")

    # Define all figures
    all_figures = {
        'event_study': lambda: plot_event_study(df, fig_dir / 'fig_event_study'),
        'trends': lambda: plot_treatment_control_means(df, fig_dir / 'fig_trends'),
        'coefficients': lambda: plot_coefficient_comparison(fig_dir / 'fig_coefficients'),
        'robustness': lambda: plot_robustness_summary(fig_dir / 'fig_robustness'),
        'distribution': lambda: plot_outcome_distribution(df, fig_dir / 'fig_distribution'),
    }

    # Determine which figures to generate
    if figures:
        figures_to_make = {k: v for k, v in all_figures.items() if k in figures}
    else:
        figures_to_make = all_figures

    # Generate figures
    generated = []
    print(f"\n  Generating {len(figures_to_make)} figures...")

    for name, func in figures_to_make.items():
        print(f"\n  Creating: {name}")
        try:
            path = func()
            if path:
                generated.append(path)
                print(f"    -> {path}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Summary
    print("\n" + "-" * 60)
    print("FIGURE SUMMARY")
    print("-" * 60)
    print(f"  Generated: {len(generated)} figures")
    print(f"  Output directory: {fig_dir}")

    if verbose and generated:
        print("\n  Files:")
        for path in generated:
            if path:
                print(f"    - {path.name}")

    # Generate QA report
    metrics = QAMetrics()
    metrics.add('n_figures_generated', len(generated))
    metrics.add('output_dir', str(fig_dir))
    if generated:
        total_size = sum(p.stat().st_size for p in generated if p and p.exists())
        metrics.add('total_size_kb', round(total_size / 1024, 1))
    generate_qa_report('s05_figures', metrics)

    print("\n" + "=" * 60)
    print("Stage 05 complete.")
    print("=" * 60)

    return generated


if __name__ == '__main__':
    main()
