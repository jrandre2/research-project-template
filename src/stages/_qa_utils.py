#!/usr/bin/env python3
"""
Quality Assurance Utilities for Pipeline Stages.

This module provides functions for generating per-stage QA reports
that track data quality metrics throughout the pipeline.

Usage
-----
    from stages._qa_utils import generate_qa_report, QAMetrics

    # At the end of a pipeline stage:
    metrics = QAMetrics()
    metrics.add('n_rows', len(df))
    metrics.add('n_columns', len(df.columns))
    metrics.add('missing_pct', df.isna().sum().sum() / df.size * 100)

    generate_qa_report('s00_ingest', metrics)
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

# Add parent for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import ENABLE_QA_REPORTS, QA_REPORTS_DIR
except ImportError:
    # Fallback if config not available
    ENABLE_QA_REPORTS = True
    QA_REPORTS_DIR = Path(__file__).parent.parent.parent / 'data_work' / 'quality'


class QAMetrics:
    """
    Container for QA metrics collected during a pipeline stage.

    Examples
    --------
    >>> metrics = QAMetrics()
    >>> metrics.add('n_rows', 1000)
    >>> metrics.add('n_columns', 15)
    >>> metrics.add_pct('missing', 2.5)
    >>> metrics.to_dict()
    {'n_rows': 1000, 'n_columns': 15, 'missing_pct': 2.5}
    """

    def __init__(self):
        self._metrics: dict[str, Any] = {}

    def add(self, name: str, value: Any) -> 'QAMetrics':
        """Add a metric."""
        self._metrics[name] = value
        return self

    def add_pct(self, name: str, value: float) -> 'QAMetrics':
        """Add a percentage metric (appends '_pct' to name)."""
        self._metrics[f'{name}_pct'] = round(value, 2)
        return self

    def add_count(self, name: str, value: int) -> 'QAMetrics':
        """Add a count metric (appends '_count' to name)."""
        self._metrics[f'{name}_count'] = value
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self._metrics.copy()

    def __len__(self) -> int:
        return len(self._metrics)

    def __repr__(self) -> str:
        return f"QAMetrics({self._metrics})"


def compute_dataframe_metrics(df) -> QAMetrics:
    """
    Compute standard QA metrics for a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to analyze

    Returns
    -------
    QAMetrics
        Metrics including row count, column count, missing values, etc.
    """
    metrics = QAMetrics()

    # Basic counts
    metrics.add('n_rows', len(df))
    metrics.add('n_columns', len(df.columns))

    # Missing values
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    metrics.add_count('missing_cells', int(missing_cells))
    if total_cells > 0:
        metrics.add_pct('missing', (missing_cells / total_cells) * 100)
    else:
        metrics.add_pct('missing', 0.0)

    # Duplicates
    n_duplicates = df.duplicated().sum()
    metrics.add_count('duplicate_rows', int(n_duplicates))
    if len(df) > 0:
        metrics.add_pct('duplicate', (n_duplicates / len(df)) * 100)
    else:
        metrics.add_pct('duplicate', 0.0)

    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    metrics.add('memory_mb', round(memory_mb, 2))

    return metrics


def generate_qa_report(
    stage_name: str,
    metrics: Union[QAMetrics, dict],
    output_dir: Optional[Path] = None,
    include_timestamp: bool = True,
) -> Optional[Path]:
    """
    Generate a QA report for a pipeline stage.

    Parameters
    ----------
    stage_name : str
        Name of the stage (e.g., 's00_ingest', 's01_link')
    metrics : QAMetrics or dict
        Metrics to include in the report
    output_dir : Path, optional
        Output directory (default: QA_REPORTS_DIR from config)
    include_timestamp : bool
        Whether to include timestamp in filename (default: True)

    Returns
    -------
    Path or None
        Path to generated report, or None if QA reports are disabled
    """
    if not ENABLE_QA_REPORTS:
        return None

    # Convert QAMetrics to dict if needed
    if isinstance(metrics, QAMetrics):
        metrics_dict = metrics.to_dict()
    else:
        metrics_dict = dict(metrics)

    # Determine output directory
    if output_dir is None:
        output_dir = QA_REPORTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if include_timestamp:
        filename = f'{stage_name}_quality_{timestamp}.csv'
    else:
        filename = f'{stage_name}_quality.csv'

    report_path = output_dir / filename

    # Import pandas here to avoid circular imports
    try:
        import pandas as pd
    except ImportError:
        # Fallback to simple CSV writing
        with open(report_path, 'w') as f:
            f.write('metric,value,stage,timestamp\n')
            for key, value in metrics_dict.items():
                f.write(f'{key},{value},{stage_name},{timestamp}\n')
        print(f"QA report saved: {report_path}")
        return report_path

    # Create DataFrame
    rows = [
        {
            'metric': key,
            'value': value,
            'stage': stage_name,
            'timestamp': timestamp,
        }
        for key, value in metrics_dict.items()
    ]

    df = pd.DataFrame(rows)
    df.to_csv(report_path, index=False)

    print(f"QA report saved: {report_path}")
    return report_path


def print_qa_summary(metrics: Union[QAMetrics, dict], stage_name: str = '') -> None:
    """
    Print a formatted summary of QA metrics.

    Parameters
    ----------
    metrics : QAMetrics or dict
        Metrics to display
    stage_name : str, optional
        Stage name for header
    """
    if isinstance(metrics, QAMetrics):
        metrics_dict = metrics.to_dict()
    else:
        metrics_dict = dict(metrics)

    if stage_name:
        print(f"\nQA Summary: {stage_name}")
    else:
        print("\nQA Summary")
    print("-" * 40)

    for key, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        elif isinstance(value, int):
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")


def check_thresholds(
    metrics: Union[QAMetrics, dict],
    thresholds: Optional[dict] = None,
) -> list[str]:
    """
    Check metrics against thresholds and return warnings.

    Parameters
    ----------
    metrics : QAMetrics or dict
        Metrics to check
    thresholds : dict, optional
        Threshold definitions. Keys should match metric names.
        Values should be tuples of (operator, value) where operator
        is 'max', 'min', 'eq', etc.

    Returns
    -------
    list[str]
        List of warning messages for threshold violations
    """
    if thresholds is None:
        try:
            from config import QA_THRESHOLDS
            thresholds = QA_THRESHOLDS
        except ImportError:
            thresholds = {}

    if isinstance(metrics, QAMetrics):
        metrics_dict = metrics.to_dict()
    else:
        metrics_dict = dict(metrics)

    warnings = []

    # Check missing percentage
    if 'missing_pct' in metrics_dict and 'max_missing_pct' in thresholds:
        if metrics_dict['missing_pct'] > thresholds['max_missing_pct']:
            warnings.append(
                f"Missing values ({metrics_dict['missing_pct']:.1f}%) exceed "
                f"threshold ({thresholds['max_missing_pct']}%)"
            )

    # Check row count
    if 'n_rows' in metrics_dict and 'min_row_count' in thresholds:
        if metrics_dict['n_rows'] < thresholds['min_row_count']:
            warnings.append(
                f"Row count ({metrics_dict['n_rows']}) below "
                f"threshold ({thresholds['min_row_count']})"
            )

    # Check duplicate percentage
    if 'duplicate_pct' in metrics_dict and 'max_duplicate_pct' in thresholds:
        if metrics_dict['duplicate_pct'] > thresholds['max_duplicate_pct']:
            warnings.append(
                f"Duplicate rows ({metrics_dict['duplicate_pct']:.1f}%) exceed "
                f"threshold ({thresholds['max_duplicate_pct']}%)"
            )

    return warnings


# =============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON PATTERNS
# =============================================================================

def qa_for_stage(
    stage_name: str,
    df,
    additional_metrics: Optional[dict] = None,
    output_file: Optional[str] = None,
) -> Optional[Path]:
    """
    Complete QA workflow for a pipeline stage.

    This is a convenience function that:
    1. Computes standard DataFrame metrics
    2. Adds any additional metrics
    3. Checks thresholds and prints warnings
    4. Generates the QA report
    5. Prints a summary

    Parameters
    ----------
    stage_name : str
        Name of the stage
    df : pandas.DataFrame
        The output DataFrame to analyze
    additional_metrics : dict, optional
        Additional metrics to include
    output_file : str, optional
        Name of the output file (for inclusion in metrics)

    Returns
    -------
    Path or None
        Path to generated report
    """
    # Compute metrics
    metrics = compute_dataframe_metrics(df)

    # Add output file info
    if output_file:
        metrics.add('output_file', str(output_file))

    # Add any additional metrics
    if additional_metrics:
        for key, value in additional_metrics.items():
            metrics.add(key, value)

    # Check thresholds
    warnings = check_thresholds(metrics)
    if warnings:
        print(f"\nQA Warnings for {stage_name}:")
        for warning in warnings:
            print(f"  WARNING: {warning}")

    # Print summary
    print_qa_summary(metrics, stage_name)

    # Generate report
    return generate_qa_report(stage_name, metrics)
