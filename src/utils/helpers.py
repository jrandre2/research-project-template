#!/usr/bin/env python3
"""
Common utility functions for the research pipeline.

This module provides shared helper functions used across multiple stages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import os


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get the project root directory."""
    # Walk up from this file to find the project root
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'src').exists() and (parent / 'manuscript_quarto').exists():
            return parent
    raise RuntimeError("Could not find project root")


def format_pvalue(p: float, threshold: float = 0.001) -> str:
    """Format p-value for display."""
    if p < threshold:
        return f"<{threshold}"
    return f"{p:.3f}"


def format_ci(lo: float, hi: float, decimals: int = 3) -> str:
    """Format confidence interval as [lo, hi]."""
    return f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"


def add_significance_stars(p: float) -> str:
    """Add significance stars based on p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def load_config(config_name: str) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_name : str
        Name of the config file (without extension) in manuscript_quarto/journal_configs/

    Returns
    -------
    dict
        Parsed configuration
    """
    import yaml
    config_path = get_project_root() / 'manuscript_quarto' / 'journal_configs' / f'{config_name}.yml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)
