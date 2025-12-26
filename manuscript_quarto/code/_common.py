"""
Common utilities for manuscript table and figure rendering.
Loads diagnostic CSVs and provides helper functions for table rendering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import Markdown, display


def show_table(df: pd.DataFrame) -> None:
    """Display a DataFrame as a properly rendered markdown table.

    This function works correctly in both HTML and PDF output formats.
    """
    display(Markdown(df.to_markdown(index=False)))


# Paths relative to manuscript_quarto/
DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR = Path(__file__).parent.parent / "figures"


def load_diagnostic(name: str) -> pd.DataFrame:
    """Load a diagnostic CSV file by name (without .csv extension)."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Diagnostic file not found: {path}")
    return pd.read_csv(path)


def format_pvalue(p: float, threshold: float = 0.001) -> str:
    """Format p-value for display."""
    if p < threshold:
        return f"<{threshold}"
    return f"{p:.3f}"


def format_ci(lo: float, hi: float, decimals: int = 3) -> str:
    """Format confidence interval as [lo, hi]."""
    return f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"


def format_percent(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def add_significance_stars(p: float) -> str:
    """Add significance stars based on p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


# Example project-specific loaders (customize for your project)
def load_main_results() -> pd.DataFrame:
    """Load main estimation results."""
    return load_diagnostic("main_results")


def load_robustness() -> pd.DataFrame:
    """Load robustness check results."""
    return load_diagnostic("robustness_results")
