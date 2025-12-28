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


def _find_project_root() -> Path:
    """Find project root by looking for characteristic files."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / 'src').exists() and (parent / 'manuscript_quarto').exists():
            return parent
    # Fallback: assume manuscript_quarto is one level down from root
    return Path(__file__).parent.parent.parent


# Project paths
PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data_work" / "diagnostics"
FIG_DIR = PROJECT_ROOT / "manuscript_quarto" / "figures"


def data_available() -> bool:
    """Check if pipeline data is available for manuscript rendering."""
    return DATA_DIR.exists() and any(DATA_DIR.glob("*.csv"))


def load_diagnostic(name: str, required: bool = True) -> pd.DataFrame:
    """Load a diagnostic CSV file by name (without .csv extension).

    Parameters
    ----------
    name : str
        Name of the diagnostic file (without .csv extension)
    required : bool
        If True, raise error when file missing. If False, return empty DataFrame.

    Returns
    -------
    pd.DataFrame
        Loaded data, or empty DataFrame if not required and missing
    """
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Diagnostic file not found: {path}\n"
                f"Run the pipeline first: python src/pipeline.py ingest_data --demo && "
                f"python src/pipeline.py run_estimation"
            )
        return pd.DataFrame()
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
