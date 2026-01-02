"""
Base Protocol and Types for Analysis Engines.

Defines the interface that all analysis engines must implement.
Uses Python's Protocol for structural subtyping (duck typing with type hints).

Usage
-----
    from analysis.base import AnalysisEngine, EstimationResult

    class MyEngine(BaseAnalysisEngine):
        @property
        def name(self) -> str:
            return 'my_engine'

        def estimate(self, data_path, specification, output_dir) -> EstimationResult:
            ...
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class EstimationResult:
    """
    Language-agnostic container for estimation results.

    This dataclass standardizes output from all engines (Python, R, Stata, Julia)
    to enable cross-engine comparison and validation.

    Attributes
    ----------
    specification : str
        Name of the specification that was run
    n_obs : int
        Number of observations
    n_units : int
        Number of unique units (for panel data)
    n_periods : int
        Number of time periods (for panel data)
    coefficients : dict[str, float]
        Estimated coefficients by variable name
    std_errors : dict[str, float]
        Standard errors by variable name
    t_stats : dict[str, float]
        T-statistics by variable name
    p_values : dict[str, float]
        P-values by variable name
    ci_lower : dict[str, float]
        Lower confidence interval bounds by variable name
    ci_upper : dict[str, float]
        Upper confidence interval bounds by variable name
    r_squared : float
        R-squared (or pseudo R-squared)
    r_squared_within : float, optional
        Within R-squared (for fixed effects models)
    fixed_effects : list[str]
        Fixed effects included in the model
    cluster_var : str, optional
        Variable used for clustering standard errors
    controls : list[str]
        Control variables included
    warnings : list[str]
        Any warnings generated during estimation
    engine : str
        Name of the engine that produced this result
    engine_version : str
        Version of the engine/package
    execution_time_seconds : float
        Time taken for estimation
    """

    specification: str
    n_obs: int
    n_units: int = 0
    n_periods: int = 0
    coefficients: dict = field(default_factory=dict)
    std_errors: dict = field(default_factory=dict)
    t_stats: dict = field(default_factory=dict)
    p_values: dict = field(default_factory=dict)
    ci_lower: dict = field(default_factory=dict)
    ci_upper: dict = field(default_factory=dict)
    r_squared: float = 0.0
    r_squared_within: Optional[float] = None
    fixed_effects: list = field(default_factory=list)
    cluster_var: Optional[str] = None
    controls: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    engine: str = 'unknown'
    engine_version: str = 'unknown'
    execution_time_seconds: float = 0.0

    def get_coefficient(self, var: str) -> Optional[float]:
        """Get coefficient for a variable, or None if not found."""
        return self.coefficients.get(var)

    def get_std_error(self, var: str) -> Optional[float]:
        """Get standard error for a variable, or None if not found."""
        return self.std_errors.get(var)

    def get_p_value(self, var: str) -> Optional[float]:
        """Get p-value for a variable, or None if not found."""
        return self.p_values.get(var)

    def is_significant(self, var: str, level: float = 0.05) -> bool:
        """Check if coefficient is significant at given level."""
        p = self.p_values.get(var)
        return p is not None and p < level

    def format_coefficient(self, var: str, decimals: int = 3) -> str:
        """Format coefficient with significance stars and SE."""
        coef = self.coefficients.get(var)
        se = self.std_errors.get(var)
        p = self.p_values.get(var)

        if coef is None:
            return ''

        # Add significance stars
        if p is not None:
            if p < 0.01:
                stars = '***'
            elif p < 0.05:
                stars = '**'
            elif p < 0.10:
                stars = '*'
            else:
                stars = ''
        else:
            stars = ''

        if se is not None:
            return f"{coef:.{decimals}f}{stars} ({se:.{decimals}f})"
        return f"{coef:.{decimals}f}{stars}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """Write result to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Path) -> 'EstimationResult':
        """Load result from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> 'EstimationResult':
        """Create from dictionary."""
        return cls(**data)


@runtime_checkable
class AnalysisEngine(Protocol):
    """
    Protocol defining the interface for analysis engines.

    All engines must implement these methods to be compatible
    with the CENTAUR estimation system.

    Attributes
    ----------
    name : str
        Engine identifier (e.g., 'python', 'r', 'stata')

    Methods
    -------
    validate_installation()
        Check if the engine is properly configured.
    estimate(data_path, specification, output_dir)
        Run estimation for a single specification.
    estimate_batch(data_path, specifications, output_dir, parallel, n_workers)
        Run estimation for multiple specifications.
    """

    @property
    def name(self) -> str:
        """Return the engine identifier (e.g., 'python', 'r', 'stata')."""
        ...

    @property
    def version(self) -> str:
        """Return the engine/package version."""
        ...

    def validate_installation(self) -> tuple[bool, str]:
        """
        Check if the engine is properly configured and available.

        Returns
        -------
        tuple[bool, str]
            (is_available, message) - True if usable, with status message
        """
        ...

    def estimate(
        self,
        data_path: Path,
        specification: dict,
        output_dir: Path,
    ) -> EstimationResult:
        """
        Run estimation for a single specification.

        Parameters
        ----------
        data_path : Path
            Path to input data file (Parquet format)
        specification : dict
            Specification dictionary with keys:
            - name: str
            - outcome: str
            - treatment: str
            - controls: list[str]
            - fixed_effects: list[str]
            - cluster: str (optional)
        output_dir : Path
            Directory for output files

        Returns
        -------
        EstimationResult
            Standardized estimation results
        """
        ...

    def estimate_batch(
        self,
        data_path: Path,
        specifications: list[dict],
        output_dir: Path,
        parallel: bool = True,
        n_workers: Optional[int] = None,
    ) -> list[EstimationResult]:
        """
        Run estimation for multiple specifications.

        Parameters
        ----------
        data_path : Path
            Path to input data file
        specifications : list[dict]
            List of specification dictionaries
        output_dir : Path
            Directory for output files
        parallel : bool
            Whether to run in parallel (if supported)
        n_workers : int, optional
            Number of parallel workers

        Returns
        -------
        list[EstimationResult]
            Results for each specification
        """
        ...


class BaseAnalysisEngine:
    """
    Base implementation with common functionality for analysis engines.

    Provides default implementations for shared behavior.
    Concrete engines should inherit from this class.
    """

    def __init__(self):
        """Initialize base engine."""
        pass

    @property
    def name(self) -> str:
        """Return engine name (must be overridden)."""
        raise NotImplementedError

    @property
    def version(self) -> str:
        """Return engine version (must be overridden)."""
        raise NotImplementedError

    def validate_installation(self) -> tuple[bool, str]:
        """Validate installation (must be overridden)."""
        raise NotImplementedError

    def estimate(
        self,
        data_path: Path,
        specification: dict,
        output_dir: Path,
    ) -> EstimationResult:
        """Run estimation (must be overridden)."""
        raise NotImplementedError

    def estimate_batch(
        self,
        data_path: Path,
        specifications: list[dict],
        output_dir: Path,
        parallel: bool = True,
        n_workers: Optional[int] = None,
    ) -> list[EstimationResult]:
        """
        Default batch estimation implementation.

        Runs specifications sequentially. Override for parallel support.
        """
        results = []
        for spec in specifications:
            try:
                result = self.estimate(data_path, spec, output_dir)
                results.append(result)
            except Exception as e:
                print(f"  ERROR in {spec.get('name', 'unknown')}: {e}")
        return results

    def _run_subprocess(
        self,
        command: list[str],
        timeout: int = 3600,
        cwd: Optional[Path] = None,
    ) -> subprocess.CompletedProcess:
        """
        Run an external command with timeout and error handling.

        Parameters
        ----------
        command : list[str]
            Command and arguments
        timeout : int
            Timeout in seconds
        cwd : Path, optional
            Working directory

        Returns
        -------
        subprocess.CompletedProcess
            Completed process result

        Raises
        ------
        RuntimeError
            If command fails or times out
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            return result
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Command timed out after {timeout}s: {' '.join(command)}"
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"Command not found: {command[0]}"
            )

    def _write_temp_spec(self, specification: dict, output_dir: Path) -> Path:
        """Write specification to temporary JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        spec_path = output_dir / f"spec_{specification.get('name', 'temp')}.json"
        with open(spec_path, 'w') as f:
            json.dump(specification, f, indent=2)
        return spec_path
