"""
R/fixest Analysis Engine.

Uses the R fixest package for high-performance fixed effects estimation.
Communication with R is via Parquet files and JSON specifications.

Usage
-----
    from analysis import get_engine

    engine = get_engine('r')
    result = engine.estimate(data_path, specification, output_dir)

Requirements
------------
- R >= 4.0
- R packages: arrow, fixest, jsonlite

Installation:
    Rscript -e "install.packages(c('arrow', 'fixest', 'jsonlite'))"
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from ..base import BaseAnalysisEngine, EstimationResult
from ..factory import register_engine


@register_engine('r')
class REngine(BaseAnalysisEngine):
    """
    R/fixest estimation engine.

    Uses subprocess to call R scripts with data passed via Parquet files
    and specifications/results passed via JSON files.

    Features
    --------
    - High-performance fixed effects via fixest::feols()
    - Clustered standard errors
    - Automatic R version and package validation
    """

    def __init__(self):
        super().__init__()
        self._r_executable = self._get_r_executable()
        self._timeout = self._get_timeout()
        self._scripts_dir = Path(__file__).parent / 'r'
        self._cached_version: Optional[str] = None

    @property
    def name(self) -> str:
        return 'r'

    @property
    def version(self) -> str:
        if self._cached_version:
            return self._cached_version

        try:
            result = subprocess.run(
                [self._r_executable, '--version'],
                capture_output=True,
                text=True,
                timeout=10,
            )
            # Parse first line for version
            first_line = result.stdout.split('\n')[0] if result.stdout else ''
            self._cached_version = first_line or 'R (version unknown)'
            return self._cached_version
        except Exception:
            return 'R (not found)'

    def validate_installation(self) -> tuple[bool, str]:
        """
        Check if R and required packages are available.

        Returns
        -------
        tuple[bool, str]
            (is_available, message)
        """
        # Check R executable exists
        if not shutil.which(self._r_executable):
            return False, f"R not found at '{self._r_executable}'. Set R_EXECUTABLE env var."

        # Check required packages
        check_script = self._scripts_dir / 'check_packages.R'
        if not check_script.exists():
            return False, f"R check script not found: {check_script}"

        try:
            result = subprocess.run(
                [self._r_executable, str(check_script)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                output = result.stdout.strip() or result.stderr.strip()
                return False, f"R packages check failed: {output}"

            # Parse package versions from output
            lines = result.stdout.strip().split('\n')
            if lines and lines[0] == 'OK':
                versions = ', '.join(lines[1:])
                return True, f"R engine ready ({versions})"

            return True, "R engine ready"

        except subprocess.TimeoutExpired:
            return False, "R package check timed out"
        except Exception as e:
            return False, f"Error checking R: {e}"

    def estimate(
        self,
        data_path: Path,
        specification: dict,
        output_dir: Path,
    ) -> EstimationResult:
        """
        Run fixed effects estimation using R/fixest.

        Parameters
        ----------
        data_path : Path
            Path to input data (Parquet format)
        specification : dict
            Specification dictionary
        output_dir : Path
            Directory for output files

        Returns
        -------
        EstimationResult
            Estimation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        spec_name = specification.get('name', 'unnamed')

        # Write specification to JSON
        spec_path = output_dir / f'spec_{spec_name}.json'
        with open(spec_path, 'w') as f:
            json.dump(specification, f, indent=2)

        # Output path for results
        result_path = output_dir / f'result_{spec_name}.json'

        # Run R script
        estimate_script = self._scripts_dir / 'estimate.R'
        if not estimate_script.exists():
            raise RuntimeError(f"R estimation script not found: {estimate_script}")

        try:
            result = subprocess.run(
                [
                    self._r_executable,
                    str(estimate_script),
                    str(data_path),
                    str(spec_path),
                    str(result_path),
                ],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                raise RuntimeError(f"R estimation failed:\n{error_msg}")

            # Parse results
            if not result_path.exists():
                raise RuntimeError(f"R did not produce output file: {result_path}")

            with open(result_path) as f:
                data = json.load(f)

            return EstimationResult.from_dict(data)

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"R estimation timed out after {self._timeout}s"
            )
        finally:
            # Cleanup temp files (keep result for debugging if needed)
            spec_path.unlink(missing_ok=True)

    def _get_r_executable(self) -> str:
        """Get R executable path from config."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from config import R_EXECUTABLE
            return R_EXECUTABLE
        except ImportError:
            return 'Rscript'

    def _get_timeout(self) -> int:
        """Get process timeout from config."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from config import EXTERNAL_PROCESS_TIMEOUT
            return EXTERNAL_PROCESS_TIMEOUT
        except ImportError:
            return 3600
