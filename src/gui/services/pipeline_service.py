"""
Pipeline service for stage discovery and execution.

Provides access to pipeline stage information and controls
stage execution via subprocess.
"""
from __future__ import annotations

import asyncio
import importlib
import re
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional

from ..config import PROJECT_ROOT, DATA_WORK_DIR
from ..models import StageInfo, StageStatus


class PipelineService:
    """Service for interacting with pipeline stages."""

    def __init__(self):
        self.stages_dir = PROJECT_ROOT / "src" / "stages"
        self.data_work_dir = DATA_WORK_DIR

    def discover_stages(self) -> list[StageInfo]:
        """
        Discover all available pipeline stages.

        Returns
        -------
        list[StageInfo]
            List of stage information, sorted by stage number.
        """
        stages = []
        stage_pattern = re.compile(r"^s(\d+[a-z]?)_(.+)\.py$")

        if not self.stages_dir.exists():
            return stages

        for file in sorted(self.stages_dir.glob("s*.py")):
            match = stage_pattern.match(file.name)
            if match and not file.name.startswith("_"):
                stage_name = file.stem
                description = self._extract_description(file)
                version = match.group(1) if match.group(1)[-1].isalpha() else None

                # Check for QA report
                has_qa = self._has_qa_report(stage_name)

                # Get status
                status = self._get_stage_status(stage_name)

                stages.append(StageInfo(
                    name=stage_name,
                    description=description,
                    version=version,
                    status=status,
                    has_qa_report=has_qa,
                    output_files=self._get_output_files(stage_name),
                ))

        return stages

    def get_stage(self, stage_name: str) -> Optional[StageInfo]:
        """Get information about a specific stage."""
        stages = self.discover_stages()
        for stage in stages:
            if stage.name == stage_name:
                return stage
        return None

    def _extract_description(self, file_path: Path) -> str:
        """Extract the Purpose line from a stage module's docstring."""
        try:
            content = file_path.read_text()
            # Look for Purpose: line in docstring
            purpose_match = re.search(
                r'Purpose:\s*(.+?)(?:\n\n|\n[A-Z])',
                content,
                re.DOTALL
            )
            if purpose_match:
                return purpose_match.group(1).strip().replace('\n', ' ')

            # Fallback: first line of docstring
            docstring_match = re.search(r'"""(.+?)"""', content, re.DOTALL)
            if docstring_match:
                first_line = docstring_match.group(1).strip().split('\n')[0]
                return first_line

            return "No description available"
        except Exception:
            return "No description available"

    def _has_qa_report(self, stage_name: str) -> bool:
        """Check if a QA report exists for this stage."""
        qa_dir = self.data_work_dir / "quality"
        if not qa_dir.exists():
            return False
        return any(qa_dir.glob(f"{stage_name}_quality_*.csv"))

    def _get_stage_status(self, stage_name: str) -> StageStatus:
        """
        Determine stage status based on output files and QA reports.

        This is a heuristic - actual status would require execution tracking.
        """
        # Check for QA report as proxy for successful completion
        if self._has_qa_report(stage_name):
            return StageStatus.SUCCESS

        # Check for known output files
        output_files = self._get_output_files(stage_name)
        if output_files:
            return StageStatus.SUCCESS

        return StageStatus.IDLE

    def _get_output_files(self, stage_name: str) -> list[str]:
        """Get list of output files for a stage."""
        outputs = []

        # Map stages to their expected outputs
        output_map = {
            "s00_ingest": ["data_raw.parquet"],
            "s01_link": ["data_linked.parquet"],
            "s02_panel": ["panel.parquet"],
            "s03_estimation": ["estimates.parquet"],
            "s04_robustness": ["robustness.parquet"],
        }

        expected = output_map.get(stage_name, [])
        for filename in expected:
            path = self.data_work_dir / filename
            if path.exists():
                outputs.append(str(path.relative_to(PROJECT_ROOT)))

        return outputs

    async def run_stage(
        self,
        stage_name: str,
        options: Optional[list[str]] = None
    ) -> AsyncIterator[str]:
        """
        Run a pipeline stage and yield log output.

        Parameters
        ----------
        stage_name : str
            Name of the stage to run.
        options : list[str], optional
            Additional command-line options.

        Yields
        ------
        str
            Lines of output from the stage execution.
        """
        cmd = [
            "python",
            str(PROJECT_ROOT / "src" / "pipeline.py"),
            "run_stage",
            stage_name,
        ]
        if options:
            cmd.extend(options)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )

        async for line in process.stdout:
            yield line.decode().rstrip()

        await process.wait()

        if process.returncode == 0:
            yield f"\n[Stage {stage_name} completed successfully]"
        else:
            yield f"\n[Stage {stage_name} failed with exit code {process.returncode}]"


# Singleton instance
_pipeline_service: Optional[PipelineService] = None


def get_pipeline_service() -> PipelineService:
    """Get the pipeline service singleton."""
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService()
    return _pipeline_service
