"""
QA service for parsing and analyzing quality reports.

Reads QA reports from data_work/quality/ and provides
metrics for the dashboard.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import DATA_WORK_DIR, PROJECT_ROOT
from ..models import QAMetric, QAReport


# Default QA thresholds (can be overridden from main config)
DEFAULT_THRESHOLDS = {
    'max_missing_pct': 5.0,
    'min_row_count': 10,
    'max_duplicate_pct': 1.0,
}


class QAService:
    """Service for accessing QA reports."""

    def __init__(self):
        self.qa_dir = DATA_WORK_DIR / "quality"
        self.thresholds = self._load_thresholds()

    def _load_thresholds(self) -> dict:
        """Load QA thresholds from config."""
        try:
            from config import QA_THRESHOLDS
            return QA_THRESHOLDS
        except ImportError:
            return DEFAULT_THRESHOLDS

    def list_reports(self, stage: Optional[str] = None) -> list[QAReport]:
        """
        List all QA reports, optionally filtered by stage.

        Parameters
        ----------
        stage : str, optional
            Filter reports to this stage only.

        Returns
        -------
        list[QAReport]
            List of QA reports, sorted by timestamp (newest first).
        """
        if not self.qa_dir.exists():
            return []

        reports = []
        pattern = f"{stage}_quality_*.csv" if stage else "*_quality_*.csv"

        for file in self.qa_dir.glob(pattern):
            report = self._parse_report(file)
            if report:
                reports.append(report)

        # Sort by timestamp, newest first
        reports.sort(key=lambda r: r.timestamp, reverse=True)
        return reports

    def get_latest_report(self, stage_name: str) -> Optional[QAReport]:
        """
        Get the most recent QA report for a stage.

        Parameters
        ----------
        stage_name : str
            Name of the stage.

        Returns
        -------
        QAReport or None
            The latest report, or None if no reports exist.
        """
        reports = self.list_reports(stage=stage_name)
        return reports[0] if reports else None

    def _parse_report(self, file_path: Path) -> Optional[QAReport]:
        """Parse a QA report CSV file."""
        try:
            metrics = []
            stage = None
            timestamp = None

            with open(file_path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if stage is None:
                        stage = row.get('stage', 'unknown')
                    if timestamp is None and row.get('timestamp'):
                        try:
                            timestamp = datetime.strptime(
                                row['timestamp'],
                                '%Y%m%d_%H%M%S'
                            )
                        except ValueError:
                            timestamp = datetime.now()

                    name = row.get('metric', '')
                    value = row.get('value', '')

                    # Try to convert to numeric
                    try:
                        if '.' in str(value):
                            value = float(value)
                        else:
                            value = int(value)
                    except (ValueError, TypeError):
                        pass

                    # Check against thresholds
                    is_warning = self._check_threshold(name, value)

                    metrics.append(QAMetric(
                        name=name,
                        value=value,
                        threshold=self.thresholds.get(name),
                        is_warning=is_warning,
                    ))

            if not stage:
                # Extract stage from filename
                stage = file_path.stem.split('_quality_')[0]

            if not timestamp:
                # Extract timestamp from filename
                try:
                    ts_str = file_path.stem.split('_quality_')[1]
                    timestamp = datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
                except (IndexError, ValueError):
                    timestamp = datetime.fromtimestamp(file_path.stat().st_mtime)

            return QAReport(
                stage=stage,
                timestamp=timestamp,
                metrics=metrics,
                file_path=str(file_path.relative_to(PROJECT_ROOT)),
            )

        except Exception as e:
            print(f"Warning: Failed to parse QA report {file_path}: {e}")
            return None

    def _check_threshold(self, metric_name: str, value) -> bool:
        """Check if a metric exceeds its threshold."""
        if not isinstance(value, (int, float)):
            return False

        if metric_name == 'missing_pct':
            return value > self.thresholds.get('max_missing_pct', 5.0)
        elif metric_name == 'duplicate_pct':
            return value > self.thresholds.get('max_duplicate_pct', 1.0)
        elif metric_name == 'n_rows':
            return value < self.thresholds.get('min_row_count', 10)

        return False


# Singleton instance
_qa_service: Optional[QAService] = None


def get_qa_service() -> QAService:
    """Get the QA service singleton."""
    global _qa_service
    if _qa_service is None:
        _qa_service = QAService()
    return _qa_service
