#!/usr/bin/env python3
"""
Tests for src/agents/migration_executor.py

Tests cover:
- ExecutionResult dataclass
- ExecutionReport dataclass
- MigrationExecutor class
- execute_migration convenience function
"""
from __future__ import annotations

import pytest
import os
import shutil
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from agents.migration_executor import (
    ExecutionResult,
    ExecutionReport,
    MigrationExecutor,
    execute_migration,
)
from agents.migration_planner import (
    MigrationStep,
    MigrationPlan,
)


# ============================================================
# EXECUTION RESULT TESTS
# ============================================================

class TestExecutionResult:
    """Tests for the ExecutionResult dataclass."""

    def test_creates_execution_result(self):
        """Test creating an execution result."""
        step = MigrationStep(1, 'setup', 'Create dirs')
        result = ExecutionResult(
            step=step,
            success=True,
            message='Created 5 directories'
        )

        assert result.step is step
        assert result.success is True
        assert result.message == 'Created 5 directories'
        assert result.error is None

    def test_failed_result_with_error(self):
        """Test failed result with error message."""
        step = MigrationStep(1, 'copy', 'Copy files')
        result = ExecutionResult(
            step=step,
            success=False,
            error='Permission denied'
        )

        assert result.success is False
        assert result.error == 'Permission denied'

    def test_duration_tracking(self):
        """Test duration field."""
        step = MigrationStep(1, 'verify', 'Run tests')
        result = ExecutionResult(
            step=step,
            success=True,
            duration_ms=1234.5
        )

        assert result.duration_ms == 1234.5

    def test_default_values(self):
        """Test default values."""
        step = MigrationStep(1, 'setup', 'Init')
        result = ExecutionResult(step=step, success=True)

        assert result.message == ""
        assert result.error is None
        assert result.duration_ms == 0


# ============================================================
# EXECUTION REPORT TESTS
# ============================================================

class TestExecutionReport:
    """Tests for the ExecutionReport dataclass."""

    def test_creates_execution_report(self):
        """Test creating an execution report."""
        plan = MigrationPlan(
            source_project='/src',
            target_location='/dst'
        )
        report = ExecutionReport(plan=plan)

        assert report.plan is plan
        assert report.results == []
        assert report.completed_at is None

    def test_started_at_auto_generated(self):
        """Test that started_at is auto-generated."""
        plan = MigrationPlan(source_project='/src', target_location='/dst')
        report = ExecutionReport(plan=plan)

        assert report.started_at is not None
        assert len(report.started_at) > 0

    def test_success_count(self):
        """Test success count calculation."""
        plan = MigrationPlan(source_project='/src', target_location='/dst')
        results = [
            ExecutionResult(MigrationStep(1, 'setup', 'A'), success=True),
            ExecutionResult(MigrationStep(2, 'copy', 'B'), success=True),
            ExecutionResult(MigrationStep(3, 'verify', 'C'), success=False),
        ]
        report = ExecutionReport(plan=plan, results=results)

        assert report.success_count == 2

    def test_failure_count(self):
        """Test failure count calculation."""
        plan = MigrationPlan(source_project='/src', target_location='/dst')
        results = [
            ExecutionResult(MigrationStep(1, 'setup', 'A'), success=True),
            ExecutionResult(MigrationStep(2, 'copy', 'B'), success=False),
            ExecutionResult(MigrationStep(3, 'verify', 'C'), success=False),
        ]
        report = ExecutionReport(plan=plan, results=results)

        assert report.failure_count == 2

    def test_overall_success_true(self):
        """Test overall success when all succeed."""
        plan = MigrationPlan(source_project='/src', target_location='/dst')
        results = [
            ExecutionResult(MigrationStep(1, 'setup', 'A'), success=True),
            ExecutionResult(MigrationStep(2, 'copy', 'B'), success=True),
        ]
        report = ExecutionReport(plan=plan, results=results)

        assert report.overall_success is True

    def test_overall_success_false(self):
        """Test overall success when any fails."""
        plan = MigrationPlan(source_project='/src', target_location='/dst')
        results = [
            ExecutionResult(MigrationStep(1, 'setup', 'A'), success=True),
            ExecutionResult(MigrationStep(2, 'copy', 'B'), success=False),
        ]
        report = ExecutionReport(plan=plan, results=results)

        assert report.overall_success is False

    def test_to_markdown(self):
        """Test markdown report generation."""
        plan = MigrationPlan(
            source_project='/project',
            target_location='/target'
        )
        step = MigrationStep(1, 'setup', 'Create dirs')
        result = ExecutionResult(
            step=step,
            success=True,
            message='Created directories',
            duration_ms=100
        )
        report = ExecutionReport(
            plan=plan,
            results=[result],
            completed_at='2024-01-01T12:00:00'
        )

        md = report.to_markdown()

        assert '# Migration Execution Report' in md
        assert '/project' in md
        assert '/target' in md
        assert 'Total Steps: 1' in md
        assert 'Successful: 1' in md
        assert 'Create dirs' in md

    def test_to_markdown_with_failure(self):
        """Test markdown report with failed step."""
        plan = MigrationPlan(source_project='/src', target_location='/dst')
        step = MigrationStep(1, 'copy', 'Copy files')
        result = ExecutionResult(
            step=step,
            success=False,
            error='File not found'
        )
        report = ExecutionReport(plan=plan, results=[result])

        md = report.to_markdown()

        assert 'Failed: 1' in md
        assert 'File not found' in md


# ============================================================
# MIGRATION EXECUTOR TESTS
# ============================================================

class TestMigrationExecutor:
    """Tests for the MigrationExecutor class."""

    def test_creates_executor(self, temp_dir):
        """Test creating a migration executor."""
        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(temp_dir / 'target')
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        assert executor.plan is plan
        assert executor.source_path == temp_dir
        assert executor.dry_run is False

    def test_dry_run_mode(self, temp_dir):
        """Test dry run mode."""
        target = temp_dir / 'target'
        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[MigrationStep(1, 'setup', 'Create directory structure')]
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir,
            dry_run=True
        )

        report = executor.execute()

        # Target directory should not be created in dry run
        assert report.overall_success
        assert 'Would' in report.results[0].message

    def test_execute_returns_report(self, temp_dir):
        """Test that execute returns a report."""
        target = temp_dir / 'target'
        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target)
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir,
            dry_run=True
        )

        report = executor.execute()

        assert isinstance(report, ExecutionReport)
        assert report.completed_at is not None

    def test_step_callback(self, temp_dir):
        """Test step callback is called."""
        target = temp_dir / 'target'
        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[MigrationStep(1, 'setup', 'Create dirs')]
        )

        callback_calls = []
        def callback(step, result):
            callback_calls.append((step, result))

        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir,
            dry_run=True,
            step_callback=callback
        )

        executor.execute()

        assert len(callback_calls) == 1
        assert callback_calls[0][0].action == 'Create dirs'

    def test_execute_setup_create_directories(self, temp_dir):
        """Test setup step creates directories."""
        target = temp_dir / 'target'
        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[MigrationStep(1, 'setup', 'Create target directory structure')]
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        report = executor.execute()

        assert report.overall_success
        assert (target / 'src' / 'stages').exists()
        assert (target / 'tests').exists()

    def test_execute_copy_files(self, temp_dir):
        """Test copy step copies files."""
        # Create source files
        src_data = temp_dir / 'data'
        src_data.mkdir()
        (src_data / 'file1.csv').write_text('a,b\n1,2')
        (src_data / 'file2.csv').write_text('c,d\n3,4')

        target = temp_dir / 'target'
        target.mkdir()

        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[MigrationStep(
                order=1,
                category='copy',
                action='Copy data files',
                source='data/*',
                target='data_raw/'
            )]
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        report = executor.execute()

        assert report.overall_success
        assert (target / 'data_raw' / 'file1.csv').exists()
        assert (target / 'data_raw' / 'file2.csv').exists()

    def test_execute_copy_no_matching_files(self, temp_dir):
        """Test copy step with no matching files."""
        target = temp_dir / 'target'
        target.mkdir()

        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[MigrationStep(
                order=1,
                category='copy',
                action='Copy files',
                source='nonexistent/*',
                target='dest/'
            )]
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        report = executor.execute()

        # Should succeed with message about no files
        assert report.overall_success
        assert 'No files' in report.results[0].message

    def test_execute_transform_creates_scaffold(self, temp_dir):
        """Test transform step creates scaffold file."""
        target = temp_dir / 'target'
        target.mkdir()

        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[MigrationStep(
                order=1,
                category='transform',
                action='Merge modules',
                source='mod1.py, mod2.py',
                target='src/stages/merged.py',
                details='Combine loading code'
            )]
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        report = executor.execute()

        assert report.overall_success
        scaffold_file = target / 'src' / 'stages' / 'merged.py'
        assert scaffold_file.exists()
        content = scaffold_file.read_text()
        assert 'mod1.py' in content
        assert 'TODO' in content

    def test_execute_generate_data_dictionary(self, temp_dir):
        """Test generate step creates data dictionary."""
        target = temp_dir / 'target'
        target.mkdir()

        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[MigrationStep(
                order=1,
                category='generate',
                action='Generate DATA_DICTIONARY.md',
                target='doc/DATA_DICTIONARY.md'
            )]
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        report = executor.execute()

        assert report.overall_success
        doc_file = target / 'doc' / 'DATA_DICTIONARY.md'
        assert doc_file.exists()
        content = doc_file.read_text()
        assert 'Data Dictionary' in content

    def test_execute_generate_methodology(self, temp_dir):
        """Test generate step creates methodology doc."""
        target = temp_dir / 'target'
        target.mkdir()

        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[MigrationStep(
                order=1,
                category='generate',
                action='Generate METHODOLOGY.md',
                target='doc/METHODOLOGY.md'
            )]
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        report = executor.execute()

        assert report.overall_success
        doc_file = target / 'doc' / 'METHODOLOGY.md'
        assert doc_file.exists()
        content = doc_file.read_text()
        assert 'Methodology' in content

    def test_execute_generate_pipeline_doc(self, temp_dir):
        """Test generate step creates pipeline doc."""
        target = temp_dir / 'target'
        target.mkdir()

        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[MigrationStep(
                order=1,
                category='generate',
                action='Generate PIPELINE.md',
                target='doc/PIPELINE.md'
            )]
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        report = executor.execute()

        assert report.overall_success
        doc_file = target / 'doc' / 'PIPELINE.md'
        assert doc_file.exists()
        content = doc_file.read_text()
        assert 'Pipeline' in content

    def test_execute_generate_pipeline_cli(self, temp_dir):
        """Test generate step creates pipeline CLI."""
        target = temp_dir / 'target'
        target.mkdir()

        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[MigrationStep(
                order=1,
                category='generate',
                action='Create pipeline.py CLI',
                target='src/pipeline.py'
            )]
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        report = executor.execute()

        assert report.overall_success
        cli_file = target / 'src' / 'pipeline.py'
        assert cli_file.exists()
        content = cli_file.read_text()
        assert 'argparse' in content
        assert 'ingest_data' in content

    def test_execute_verify_dry_run(self, temp_dir):
        """Test verify step in dry run mode."""
        target = temp_dir / 'target'
        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[MigrationStep(
                order=1,
                category='verify',
                action='Verify imports resolve'
            )]
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir,
            dry_run=True
        )

        report = executor.execute()

        assert report.overall_success
        assert 'dry run' in report.results[0].message.lower()

    def test_execute_stops_on_non_verify_failure(self, temp_dir):
        """Test execution stops on non-verify failure."""
        target = temp_dir / 'target'
        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[
                MigrationStep(
                    order=1,
                    category='copy',
                    action='Copy files',
                    # Missing source/target to trigger failure
                ),
                MigrationStep(
                    order=2,
                    category='setup',
                    action='Should not run'
                ),
            ]
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        report = executor.execute()

        # Should only have one result (stopped after first failure)
        assert len(report.results) == 1
        assert report.results[0].success is False

    def test_execute_continues_on_verify_failure(self, temp_dir):
        """Test execution continues on verify failure."""
        target = temp_dir / 'target'
        target.mkdir()
        (target / 'src' / 'stages').mkdir(parents=True)

        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[
                MigrationStep(
                    order=1,
                    category='verify',
                    action='Verify imports resolve'
                ),
                MigrationStep(
                    order=2,
                    category='verify',
                    action='Check documentation links'
                ),
            ]
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        report = executor.execute()

        # Both verify steps should run
        assert len(report.results) == 2

    def test_unknown_category_fails(self, temp_dir):
        """Test that unknown category fails gracefully."""
        target = temp_dir / 'target'
        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[MigrationStep(
                order=1,
                category='unknown_category',
                action='Do something'
            )]
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        report = executor.execute()

        assert report.results[0].success is False
        assert 'Unknown category' in report.results[0].error

    def test_step_marked_completed_on_success(self, temp_dir):
        """Test that step is marked completed on success."""
        target = temp_dir / 'target'
        step = MigrationStep(1, 'setup', 'Create directory structure')
        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[step]
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        executor.execute()

        assert step.completed is True


# ============================================================
# HELPER METHOD TESTS
# ============================================================

class TestMigrationExecutorHelpers:
    """Tests for MigrationExecutor helper methods."""

    def test_find_files_with_glob(self, temp_dir):
        """Test finding files with glob pattern."""
        # Create files
        (temp_dir / 'data').mkdir()
        (temp_dir / 'data' / 'file1.csv').write_text('a')
        (temp_dir / 'data' / 'file2.csv').write_text('b')

        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(temp_dir / 'target')
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        files = executor._find_files('data/*')

        assert len(files) >= 2

    def test_find_files_exact_path(self, temp_dir):
        """Test finding file with exact path."""
        (temp_dir / 'config.py').write_text('# Config')

        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(temp_dir / 'target')
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        files = executor._find_files('config.py')

        assert len(files) == 1
        assert files[0].name == 'config.py'

    def test_find_files_nonexistent(self, temp_dir):
        """Test finding nonexistent files."""
        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(temp_dir / 'target')
        )
        executor = MigrationExecutor(
            plan=plan,
            source_path=temp_dir
        )

        files = executor._find_files('nonexistent.xyz')

        assert len(files) == 0


# ============================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================

class TestExecuteMigration:
    """Tests for the execute_migration convenience function."""

    def test_executes_migration(self, temp_dir):
        """Test convenience function."""
        target = temp_dir / 'target'
        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[MigrationStep(1, 'setup', 'Create directory structure')]
        )

        report = execute_migration(
            plan=plan,
            source_path=temp_dir,
            dry_run=True
        )

        assert isinstance(report, ExecutionReport)
        assert report.completed_at is not None

    def test_executes_with_verbose(self, temp_dir, capsys):
        """Test verbose output."""
        target = temp_dir / 'target'
        plan = MigrationPlan(
            source_project=str(temp_dir),
            target_location=str(target),
            steps=[MigrationStep(1, 'setup', 'Create directory structure')]
        )

        execute_migration(
            plan=plan,
            source_path=temp_dir,
            dry_run=True,
            verbose=True
        )

        captured = capsys.readouterr()
        assert 'Create directory structure' in captured.out

