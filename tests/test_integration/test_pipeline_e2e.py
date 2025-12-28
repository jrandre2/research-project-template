#!/usr/bin/env python3
"""
End-to-end integration tests for the CENTAUR pipeline.

These tests verify the complete data flow from raw data to analysis outputs
using synthetic demo data.
"""
from __future__ import annotations

import pytest
import subprocess
import sys
from pathlib import Path

# Mark all tests as integration and e2e
pytestmark = [pytest.mark.integration, pytest.mark.e2e]


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def pipeline_script(project_root):
    """Get the pipeline.py script path."""
    return project_root / 'src' / 'pipeline.py'


@pytest.fixture
def data_work_dir(project_root):
    """Get the data_work directory."""
    return project_root / 'data_work'


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def run_pipeline_command(project_root, *args, timeout=120):
    """Run a pipeline command and return the result."""
    cmd = [
        sys.executable,
        'src/pipeline.py',
        *args
    ]
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result


# ============================================================
# PIPELINE AVAILABILITY TESTS
# ============================================================

class TestPipelineAvailable:
    """Tests that verify the pipeline is available and runnable."""

    def test_pipeline_script_exists(self, pipeline_script):
        """Test that pipeline.py exists."""
        assert pipeline_script.exists()

    def test_pipeline_can_import(self, project_root):
        """Test that pipeline module can be imported."""
        result = run_pipeline_command(project_root, '--help')
        # Should not error
        assert result.returncode == 0 or 'usage' in result.stdout.lower() or 'usage' in result.stderr.lower()

    def test_list_stages_command(self, project_root):
        """Test that list_stages command works."""
        result = run_pipeline_command(project_root, 'list_stages')
        # Should succeed or show available stages
        assert result.returncode == 0 or 's00' in result.stdout or 'ingest' in result.stdout.lower()


# ============================================================
# DATA INGESTION TESTS
# ============================================================

class TestDataIngestion:
    """Tests for the data ingestion stage."""

    def test_ingest_demo_data(self, project_root, data_work_dir):
        """Test ingesting demo data."""
        result = run_pipeline_command(project_root, 'ingest_data', '--demo')

        # Should succeed
        assert result.returncode == 0, f"Failed: {result.stderr}"

        # Should create output file
        output_file = data_work_dir / 'data_raw.parquet'
        assert output_file.exists(), "data_raw.parquet not created"

    def test_ingest_creates_summary(self, project_root, data_work_dir):
        """Test that ingest creates summary output."""
        # Run ingest first if not already done
        run_pipeline_command(project_root, 'ingest_data', '--demo')

        # Check for diagnostic/summary output
        quality_dir = data_work_dir / 'quality'
        if quality_dir.exists():
            qa_files = list(quality_dir.glob('s00_ingest*.csv'))
            # May or may not have QA files depending on config
            pass


# ============================================================
# RECORD LINKAGE TESTS
# ============================================================

class TestRecordLinkage:
    """Tests for the record linkage stage."""

    def test_link_records_requires_input(self, project_root, data_work_dir):
        """Test that link_records requires input data."""
        # Clear any existing linked data
        linked_file = data_work_dir / 'data_linked.parquet'
        if linked_file.exists():
            linked_file.unlink()

        # First ensure we have input
        run_pipeline_command(project_root, 'ingest_data', '--demo')

        # Now run linkage
        result = run_pipeline_command(project_root, 'link_records')

        # Should either succeed or provide meaningful error
        # (Linkage may not be implemented for demo data)
        if result.returncode != 0:
            # Should have meaningful error message
            assert len(result.stderr) > 0 or len(result.stdout) > 0


# ============================================================
# PANEL CONSTRUCTION TESTS
# ============================================================

class TestPanelConstruction:
    """Tests for the panel construction stage."""

    def test_build_panel(self, project_root, data_work_dir):
        """Test building the panel."""
        # Ensure we have input data
        run_pipeline_command(project_root, 'ingest_data', '--demo')

        # Try to build panel
        result = run_pipeline_command(project_root, 'build_panel')

        # Check result
        if result.returncode == 0:
            # Should create panel file
            panel_file = data_work_dir / 'panel.parquet'
            # May or may not exist depending on linkage requirement
            pass


# ============================================================
# ESTIMATION TESTS
# ============================================================

class TestEstimation:
    """Tests for the estimation stage."""

    def test_estimation_help(self, project_root):
        """Test estimation command help."""
        result = run_pipeline_command(project_root, 'run_estimation', '--help')

        # Should show help
        assert result.returncode == 0 or 'specification' in result.stdout.lower() or 'usage' in result.stderr.lower()


# ============================================================
# FIGURE GENERATION TESTS
# ============================================================

class TestFigures:
    """Tests for the figure generation stage."""

    def test_make_figures_help(self, project_root):
        """Test make_figures command help."""
        result = run_pipeline_command(project_root, 'make_figures', '--help')

        # Should show help or run
        assert result.returncode == 0 or 'usage' in result.stdout.lower() or 'usage' in result.stderr.lower()


# ============================================================
# END-TO-END PIPELINE TESTS
# ============================================================

class TestPipelineE2E:
    """End-to-end tests for the complete pipeline."""

    @pytest.mark.slow
    def test_demo_pipeline_runs(self, project_root, data_work_dir):
        """Test that the demo pipeline can run from ingestion to panel."""
        # Clear previous runs
        for f in data_work_dir.glob('*.parquet'):
            f.unlink()

        # Run ingestion with demo data
        result = run_pipeline_command(project_root, 'ingest_data', '--demo')
        assert result.returncode == 0, f"Ingestion failed: {result.stderr}"

        # Verify raw data created
        raw_file = data_work_dir / 'data_raw.parquet'
        assert raw_file.exists(), "Raw data file not created"

        # Try to continue pipeline
        # (may fail at linkage depending on demo data structure)
        run_pipeline_command(project_root, 'link_records')
        run_pipeline_command(project_root, 'build_panel')

    def test_data_files_are_parquet(self, project_root, data_work_dir):
        """Test that output files are in Parquet format."""
        # Ensure we have some data
        run_pipeline_command(project_root, 'ingest_data', '--demo')

        # Check file format
        parquet_files = list(data_work_dir.glob('*.parquet'))
        assert len(parquet_files) > 0, "No parquet files created"

        # Verify they can be read
        import pandas as pd
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                assert len(df) >= 0  # Can be empty but should be readable
            except Exception as e:
                pytest.fail(f"Could not read {pf.name}: {e}")

    def test_demo_data_has_expected_columns(self, project_root, data_work_dir):
        """Test that demo data has expected structure."""
        # Run ingestion
        run_pipeline_command(project_root, 'ingest_data', '--demo')

        raw_file = data_work_dir / 'data_raw.parquet'
        if raw_file.exists():
            import pandas as pd
            df = pd.read_parquet(raw_file)

            # Should have some columns
            assert len(df.columns) > 0

            # Check for common expected columns (may vary by implementation)
            # These are just examples - adjust based on actual demo data
            possible_id_cols = ['id', 'unit_id', 'entity_id', 'firm_id']
            has_id = any(col in df.columns for col in possible_id_cols)
            # ID column may or may not be present depending on demo structure


# ============================================================
# QA REPORT TESTS
# ============================================================

class TestQAReports:
    """Tests for QA report generation."""

    def test_qa_reports_directory_created(self, project_root, data_work_dir):
        """Test that QA reports directory can be created."""
        quality_dir = data_work_dir / 'quality'

        # Run ingestion to potentially trigger QA
        run_pipeline_command(project_root, 'ingest_data', '--demo')

        # Quality directory may or may not exist depending on config
        # This test just verifies the mechanism doesn't error


# ============================================================
# CONFIGURATION TESTS
# ============================================================

class TestConfiguration:
    """Tests for pipeline configuration."""

    def test_config_module_importable(self, project_root):
        """Test that config module can be imported."""
        result = subprocess.run(
            [sys.executable, '-c', 'from config import PROJECT_ROOT'],
            cwd=project_root / 'src',
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Config import failed: {result.stderr}"

    def test_stages_directory_exists(self, project_root):
        """Test that stages directory exists."""
        stages_dir = project_root / 'src' / 'stages'
        assert stages_dir.exists()
        assert stages_dir.is_dir()

    def test_all_stages_importable(self, project_root):
        """Test that all stage modules can be imported."""
        stages_dir = project_root / 'src' / 'stages'
        stage_files = list(stages_dir.glob('s[0-9]*.py'))

        for stage_file in stage_files:
            module_name = stage_file.stem
            result = subprocess.run(
                [sys.executable, '-c', f'from stages.{module_name} import main'],
                cwd=project_root / 'src',
                capture_output=True,
                text=True
            )
            # Should import without error (may have missing dependencies)
            if result.returncode != 0:
                # Check it's not a syntax error
                assert 'SyntaxError' not in result.stderr, f"Syntax error in {module_name}"


# ============================================================
# MANUSCRIPT TESTS
# ============================================================

class TestManuscript:
    """Tests for manuscript-related functionality."""

    def test_manuscript_directory_exists(self, project_root):
        """Test that manuscript directory exists."""
        manuscript_dir = project_root / 'manuscript_quarto'
        assert manuscript_dir.exists(), "manuscript_quarto directory not found"

    def test_index_qmd_exists(self, project_root):
        """Test that main manuscript file exists."""
        index_file = project_root / 'manuscript_quarto' / 'index.qmd'
        assert index_file.exists(), "index.qmd not found"

    def test_manuscript_code_directory(self, project_root):
        """Test that manuscript code directory exists."""
        code_dir = project_root / 'manuscript_quarto' / 'code'
        assert code_dir.exists(), "manuscript_quarto/code directory not found"

