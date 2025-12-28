"""Tests for Stage 09: AI-Assisted Manuscript Writing."""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestDraftResultsDryRun:
    """Tests for draft_results dry-run mode."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create sample diagnostic CSV."""
        data = pd.DataFrame({
            'variable': ['treatment', 'control'],
            'coefficient': [0.15, 0.03],
            'p_value': [0.003, 0.134],
        })
        csv_path = tmp_path / 'diagnostics' / 'test_results.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(csv_path, index=False)
        return tmp_path

    def test_dry_run_returns_none(self, sample_csv, capsys):
        """Test dry-run mode returns None."""
        from stages import s09_writing

        # Patch DIAGNOSTICS_DIR to use our temp directory
        with patch.object(s09_writing, 'DIAGNOSTICS_DIR', sample_csv / 'diagnostics'):
            result = s09_writing.draft_results(
                table_name='test_results',
                dry_run=True,
            )

        assert result is None

    def test_dry_run_shows_prompt(self, sample_csv, capsys):
        """Test dry-run mode shows prompt."""
        from stages import s09_writing

        with patch.object(s09_writing, 'DIAGNOSTICS_DIR', sample_csv / 'diagnostics'):
            s09_writing.draft_results(
                table_name='test_results',
                dry_run=True,
            )

        captured = capsys.readouterr()
        assert 'DRY RUN' in captured.out
        assert 'treatment' in captured.out


class TestDraftCaptionsDryRun:
    """Tests for draft_captions dry-run mode."""

    @pytest.fixture
    def sample_figures(self, tmp_path):
        """Create sample figure files."""
        figures_dir = tmp_path / 'manuscript_quarto' / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy figure files
        (figures_dir / 'fig_main_results.png').write_bytes(b'fake png')
        (figures_dir / 'fig_robustness.png').write_bytes(b'fake png')

        return tmp_path

    def test_dry_run_returns_none(self, sample_figures, capsys):
        """Test dry-run mode returns None."""
        from stages import s09_writing
        from config import MANUSCRIPTS

        # Create mock manuscript config
        mock_manuscripts = {
            'main': {
                'name': 'Test Manuscript',
                'dir': sample_figures / 'manuscript_quarto',
                'reviews_dir': sample_figures / 'reviews',
                'archive_dir': sample_figures / 'archive',
            }
        }

        with patch.object(s09_writing, 'MANUSCRIPTS', mock_manuscripts):
            result = s09_writing.draft_captions(
                figure_pattern='*.png',
                dry_run=True,
            )

        assert result is None


class TestDraftAbstractDryRun:
    """Tests for draft_abstract dry-run mode."""

    @pytest.fixture
    def sample_manuscript(self, tmp_path):
        """Create sample manuscript file."""
        manuscript_dir = tmp_path / 'manuscript_quarto'
        manuscript_dir.mkdir(parents=True, exist_ok=True)

        qmd_content = """---
title: Test Paper
---

## Introduction

This paper studies the effect of X on Y.

## Methods

We use a regression approach with fixed effects.

## Results

We find a positive effect of 0.15 (p < 0.01).

## Conclusion

This has important implications for policy.
"""
        (manuscript_dir / 'index.qmd').write_text(qmd_content)
        return tmp_path

    def test_dry_run_returns_none(self, sample_manuscript, capsys):
        """Test dry-run mode returns None."""
        from stages import s09_writing

        mock_manuscripts = {
            'main': {
                'name': 'Test Manuscript',
                'dir': sample_manuscript / 'manuscript_quarto',
                'reviews_dir': sample_manuscript / 'reviews',
                'archive_dir': sample_manuscript / 'archive',
            }
        }

        with patch.object(s09_writing, 'MANUSCRIPTS', mock_manuscripts):
            result = s09_writing.draft_abstract(
                dry_run=True,
            )

        assert result is None

    def test_dry_run_extracts_sections(self, sample_manuscript, capsys):
        """Test dry-run mode shows extracted sections."""
        from stages import s09_writing

        mock_manuscripts = {
            'main': {
                'name': 'Test Manuscript',
                'dir': sample_manuscript / 'manuscript_quarto',
                'reviews_dir': sample_manuscript / 'reviews',
                'archive_dir': sample_manuscript / 'archive',
            }
        }

        with patch.object(s09_writing, 'MANUSCRIPTS', mock_manuscripts):
            s09_writing.draft_abstract(dry_run=True)

        captured = capsys.readouterr()
        assert 'DRY RUN' in captured.out


class TestOutputFormat:
    """Tests for draft output file format."""

    def test_draft_header_format(self):
        """Test draft header contains required metadata."""
        from stages.s09_writing import _create_draft_header

        header = _create_draft_header(
            source='test.csv',
            provider_name='anthropic',
            model_name='claude-test',
        )

        assert 'AI-Generated Draft' in header
        assert 'test.csv' in header
        assert 'anthropic' in header
        assert 'claude-test' in header
        assert 'REQUIRES HUMAN REVIEW' in header

    def test_output_path_format(self):
        """Test output path naming convention."""
        from stages.s09_writing import _get_output_path

        # Patch DRAFTS_DIR to avoid file system
        from stages import s09_writing
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(s09_writing, 'DRAFTS_DIR', Path(tmp)):
                path = _get_output_path('results', 'main')

        assert 'results' in path.name
        assert 'main' in path.name
        assert path.suffix == '.md'


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_table_raises_error(self, tmp_path):
        """Test FileNotFoundError for missing table."""
        from stages import s09_writing

        with patch.object(s09_writing, 'DIAGNOSTICS_DIR', tmp_path):
            with pytest.raises(FileNotFoundError):
                s09_writing.draft_results(
                    table_name='nonexistent',
                    dry_run=True,
                )

    def test_invalid_manuscript_raises_error(self):
        """Test ValueError for invalid manuscript."""
        from stages import s09_writing

        with pytest.raises(ValueError, match='Unknown manuscript'):
            s09_writing.get_manuscript_paths('nonexistent_manuscript')
