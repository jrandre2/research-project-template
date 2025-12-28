#!/usr/bin/env python3
"""
Tests for src/pipeline.py

Tests cover:
- CLI argument parsing
- Command routing
- Environment checks
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestParseArgs:
    """Tests for argument parsing."""

    def test_ingest_data_command(self):
        """Parse ingest_data command."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'ingest_data']):
            args = parse_args()
            assert args.cmd == 'ingest_data'

    def test_run_estimation_defaults(self):
        """Parse run_estimation with defaults."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'run_estimation']):
            args = parse_args()
            assert args.cmd == 'run_estimation'
            assert args.specification == 'baseline'
            assert args.sample == 'full'

    def test_run_estimation_with_options(self):
        """Parse run_estimation with custom options."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'run_estimation', '-s', 'robust', '--sample', 'subset']):
            args = parse_args()
            assert args.specification == 'robust'
            assert args.sample == 'subset'

    def test_review_new_discipline(self):
        """Parse review_new with discipline option (deprecated, maps to focus)."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'review_new', '--discipline', 'economics']):
            args = parse_args()
            assert args.focus == 'economics'

    def test_journal_parse_requires_input(self):
        """journal_parse requires --input argument."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'journal_parse']):
            with pytest.raises(SystemExit):
                parse_args()

    def test_journal_parse_with_url(self):
        """Parse journal_parse with URL source."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'journal_parse', '--url', 'https://example.com']):
            args = parse_args()
            assert args.url == 'https://example.com'

    def test_journal_fetch(self):
        """Parse journal_fetch command."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'journal_fetch', '--url', 'https://example.com']):
            args = parse_args()
            assert args.cmd == 'journal_fetch'

    def test_audit_data_options(self):
        """Parse audit_data with options."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'audit_data', '--full', '--report']):
            args = parse_args()
            assert args.cmd == 'audit_data'
            assert args.full is True
            assert args.report is True


class TestEnvironmentCheck:
    """Tests for environment validation."""

    def test_ensure_env_with_venv(self):
        """Environment check passes with valid venv."""
        from pipeline import ensure_env
        with patch.dict('os.environ', {'VIRTUAL_ENV': '/path/to/.venv'}):
            # Should not raise
            ensure_env()

    def test_ensure_env_without_venv(self):
        """Environment check fails without venv."""
        from pipeline import ensure_env
        with patch.dict('os.environ', {'VIRTUAL_ENV': ''}, clear=True):
            with pytest.raises(SystemExit):
                ensure_env()


class TestCommandRouting:
    """Tests for command routing in main()."""

    def test_routes_to_correct_stage(self):
        """Commands route to correct stage modules."""
        # This is tested indirectly through integration tests
        # Unit tests would require mocking all stage imports
        pass


class TestRunStageCommand:
    """Tests for run_stage command."""

    def test_run_stage_parses(self):
        """Parse run_stage command."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'run_stage', 's00_ingest']):
            args = parse_args()
            assert args.cmd == 'run_stage'
            assert args.stage_name == 's00_ingest'

    def test_run_stage_with_version(self):
        """Parse run_stage with versioned stage name."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'run_stage', 's00b_standardize']):
            args = parse_args()
            assert args.stage_name == 's00b_standardize'

    def test_run_stage_requires_name(self):
        """run_stage requires stage_name argument."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'run_stage']):
            with pytest.raises(SystemExit):
                parse_args()


class TestListStagesCommand:
    """Tests for list_stages command."""

    def test_list_stages_parses(self):
        """Parse list_stages command."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'list_stages']):
            args = parse_args()
            assert args.cmd == 'list_stages'

    def test_list_stages_with_prefix(self):
        """Parse list_stages with prefix filter."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'list_stages', '--prefix', 's00']):
            args = parse_args()
            assert args.prefix == 's00'

    def test_list_stages_prefix_short_option(self):
        """Parse list_stages with -p short option."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'list_stages', '-p', 's01']):
            args = parse_args()
            assert args.prefix == 's01'


class TestDiscoverStages:
    """Tests for stage discovery functions."""

    def test_discover_stages_returns_list(self):
        """discover_stages() returns a list."""
        from pipeline import discover_stages
        stages = discover_stages()
        assert isinstance(stages, list)

    def test_discover_stages_finds_core_stages(self):
        """discover_stages() finds core pipeline stages."""
        from pipeline import discover_stages
        stages = discover_stages()
        # discover_stages returns list of (name, docstring) tuples
        stage_names = [s[0] for s in stages]

        # Core stages should be found
        assert 's00_ingest' in stage_names
        assert 's07_reviews' in stage_names

    def test_list_available_stages_format(self):
        """list_available_stages() formats output correctly."""
        from pipeline import list_available_stages
        # Should not raise
        list_available_stages(prefix=None)


class TestManuscriptArgument:
    """Tests for --manuscript argument on review commands."""

    def test_review_status_manuscript_arg(self):
        """review_status accepts --manuscript argument."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'review_status', '--manuscript', 'main']):
            args = parse_args()
            assert args.manuscript == 'main'

    def test_review_status_manuscript_short(self):
        """review_status accepts -m short option."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'review_status', '-m', 'main']):
            args = parse_args()
            assert args.manuscript == 'main'

    def test_review_new_manuscript_arg(self):
        """review_new accepts --manuscript argument."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'review_new', '--manuscript', 'main']):
            args = parse_args()
            assert args.manuscript == 'main'

    def test_review_verify_manuscript_arg(self):
        """review_verify accepts --manuscript argument."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'review_verify', '--manuscript', 'main']):
            args = parse_args()
            assert args.manuscript == 'main'

    def test_review_archive_manuscript_arg(self):
        """review_archive accepts --manuscript argument."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'review_archive', '--manuscript', 'main']):
            args = parse_args()
            assert args.manuscript == 'main'


class TestFocusArgument:
    """Tests for --focus argument on review_new command."""

    def test_review_new_focus_arg(self):
        """review_new accepts --focus argument."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'review_new', '--focus', 'methods']):
            args = parse_args()
            assert args.focus == 'methods'

    def test_review_new_focus_short(self):
        """review_new accepts -f short option."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'review_new', '-f', 'policy']):
            args = parse_args()
            assert args.focus == 'policy'

    def test_review_new_discipline_deprecated(self):
        """review_new accepts --discipline as deprecated alias."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'review_new', '--discipline', 'economics']):
            args = parse_args()
            # --discipline maps to focus attribute (dest='focus')
            assert args.focus == 'economics'

    def test_review_new_combined_args(self):
        """review_new accepts both --manuscript and --focus."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'review_new', '-m', 'main', '-f', 'clarity']):
            args = parse_args()
            assert args.manuscript == 'main'
            assert args.focus == 'clarity'


# ============================================================
# CACHING AND PARALLEL EXECUTION CLI TESTS
# ============================================================

class TestCachingFlags:
    """Tests for --no-cache flag on estimation/robustness/figures commands."""

    def test_run_estimation_no_cache_flag(self):
        """run_estimation accepts --no-cache flag."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'run_estimation', '--no-cache']):
            args = parse_args()
            assert args.no_cache is True

    def test_run_estimation_cache_default(self):
        """run_estimation has caching enabled by default."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'run_estimation']):
            args = parse_args()
            assert args.no_cache is False

    def test_estimate_robustness_no_cache_flag(self):
        """estimate_robustness accepts --no-cache flag."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'estimate_robustness', '--no-cache']):
            args = parse_args()
            assert args.no_cache is True

    def test_make_figures_no_cache_flag(self):
        """make_figures accepts --no-cache flag."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'make_figures', '--no-cache']):
            args = parse_args()
            assert args.no_cache is True


class TestParallelExecutionFlags:
    """Tests for --sequential and --workers flags."""

    def test_run_estimation_sequential_flag(self):
        """run_estimation accepts --sequential flag."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'run_estimation', '--sequential']):
            args = parse_args()
            assert args.sequential is True

    def test_run_estimation_parallel_default(self):
        """run_estimation has parallel enabled by default."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'run_estimation']):
            args = parse_args()
            assert args.sequential is False

    def test_run_estimation_workers_flag(self):
        """run_estimation accepts --workers flag."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'run_estimation', '--workers', '4']):
            args = parse_args()
            assert args.workers == 4

    def test_run_estimation_workers_short_flag(self):
        """run_estimation accepts -w short flag."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'run_estimation', '-w', '2']):
            args = parse_args()
            assert args.workers == 2

    def test_run_estimation_workers_default_none(self):
        """run_estimation workers defaults to None (CPU count)."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'run_estimation']):
            args = parse_args()
            assert args.workers is None

    def test_estimate_robustness_sequential_flag(self):
        """estimate_robustness accepts --sequential flag."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'estimate_robustness', '--sequential']):
            args = parse_args()
            assert args.sequential is True

    def test_estimate_robustness_workers_flag(self):
        """estimate_robustness accepts --workers flag."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'estimate_robustness', '-w', '8']):
            args = parse_args()
            assert args.workers == 8

    def test_make_figures_sequential_flag(self):
        """make_figures accepts --sequential flag."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'make_figures', '--sequential']):
            args = parse_args()
            assert args.sequential is True

    def test_make_figures_workers_flag(self):
        """make_figures accepts --workers flag."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'make_figures', '--workers', '3']):
            args = parse_args()
            assert args.workers == 3


class TestCacheManagementCommand:
    """Tests for the cache management command."""

    def test_cache_stats_command(self):
        """cache stats command parses correctly."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'cache', 'stats']):
            args = parse_args()
            assert args.cmd == 'cache'
            assert args.action == 'stats'

    def test_cache_clear_command(self):
        """cache clear command parses correctly."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'cache', 'clear']):
            args = parse_args()
            assert args.cmd == 'cache'
            assert args.action == 'clear'

    def test_cache_clear_with_stage(self):
        """cache clear accepts --stage option."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'cache', 'clear', '--stage', 's03_estimation']):
            args = parse_args()
            assert args.action == 'clear'
            assert args.stage == 's03_estimation'

    def test_cache_clear_stage_short_option(self):
        """cache clear accepts -s short option."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'cache', 'clear', '-s', 's04_robustness']):
            args = parse_args()
            assert args.stage == 's04_robustness'

    def test_cache_requires_action(self):
        """cache command requires action argument."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'cache']):
            with pytest.raises(SystemExit):
                parse_args()


class TestCombinedCacheParallelFlags:
    """Tests for combining cache and parallel flags."""

    def test_run_estimation_all_flags(self):
        """run_estimation accepts all cache/parallel flags together."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'run_estimation',
                               '--no-cache', '--sequential', '-w', '2',
                               '-s', 'robust', '--sample', 'subset']):
            args = parse_args()
            assert args.no_cache is True
            assert args.sequential is True
            assert args.workers == 2
            assert args.specification == 'robust'
            assert args.sample == 'subset'

    def test_estimate_robustness_all_flags(self):
        """estimate_robustness accepts all cache/parallel flags."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'estimate_robustness',
                               '--no-cache', '--sequential', '--workers', '4']):
            args = parse_args()
            assert args.no_cache is True
            assert args.sequential is True
            assert args.workers == 4

    def test_make_figures_all_flags(self):
        """make_figures accepts all cache/parallel flags."""
        from pipeline import parse_args
        with patch('sys.argv', ['pipeline.py', 'make_figures',
                               '--no-cache', '--sequential', '-w', '1']):
            args = parse_args()
            assert args.no_cache is True
            assert args.sequential is True
            assert args.workers == 1
