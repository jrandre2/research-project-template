#!/usr/bin/env python3
"""
Tests for multi-manuscript support in src/stages/s07_reviews.py

Tests cover:
- Manuscript configuration and paths
- Focus prompts (replacing discipline prompts)
- Review functions with manuscript parameter
- Journal compliance checks
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestManuscriptConfiguration:
    """Tests for manuscript registry and configuration."""

    def test_manuscripts_dict_exists(self):
        """MANUSCRIPTS dictionary should be defined."""
        from stages.s07_reviews import MANUSCRIPTS
        assert isinstance(MANUSCRIPTS, dict)
        assert len(MANUSCRIPTS) > 0

    def test_default_manuscript_exists(self):
        """DEFAULT_MANUSCRIPT should be in MANUSCRIPTS."""
        from stages.s07_reviews import MANUSCRIPTS, DEFAULT_MANUSCRIPT
        assert DEFAULT_MANUSCRIPT in MANUSCRIPTS

    def test_manuscript_has_required_keys(self):
        """Each manuscript config should have required keys."""
        from stages.s07_reviews import MANUSCRIPTS
        required_keys = ['name', 'dir', 'reviews_dir', 'archive_dir']
        for name, config in MANUSCRIPTS.items():
            for key in required_keys:
                assert key in config, f"Manuscript '{name}' missing key '{key}'"

    def test_manuscript_paths_are_path_objects(self):
        """Manuscript path values should be Path objects."""
        from stages.s07_reviews import MANUSCRIPTS
        path_keys = ['dir', 'reviews_dir', 'archive_dir']
        for name, config in MANUSCRIPTS.items():
            for key in path_keys:
                assert isinstance(config[key], Path), \
                    f"Manuscript '{name}' key '{key}' should be Path"


class TestGetManuscriptPaths:
    """Tests for get_manuscript_paths() function."""

    def test_get_manuscript_paths_default(self):
        """get_manuscript_paths() returns paths for default manuscript."""
        from stages.s07_reviews import get_manuscript_paths, DEFAULT_MANUSCRIPT
        paths = get_manuscript_paths()
        assert paths is not None
        assert 'name' in paths
        assert 'manuscript_dir' in paths

    def test_get_manuscript_paths_specific(self):
        """get_manuscript_paths() returns paths for specific manuscript."""
        from stages.s07_reviews import get_manuscript_paths
        paths = get_manuscript_paths('main')
        assert paths is not None
        assert paths['name'] == 'Main Manuscript'

    def test_get_manuscript_paths_unknown_raises(self):
        """get_manuscript_paths() raises ValueError for unknown manuscript."""
        from stages.s07_reviews import get_manuscript_paths
        with pytest.raises(ValueError) as exc_info:
            get_manuscript_paths('nonexistent')
        assert 'Unknown manuscript' in str(exc_info.value)

    def test_get_manuscript_paths_shows_available(self):
        """Error message lists available manuscripts."""
        from stages.s07_reviews import get_manuscript_paths
        with pytest.raises(ValueError) as exc_info:
            get_manuscript_paths('nonexistent')
        assert 'main' in str(exc_info.value)  # Should list available manuscripts


class TestFocusPrompts:
    """Tests for focus prompts configuration."""

    def test_focus_prompts_exists(self):
        """FOCUS_PROMPTS dictionary should be defined."""
        from stages.s07_reviews import FOCUS_PROMPTS
        assert isinstance(FOCUS_PROMPTS, dict)

    def test_focus_prompts_has_core_disciplines(self):
        """FOCUS_PROMPTS should include core discipline prompts."""
        from stages.s07_reviews import FOCUS_PROMPTS
        core_prompts = ['economics', 'engineering', 'social_sciences', 'general']
        for prompt in core_prompts:
            assert prompt in FOCUS_PROMPTS, f"Missing core prompt: {prompt}"

    def test_focus_prompts_has_new_focus_areas(self):
        """FOCUS_PROMPTS should include new focus areas."""
        from stages.s07_reviews import FOCUS_PROMPTS
        new_prompts = ['methods', 'policy', 'clarity']
        for prompt in new_prompts:
            assert prompt in FOCUS_PROMPTS, f"Missing new focus: {prompt}"

    def test_focus_prompts_are_strings(self):
        """All focus prompts should be non-empty strings."""
        from stages.s07_reviews import FOCUS_PROMPTS
        for name, prompt in FOCUS_PROMPTS.items():
            assert isinstance(prompt, str), f"Prompt '{name}' should be string"
            assert len(prompt) > 0, f"Prompt '{name}' should not be empty"

    def test_focus_prompts_are_generic(self):
        """Focus prompts should not contain project-specific content."""
        from stages.s07_reviews import FOCUS_PROMPTS
        # Check prompts don't reference specific projects
        specific_terms = ['capacity-sem', 'RISBS']
        for name, prompt in FOCUS_PROMPTS.items():
            for term in specific_terms:
                assert term not in prompt, \
                    f"Prompt '{name}' contains project-specific term '{term}'"


class TestReviewFunctionsManuscriptParam:
    """Tests for manuscript parameter in review functions."""

    def test_status_accepts_manuscript_param(self):
        """status() function should accept manuscript parameter."""
        from stages.s07_reviews import status
        import inspect
        sig = inspect.signature(status)
        assert 'manuscript' in sig.parameters

    def test_new_cycle_accepts_manuscript_param(self):
        """new_cycle() function should accept manuscript parameter."""
        from stages.s07_reviews import new_cycle
        import inspect
        sig = inspect.signature(new_cycle)
        assert 'manuscript' in sig.parameters

    def test_new_cycle_accepts_focus_param(self):
        """new_cycle() function should accept focus parameter."""
        from stages.s07_reviews import new_cycle
        import inspect
        sig = inspect.signature(new_cycle)
        assert 'focus' in sig.parameters

    def test_verify_accepts_manuscript_param(self):
        """verify() function should accept manuscript parameter."""
        from stages.s07_reviews import verify
        import inspect
        sig = inspect.signature(verify)
        assert 'manuscript' in sig.parameters

    def test_archive_accepts_manuscript_param(self):
        """archive() function should accept manuscript parameter."""
        from stages.s07_reviews import archive
        import inspect
        sig = inspect.signature(archive)
        assert 'manuscript' in sig.parameters


class TestJournalComplianceChecks:
    """Tests for journal compliance checking functions."""

    def test_count_manuscript_words_exists(self):
        """count_manuscript_words() function should exist."""
        from stages.s07_reviews import count_manuscript_words
        assert callable(count_manuscript_words)

    def test_check_self_references_exists(self):
        """check_self_references() function should exist."""
        from stages.s07_reviews import check_self_references
        assert callable(check_self_references)

    def test_check_abstract_length_exists(self):
        """check_abstract_length() function should exist."""
        from stages.s07_reviews import check_abstract_length
        assert callable(check_abstract_length)

    def test_count_manuscript_words_returns_int(self, temp_dir):
        """count_manuscript_words() should return integer count."""
        from stages.s07_reviews import count_manuscript_words

        # Create a mock manuscript directory with a .qmd file
        manuscript_dir = temp_dir / 'manuscript'
        manuscript_dir.mkdir()
        qmd_file = manuscript_dir / 'test.qmd'
        qmd_file.write_text("""---
title: Test
---

This is a test manuscript with some words.
""")

        word_count = count_manuscript_words(manuscript_dir)
        assert isinstance(word_count, int)
        assert word_count > 0


class TestBackwardCompatibility:
    """Tests for backward compatibility with discipline parameter."""

    def test_discipline_prompts_alias_exists(self):
        """DISCIPLINE_PROMPTS should exist as alias or be accessible."""
        from stages.s07_reviews import FOCUS_PROMPTS
        # Core disciplines should work with both old and new terminology
        assert 'economics' in FOCUS_PROMPTS
        assert 'engineering' in FOCUS_PROMPTS
        assert 'social_sciences' in FOCUS_PROMPTS
        assert 'general' in FOCUS_PROMPTS


class TestReviewDirectoryStructure:
    """Tests for review directory organization."""

    def test_reviews_dir_path_construction(self):
        """Reviews directory should be correctly constructed."""
        from stages.s07_reviews import get_manuscript_paths
        paths = get_manuscript_paths('main')
        reviews_dir = paths['reviews_dir']

        # Should be under doc/reviews/
        assert 'reviews' in str(reviews_dir)

    def test_archive_dir_under_reviews(self):
        """Archive directory should be under reviews directory."""
        from stages.s07_reviews import get_manuscript_paths
        paths = get_manuscript_paths('main')
        reviews_dir = paths['reviews_dir']
        archive_dir = paths['archive_dir']

        # Archive should be a subdirectory of reviews
        assert archive_dir.is_relative_to(reviews_dir)
