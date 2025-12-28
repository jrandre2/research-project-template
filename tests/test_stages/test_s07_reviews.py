#!/usr/bin/env python3
"""
Tests for src/stages/s07_reviews.py

Tests cover:
- Manuscript configuration
- Focus prompts
- Path resolution
"""
from __future__ import annotations

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from stages.s07_reviews import (
    MANUSCRIPTS,
    DEFAULT_MANUSCRIPT,
    FOCUS_PROMPTS,
    get_manuscript_paths,
)


# ============================================================
# MANUSCRIPT CONFIGURATION TESTS
# ============================================================

class TestManuscriptConfiguration:
    """Tests for manuscript configuration."""

    def test_default_manuscript_exists(self):
        """Test that default manuscript is configured."""
        assert DEFAULT_MANUSCRIPT in MANUSCRIPTS

    def test_main_manuscript_configured(self):
        """Test that main manuscript is configured."""
        assert 'main' in MANUSCRIPTS

    def test_manuscript_has_required_keys(self):
        """Test manuscripts have required configuration keys."""
        required_keys = ['name', 'title', 'dir', 'reviews_dir', 'archive_dir', 'description']

        for name, config in MANUSCRIPTS.items():
            for key in required_keys:
                assert key in config, f"Manuscript '{name}' missing key '{key}'"


# ============================================================
# GET MANUSCRIPT PATHS TESTS
# ============================================================

class TestGetManuscriptPaths:
    """Tests for the get_manuscript_paths function."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        paths = get_manuscript_paths('main')
        assert isinstance(paths, dict)

    def test_contains_required_keys(self):
        """Test that returned dict contains required keys."""
        paths = get_manuscript_paths('main')

        required = ['manuscript_dir', 'tracker_file', 'reviews_dir', 'archive_dir', 'name', 'title']
        for key in required:
            assert key in paths, f"Missing key '{key}'"

    def test_uses_default_when_none(self):
        """Test that None uses default manuscript."""
        paths = get_manuscript_paths(None)
        assert paths is not None

    def test_raises_for_unknown_manuscript(self):
        """Test raises ValueError for unknown manuscript."""
        with pytest.raises(ValueError, match="Unknown manuscript"):
            get_manuscript_paths('nonexistent_manuscript')

    def test_paths_are_path_objects(self):
        """Test that path values are Path objects."""
        paths = get_manuscript_paths('main')

        assert isinstance(paths['manuscript_dir'], Path)
        assert isinstance(paths['tracker_file'], Path)
        assert isinstance(paths['reviews_dir'], Path)


# ============================================================
# FOCUS PROMPTS TESTS
# ============================================================

class TestFocusPrompts:
    """Tests for focus prompts."""

    def test_has_discipline_prompts(self):
        """Test that discipline-based prompts exist."""
        assert 'economics' in FOCUS_PROMPTS
        assert 'engineering' in FOCUS_PROMPTS
        assert 'social_sciences' in FOCUS_PROMPTS
        assert 'general' in FOCUS_PROMPTS

    def test_has_focus_prompts(self):
        """Test that focus-specific prompts exist."""
        assert 'methods' in FOCUS_PROMPTS

    def test_prompts_are_strings(self):
        """Test that prompts are non-empty strings."""
        for name, prompt in FOCUS_PROMPTS.items():
            assert isinstance(prompt, str), f"Prompt '{name}' is not a string"
            assert len(prompt) > 50, f"Prompt '{name}' is too short"

    def test_prompts_have_instructions(self):
        """Test that prompts contain review instructions."""
        for name, prompt in FOCUS_PROMPTS.items():
            # Should have some kind of structure
            assert 'review' in prompt.lower() or 'comment' in prompt.lower(), \
                f"Prompt '{name}' missing review instructions"


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestReviewsIntegration:
    """Integration tests for review management."""

    def test_manuscript_paths_structure(self):
        """Test complete manuscript paths structure."""
        for manuscript_name in MANUSCRIPTS.keys():
            paths = get_manuscript_paths(manuscript_name)

            # Tracker file should be in manuscript dir
            assert paths['tracker_file'].parent == paths['manuscript_dir']

            # Reviews dir should be a subdirectory
            assert 'reviews' in str(paths['reviews_dir']).lower()
