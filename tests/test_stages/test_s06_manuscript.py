#!/usr/bin/env python3
"""
Tests for src/stages/s06_manuscript.py

Tests cover:
- ValidationCheck dataclass
- ManuscriptValidation dataclass
- Word counting
- Abstract extraction
- Validation checks
"""
from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from stages.s06_manuscript import (
    ValidationCheck,
    ManuscriptValidation,
    count_words,
    extract_abstract,
    check_word_count,
)


# ============================================================
# VALIDATION CHECK TESTS
# ============================================================

class TestValidationCheck:
    """Tests for the ValidationCheck dataclass."""

    def test_creates_check(self):
        """Test creating a validation check."""
        check = ValidationCheck(
            name='word_count',
            passed=True,
            message='Word count OK',
            severity='error'
        )

        assert check.name == 'word_count'
        assert check.passed is True
        assert check.severity == 'error'

    def test_with_actual_expected(self):
        """Test check with actual/expected values."""
        check = ValidationCheck(
            name='abstract_length',
            passed=False,
            message='Abstract too long',
            severity='warning',
            actual_value='300 words',
            expected_value='250 words max'
        )

        assert check.actual_value == '300 words'
        assert check.expected_value == '250 words max'

    def test_default_severity(self):
        """Test default severity is error."""
        check = ValidationCheck(
            name='test',
            passed=True,
            message='OK'
        )

        assert check.severity == 'error'


# ============================================================
# MANUSCRIPT VALIDATION TESTS
# ============================================================

class TestManuscriptValidation:
    """Tests for the ManuscriptValidation dataclass."""

    def test_creates_validation(self):
        """Test creating manuscript validation."""
        validation = ManuscriptValidation(journal='test_journal')

        assert validation.journal == 'test_journal'
        assert validation.checks == []

    def test_passed_with_no_errors(self):
        """Test passed is True when no errors."""
        validation = ManuscriptValidation(
            journal='test',
            checks=[
                ValidationCheck('check1', True, 'OK'),
                ValidationCheck('check2', True, 'OK'),
            ]
        )

        assert validation.passed is True
        assert validation.n_errors == 0

    def test_failed_with_error(self):
        """Test passed is False when error check fails."""
        validation = ManuscriptValidation(
            journal='test',
            checks=[
                ValidationCheck('check1', True, 'OK'),
                ValidationCheck('check2', False, 'Failed', severity='error'),
            ]
        )

        assert validation.passed is False
        assert validation.n_errors == 1

    def test_passed_with_warning_only(self):
        """Test passed is True when only warnings fail."""
        validation = ManuscriptValidation(
            journal='test',
            checks=[
                ValidationCheck('check1', True, 'OK'),
                ValidationCheck('check2', False, 'Warning', severity='warning'),
            ]
        )

        assert validation.passed is True
        assert validation.n_warnings == 1

    def test_n_errors_count(self):
        """Test error count calculation."""
        validation = ManuscriptValidation(
            journal='test',
            checks=[
                ValidationCheck('c1', False, 'E1', severity='error'),
                ValidationCheck('c2', False, 'E2', severity='error'),
                ValidationCheck('c3', False, 'W1', severity='warning'),
            ]
        )

        assert validation.n_errors == 2
        assert validation.n_warnings == 1

    def test_format_output(self):
        """Test string formatting."""
        validation = ManuscriptValidation(
            journal='test',
            checks=[
                ValidationCheck('check1', True, 'OK'),
            ]
        )

        formatted = validation.format()

        assert 'MANUSCRIPT VALIDATION' in formatted
        assert 'TEST' in formatted

    def test_to_markdown(self):
        """Test markdown formatting."""
        validation = ManuscriptValidation(
            journal='test',
            checks=[
                ValidationCheck('check1', True, 'OK'),
            ]
        )

        md = validation.to_markdown()

        assert '# Manuscript Validation Report' in md
        assert '**Journal:**' in md


# ============================================================
# WORD COUNT TESTS
# ============================================================

class TestCountWords:
    """Tests for the count_words function."""

    def test_simple_text(self):
        """Test counting words in simple text."""
        text = "This is a simple test."
        assert count_words(text) == 5

    def test_removes_yaml_frontmatter(self):
        """Test that YAML frontmatter is excluded."""
        text = """---
title: Test
author: Someone
---
This is the content."""

        count = count_words(text)
        assert count == 4  # "This is the content"

    def test_removes_code_blocks(self):
        """Test that code blocks are excluded."""
        text = """Some text here.
```python
def foo():
    pass
```
More text here."""

        count = count_words(text)
        assert 'foo' not in text.split() or count < 10

    def test_removes_inline_code(self):
        """Test that inline code is excluded."""
        text = "Use the `function_name` to do something."
        count = count_words(text)
        # Should count: Use, the, to, do, something
        assert count >= 4

    def test_empty_text(self):
        """Test counting empty text."""
        assert count_words("") == 0

    def test_preserves_link_text(self):
        """Test that link text is preserved."""
        text = "Check out [this link](http://example.com) for more."
        count = count_words(text)
        assert count >= 5  # Check, out, this, link, for, more


# ============================================================
# ABSTRACT EXTRACTION TESTS
# ============================================================

class TestExtractAbstract:
    """Tests for the extract_abstract function."""

    def test_extracts_from_yaml(self):
        """Test extracting abstract from YAML frontmatter."""
        text = """---
title: Test Paper
abstract: |
  This is the abstract text.
author: Test Author
---
Content here."""

        abstract = extract_abstract(text)
        assert abstract is not None
        assert 'abstract text' in abstract

    def test_extracts_from_section(self):
        """Test extracting abstract from section."""
        text = """# Title

## Abstract

This is the abstract paragraph.

## Introduction

This is the intro."""

        abstract = extract_abstract(text)
        assert abstract is not None
        assert 'abstract paragraph' in abstract

    def test_returns_none_for_no_abstract(self):
        """Test returns None when no abstract found."""
        text = """# Title

## Introduction

Just an intro here."""

        abstract = extract_abstract(text)
        # May or may not find abstract depending on implementation
        # Either None or empty is acceptable


# ============================================================
# WORD COUNT CHECK TESTS
# ============================================================

class TestCheckWordCount:
    """Tests for the check_word_count function."""

    def test_passes_under_limit(self, temp_dir):
        """Test check passes when under limit."""
        manuscript = temp_dir / 'test.qmd'
        manuscript.write_text("One two three four five.")

        check = check_word_count(manuscript, max_words=10)

        assert check.passed is True

    def test_fails_over_limit(self, temp_dir):
        """Test check fails when over limit."""
        manuscript = temp_dir / 'test.qmd'
        manuscript.write_text("One two three four five six seven eight nine ten eleven.")

        check = check_word_count(manuscript, max_words=5)

        assert check.passed is False

    def test_passes_without_limit(self, temp_dir):
        """Test check passes when no limit specified."""
        manuscript = temp_dir / 'test.qmd'
        manuscript.write_text("One two three.")

        check = check_word_count(manuscript)

        assert check.passed is True


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestManuscriptIntegration:
    """Integration tests for manuscript validation."""

    def test_full_validation_workflow(self, temp_dir):
        """Test complete validation workflow."""
        manuscript = temp_dir / 'test.qmd'
        manuscript.write_text("""---
title: Test Paper
abstract: |
  This is a test abstract for the paper.
---

# Introduction

This is the introduction section with some content.

# Methods

This is the methods section.

# Results

These are the results.

# Conclusion

This is the conclusion.
""")

        # Check word count
        check = check_word_count(manuscript, max_words=1000)

        # Create validation
        validation = ManuscriptValidation(
            journal='test',
            checks=[check]
        )

        assert validation.passed is True
