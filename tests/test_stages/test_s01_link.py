#!/usr/bin/env python3
"""
Tests for src/stages/s01_link.py

Tests cover:
- LinkageResult dataclass
- Exact matching (exact_match)
- Fuzzy matching (fuzzy_match)
- String similarity functions
- Linkage diagnostics generation
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from stages.s01_link import (
    LinkageResult,
    exact_match,
    fuzzy_match,
    generate_linkage_diagnostics,
    _levenshtein_similarity,
    _jaro_winkler_similarity,
    DEFAULT_KEY_COLUMNS,
)


# ============================================================
# LINKAGE RESULT TESTS
# ============================================================

class TestLinkageResult:
    """Tests for the LinkageResult dataclass."""

    def test_creates_result(self):
        """Test creating a linkage result."""
        result = LinkageResult(
            source_name='test_source',
            n_source=100,
            n_matched=80,
            n_unmatched=20,
            match_type='exact',
            key_columns=['id']
        )

        assert result.source_name == 'test_source'
        assert result.n_source == 100
        assert result.n_matched == 80
        assert result.n_unmatched == 20
        assert result.match_type == 'exact'

    def test_match_rate_calculation(self):
        """Test match rate property calculation."""
        result = LinkageResult(
            source_name='test',
            n_source=100,
            n_matched=75,
            n_unmatched=25,
            match_type='exact'
        )

        assert result.match_rate == 0.75

    def test_match_rate_zero_total(self):
        """Test match rate with zero total."""
        result = LinkageResult(
            source_name='test',
            n_source=0,
            n_matched=0,
            n_unmatched=0,
            match_type='exact'
        )

        assert result.match_rate == 0.0

    def test_match_rate_perfect(self):
        """Test perfect match rate."""
        result = LinkageResult(
            source_name='test',
            n_source=100,
            n_matched=100,
            n_unmatched=0,
            match_type='exact'
        )

        assert result.match_rate == 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = LinkageResult(
            source_name='test',
            n_source=100,
            n_matched=80,
            n_unmatched=20,
            match_type='fuzzy',
            key_columns=['id', 'name']
        )

        d = result.to_dict()

        assert d['source'] == 'test'
        assert d['n_source'] == 100
        assert d['n_matched'] == 80
        assert d['n_unmatched'] == 20
        assert d['match_rate'] == 0.8
        assert d['match_type'] == 'fuzzy'
        assert d['key_columns'] == 'id,name'

    def test_default_key_columns(self):
        """Test default empty key columns."""
        result = LinkageResult(
            source_name='test',
            n_source=100,
            n_matched=80,
            n_unmatched=20,
            match_type='exact'
        )

        assert result.key_columns == []


# ============================================================
# EXACT MATCH TESTS
# ============================================================

class TestExactMatch:
    """Tests for the exact_match function."""

    def test_exact_match_all_matched(self):
        """Test exact matching when all records match."""
        df_left = pd.DataFrame({
            'id': [1, 2, 3],
            'value': ['a', 'b', 'c']
        })
        df_right = pd.DataFrame({
            'id': [1, 2, 3],
            'extra': ['x', 'y', 'z']
        })

        merged, result = exact_match(df_left, df_right, on='id')

        assert len(merged) == 3
        assert result.n_matched == 3
        assert result.n_unmatched == 0
        assert result.match_type == 'exact'

    def test_exact_match_partial(self):
        """Test exact matching with partial matches."""
        df_left = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': ['a', 'b', 'c', 'd', 'e']
        })
        df_right = pd.DataFrame({
            'id': [1, 3, 5],
            'extra': ['x', 'y', 'z']
        })

        merged, result = exact_match(df_left, df_right, on='id', how='left')

        assert len(merged) == 5
        assert result.n_matched == 3
        assert result.n_unmatched == 2

    def test_exact_match_no_matches(self):
        """Test exact matching with no matches."""
        df_left = pd.DataFrame({
            'id': [1, 2, 3],
            'value': ['a', 'b', 'c']
        })
        df_right = pd.DataFrame({
            'id': [4, 5, 6],
            'extra': ['x', 'y', 'z']
        })

        merged, result = exact_match(df_left, df_right, on='id', how='left')

        assert len(merged) == 3
        assert result.n_matched == 0
        assert result.n_unmatched == 3

    def test_exact_match_multiple_columns(self):
        """Test exact matching on multiple columns."""
        df_left = pd.DataFrame({
            'id': [1, 1, 2],
            'period': [1, 2, 1],
            'value': ['a', 'b', 'c']
        })
        df_right = pd.DataFrame({
            'id': [1, 2],
            'period': [1, 1],
            'extra': ['x', 'y']
        })

        merged, result = exact_match(df_left, df_right, on=['id', 'period'], how='left')

        assert result.n_matched == 2
        assert result.key_columns == ['id', 'period']

    def test_exact_match_inner_join(self):
        """Test exact matching with inner join."""
        df_left = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'value': ['a', 'b', 'c', 'd']
        })
        df_right = pd.DataFrame({
            'id': [2, 3],
            'extra': ['x', 'y']
        })

        merged, result = exact_match(df_left, df_right, on='id', how='inner')

        assert len(merged) == 2
        assert result.n_matched == 2

    def test_exact_match_suffixes(self):
        """Test exact matching with custom suffixes."""
        df_left = pd.DataFrame({
            'id': [1, 2],
            'name': ['a', 'b']
        })
        df_right = pd.DataFrame({
            'id': [1, 2],
            'name': ['x', 'y']
        })

        merged, result = exact_match(
            df_left, df_right, on='id',
            suffixes=('_left', '_right')
        )

        assert 'name_left' in merged.columns
        assert 'name_right' in merged.columns

    def test_exact_match_string_column(self):
        """Test exact matching accepts single string column."""
        df_left = pd.DataFrame({'id': [1, 2]})
        df_right = pd.DataFrame({'id': [1, 2], 'extra': [10, 20]})

        merged, result = exact_match(df_left, df_right, on='id')

        assert result.key_columns == ['id']


# ============================================================
# FUZZY MATCH TESTS
# ============================================================

class TestFuzzyMatch:
    """Tests for the fuzzy_match function."""

    def test_fuzzy_match_exact_strings(self):
        """Test fuzzy matching with exact string matches."""
        df_left = pd.DataFrame({
            'id': [1, 2],
            'name': ['apple', 'banana']
        })
        df_right = pd.DataFrame({
            'code': ['apple', 'banana'],
            'value': [10, 20]
        })

        merged, result = fuzzy_match(
            df_left, df_right,
            left_on='name', right_on='code',
            threshold=0.9
        )

        assert result.n_matched == 2

    def test_fuzzy_match_similar_strings(self):
        """Test fuzzy matching with similar strings."""
        df_left = pd.DataFrame({
            'id': [1],
            'name': ['apple']
        })
        df_right = pd.DataFrame({
            'code': ['aple'],  # Typo
            'value': [10]
        })

        merged, result = fuzzy_match(
            df_left, df_right,
            left_on='name', right_on='code',
            threshold=0.7
        )

        assert result.n_matched == 1

    def test_fuzzy_match_no_matches(self):
        """Test fuzzy matching with no matches."""
        df_left = pd.DataFrame({
            'id': [1],
            'name': ['apple']
        })
        df_right = pd.DataFrame({
            'code': ['xyz'],
            'value': [10]
        })

        merged, result = fuzzy_match(
            df_left, df_right,
            left_on='name', right_on='code',
            threshold=0.8
        )

        assert result.n_matched == 0

    def test_fuzzy_match_contains_method(self):
        """Test fuzzy matching with contains method."""
        df_left = pd.DataFrame({
            'id': [1],
            'name': ['apple']
        })
        df_right = pd.DataFrame({
            'code': ['green apple pie'],
            'value': [10]
        })

        merged, result = fuzzy_match(
            df_left, df_right,
            left_on='name', right_on='code',
            method='contains',
            threshold=0.5
        )

        assert result.n_matched == 1

    def test_fuzzy_match_returns_linkage_result(self):
        """Test that fuzzy match returns LinkageResult."""
        df_left = pd.DataFrame({'id': [1], 'name': ['test']})
        df_right = pd.DataFrame({'code': ['test'], 'value': [10]})

        merged, result = fuzzy_match(
            df_left, df_right,
            left_on='name', right_on='code'
        )

        assert isinstance(result, LinkageResult)
        assert 'fuzzy' in result.match_type


# ============================================================
# STRING SIMILARITY TESTS
# ============================================================

class TestLevenshteinSimilarity:
    """Tests for the Levenshtein similarity function."""

    def test_identical_strings(self):
        """Test similarity of identical strings."""
        assert _levenshtein_similarity('hello', 'hello') == 1.0

    def test_empty_strings(self):
        """Test similarity of empty strings."""
        assert _levenshtein_similarity('', '') == 1.0

    def test_one_empty_string(self):
        """Test similarity when one string is empty."""
        assert _levenshtein_similarity('hello', '') == 0.0
        assert _levenshtein_similarity('', 'hello') == 0.0

    def test_one_character_difference(self):
        """Test similarity with one character difference."""
        sim = _levenshtein_similarity('hello', 'hallo')
        assert 0.7 < sim < 1.0

    def test_completely_different(self):
        """Test similarity of completely different strings."""
        sim = _levenshtein_similarity('abc', 'xyz')
        assert sim == 0.0

    def test_case_insensitive(self):
        """Test case insensitivity."""
        sim = _levenshtein_similarity('Hello', 'hello')
        assert sim == 1.0


class TestJaroWinklerSimilarity:
    """Tests for the Jaro-Winkler similarity function."""

    def test_identical_strings(self):
        """Test similarity of identical strings."""
        assert _jaro_winkler_similarity('hello', 'hello') == 1.0

    def test_empty_strings(self):
        """Test similarity of empty strings (identical = 1.0)."""
        # Two identical empty strings have perfect similarity
        assert _jaro_winkler_similarity('', '') == 1.0

    def test_one_empty_string(self):
        """Test similarity when one string is empty."""
        assert _jaro_winkler_similarity('hello', '') == 0.0

    def test_similar_prefix(self):
        """Test that common prefix boosts similarity."""
        sim1 = _jaro_winkler_similarity('martha', 'marhta')
        sim2 = _jaro_winkler_similarity('dwayne', 'duane')

        assert sim1 > 0.8
        assert sim2 > 0.7

    def test_case_insensitive(self):
        """Test case insensitivity."""
        sim = _jaro_winkler_similarity('Hello', 'hello')
        assert sim == 1.0


# ============================================================
# DIAGNOSTICS TESTS
# ============================================================

class TestGenerateLinkageDiagnostics:
    """Tests for the generate_linkage_diagnostics function."""

    def test_generates_summary(self, temp_dir):
        """Test generating diagnostics summary."""
        results = [
            LinkageResult('source1', 100, 80, 20, 'exact', ['id']),
            LinkageResult('source2', 50, 40, 10, 'fuzzy', ['name']),
        ]

        diag_dir = temp_dir / 'diagnostics'
        diag_dir.mkdir()

        summary_df = generate_linkage_diagnostics(results, diag_dir)

        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 2
        assert 'source' in summary_df.columns
        assert 'match_rate' in summary_df.columns

    def test_empty_results(self, temp_dir):
        """Test generating diagnostics with empty results."""
        diag_dir = temp_dir / 'diagnostics'
        diag_dir.mkdir()

        summary_df = generate_linkage_diagnostics([], diag_dir)

        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 0

    def test_creates_directory_if_missing(self, temp_dir):
        """Test that directory is created if missing."""
        results = [LinkageResult('test', 10, 8, 2, 'exact')]
        diag_dir = temp_dir / 'new_diagnostics'

        summary_df = generate_linkage_diagnostics(results, diag_dir)

        assert diag_dir.exists()


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestLinkStageIntegration:
    """Integration tests for the linkage stage."""

    def test_link_two_sources_exact(self):
        """Test linking two data sources with exact match."""
        primary = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['a', 'b', 'c', 'd', 'e']
        })
        secondary = pd.DataFrame({
            'id': [1, 2, 3],
            'score': [100, 200, 300]
        })
        secondary.name = 'scores'

        merged, result = exact_match(primary, secondary, on='id', how='left')

        assert len(merged) == 5
        assert 'score' in merged.columns
        assert result.n_matched == 3
        assert result.n_unmatched == 2

    def test_multiple_linkages(self):
        """Test performing multiple linkages sequentially."""
        primary = pd.DataFrame({
            'id': [1, 2, 3],
            'value': ['a', 'b', 'c']
        })
        source1 = pd.DataFrame({
            'id': [1, 2],
            'extra1': [10, 20]
        })
        source2 = pd.DataFrame({
            'id': [2, 3],
            'extra2': [100, 200]
        })

        # First linkage
        df, result1 = exact_match(primary, source1, on='id', how='left')
        # Second linkage
        df, result2 = exact_match(df, source2, on='id', how='left')

        assert len(df) == 3
        assert 'extra1' in df.columns
        assert 'extra2' in df.columns
        assert result1.n_matched == 2
        assert result2.n_matched == 2

    def test_linkage_result_tracking(self, temp_dir):
        """Test tracking multiple linkage results."""
        results = []

        # Simulate multiple linkages
        for i in range(3):
            result = LinkageResult(
                source_name=f'source_{i}',
                n_source=100,
                n_matched=80 - i * 10,
                n_unmatched=20 + i * 10,
                match_type='exact',
                key_columns=['id']
            )
            results.append(result)

        diag_dir = temp_dir / 'diagnostics'
        diag_dir.mkdir()

        summary = generate_linkage_diagnostics(results, diag_dir)

        assert len(summary) == 3
        assert summary['match_rate'].iloc[0] == 0.8
