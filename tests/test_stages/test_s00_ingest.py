#!/usr/bin/env python3
"""
Tests for src/stages/s00_ingest.py

Tests cover:
- File discovery (find_input_files)
- Data loading from various formats (load_all_sources)
- Data cleaning (clean_data)
- Type conversion (convert_types)
- Validation (validate_input)
- Demo data generation (generate_demo_data)
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from stages.s00_ingest import (
    find_input_files,
    load_all_sources,
    clean_data,
    convert_types,
    validate_input,
    generate_demo_data,
    INPUT_PATTERNS,
    REQUIRED_COLUMNS,
)


# ============================================================
# FIND INPUT FILES TESTS
# ============================================================

class TestFindInputFiles:
    """Tests for the find_input_files function."""

    def test_finds_csv_files(self, temp_dir):
        """Test finding CSV files in directory."""
        # Create test CSV files
        (temp_dir / 'data1.csv').write_text('a,b\n1,2')
        (temp_dir / 'data2.csv').write_text('a,b\n3,4')

        files = find_input_files(temp_dir)

        assert len(files) == 2
        assert all(f.suffix == '.csv' for f in files)

    def test_finds_parquet_files(self, temp_dir, sample_df):
        """Test finding Parquet files in directory."""
        sample_df.to_parquet(temp_dir / 'data.parquet')

        files = find_input_files(temp_dir)

        assert len(files) == 1
        assert files[0].suffix == '.parquet'

    def test_finds_xlsx_files(self, temp_dir, sample_df):
        """Test finding Excel files in directory."""
        pytest.importorskip('openpyxl')
        sample_df.to_excel(temp_dir / 'data.xlsx', index=False)

        files = find_input_files(temp_dir)

        assert len(files) == 1
        assert files[0].suffix == '.xlsx'

    def test_finds_multiple_formats(self, temp_dir, sample_df):
        """Test finding files of multiple formats."""
        (temp_dir / 'data1.csv').write_text('a,b\n1,2')
        sample_df.to_parquet(temp_dir / 'data2.parquet')

        files = find_input_files(temp_dir)

        assert len(files) == 2
        suffixes = {f.suffix for f in files}
        assert suffixes == {'.csv', '.parquet'}

    def test_returns_empty_for_no_data(self, temp_dir):
        """Test empty result for directory without data files."""
        files = find_input_files(temp_dir)

        assert len(files) == 0
        assert isinstance(files, list)

    def test_ignores_non_matching_files(self, temp_dir):
        """Test that non-matching files are ignored."""
        (temp_dir / 'readme.txt').write_text('documentation')
        (temp_dir / 'script.py').write_text('# python')
        (temp_dir / 'data.csv').write_text('a,b\n1,2')

        files = find_input_files(temp_dir)

        assert len(files) == 1
        assert files[0].name == 'data.csv'

    def test_returns_sorted_files(self, temp_dir):
        """Test that files are returned in sorted order."""
        (temp_dir / 'c_data.csv').write_text('a,b\n1,2')
        (temp_dir / 'a_data.csv').write_text('a,b\n3,4')
        (temp_dir / 'b_data.csv').write_text('a,b\n5,6')

        files = find_input_files(temp_dir)

        names = [f.name for f in files]
        assert names == ['a_data.csv', 'b_data.csv', 'c_data.csv']

    def test_custom_patterns(self, temp_dir):
        """Test finding files with custom patterns."""
        (temp_dir / 'data.csv').write_text('a,b\n1,2')
        (temp_dir / 'data.json').write_text('{}')

        # Only look for JSON files
        files = find_input_files(temp_dir, patterns=['*.json'])

        assert len(files) == 1
        assert files[0].suffix == '.json'

    def test_uses_default_patterns(self, temp_dir):
        """Test that default patterns are used when none specified."""
        (temp_dir / 'data.csv').write_text('a,b\n1,2')

        files = find_input_files(temp_dir)

        assert len(files) == 1


# ============================================================
# LOAD ALL SOURCES TESTS
# ============================================================

class TestLoadAllSources:
    """Tests for the load_all_sources function."""

    def test_loads_single_csv(self, temp_dir):
        """Test loading a single CSV file."""
        csv_path = temp_dir / 'data.csv'
        csv_path.write_text('id,value\n1,100\n2,200')

        result = load_all_sources([csv_path])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ['id', 'value']

    def test_loads_multiple_csvs_concatenated(self, temp_dir):
        """Test loading and concatenating multiple CSV files."""
        csv1 = temp_dir / 'data1.csv'
        csv2 = temp_dir / 'data2.csv'
        csv1.write_text('id,value\n1,100\n2,200')
        csv2.write_text('id,value\n3,300\n4,400')

        result = load_all_sources([csv1, csv2], concat=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

    def test_loads_as_dict_when_concat_false(self, temp_dir):
        """Test loading files as dictionary when concat=False."""
        csv1 = temp_dir / 'data1.csv'
        csv2 = temp_dir / 'data2.csv'
        csv1.write_text('id,value\n1,100')
        csv2.write_text('id,value\n2,200')

        result = load_all_sources([csv1, csv2], concat=False)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert 'data1' in result
        assert 'data2' in result

    def test_loads_parquet(self, temp_dir, sample_df):
        """Test loading a Parquet file."""
        parquet_path = temp_dir / 'data.parquet'
        sample_df.to_parquet(parquet_path)

        result = load_all_sources([parquet_path])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_df)

    def test_handles_empty_file_list(self):
        """Test handling of empty file list."""
        result = load_all_sources([], concat=True)

        # When concat=True and no files, returns empty dict (falsy)
        assert not result or isinstance(result, (dict, pd.DataFrame))


# ============================================================
# CLEAN DATA TESTS
# ============================================================

class TestCleanData:
    """Tests for the clean_data function."""

    def test_drops_completely_empty_rows(self):
        """Test dropping rows that are entirely empty."""
        df = pd.DataFrame({
            'a': [1, np.nan, 3],
            'b': [4, np.nan, 6]
        })

        result = clean_data(df)

        assert len(result) == 2
        assert 1 in result['a'].values
        assert 3 in result['a'].values

    def test_keeps_partially_empty_rows(self):
        """Test that rows with some values are kept."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, np.nan, 6]
        })

        result = clean_data(df)

        assert len(result) == 3

    def test_drops_duplicate_rows(self):
        """Test dropping duplicate rows."""
        df = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': [4, 5, 5, 6]
        })

        result = clean_data(df, drop_duplicates=True)

        assert len(result) == 3

    def test_keeps_duplicates_when_disabled(self):
        """Test keeping duplicates when drop_duplicates=False."""
        df = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': [4, 5, 5, 6]
        })

        result = clean_data(df, drop_duplicates=False)

        assert len(result) == 4

    def test_resets_index(self):
        """Test that index is reset."""
        df = pd.DataFrame({
            'a': [1, 2, 3]
        }, index=[10, 20, 30])

        result = clean_data(df, reset_index=True)

        assert list(result.index) == [0, 1, 2]

    def test_preserves_index_when_disabled(self):
        """Test preserving index when reset_index=False."""
        df = pd.DataFrame({
            'a': [1, 2, 3]
        }, index=[10, 20, 30])

        result = clean_data(df, reset_index=False)

        assert list(result.index) == [10, 20, 30]

    def test_returns_dataframe(self, sample_df):
        """Test that result is always a DataFrame."""
        result = clean_data(sample_df)

        assert isinstance(result, pd.DataFrame)


# ============================================================
# CONVERT TYPES TESTS
# ============================================================

class TestConvertTypes:
    """Tests for the convert_types function."""

    def test_converts_to_int(self):
        """Test converting column to integer."""
        df = pd.DataFrame({'id': ['1', '2', '3']})

        result = convert_types(df, {'id': 'int64'})

        assert result['id'].dtype == 'int64'

    def test_converts_to_float(self):
        """Test converting column to float."""
        df = pd.DataFrame({'value': ['1.5', '2.5', '3.5']})

        result = convert_types(df, {'value': 'float64'})

        assert result['value'].dtype == 'float64'

    def test_skips_missing_columns(self):
        """Test that missing columns are skipped without error."""
        df = pd.DataFrame({'a': [1, 2, 3]})

        result = convert_types(df, {'missing_col': 'int64'})

        assert 'a' in result.columns
        assert 'missing_col' not in result.columns

    def test_handles_conversion_errors(self):
        """Test graceful handling of conversion errors."""
        df = pd.DataFrame({'id': ['a', 'b', 'c']})

        # Should not raise, just warn
        result = convert_types(df, {'id': 'int64'})

        assert isinstance(result, pd.DataFrame)

    def test_uses_default_type_map(self):
        """Test using default COLUMN_TYPES."""
        df = pd.DataFrame({'id': [1.0, 2.0, 3.0]})

        result = convert_types(df)

        # Default COLUMN_TYPES has 'id': 'int64'
        assert result['id'].dtype == 'int64'

    def test_preserves_unconverted_columns(self):
        """Test that columns not in type_map are preserved."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['a', 'b', 'c']
        })

        result = convert_types(df, {'id': 'int64'})

        assert 'name' in result.columns
        assert result['name'].dtype == 'object'


# ============================================================
# VALIDATE INPUT TESTS
# ============================================================

class TestValidateInput:
    """Tests for the validate_input function."""

    def test_passes_for_valid_data(self):
        """Test validation passes for valid data."""
        df = pd.DataFrame({'id': [1, 2, 3], 'value': [4, 5, 6]})

        result = validate_input(df)

        assert result is True

    def test_passes_for_non_empty_data(self):
        """Test validation passes for non-empty data."""
        df = pd.DataFrame({'id': [1]})

        result = validate_input(df)

        assert result is True

    def test_validates_row_count(self):
        """Test validation checks row count."""
        df = pd.DataFrame({'id': []})

        result = validate_input(df)

        assert result is False

    def test_validates_required_columns(self):
        """Test validation checks required columns."""
        # Create data with missing values in required column
        if REQUIRED_COLUMNS:
            df = pd.DataFrame({
                REQUIRED_COLUMNS[0]: [1, None, 3],
                'other': [4, 5, 6]
            })

            result = validate_input(df)

            assert result is False


# ============================================================
# GENERATE DEMO DATA TESTS
# ============================================================

class TestGenerateDemoData:
    """Tests for the generate_demo_data function."""

    def test_returns_dataframe(self):
        """Test that demo data is a DataFrame."""
        result = generate_demo_data()

        assert isinstance(result, pd.DataFrame)

    def test_has_expected_size(self):
        """Test demo data has expected dimensions."""
        result = generate_demo_data()

        # 500 units * 24 periods = 12000 rows
        assert len(result) == 12000

    def test_has_id_column(self):
        """Test demo data has id column."""
        result = generate_demo_data()

        assert 'id' in result.columns

    def test_has_required_columns(self):
        """Test demo data has all required columns."""
        result = generate_demo_data()

        for col in REQUIRED_COLUMNS:
            assert col in result.columns

    def test_is_deterministic(self):
        """Test demo data generation is deterministic (same seed)."""
        result1 = generate_demo_data()
        result2 = generate_demo_data()

        pd.testing.assert_frame_equal(result1, result2)

    def test_contains_treatment_column(self):
        """Test demo data contains treatment indicator."""
        result = generate_demo_data()

        assert 'treatment' in result.columns
        assert set(result['treatment'].unique()).issubset({0, 1})


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIngestStageIntegration:
    """Integration tests for the ingestion stage."""

    def test_full_pipeline_with_csv(self, temp_dir, sample_df):
        """Test full ingestion pipeline with CSV input."""
        # Setup
        raw_dir = temp_dir / 'data_raw'
        work_dir = temp_dir / 'data_work'
        raw_dir.mkdir()
        work_dir.mkdir()

        # Add id column if not present
        if 'id' not in sample_df.columns:
            sample_df['id'] = range(1, len(sample_df) + 1)

        csv_path = raw_dir / 'test_data.csv'
        sample_df.to_csv(csv_path, index=False)

        # Find and load
        files = find_input_files(raw_dir)
        assert len(files) == 1

        df = load_all_sources(files)
        assert len(df) == len(sample_df)

        # Clean
        df = clean_data(df)
        df = convert_types(df)

        # Validate
        is_valid = validate_input(df)
        assert is_valid is True

    def test_full_pipeline_with_parquet(self, temp_dir, sample_df):
        """Test full ingestion pipeline with Parquet input."""
        # Setup
        raw_dir = temp_dir / 'data_raw'
        raw_dir.mkdir()

        # Add id column if not present
        if 'id' not in sample_df.columns:
            sample_df['id'] = range(1, len(sample_df) + 1)

        parquet_path = raw_dir / 'test_data.parquet'
        sample_df.to_parquet(parquet_path)

        # Find and load
        files = find_input_files(raw_dir)
        df = load_all_sources(files)

        # Clean and validate
        df = clean_data(df)
        df = convert_types(df)
        is_valid = validate_input(df)

        assert is_valid is True

    def test_empty_directory_returns_empty_list(self, temp_dir):
        """Test that empty directory returns empty file list."""
        files = find_input_files(temp_dir)

        assert files == []
