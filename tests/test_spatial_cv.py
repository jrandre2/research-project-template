#!/usr/bin/env python3
"""
Tests for spatial cross-validation module.

Tests cover:
- Grouping methods that don't require geopandas
- Cross-validation functionality
- Leakage quantification
- Error handling for missing dependencies
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def geographic_df() -> pd.DataFrame:
    """Create a sample DataFrame with geographic coordinates."""
    np.random.seed(42)
    n = 100

    # Generate coordinates roughly in a rectangular area
    lats = np.random.uniform(40.0, 42.0, n)  # Latitude range
    lons = np.random.uniform(-100.0, -98.0, n)  # Longitude range

    # Generate outcome correlated with location (spatial pattern)
    outcome = lats + lons * 0.5 + np.random.randn(n) * 0.5

    # Generate features
    feature1 = lats * 0.3 + np.random.randn(n) * 0.2
    feature2 = lons * -0.2 + np.random.randn(n) * 0.3

    return pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'outcome': outcome,
        'feature1': feature1,
        'feature2': feature2,
        'zip': [f'{69000 + i % 100:05d}' for i in range(n)],
    })


@pytest.fixture
def simple_xy():
    """Create simple X, y arrays for testing."""
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 5)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
    lats = np.random.uniform(40, 42, n)
    lons = np.random.uniform(-100, -98, n)
    return X, y, lats, lons


# ============================================================
# IMPORT TESTS
# ============================================================

def test_import_spatial_cv():
    """Test that spatial_cv module can be imported."""
    from utils.spatial_cv import SpatialCVManager
    assert SpatialCVManager is not None


def test_import_convenience_functions():
    """Test that convenience functions can be imported."""
    from utils.spatial_cv import (
        create_spatial_groups_simple,
        compare_spatial_vs_random_cv,
    )
    assert create_spatial_groups_simple is not None
    assert compare_spatial_vs_random_cv is not None


# ============================================================
# GROUPING METHOD TESTS
# ============================================================

class TestGroupingMethods:
    """Tests for spatial grouping methods."""

    def test_kmeans_grouping(self, geographic_df):
        """Test k-means grouping creates correct number of groups."""
        from utils.spatial_cv import SpatialCVManager

        manager = SpatialCVManager(n_groups=5, method='kmeans')
        groups = manager.create_groups_from_coordinates(
            geographic_df['latitude'].values,
            geographic_df['longitude'].values,
            verbose=False,
        )

        assert len(groups) == len(geographic_df)
        assert len(np.unique(groups)) == 5
        assert groups.min() >= 0
        assert groups.max() <= 4

    def test_balanced_kmeans_grouping(self, geographic_df):
        """Test balanced k-means creates roughly equal group sizes."""
        from utils.spatial_cv import SpatialCVManager

        manager = SpatialCVManager(n_groups=5, method='balanced_kmeans')
        groups = manager.create_groups_from_coordinates(
            geographic_df['latitude'].values,
            geographic_df['longitude'].values,
            verbose=False,
        )

        unique, counts = np.unique(groups, return_counts=True)

        # Check balance: max group size should be at most 1 more than min
        assert counts.max() - counts.min() <= 1

    def test_geographic_bands_grouping(self, geographic_df):
        """Test geographic bands creates latitude-based groups."""
        from utils.spatial_cv import SpatialCVManager

        manager = SpatialCVManager(n_groups=5, method='geographic_bands')
        groups = manager.create_groups_from_coordinates(
            geographic_df['latitude'].values,
            geographic_df['longitude'].values,
            verbose=False,
        )

        # Lower latitude should have lower group numbers (roughly)
        df = geographic_df.copy()
        df['group'] = groups
        group_means = df.groupby('group')['latitude'].mean()

        # Groups should be roughly ordered by latitude
        assert group_means.is_monotonic_increasing or len(group_means) == 1

    def test_longitude_bands_grouping(self, geographic_df):
        """Test longitude bands creates longitude-based groups."""
        from utils.spatial_cv import SpatialCVManager

        manager = SpatialCVManager(n_groups=5, method='longitude_bands')
        groups = manager.create_groups_from_coordinates(
            geographic_df['latitude'].values,
            geographic_df['longitude'].values,
            verbose=False,
        )

        assert len(np.unique(groups)) == 5

    def test_spatial_blocks_grouping(self, geographic_df):
        """Test spatial blocks creates grid-based groups."""
        from utils.spatial_cv import SpatialCVManager

        manager = SpatialCVManager(n_groups=4, method='spatial_blocks')
        groups = manager.create_groups_from_coordinates(
            geographic_df['latitude'].values,
            geographic_df['longitude'].values,
            verbose=False,
        )

        assert len(groups) == len(geographic_df)
        # Spatial blocks maps to n_groups via modulo
        assert groups.max() < 4

    def test_zip_digit_grouping(self, geographic_df):
        """Test ZIP code digit-based grouping."""
        from utils.spatial_cv import SpatialCVManager

        manager = SpatialCVManager(n_groups=5, method='zip_digit')
        groups = manager.create_groups_from_zip_codes(
            geographic_df['zip'],
            digit_position=3,
            verbose=False,
        )

        assert len(groups) == len(geographic_df)
        assert len(np.unique(groups)) <= 5

    def test_contiguity_requires_geodataframe(self, geographic_df):
        """Test that contiguity methods raise error without GeoDataFrame."""
        from utils.spatial_cv import SpatialCVManager

        manager = SpatialCVManager(n_groups=5, method='contiguity_queen')

        with pytest.raises(ValueError, match="requires polygon geometry"):
            manager.create_groups_from_coordinates(
                geographic_df['latitude'].values,
                geographic_df['longitude'].values,
                verbose=False,
            )

    def test_unknown_method_raises_error(self, geographic_df):
        """Test that unknown methods raise ValueError."""
        from utils.spatial_cv import SpatialCVManager

        manager = SpatialCVManager(n_groups=5, method='unknown_method')

        with pytest.raises(ValueError, match="Unknown"):
            manager.create_groups_from_coordinates(
                geographic_df['latitude'].values,
                geographic_df['longitude'].values,
                verbose=False,
            )


# ============================================================
# CROSS-VALIDATION TESTS
# ============================================================

class TestCrossValidation:
    """Tests for spatial cross-validation functionality."""

    def test_split_generates_correct_folds(self, simple_xy):
        """Test that split generates correct number of folds."""
        from utils.spatial_cv import SpatialCVManager

        X, y, lats, lons = simple_xy
        manager = SpatialCVManager(n_groups=5, method='kmeans')
        manager.create_groups_from_coordinates(lats, lons, verbose=False)

        folds = list(manager.split(X, y))
        assert len(folds) == 5

        for train_idx, test_idx in folds:
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0
            # All indices covered
            assert len(train_idx) + len(test_idx) == len(y)

    def test_cross_validate_returns_results(self, simple_xy):
        """Test that cross_validate returns expected structure."""
        from utils.spatial_cv import SpatialCVManager
        from sklearn.linear_model import Ridge

        X, y, lats, lons = simple_xy
        manager = SpatialCVManager(n_groups=5, method='kmeans')
        manager.create_groups_from_coordinates(lats, lons, verbose=False)

        model = Ridge(alpha=1.0)
        results = manager.cross_validate(model, X, y)

        assert 'scores' in results
        assert 'mean' in results
        assert 'std' in results
        assert 'fold_details' in results

        assert len(results['scores']) == 5
        assert isinstance(results['mean'], float)
        assert isinstance(results['std'], float)

    def test_cross_validate_with_scaling(self, simple_xy):
        """Test cross-validation with and without feature scaling."""
        from utils.spatial_cv import SpatialCVManager
        from sklearn.linear_model import Ridge

        X, y, lats, lons = simple_xy
        manager = SpatialCVManager(n_groups=5, method='kmeans')
        manager.create_groups_from_coordinates(lats, lons, verbose=False)

        model = Ridge(alpha=1.0)

        results_scaled = manager.cross_validate(model, X, y, scale_features=True)
        results_unscaled = manager.cross_validate(model, X, y, scale_features=False)

        # Both should return valid results
        assert len(results_scaled['scores']) == 5
        assert len(results_unscaled['scores']) == 5

    def test_split_without_groups_raises_error(self, simple_xy):
        """Test that split raises error if groups not created."""
        from utils.spatial_cv import SpatialCVManager

        X, y, _, _ = simple_xy
        manager = SpatialCVManager(n_groups=5, method='kmeans')
        # Don't create groups

        with pytest.raises(ValueError, match="Must create spatial groups"):
            list(manager.split(X, y))


# ============================================================
# LEAKAGE QUANTIFICATION TESTS
# ============================================================

class TestLeakageQuantification:
    """Tests for leakage quantification functionality."""

    def test_compare_to_random_cv_returns_results(self, simple_xy):
        """Test that compare_to_random_cv returns expected structure."""
        from utils.spatial_cv import SpatialCVManager
        from sklearn.linear_model import Ridge

        X, y, lats, lons = simple_xy
        manager = SpatialCVManager(n_groups=5, method='kmeans')
        manager.create_groups_from_coordinates(lats, lons, verbose=False)

        model = Ridge(alpha=1.0)
        comparison = manager.compare_to_random_cv(model, X, y)

        assert 'spatial_cv' in comparison
        assert 'random_cv' in comparison
        assert 'leakage' in comparison
        assert 'leakage_pct' in comparison

        assert 'mean' in comparison['spatial_cv']
        assert 'std' in comparison['spatial_cv']
        assert 'mean' in comparison['random_cv']
        assert 'std' in comparison['random_cv']

    def test_convenience_compare_function(self, simple_xy):
        """Test the convenience compare function."""
        from utils.spatial_cv import compare_spatial_vs_random_cv
        from sklearn.linear_model import Ridge

        X, y, lats, lons = simple_xy
        model = Ridge(alpha=1.0)

        results = compare_spatial_vs_random_cv(
            model, X, y, lats, lons,
            n_groups=5,
            method='kmeans',
            verbose=False,
        )

        assert 'leakage' in results
        assert isinstance(results['leakage'], float)


# ============================================================
# GROUP STATISTICS TESTS
# ============================================================

class TestGroupStatistics:
    """Tests for group statistics functionality."""

    def test_get_group_statistics(self, simple_xy):
        """Test that get_group_statistics returns expected structure."""
        from utils.spatial_cv import SpatialCVManager

        _, _, lats, lons = simple_xy
        manager = SpatialCVManager(n_groups=5, method='kmeans')
        manager.create_groups_from_coordinates(lats, lons, verbose=False)

        stats = manager.get_group_statistics()

        assert stats is not None
        assert 'n_groups' in stats
        assert 'group_sizes' in stats
        assert 'min_size' in stats
        assert 'max_size' in stats
        assert 'mean_size' in stats
        assert 'balance_ratio' in stats

        assert stats['n_groups'] == 5
        assert 0 < stats['balance_ratio'] <= 1

    def test_get_group_statistics_without_groups(self):
        """Test that get_group_statistics returns None if groups not created."""
        from utils.spatial_cv import SpatialCVManager

        manager = SpatialCVManager(n_groups=5, method='kmeans')
        stats = manager.get_group_statistics()

        assert stats is None


# ============================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_spatial_groups_simple(self, geographic_df):
        """Test the simple convenience function."""
        from utils.spatial_cv import create_spatial_groups_simple

        groups = create_spatial_groups_simple(
            geographic_df,
            lat_col='latitude',
            lon_col='longitude',
            n_groups=5,
            method='kmeans',
        )

        assert len(groups) == len(geographic_df)
        assert len(np.unique(groups)) == 5


# ============================================================
# EDGE CASE TESTS
# ============================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_dataset(self):
        """Test behavior with small dataset."""
        from utils.spatial_cv import SpatialCVManager

        np.random.seed(42)
        n = 10  # Very small
        lats = np.random.uniform(40, 42, n)
        lons = np.random.uniform(-100, -98, n)

        manager = SpatialCVManager(n_groups=3, method='kmeans')
        groups = manager.create_groups_from_coordinates(lats, lons, verbose=False)

        # Should still work
        assert len(groups) == n

    def test_two_groups(self):
        """Test with n_groups=2 (minimum for cross-validation)."""
        from utils.spatial_cv import SpatialCVManager

        np.random.seed(42)
        lats = np.random.uniform(40, 42, 50)
        lons = np.random.uniform(-100, -98, 50)

        manager = SpatialCVManager(n_groups=2, method='kmeans')
        groups = manager.create_groups_from_coordinates(lats, lons, verbose=False)

        assert len(np.unique(groups)) == 2

    def test_series_input(self, geographic_df):
        """Test that pandas Series input works."""
        from utils.spatial_cv import SpatialCVManager

        manager = SpatialCVManager(n_groups=5, method='kmeans')
        groups = manager.create_groups_from_coordinates(
            geographic_df['latitude'],  # pandas Series
            geographic_df['longitude'],  # pandas Series
            verbose=False,
        )

        assert len(groups) == len(geographic_df)

    def test_list_input(self):
        """Test that list input works."""
        from utils.spatial_cv import SpatialCVManager

        lats = [40.0, 40.5, 41.0, 41.5, 42.0]
        lons = [-100.0, -99.5, -99.0, -98.5, -98.0]

        manager = SpatialCVManager(n_groups=2, method='kmeans')
        groups = manager.create_groups_from_coordinates(lats, lons, verbose=False)

        assert len(groups) == 5
