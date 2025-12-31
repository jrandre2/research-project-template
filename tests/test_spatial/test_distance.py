"""Tests for spatial distance calculations."""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spatial.core.distance import (
    haversine_distance,
    EARTH_RADIUS_M,
)

# Try to import geopandas-dependent functions
try:
    import geopandas as gpd
    from shapely.geometry import Point

    from spatial.core.distance import (
        haversine_matrix,
        nearest_neighbor,
        distance_to_nearest,
        distance_band_neighbors,
    )

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


class TestHaversineDistance:
    """Tests for haversine_distance function."""

    def test_nyc_to_la(self):
        """Test NYC to LA distance (well-known value)."""
        # NYC: 40.7128° N, 74.0060° W
        # LA: 34.0522° N, 118.2437° W
        dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)

        # Expected: ~3,944 km (varies slightly by Earth radius used)
        dist_km = dist / 1000
        assert 3900 < dist_km < 4000, f"Expected ~3944 km, got {dist_km:.0f} km"

    def test_same_point(self):
        """Distance from point to itself should be zero."""
        dist = haversine_distance(40.7128, -74.0060, 40.7128, -74.0060)
        assert dist == 0.0

    def test_equator_distance(self):
        """Test distance along equator (simple case)."""
        # 1 degree of longitude at equator ≈ 111.32 km
        dist = haversine_distance(0, 0, 0, 1)
        dist_km = dist / 1000
        assert 110 < dist_km < 112, f"Expected ~111 km, got {dist_km:.1f} km"

    def test_poles_to_equator(self):
        """Test distance from pole to equator."""
        # North pole to equator = 90 degrees = ~10,000 km
        dist = haversine_distance(90, 0, 0, 0)
        dist_km = dist / 1000
        assert 9900 < dist_km < 10100, f"Expected ~10,000 km, got {dist_km:.0f} km"

    def test_symmetry(self):
        """Distance should be symmetric."""
        dist1 = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        dist2 = haversine_distance(34.0522, -118.2437, 40.7128, -74.0060)
        assert abs(dist1 - dist2) < 1e-10

    def test_custom_radius(self):
        """Test with custom radius (e.g., Moon)."""
        moon_radius = 1_737_400  # meters
        dist = haversine_distance(0, 0, 0, 90, radius=moon_radius)
        # Quarter circumference of Moon
        expected = np.pi * moon_radius / 2
        assert abs(dist - expected) < 1, f"Expected {expected:.0f} m, got {dist:.0f} m"


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")
class TestHaversineMatrix:
    """Tests for haversine_matrix function."""

    @pytest.fixture
    def sample_points(self):
        """Create sample point GeoDataFrame."""
        points = [
            Point(-74.0060, 40.7128),  # NYC
            Point(-118.2437, 34.0522),  # LA
            Point(-87.6298, 41.8781),  # Chicago
        ]
        return gpd.GeoDataFrame(
            {"name": ["NYC", "LA", "Chicago"]},
            geometry=points,
            crs="EPSG:4326",
        )

    def test_matrix_shape(self, sample_points):
        """Test output matrix shape."""
        dist_matrix = haversine_matrix(sample_points)
        assert dist_matrix.shape == (3, 3)

    def test_diagonal_zeros(self, sample_points):
        """Diagonal should be zeros (self-distance)."""
        dist_matrix = haversine_matrix(sample_points)
        np.testing.assert_array_almost_equal(
            np.diag(dist_matrix), [0, 0, 0], decimal=10
        )

    def test_symmetry(self, sample_points):
        """Matrix should be symmetric."""
        dist_matrix = haversine_matrix(sample_points)
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)

    def test_known_distance(self, sample_points):
        """Check known NYC-LA distance."""
        dist_matrix = haversine_matrix(sample_points)
        nyc_la_dist = dist_matrix[0, 1] / 1000  # km
        assert 3900 < nyc_la_dist < 4000

    def test_two_gdfs(self, sample_points):
        """Test with two different GeoDataFrames."""
        gdf1 = sample_points.iloc[:1]  # Just NYC
        gdf2 = sample_points.iloc[1:]  # LA and Chicago

        dist_matrix = haversine_matrix(gdf1, gdf2)
        assert dist_matrix.shape == (1, 2)


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")
class TestNearestNeighbor:
    """Tests for nearest_neighbor function."""

    @pytest.fixture
    def cities(self):
        """Create sample cities GeoDataFrame."""
        points = [
            Point(-74.0060, 40.7128),  # NYC
            Point(-118.2437, 34.0522),  # LA
            Point(-87.6298, 41.8781),  # Chicago
            Point(-122.4194, 37.7749),  # SF
        ]
        return gpd.GeoDataFrame(
            {"name": ["NYC", "LA", "Chicago", "SF"]},
            geometry=points,
            crs="EPSG:4326",
        )

    def test_k1(self, cities):
        """Test finding 1 nearest neighbor."""
        result = nearest_neighbor(cities.iloc[:1], cities.iloc[1:], k=1)

        assert len(result) == 1
        assert "source_idx" in result.columns
        assert "target_idx" in result.columns
        assert "distance" in result.columns
        assert "rank" in result.columns

    def test_k2(self, cities):
        """Test finding 2 nearest neighbors."""
        result = nearest_neighbor(cities.iloc[:1], cities.iloc[1:], k=2)
        assert len(result) == 2
        assert list(result["rank"]) == [1, 2]

    def test_max_distance(self, cities):
        """Test max_distance filter."""
        # Very small distance should return no results
        result = nearest_neighbor(cities.iloc[:1], cities.iloc[1:], max_distance=100)
        assert len(result) == 0

        # Very large distance should return all (need k=3 to get all 3 neighbors)
        result = nearest_neighbor(
            cities.iloc[:1], cities.iloc[1:], k=3, max_distance=10_000_000
        )
        assert len(result) == 3


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")
class TestDistanceToNearest:
    """Tests for distance_to_nearest function."""

    @pytest.fixture
    def cities(self):
        """Create sample cities GeoDataFrame."""
        points = [
            Point(-74.0060, 40.7128),  # NYC
            Point(-118.2437, 34.0522),  # LA
            Point(-87.6298, 41.8781),  # Chicago
        ]
        return gpd.GeoDataFrame(
            {"name": ["NYC", "LA", "Chicago"]},
            geometry=points,
            crs="EPSG:4326",
        )

    def test_returns_series(self, cities):
        """Should return a pandas Series."""
        result = distance_to_nearest(cities.iloc[:2], cities.iloc[2:])
        assert isinstance(result, pd.Series)
        assert len(result) == 2

    def test_index_preserved(self, cities):
        """Index should match source GeoDataFrame."""
        gdf_from = cities.iloc[:2]
        result = distance_to_nearest(gdf_from, cities.iloc[2:])
        assert list(result.index) == list(gdf_from.index)
