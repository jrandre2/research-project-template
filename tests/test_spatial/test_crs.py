"""Tests for CRS (Coordinate Reference System) utilities."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spatial.core.crs import estimate_utm_zone, get_utm_crs

# Try to import geopandas-dependent functions
try:
    import geopandas as gpd
    from shapely.geometry import Point

    from spatial.core.crs import (
        ensure_crs,
        to_projected,
        get_crs_info,
        crs_matches,
        WGS84,
    )

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


class TestEstimateUtmZone:
    """Tests for estimate_utm_zone function."""

    def test_nyc(self):
        """NYC should be in UTM zone 18."""
        zone = estimate_utm_zone(-74.0)
        assert zone == 18

    def test_la(self):
        """LA should be in UTM zone 11."""
        zone = estimate_utm_zone(-118.2)
        assert zone == 11

    def test_london(self):
        """London should be in UTM zone 30."""
        zone = estimate_utm_zone(-0.1)
        assert zone == 30

    def test_tokyo(self):
        """Tokyo should be in UTM zone 54."""
        zone = estimate_utm_zone(139.7)
        assert zone == 54

    def test_boundary_low(self):
        """Test at -180 longitude."""
        zone = estimate_utm_zone(-180)
        assert zone == 1

    def test_boundary_high(self):
        """Test at 180 longitude."""
        zone = estimate_utm_zone(180)
        assert zone == 60


class TestGetUtmCrs:
    """Tests for get_utm_crs function."""

    def test_nyc_northern(self):
        """NYC (northern hemisphere) should use 326XX."""
        crs = get_utm_crs(-74.0, 40.7)
        assert crs == "EPSG:32618"

    def test_sydney_southern(self):
        """Sydney (southern hemisphere) should use 327XX."""
        crs = get_utm_crs(151.2, -33.9)
        assert crs == "EPSG:32756"

    def test_equator_positive_lat(self):
        """Just north of equator should use 326XX."""
        crs = get_utm_crs(0, 0.1)
        assert crs.startswith("EPSG:326")

    def test_equator_negative_lat(self):
        """Just south of equator should use 327XX."""
        crs = get_utm_crs(0, -0.1)
        assert crs.startswith("EPSG:327")


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")
class TestEnsureCrs:
    """Tests for ensure_crs function."""

    @pytest.fixture
    def sample_gdf(self):
        """Create sample GeoDataFrame with WGS84 CRS."""
        points = [Point(-74.0, 40.7), Point(-118.2, 34.1)]
        return gpd.GeoDataFrame(
            {"name": ["NYC", "LA"]}, geometry=points, crs="EPSG:4326"
        )

    def test_same_crs(self, sample_gdf):
        """No reprojection needed when CRS matches."""
        result = ensure_crs(sample_gdf, "EPSG:4326")
        assert result.crs.to_epsg() == 4326

    def test_reproject(self, sample_gdf):
        """Should reproject when CRS differs."""
        result = ensure_crs(sample_gdf, "EPSG:3857")
        assert result.crs.to_epsg() == 3857

    def test_no_crs_error(self):
        """Should raise error when no CRS and allow_override=False."""
        points = [Point(-74.0, 40.7)]
        gdf = gpd.GeoDataFrame({"name": ["NYC"]}, geometry=points)
        gdf = gdf.set_crs(None)  # Explicitly remove CRS

        with pytest.raises(ValueError, match="has no CRS"):
            ensure_crs(gdf, "EPSG:4326", allow_override=False)

    def test_no_crs_override(self):
        """Should assign CRS when allow_override=True."""
        points = [Point(-74.0, 40.7)]
        gdf = gpd.GeoDataFrame({"name": ["NYC"]}, geometry=points)

        with pytest.warns(UserWarning):
            result = ensure_crs(gdf, "EPSG:4326", allow_override=True)

        assert result.crs.to_epsg() == 4326

    def test_returns_copy(self, sample_gdf):
        """Should return a copy, not modify original."""
        result = ensure_crs(sample_gdf, "EPSG:3857")
        assert sample_gdf.crs.to_epsg() == 4326
        assert result.crs.to_epsg() == 3857


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")
class TestToProjected:
    """Tests for to_projected function."""

    @pytest.fixture
    def nyc_gdf(self):
        """Create sample GeoDataFrame near NYC."""
        points = [Point(-74.0, 40.7), Point(-74.1, 40.8)]
        return gpd.GeoDataFrame(
            {"name": ["Point1", "Point2"]}, geometry=points, crs="EPSG:4326"
        )

    def test_auto_utm(self, nyc_gdf):
        """Should automatically select appropriate UTM zone."""
        result = to_projected(nyc_gdf)
        # NYC should be UTM zone 18N (EPSG:32618)
        assert result.crs.to_epsg() == 32618

    def test_specify_zone(self, nyc_gdf):
        """Should use specified UTM zone."""
        result = to_projected(nyc_gdf, utm_zone=17)
        assert result.crs.to_epsg() == 32617

    def test_is_projected(self, nyc_gdf):
        """Result should be in projected CRS."""
        result = to_projected(nyc_gdf)
        assert result.crs.is_projected


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")
class TestGetCrsInfo:
    """Tests for get_crs_info function."""

    def test_wgs84(self):
        """Test info for WGS84 CRS."""
        points = [Point(-74.0, 40.7)]
        gdf = gpd.GeoDataFrame({"name": ["NYC"]}, geometry=points, crs="EPSG:4326")

        info = get_crs_info(gdf)

        assert info["epsg"] == 4326
        assert info["is_geographic"] is True
        assert info["is_projected"] is False
        assert info["units"] == "degree"

    def test_utm(self):
        """Test info for UTM CRS."""
        points = [Point(-74.0, 40.7)]
        gdf = gpd.GeoDataFrame({"name": ["NYC"]}, geometry=points, crs="EPSG:32618")

        info = get_crs_info(gdf)

        assert info["epsg"] == 32618
        assert info["is_geographic"] is False
        assert info["is_projected"] is True
        assert info["units"] == "metre"

    def test_no_crs(self):
        """Test info when no CRS is set."""
        points = [Point(-74.0, 40.7)]
        gdf = gpd.GeoDataFrame({"name": ["NYC"]}, geometry=points)

        info = get_crs_info(gdf)

        assert info["crs"] is None
        assert info["epsg"] is None


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")
class TestCrsMatches:
    """Tests for crs_matches function."""

    def test_same_crs(self):
        """Should return True for matching CRS."""
        points = [Point(-74.0, 40.7)]
        gdf1 = gpd.GeoDataFrame({"name": ["A"]}, geometry=points, crs="EPSG:4326")
        gdf2 = gpd.GeoDataFrame({"name": ["B"]}, geometry=points, crs="EPSG:4326")

        assert crs_matches(gdf1, gdf2) is True

    def test_different_crs(self):
        """Should return False for different CRS."""
        points = [Point(-74.0, 40.7)]
        gdf1 = gpd.GeoDataFrame({"name": ["A"]}, geometry=points, crs="EPSG:4326")
        gdf2 = gpd.GeoDataFrame({"name": ["B"]}, geometry=points, crs="EPSG:3857")

        assert crs_matches(gdf1, gdf2) is False

    def test_no_crs(self):
        """Should return False if either has no CRS."""
        points = [Point(-74.0, 40.7)]
        gdf1 = gpd.GeoDataFrame({"name": ["A"]}, geometry=points, crs="EPSG:4326")
        gdf2 = gpd.GeoDataFrame({"name": ["B"]}, geometry=points)

        assert crs_matches(gdf1, gdf2) is False
        assert crs_matches(gdf2, gdf1) is False
