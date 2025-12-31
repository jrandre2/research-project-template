"""Tests for spatial data I/O utilities."""

import pytest
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spatial.core.io import SPATIAL_FORMATS

# Try to import geopandas-dependent functions
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    from spatial.core.io import (
        load_spatial,
        save_spatial,
        has_geometry,
        list_layers,
    )

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


class TestSpatialFormats:
    """Tests for supported spatial formats."""

    def test_gpkg_supported(self):
        """GeoPackage should be supported."""
        assert ".gpkg" in SPATIAL_FORMATS
        assert SPATIAL_FORMATS[".gpkg"] == "GPKG"

    def test_shp_supported(self):
        """Shapefile should be supported."""
        assert ".shp" in SPATIAL_FORMATS
        assert SPATIAL_FORMATS[".shp"] == "ESRI Shapefile"

    def test_geojson_supported(self):
        """GeoJSON should be supported."""
        assert ".geojson" in SPATIAL_FORMATS
        assert SPATIAL_FORMATS[".geojson"] == "GeoJSON"

    def test_json_supported(self):
        """JSON (GeoJSON) should be supported."""
        assert ".json" in SPATIAL_FORMATS
        assert SPATIAL_FORMATS[".json"] == "GeoJSON"


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")
class TestLoadSpatial:
    """Tests for load_spatial function."""

    @pytest.fixture
    def sample_gdf(self):
        """Create sample GeoDataFrame."""
        points = [Point(-74.0, 40.7), Point(-118.2, 34.1)]
        return gpd.GeoDataFrame(
            {"name": ["NYC", "LA"], "value": [100, 200]},
            geometry=points,
            crs="EPSG:4326",
        )

    def test_load_geojson(self, sample_gdf, tmp_path):
        """Should load GeoJSON files."""
        filepath = tmp_path / "test.geojson"
        sample_gdf.to_file(filepath, driver="GeoJSON")

        result = load_spatial(filepath)

        assert len(result) == 2
        assert "name" in result.columns
        assert result.crs.to_epsg() == 4326

    def test_load_gpkg(self, sample_gdf, tmp_path):
        """Should load GeoPackage files."""
        filepath = tmp_path / "test.gpkg"
        sample_gdf.to_file(filepath, driver="GPKG")

        result = load_spatial(filepath)

        assert len(result) == 2
        assert "name" in result.columns

    def test_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_spatial(tmp_path / "nonexistent.gpkg")

    def test_unsupported_format(self, tmp_path):
        """Should raise ValueError for unsupported formats."""
        filepath = tmp_path / "test.xyz"
        filepath.touch()

        with pytest.raises(ValueError, match="Unsupported spatial format"):
            load_spatial(filepath)


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")
class TestSaveSpatial:
    """Tests for save_spatial function."""

    @pytest.fixture
    def sample_gdf(self):
        """Create sample GeoDataFrame."""
        points = [Point(-74.0, 40.7), Point(-118.2, 34.1)]
        return gpd.GeoDataFrame(
            {"name": ["NYC", "LA"], "value": [100, 200]},
            geometry=points,
            crs="EPSG:4326",
        )

    def test_save_geojson(self, sample_gdf, tmp_path):
        """Should save to GeoJSON format."""
        filepath = tmp_path / "output.geojson"
        result_path = save_spatial(sample_gdf, filepath)

        assert result_path.exists()
        assert result_path == filepath

        # Verify can be loaded back
        loaded = gpd.read_file(filepath)
        assert len(loaded) == 2

    def test_save_gpkg(self, sample_gdf, tmp_path):
        """Should save to GeoPackage format."""
        filepath = tmp_path / "output.gpkg"
        result_path = save_spatial(sample_gdf, filepath)

        assert result_path.exists()

        loaded = gpd.read_file(filepath)
        assert len(loaded) == 2

    def test_save_creates_parent_dirs(self, sample_gdf, tmp_path):
        """Should create parent directories if they don't exist."""
        filepath = tmp_path / "subdir" / "nested" / "output.geojson"
        result_path = save_spatial(sample_gdf, filepath)

        assert result_path.exists()
        assert result_path.parent.exists()

    def test_save_with_layer(self, sample_gdf, tmp_path):
        """Should save with specified layer name (GeoPackage)."""
        filepath = tmp_path / "output.gpkg"
        save_spatial(sample_gdf, filepath, layer="my_layer")

        # Verify layer was created
        import fiona

        layers = fiona.listlayers(filepath)
        assert "my_layer" in layers

    def test_unsupported_format(self, sample_gdf, tmp_path):
        """Should raise ValueError for unsupported formats."""
        filepath = tmp_path / "output.xyz"

        with pytest.raises(ValueError, match="Cannot determine driver"):
            save_spatial(sample_gdf, filepath)


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")
class TestHasGeometry:
    """Tests for has_geometry function."""

    def test_geodataframe_with_geometry(self):
        """Should return True for GeoDataFrame with geometry."""
        points = [Point(-74.0, 40.7)]
        gdf = gpd.GeoDataFrame({"name": ["NYC"]}, geometry=points, crs="EPSG:4326")

        assert has_geometry(gdf) is True

    def test_geodataframe_without_geometry(self):
        """Should return False for GeoDataFrame with all null geometry."""
        import pandas as pd

        gdf = gpd.GeoDataFrame({"name": ["NYC"]})
        gdf["geometry"] = None

        assert has_geometry(gdf) is False

    def test_regular_dataframe(self):
        """Should return False for regular DataFrame without geometry column."""
        import pandas as pd

        df = pd.DataFrame({"name": ["NYC"], "value": [100]})

        assert has_geometry(df) is False

    def test_dataframe_with_geometry_column(self):
        """Should return True for DataFrame with 'geometry' column."""
        import pandas as pd

        df = pd.DataFrame({"name": ["NYC"], "geometry": ["POINT(-74.0 40.7)"]})

        assert has_geometry(df) is True


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")
class TestListLayers:
    """Tests for list_layers function."""

    @pytest.fixture
    def multi_layer_gpkg(self, tmp_path):
        """Create GeoPackage with multiple layers."""
        filepath = tmp_path / "multi.gpkg"

        # Create two layers
        points = [Point(-74.0, 40.7)]
        gdf1 = gpd.GeoDataFrame({"name": ["NYC"]}, geometry=points, crs="EPSG:4326")
        gdf2 = gpd.GeoDataFrame({"name": ["LA"]}, geometry=points, crs="EPSG:4326")

        gdf1.to_file(filepath, layer="layer1", driver="GPKG")
        gdf2.to_file(filepath, layer="layer2", driver="GPKG")

        return filepath

    def test_list_layers(self, multi_layer_gpkg):
        """Should list all layers in a GeoPackage."""
        layers = list_layers(multi_layer_gpkg)

        assert "layer1" in layers
        assert "layer2" in layers
        assert len(layers) == 2

    def test_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="not found"):
            list_layers(tmp_path / "nonexistent.gpkg")


@pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")
class TestRoundTrip:
    """Integration tests for load/save round trips."""

    @pytest.fixture
    def complex_gdf(self):
        """Create a more complex GeoDataFrame with various data types."""
        geometries = [
            Point(-74.0, 40.7),
            Point(-118.2, 34.1),
            Point(-87.6, 41.9),
        ]
        return gpd.GeoDataFrame(
            {
                "name": ["NYC", "LA", "Chicago"],
                "population": [8336817, 3979576, 2693976],
                "area_km2": [783.8, 1213.9, 606.1],
            },
            geometry=geometries,
            crs="EPSG:4326",
        )

    @pytest.mark.parametrize("extension", [".geojson", ".gpkg"])
    def test_round_trip(self, complex_gdf, tmp_path, extension):
        """Data should survive a save/load round trip."""
        filepath = tmp_path / f"test{extension}"

        save_spatial(complex_gdf, filepath)
        loaded = load_spatial(filepath)

        assert len(loaded) == len(complex_gdf)
        assert set(loaded.columns) == set(complex_gdf.columns)
        assert loaded.crs.to_epsg() == 4326
