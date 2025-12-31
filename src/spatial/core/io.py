"""
Spatial data I/O utilities.

Provides functions for loading and saving spatial data in various formats
including GeoPackage, Shapefile, and GeoJSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None


# Supported file extensions and their drivers
SPATIAL_FORMATS = {
    ".gpkg": "GPKG",
    ".shp": "ESRI Shapefile",
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
}


def _check_geopandas() -> None:
    """Raise ImportError if geopandas is not available."""
    if not HAS_GEOPANDAS:
        raise ImportError(
            "geopandas is required for spatial operations. "
            "Install with: pip install geopandas"
        )


def load_spatial(
    path: Union[str, Path],
    layer: Optional[str] = None,
    **kwargs,
) -> "gpd.GeoDataFrame":
    """
    Load spatial data from file.

    Automatically detects format based on file extension. Supports GeoPackage,
    Shapefile, and GeoJSON formats.

    Parameters
    ----------
    path : str or Path
        Path to the spatial data file.
    layer : str, optional
        Layer name for multi-layer formats (e.g., GeoPackage).
        If not specified, reads the first layer.
    **kwargs
        Additional arguments passed to geopandas.read_file().

    Returns
    -------
    gpd.GeoDataFrame
        The loaded spatial data.

    Raises
    ------
    ImportError
        If geopandas is not installed.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is not supported.

    Examples
    --------
    >>> gdf = load_spatial('data/counties.gpkg')
    >>> gdf = load_spatial('data/points.shp')
    >>> gdf = load_spatial('data/boundaries.geojson')
    """
    _check_geopandas()

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Spatial file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SPATIAL_FORMATS:
        supported = ", ".join(SPATIAL_FORMATS.keys())
        raise ValueError(
            f"Unsupported spatial format: {ext}. "
            f"Supported formats: {supported}"
        )

    # Build read arguments
    read_kwargs = kwargs.copy()
    if layer is not None:
        read_kwargs["layer"] = layer

    gdf = gpd.read_file(path, **read_kwargs)

    return gdf


def save_spatial(
    gdf: "gpd.GeoDataFrame",
    path: Union[str, Path],
    layer: Optional[str] = None,
    driver: Optional[str] = None,
    **kwargs,
) -> Path:
    """
    Save spatial data to file.

    Automatically selects driver based on file extension unless explicitly
    specified.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The spatial data to save.
    path : str or Path
        Output file path.
    layer : str, optional
        Layer name for multi-layer formats (e.g., GeoPackage).
    driver : str, optional
        Output driver. Auto-detected from extension if not specified.
    **kwargs
        Additional arguments passed to geopandas.to_file().

    Returns
    -------
    Path
        The path to the saved file.

    Examples
    --------
    >>> save_spatial(gdf, 'output/results.gpkg')
    >>> save_spatial(gdf, 'output/results.shp')
    >>> save_spatial(gdf, 'output/results.geojson')
    """
    _check_geopandas()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()

    # Determine driver
    if driver is None:
        if ext not in SPATIAL_FORMATS:
            supported = ", ".join(SPATIAL_FORMATS.keys())
            raise ValueError(
                f"Cannot determine driver for extension: {ext}. "
                f"Supported formats: {supported}"
            )
        driver = SPATIAL_FORMATS[ext]

    # Build write arguments
    write_kwargs = kwargs.copy()
    write_kwargs["driver"] = driver
    if layer is not None:
        write_kwargs["layer"] = layer

    gdf.to_file(path, **write_kwargs)

    return path


def has_geometry(df: Union[pd.DataFrame, "gpd.GeoDataFrame"]) -> bool:
    """
    Check if a DataFrame has a geometry column.

    Parameters
    ----------
    df : pd.DataFrame or gpd.GeoDataFrame
        The DataFrame to check.

    Returns
    -------
    bool
        True if the DataFrame has a geometry column, False otherwise.

    Examples
    --------
    >>> df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    >>> has_geometry(df)
    False
    >>> gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    >>> has_geometry(gdf)
    True
    """
    if not HAS_GEOPANDAS:
        return False

    if isinstance(df, gpd.GeoDataFrame):
        return df.geometry is not None and not df.geometry.isna().all()

    # Check for geometry-like column in regular DataFrame
    if "geometry" in df.columns:
        return True

    return False


def list_layers(path: Union[str, Path]) -> list[str]:
    """
    List available layers in a spatial data file.

    Useful for multi-layer formats like GeoPackage.

    Parameters
    ----------
    path : str or Path
        Path to the spatial data file.

    Returns
    -------
    list[str]
        List of layer names.

    Examples
    --------
    >>> layers = list_layers('data/multi_layer.gpkg')
    >>> print(layers)
    ['counties', 'states', 'points']
    """
    _check_geopandas()

    import fiona

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Spatial file not found: {path}")

    return fiona.listlayers(path)
