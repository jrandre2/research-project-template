"""
Geospatial analysis utilities for CENTAUR.

This module provides spatial data I/O, distance calculations, CRS handling,
and spatial analysis tools for research workflows.

Example usage:
    from spatial import load_spatial, haversine_distance, ensure_crs

    # Load spatial data
    gdf = load_spatial('data/counties.gpkg')

    # Calculate distance between two points
    dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)

    # Ensure consistent CRS
    gdf = ensure_crs(gdf, target_crs="EPSG:4326")
"""

from spatial.core.io import load_spatial, save_spatial, has_geometry, list_layers
from spatial.core.distance import (
    haversine_distance,
    haversine_matrix,
    nearest_neighbor,
    distance_to_nearest,
    distance_band_neighbors,
)
from spatial.core.crs import (
    ensure_crs,
    to_projected,
    estimate_utm_zone,
    get_utm_crs,
    get_crs_info,
    crs_matches,
)

__all__ = [
    # I/O
    "load_spatial",
    "save_spatial",
    "has_geometry",
    "list_layers",
    # Distance
    "haversine_distance",
    "haversine_matrix",
    "nearest_neighbor",
    "distance_to_nearest",
    "distance_band_neighbors",
    # CRS
    "ensure_crs",
    "to_projected",
    "estimate_utm_zone",
    "get_utm_crs",
    "get_crs_info",
    "crs_matches",
]
