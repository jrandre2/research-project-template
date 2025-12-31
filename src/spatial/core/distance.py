"""
Distance calculation utilities.

Provides functions for computing distances between geographic coordinates
including great-circle (haversine) distance and distance matrices.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None


# Earth's radius in meters (WGS84 mean radius)
EARTH_RADIUS_M = 6_371_008.8


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius: float = EARTH_RADIUS_M,
) -> float:
    """
    Calculate great-circle distance between two points using the Haversine formula.

    Parameters
    ----------
    lat1 : float
        Latitude of the first point in degrees.
    lon1 : float
        Longitude of the first point in degrees.
    lat2 : float
        Latitude of the second point in degrees.
    lon2 : float
        Longitude of the second point in degrees.
    radius : float, optional
        Radius of the sphere in meters. Default is Earth's mean radius.

    Returns
    -------
    float
        Distance between the two points in meters.

    Examples
    --------
    >>> # Distance from NYC to LA
    >>> dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
    >>> print(f"{dist / 1000:.0f} km")
    3936 km
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return radius * c


def haversine_matrix(
    gdf1: "gpd.GeoDataFrame",
    gdf2: Optional["gpd.GeoDataFrame"] = None,
    radius: float = EARTH_RADIUS_M,
) -> np.ndarray:
    """
    Compute pairwise distance matrix between points.

    Parameters
    ----------
    gdf1 : gpd.GeoDataFrame
        First set of points. Must have Point geometry in WGS84 (EPSG:4326).
    gdf2 : gpd.GeoDataFrame, optional
        Second set of points. If None, computes distances within gdf1.
    radius : float, optional
        Radius of the sphere in meters. Default is Earth's mean radius.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (len(gdf1), len(gdf2)) in meters.
        If gdf2 is None, returns (len(gdf1), len(gdf1)).

    Notes
    -----
    Memory complexity is O(n1 * n2). For large datasets, this can be
    prohibitive:

    - 10,000 × 10,000 points = ~800 MB
    - 50,000 × 50,000 points = ~20 GB

    For large datasets, consider using `nearest_neighbor()` with a
    `max_distance` parameter instead.

    Examples
    --------
    >>> # Pairwise distances within a GeoDataFrame
    >>> dist_matrix = haversine_matrix(gdf)
    >>> print(dist_matrix.shape)
    (100, 100)

    >>> # Distances from points to reference locations
    >>> dist_matrix = haversine_matrix(points, reference_points)
    """
    if not HAS_GEOPANDAS:
        raise ImportError(
            "geopandas is required for haversine_matrix. "
            "Install with: pip install geopandas"
        )

    # Extract coordinates
    coords1 = np.column_stack([gdf1.geometry.y, gdf1.geometry.x])  # lat, lon

    if gdf2 is None:
        coords2 = coords1
    else:
        coords2 = np.column_stack([gdf2.geometry.y, gdf2.geometry.x])

    n1 = len(coords1)
    n2 = len(coords2)

    # Vectorized haversine calculation
    lat1 = np.radians(coords1[:, 0])[:, np.newaxis]  # (n1, 1)
    lon1 = np.radians(coords1[:, 1])[:, np.newaxis]  # (n1, 1)
    lat2 = np.radians(coords2[:, 0])[np.newaxis, :]  # (1, n2)
    lon2 = np.radians(coords2[:, 1])[np.newaxis, :]  # (1, n2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return radius * c


def nearest_neighbor(
    gdf_from: "gpd.GeoDataFrame",
    gdf_to: "gpd.GeoDataFrame",
    k: int = 1,
    max_distance: Optional[float] = None,
    return_distance: bool = True,
) -> pd.DataFrame:
    """
    Find k nearest neighbors for each point.

    Parameters
    ----------
    gdf_from : gpd.GeoDataFrame
        Source points to find neighbors for.
    gdf_to : gpd.GeoDataFrame
        Target points to search among.
    k : int, optional
        Number of nearest neighbors to find. Default is 1.
    max_distance : float, optional
        Maximum distance in meters. Points beyond this are excluded.
    return_distance : bool, optional
        Whether to include distance in output. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'source_idx': Index from gdf_from
        - 'target_idx': Index from gdf_to (for each of k neighbors)
        - 'distance': Distance in meters (if return_distance=True)
        - 'rank': Neighbor rank (1 = nearest, 2 = second nearest, etc.)

    Examples
    --------
    >>> # Find nearest hospital for each census tract
    >>> result = nearest_neighbor(tracts, hospitals, k=3)
    >>> result.head()
       source_idx  target_idx     distance  rank
    0           0          42      1523.45     1
    1           0          17      2891.23     2
    2           0          33      3456.78     3
    """
    if not HAS_GEOPANDAS:
        raise ImportError(
            "geopandas is required for nearest_neighbor. "
            "Install with: pip install geopandas"
        )

    # Compute full distance matrix
    dist_matrix = haversine_matrix(gdf_from, gdf_to)

    n_from = len(gdf_from)
    k = min(k, len(gdf_to))

    results = []

    for i in range(n_from):
        distances = dist_matrix[i]

        # Get indices of k smallest distances
        if max_distance is not None:
            valid_mask = distances <= max_distance
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                continue
            valid_distances = distances[valid_indices]
            sorted_order = np.argsort(valid_distances)[:k]
            nearest_indices = valid_indices[sorted_order]
            nearest_distances = valid_distances[sorted_order]
        else:
            sorted_order = np.argsort(distances)[:k]
            nearest_indices = sorted_order
            nearest_distances = distances[sorted_order]

        for rank, (idx, dist) in enumerate(zip(nearest_indices, nearest_distances), 1):
            row = {
                "source_idx": gdf_from.index[i],
                "target_idx": gdf_to.index[idx],
                "rank": rank,
            }
            if return_distance:
                row["distance"] = dist
            results.append(row)

    df = pd.DataFrame(results)

    # Ensure proper column order
    cols = ["source_idx", "target_idx"]
    if return_distance:
        cols.append("distance")
    cols.append("rank")

    return df[cols] if len(df) > 0 else pd.DataFrame(columns=cols)


def distance_to_nearest(
    gdf_from: "gpd.GeoDataFrame",
    gdf_to: "gpd.GeoDataFrame",
) -> pd.Series:
    """
    Compute distance to nearest point for each source point.

    Convenience function equivalent to nearest_neighbor(gdf_from, gdf_to, k=1).

    Parameters
    ----------
    gdf_from : gpd.GeoDataFrame
        Source points.
    gdf_to : gpd.GeoDataFrame
        Target points.

    Returns
    -------
    pd.Series
        Series with distance to nearest target for each source point.
        Index matches gdf_from.index.

    Examples
    --------
    >>> gdf['distance_to_hospital'] = distance_to_nearest(gdf, hospitals)
    """
    dist_matrix = haversine_matrix(gdf_from, gdf_to)
    min_distances = dist_matrix.min(axis=1)
    return pd.Series(min_distances, index=gdf_from.index, name="distance_to_nearest")


def distance_band_neighbors(
    gdf: "gpd.GeoDataFrame",
    threshold: float,
    include_self: bool = False,
) -> dict[int, list[int]]:
    """
    Find all neighbors within a distance threshold.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Points to compute neighborhood for.
    threshold : float
        Distance threshold in meters.
    include_self : bool, optional
        Whether to include self as a neighbor (distance=0). Default is False.

    Returns
    -------
    dict[int, list[int]]
        Dictionary mapping each index to list of neighbor indices.

    Examples
    --------
    >>> neighbors = distance_band_neighbors(gdf, threshold=5000)  # 5km
    >>> print(f"Point 0 has neighbors: {neighbors[0]}")
    """
    dist_matrix = haversine_matrix(gdf)

    neighbors = {}
    for i, idx in enumerate(gdf.index):
        within_threshold = dist_matrix[i] <= threshold
        if not include_self:
            within_threshold[i] = False
        neighbor_positions = np.where(within_threshold)[0]
        neighbors[idx] = [gdf.index[j] for j in neighbor_positions]

    return neighbors
