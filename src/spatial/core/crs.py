"""
Coordinate Reference System (CRS) utilities.

Provides functions for handling and transforming coordinate reference systems.
"""

from __future__ import annotations

from typing import Optional, Union
import warnings

try:
    import geopandas as gpd
    from pyproj import CRS
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None
    CRS = None


# Common CRS codes
WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"


def _check_geopandas() -> None:
    """Raise ImportError if geopandas is not available."""
    if not HAS_GEOPANDAS:
        raise ImportError(
            "geopandas and pyproj are required for CRS operations. "
            "Install with: pip install geopandas pyproj"
        )


def ensure_crs(
    gdf: "gpd.GeoDataFrame",
    target_crs: str = WGS84,
    allow_override: bool = False,
) -> "gpd.GeoDataFrame":
    """
    Ensure a GeoDataFrame has the specified CRS.

    If the GeoDataFrame has a different CRS, it will be reprojected.
    If the GeoDataFrame has no CRS, either the target CRS will be assigned
    (if allow_override=True) or an error will be raised.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to check/transform.
    target_crs : str, optional
        Target CRS as EPSG code or proj4 string. Default is WGS84 (EPSG:4326).
    allow_override : bool, optional
        If True and the GeoDataFrame has no CRS, assign the target CRS
        without reprojection. Default is False.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with the target CRS.

    Raises
    ------
    ValueError
        If the GeoDataFrame has no CRS and allow_override is False.

    Warnings
    --------
    Using ``allow_override=True`` assumes the coordinates are already in
    the target CRS. This can produce incorrect results if the coordinates
    are actually in a different CRS. Only use this when you are certain
    of the original coordinate system.

    Examples
    --------
    >>> gdf = ensure_crs(gdf, target_crs="EPSG:4326")
    >>> gdf = ensure_crs(gdf, target_crs="EPSG:32618")  # UTM Zone 18N
    """
    _check_geopandas()

    # Make a copy to avoid modifying the original
    gdf = gdf.copy()

    if gdf.crs is None:
        if allow_override:
            gdf = gdf.set_crs(target_crs)
            warnings.warn(
                f"GeoDataFrame had no CRS. Assigned {target_crs} without reprojection.",
                UserWarning,
            )
        else:
            raise ValueError(
                "GeoDataFrame has no CRS. Set allow_override=True to assign "
                f"{target_crs} without reprojection, or set CRS explicitly."
            )
    elif not gdf.crs.equals(CRS.from_user_input(target_crs)):
        gdf = gdf.to_crs(target_crs)

    return gdf


def estimate_utm_zone(lon: float) -> int:
    """
    Estimate the appropriate UTM zone for a given longitude.

    Parameters
    ----------
    lon : float
        Longitude in degrees (WGS84).

    Returns
    -------
    int
        UTM zone number (1-60).

    Examples
    --------
    >>> estimate_utm_zone(-74.0)  # New York
    18
    >>> estimate_utm_zone(-122.4)  # San Francisco
    10
    """
    # UTM zones are 6 degrees wide, starting at -180
    zone = int((lon + 180) / 6) + 1
    return min(max(zone, 1), 60)


def get_utm_crs(lon: float, lat: float) -> str:
    """
    Get the appropriate UTM CRS for a given location.

    Parameters
    ----------
    lon : float
        Longitude in degrees (WGS84).
    lat : float
        Latitude in degrees (WGS84).

    Returns
    -------
    str
        EPSG code for the appropriate UTM zone.

    Examples
    --------
    >>> get_utm_crs(-74.0, 40.7)  # NYC
    'EPSG:32618'
    >>> get_utm_crs(-122.4, 37.8)  # SF
    'EPSG:32610'
    >>> get_utm_crs(0.0, -34.0)  # Southern hemisphere
    'EPSG:32731'
    """
    zone = estimate_utm_zone(lon)

    if lat >= 0:
        # Northern hemisphere: EPSG 326XX
        return f"EPSG:326{zone:02d}"
    else:
        # Southern hemisphere: EPSG 327XX
        return f"EPSG:327{zone:02d}"


def to_projected(
    gdf: "gpd.GeoDataFrame",
    utm_zone: Optional[int] = None,
) -> "gpd.GeoDataFrame":
    """
    Convert a GeoDataFrame to an appropriate projected CRS.

    For accurate distance and area calculations, geographic coordinates
    (lat/lon) should be projected. This function automatically selects
    an appropriate UTM zone based on the data's centroid.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with geographic coordinates (typically WGS84).
    utm_zone : int, optional
        Specific UTM zone to use. If None, automatically determined
        from the data's centroid.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame in projected coordinates (UTM).

    Examples
    --------
    >>> # Automatically select UTM zone
    >>> gdf_utm = to_projected(gdf)

    >>> # Force specific UTM zone
    >>> gdf_utm = to_projected(gdf, utm_zone=18)
    """
    _check_geopandas()

    # Ensure we have WGS84 for UTM zone calculation
    if gdf.crs is None or not gdf.crs.is_geographic:
        gdf_wgs84 = ensure_crs(gdf, WGS84, allow_override=True)
    else:
        gdf_wgs84 = gdf

    # Calculate centroid for UTM zone estimation
    centroid = gdf_wgs84.geometry.unary_union.centroid
    lon, lat = centroid.x, centroid.y

    if utm_zone is not None:
        # Use specified zone
        if lat >= 0:
            target_crs = f"EPSG:326{utm_zone:02d}"
        else:
            target_crs = f"EPSG:327{utm_zone:02d}"
    else:
        target_crs = get_utm_crs(lon, lat)

    return gdf.to_crs(target_crs)


def get_crs_info(gdf: "gpd.GeoDataFrame") -> dict:
    """
    Get information about a GeoDataFrame's CRS.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to inspect.

    Returns
    -------
    dict
        Dictionary with CRS information:
        - 'crs': CRS object or None
        - 'epsg': EPSG code if available, else None
        - 'is_geographic': True if lat/lon coordinates
        - 'is_projected': True if projected coordinates
        - 'units': Coordinate units (e.g., 'degree', 'metre')

    Examples
    --------
    >>> info = get_crs_info(gdf)
    >>> print(info)
    {'crs': <CRS: EPSG:4326>, 'epsg': 4326, 'is_geographic': True,
     'is_projected': False, 'units': 'degree'}
    """
    _check_geopandas()

    if gdf.crs is None:
        return {
            "crs": None,
            "epsg": None,
            "is_geographic": None,
            "is_projected": None,
            "units": None,
        }

    crs = gdf.crs

    # Get units
    try:
        units = crs.axis_info[0].unit_name
    except (AttributeError, IndexError):
        units = None

    return {
        "crs": crs,
        "epsg": crs.to_epsg(),
        "is_geographic": crs.is_geographic,
        "is_projected": crs.is_projected,
        "units": units,
    }


def crs_matches(
    gdf1: "gpd.GeoDataFrame",
    gdf2: "gpd.GeoDataFrame",
) -> bool:
    """
    Check if two GeoDataFrames have the same CRS.

    Parameters
    ----------
    gdf1 : gpd.GeoDataFrame
        First GeoDataFrame.
    gdf2 : gpd.GeoDataFrame
        Second GeoDataFrame.

    Returns
    -------
    bool
        True if both have the same CRS, False otherwise.
        Also returns False if either has no CRS.

    Examples
    --------
    >>> if not crs_matches(gdf1, gdf2):
    ...     gdf2 = ensure_crs(gdf2, gdf1.crs)
    """
    _check_geopandas()

    if gdf1.crs is None or gdf2.crs is None:
        return False

    return gdf1.crs.equals(gdf2.crs)
