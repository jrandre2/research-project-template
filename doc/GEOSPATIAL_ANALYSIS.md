# Geospatial Analysis Module

**Related**: [PIPELINE.md](PIPELINE.md) | [ARCHITECTURE.md](ARCHITECTURE.md) | [design/GEOSPATIAL_MODULE.md](design/GEOSPATIAL_MODULE.md)
**Status**: Active
**Last Updated**: 2025-12-30

The `src/spatial/` module provides core utilities for working with geographic data in research projects. This document covers the available functionality and usage patterns.

## Quick Start

```python
# Run from src/ directory, or set PYTHONPATH=src
from spatial import load_spatial, save_spatial, haversine_distance, ensure_crs

# Load spatial data
gdf = load_spatial('data_raw/counties.gpkg')

# Ensure proper CRS
gdf = ensure_crs(gdf, 'EPSG:4326')

# Calculate distance between two points (in meters)
dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
print(f"NYC to LA: {dist/1000:.0f} km")  # ~3,936 km
```

## Installation

The spatial module requires optional dependencies. Install them with:

```bash
pip install geopandas shapely pyproj
```

Or install from requirements.txt (dependencies are already included).

### Troubleshooting Installation

**Windows: GDAL/GEOS errors**

Windows users often encounter errors with GDAL/GEOS dependencies. Use conda instead:

```bash
conda install -c conda-forge geopandas
```

**macOS: Missing libspatialindex**

```bash
brew install spatialindex
pip install geopandas
```

**Linux: Missing system libraries**

```bash
# Ubuntu/Debian
sudo apt-get install libgdal-dev libgeos-dev libproj-dev

# Then install Python packages
pip install geopandas shapely pyproj
```

**Import errors after installation**

- Ensure virtual environment is activated: `source .venv/bin/activate`
- Verify installation: `python -c "import geopandas; print(geopandas.__version__)"`
- Check for version conflicts: `pip check`

## Module Structure

```
src/spatial/
├── __init__.py           # Public API exports
└── core/
    ├── io.py             # Spatial data I/O
    ├── distance.py       # Distance calculations
    └── crs.py            # CRS handling
```

## Core Functions

### Data I/O (`spatial.core.io`)

#### `load_spatial(path, layer=None)`

Load spatial data from file. Auto-detects format based on extension.

```python
from spatial import load_spatial

# Load GeoPackage
gdf = load_spatial('data/counties.gpkg')

# Load specific layer from GeoPackage
gdf = load_spatial('data/multi_layer.gpkg', layer='boundaries')

# Load Shapefile
gdf = load_spatial('data/points.shp')

# Load GeoJSON
gdf = load_spatial('data/regions.geojson')
```

**Supported formats:**
- GeoPackage (`.gpkg`) - Recommended for most uses
- Shapefile (`.shp`) - Legacy format, common in GIS
- GeoJSON (`.geojson`, `.json`) - Web-friendly, human-readable

#### `save_spatial(gdf, path, layer=None)`

Save GeoDataFrame to file.

```python
from spatial import save_spatial

# Save to GeoPackage
save_spatial(gdf, 'output/results.gpkg')

# Save with layer name
save_spatial(gdf, 'output/results.gpkg', layer='analysis_results')

# Creates parent directories automatically
save_spatial(gdf, 'output/nested/deep/results.geojson')
```

#### `has_geometry(df)`

Check if a DataFrame has geometry.

```python
from spatial import has_geometry

if has_geometry(df):
    print("Has spatial data")
```

#### `list_layers(path)`

List available layers in a multi-layer spatial file (e.g., GeoPackage).

```python
from spatial import list_layers

layers = list_layers('data/multi_layer.gpkg')
print(layers)  # ['counties', 'states', 'cities']
```

### Distance Calculations (`spatial.core.distance`)

#### `haversine_distance(lat1, lon1, lat2, lon2)`

Calculate great-circle distance between two points using the Haversine formula.

```python
from spatial import haversine_distance

# NYC to LA
dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
print(f"{dist/1000:.0f} km")  # 3,936 km
```

**Parameters:**
- `lat1, lon1`: First point coordinates (degrees)
- `lat2, lon2`: Second point coordinates (degrees)

**Returns:** Distance in meters

#### `haversine_matrix(gdf1, gdf2=None)`

Compute pairwise distance matrix between points.

```python
from spatial.core.distance import haversine_matrix

# Pairwise distances within a single GeoDataFrame
dist_matrix = haversine_matrix(gdf)
print(dist_matrix.shape)  # (n, n)

# Distances from points to reference locations
dist_matrix = haversine_matrix(observations, reference_points)
print(dist_matrix.shape)  # (n_obs, n_ref)
```

#### `nearest_neighbor(gdf_from, gdf_to, k=1, max_distance=None)`

Find k nearest neighbors for each point.

```python
from spatial.core.distance import nearest_neighbor

# Find nearest hospital for each census tract
result = nearest_neighbor(tracts, hospitals, k=3)

# Result DataFrame:
#    source_idx  target_idx  distance  rank
# 0          0          42   1523.45     1
# 1          0          17   2891.23     2
# 2          0          33   3456.78     3
```

#### `distance_to_nearest(gdf_from, gdf_to)`

Get distance to nearest point for each source (convenience function).

```python
from spatial.core.distance import distance_to_nearest

# Add distance column to data
gdf['distance_to_hospital'] = distance_to_nearest(gdf, hospitals)
```

#### `distance_band_neighbors(gdf, threshold, include_self=False)`

Find all neighbors within a distance threshold.

```python
from spatial.core.distance import distance_band_neighbors

# All neighbors within 5km
neighbors = distance_band_neighbors(gdf, threshold=5000)
print(f"Point 0 has neighbors: {neighbors[0]}")
```

### CRS Handling (`spatial.core.crs`)

#### `ensure_crs(gdf, target_crs='EPSG:4326', allow_override=False)`

Ensure a GeoDataFrame has the specified CRS.

```python
from spatial import ensure_crs

# Reproject to WGS84
gdf = ensure_crs(gdf, 'EPSG:4326')

# Reproject to UTM Zone 18N
gdf = ensure_crs(gdf, 'EPSG:32618')

# Assign CRS to data without one (use with caution)
gdf = ensure_crs(gdf, 'EPSG:4326', allow_override=True)
```

#### `to_projected(gdf, utm_zone=None)`

Convert to appropriate projected CRS for accurate distance/area calculations.

```python
from spatial.core.crs import to_projected

# Auto-detect appropriate UTM zone
gdf_projected = to_projected(gdf)

# Force specific UTM zone
gdf_projected = to_projected(gdf, utm_zone=18)
```

#### `estimate_utm_zone(lon)`

Get the appropriate UTM zone for a longitude.

```python
from spatial import estimate_utm_zone

zone = estimate_utm_zone(-74.0)  # NYC -> 18
zone = estimate_utm_zone(-122.4)  # SF -> 10
```

#### `get_utm_crs(lon, lat)`

Get the EPSG code for the appropriate UTM zone for a location.

```python
from spatial import get_utm_crs

# Northern hemisphere
crs = get_utm_crs(-74.0, 40.7)  # NYC -> 'EPSG:32618'

# Southern hemisphere
crs = get_utm_crs(151.2, -33.9)  # Sydney -> 'EPSG:32756'
```

#### `get_crs_info(gdf)`

Get detailed CRS information.

```python
from spatial.core.crs import get_crs_info

info = get_crs_info(gdf)
# {'crs': <CRS>, 'epsg': 4326, 'is_geographic': True,
#  'is_projected': False, 'units': 'degree'}
```

#### `crs_matches(gdf1, gdf2)`

Check if two GeoDataFrames have the same CRS.

```python
from spatial.core.crs import crs_matches

if not crs_matches(gdf1, gdf2):
    gdf2 = ensure_crs(gdf2, gdf1.crs)
```

## Common Workflows

### Loading and Preparing Spatial Data

```python
from spatial import load_spatial, ensure_crs
from spatial.core.crs import to_projected

# Load data
counties = load_spatial('data_raw/counties.gpkg')

# Ensure WGS84 for storage/display
counties = ensure_crs(counties, 'EPSG:4326')

# Project for area calculations
counties_proj = to_projected(counties)
counties_proj['area_km2'] = counties_proj.area / 1e6
```

### Spatial Joins with Distance

```python
from spatial import load_spatial
from spatial.core.distance import nearest_neighbor, distance_to_nearest

# Load data
obs = load_spatial('data/observations.gpkg')
hospitals = load_spatial('data/hospitals.gpkg')

# Add distance to nearest hospital
obs['dist_to_hospital'] = distance_to_nearest(obs, hospitals)

# Get 3 nearest hospitals for each observation
neighbors = nearest_neighbor(obs, hospitals, k=3)
```

### Working with Multiple Layers

```python
from spatial import load_spatial, save_spatial
from spatial.core.io import list_layers

# List available layers
layers = list_layers('data/multi_layer.gpkg')
print(layers)  # ['counties', 'states', 'cities']

# Load specific layer
counties = load_spatial('data/multi_layer.gpkg', layer='counties')

# Save multiple layers to same file
save_spatial(counties, 'output/analysis.gpkg', layer='counties')
save_spatial(results, 'output/analysis.gpkg', layer='results')
```

## Configuration

Spatial settings in `src/config.py`:

```python
# Enable/disable spatial module
SPATIAL_ENABLED = True

# Default CRS for loaded data
SPATIAL_DEFAULT_CRS = "EPSG:4326"

# Spatial data directory
SPATIAL_DATA_DIR = DATA_WORK_DIR / 'spatial'

# Cache for geocoding results
SPATIAL_CACHE_DIR = CACHE_DIR / 'spatial'

# Geocoding settings
GEOCODING_PROVIDER = "census"
GEOCODING_CACHE_TTL_DAYS = 30
```

### Configuration Settings Explained

| Setting | Default | Purpose |
|---------|---------|---------|
| `SPATIAL_ENABLED` | `True` | Enable/disable spatial module features |
| `SPATIAL_DEFAULT_CRS` | `"EPSG:4326"` | Default CRS (WGS84) for loaded data |
| `SPATIAL_DATA_DIR` | `data_work/spatial/` | Directory for spatial analysis outputs |
| `SPATIAL_CACHE_DIR` | `.cache/spatial/` | Cache directory for geocoding results |
| `GEOCODING_PROVIDER` | `"census"` | Provider for geocoding (future feature) |
| `GEOCODING_CACHE_TTL_DAYS` | `30` | Days to cache geocoding results |

## Pipeline Integration

### Using Spatial Functions in Pipeline Stages

The spatial module integrates with CENTAUR pipeline stages. Here are common patterns:

**Example: Spatial join in s01_link.py**

```python
from spatial import load_spatial, ensure_crs
from spatial.core.distance import nearest_neighbor
from config import DATA_RAW_DIR

# Load observation points and reference locations
obs = load_spatial(DATA_RAW_DIR / 'observations.gpkg')
facilities = load_spatial(DATA_RAW_DIR / 'facilities.gpkg')

# Ensure matching CRS before spatial operations
obs = ensure_crs(obs, 'EPSG:4326')
facilities = ensure_crs(facilities, 'EPSG:4326')

# Find nearest facility for each observation
matches = nearest_neighbor(obs, facilities, k=1)
obs['facility_id'] = matches.set_index('source_idx')['target_idx']
```

**Example: Distance-based sample selection**

```python
from spatial.core.distance import distance_to_nearest

# Calculate distance to treatment sites
control['dist_to_treatment'] = distance_to_nearest(control, treatment)

# Select controls within 50km of treatment
nearby_controls = control[control['dist_to_treatment'] <= 50000]
```

**Example: Spatial panel construction in s02_panel.py**

```python
from spatial import load_spatial
from spatial.core.crs import to_projected

# Load administrative boundaries
counties = load_spatial(DATA_RAW_DIR / 'counties.gpkg')

# Project for accurate area calculations
counties_proj = to_projected(counties)
counties_proj['area_km2'] = counties_proj.area / 1e6

# Merge with panel data
panel = panel.merge(
    counties_proj[['fips', 'area_km2']],
    on='fips',
    how='left'
)
```

## Performance Guidelines

### Distance Matrix Memory

The `haversine_matrix()` function creates an O(n²) memory structure:

| Points | Matrix Size | Approximate Memory |
|--------|-------------|-------------------|
| 1,000 | 1M cells | ~8 MB |
| 10,000 | 100M cells | ~800 MB |
| 50,000 | 2.5B cells | ~20 GB |
| 100,000 | 10B cells | ~80 GB |

**Recommendations for large datasets:**

- Use `nearest_neighbor()` with `max_distance` parameter to limit comparisons
- Use `distance_band_neighbors()` for threshold-based neighbor finding
- Process in chunks if full matrix is needed

### Projection Performance

| Operation | Geographic CRS | Projected CRS |
|-----------|---------------|---------------|
| File I/O | Faster | Slower |
| Distance calc | Slower, less accurate | Faster, more accurate |
| Area calc | Inaccurate | Accurate |

**Recommendations:**

- Store data in WGS84 (EPSG:4326) for compatibility
- Project to UTM for distance/area calculations using `to_projected()`
- Cache projected versions if reused frequently

### Spatial Index Performance

GeoPandas automatically uses R-tree spatial indexes for:

- `sjoin()` - Spatial joins
- `intersects()`, `within()` - Spatial predicates

For custom operations, ensure geometries are valid:

```python
# Check for invalid geometries
invalid_count = (~gdf.geometry.is_valid).sum()
if invalid_count > 0:
    gdf['geometry'] = gdf.geometry.buffer(0)  # Fix common issues
```

## Best Practices

### CRS Management

1. **Always check CRS** when loading data: `print(gdf.crs)`
2. **Store in WGS84** (EPSG:4326) for compatibility
3. **Project for calculations** - geographic coordinates give inaccurate distances/areas
4. **Use `to_projected()`** for automatic UTM zone selection

### Performance

1. **Use GeoPackage** (.gpkg) for large datasets - faster than Shapefile
2. **Compute distance matrices once** and reuse
3. **Use spatial indexes** for large join operations (built into geopandas)

### Common Pitfalls

1. **Lat/Lon order**: Many functions expect `(lat, lon)`, but Shapely uses `(lon, lat)`. The `haversine_distance` function uses `(lat, lon)` order.

2. **Distance units**: All distances are in meters by default.

3. **CRS mismatch**: Always ensure CRS matches before spatial operations:
   ```python
   if not crs_matches(gdf1, gdf2):
       gdf2 = ensure_crs(gdf2, gdf1.crs)
   result = gdf1.sjoin(gdf2)
   ```

## Troubleshooting

### Common Errors and Solutions

**"GeoDataFrame has no CRS"**

```python
# Solution: Assign CRS (use with caution - only if you know the actual CRS)
gdf = ensure_crs(gdf, 'EPSG:4326', allow_override=True)
```

**"CRS mismatch" in spatial operations**

```python
from spatial import ensure_crs, crs_matches

# Solution: Ensure matching CRS before operations
if not crs_matches(gdf1, gdf2):
    gdf2 = ensure_crs(gdf2, gdf1.crs)
```

**Invalid geometry errors**

```python
# Check for invalid geometries
invalid = gdf[~gdf.geometry.is_valid]
print(f"Found {len(invalid)} invalid geometries")

# Fix common issues (self-intersections, etc.)
gdf['geometry'] = gdf.geometry.buffer(0)
```

**Coordinate order confusion**

- Shapely/GeoPandas uses (longitude, latitude) order: `Point(-74.0, 40.7)`
- `haversine_distance()` uses (latitude, longitude) order: `haversine_distance(40.7, -74.0, ...)`

This is a common source of errors. Remember: Shapely is (x, y) = (lon, lat).

**Empty results from spatial join**

```python
# Debug checklist:
print(f"GDF1 CRS: {gdf1.crs}")
print(f"GDF2 CRS: {gdf2.crs}")
print(f"GDF1 bounds: {gdf1.total_bounds}")
print(f"GDF2 bounds: {gdf2.total_bounds}")
print(f"GDF1 empty geometries: {gdf1.geometry.is_empty.sum()}")
print(f"GDF2 empty geometries: {gdf2.geometry.is_empty.sum()}")
```

**Memory errors with distance matrix**

```python
# Instead of full matrix for large datasets:
# dist_matrix = haversine_matrix(large_gdf)  # May fail

# Use nearest neighbor with distance limit:
result = nearest_neighbor(large_gdf, reference, k=5, max_distance=50000)
```

**File format errors**

```python
# Check supported formats
from spatial.core.io import SPATIAL_FORMATS
print(SPATIAL_FORMATS)  # {'.gpkg': 'GPKG', '.shp': 'ESRI Shapefile', ...}
```

### Debugging Tips

1. **Always print CRS** when loading data: `print(gdf.crs)`
2. **Check geometry validity**: `print(gdf.geometry.is_valid.all())`
3. **Inspect bounds**: `print(gdf.total_bounds)` to verify data location
4. **Use small samples** for testing: `gdf.head(100)` before full operations

## Future Phases

The spatial module will be expanded in future phases:

- **Phase 2**: Geocoding and address matching
- **Phase 3**: Spatial weights matrices
- **Phase 4**: Spatial econometrics (Conley SE, spatial lag/error models, panel models, zonal statistics)
- **Phase 5**: Visualization and mapping

See `doc/design/GEOSPATIAL_MODULE.md` for the full roadmap.

## Related Documentation

- [PIPELINE.md](PIPELINE.md) - Pipeline stage documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [design/GEOSPATIAL_MODULE.md](design/GEOSPATIAL_MODULE.md) - Full geospatial design document
