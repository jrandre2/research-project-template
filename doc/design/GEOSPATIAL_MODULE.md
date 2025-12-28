# Geospatial Analysis Module Design Plan

**Feature**: Comprehensive geospatial data handling and spatial econometrics for CENTAUR
**Status**: Design Document
**Created**: 2025-12-27

---

## Executive Summary

Add comprehensive geospatial analysis capabilities to CENTAUR through a utility-based architecture that **enhances existing pipeline stages** rather than creating a parallel track. The module provides: (1) spatial data I/O for vector and raster formats, (2) geocoding and spatial record linkage, (3) spatial panel construction with distance/adjacency measures, (4) spatial econometric methods (Conley SE, spatial lag/error models), and (5) publication-ready geospatial visualizations.

---

## Architecture Overview

Unlike the qualitative module (parallel track), the geospatial module **integrates into existing stages**:

```text
                    ┌─────────────────────────────────────┐
                    │       src/spatial/ (utilities)      │
                    │  ┌─────────┐ ┌─────────┐ ┌───────┐  │
                    │  │  core   │ │  econom │ │  viz  │  │
                    │  └────┬────┘ └────┬────┘ └───┬───┘  │
                    └───────┼───────────┼──────────┼──────┘
                            │           │          │
        ┌───────────────────┼───────────┼──────────┼───────────────────┐
        ▼                   ▼           ▼          ▼                   │
   ┌─────────┐        ┌─────────┐  ┌─────────┐  ┌─────────┐            │
   │s00_ingest│───────►│s01_link │──►│s02_panel│──►│s03_estim│           │
   │(+spatial │        │(+spatial│  │(+spatial│  │(+spatial│           │
   │ formats) │        │ match)  │  │ weights)│  │ econom) │           │
   └─────────┘        └─────────┘  └─────────┘  └─────────┘            │
        │                                              │               │
        │                                              ▼               │
        │                                        ┌─────────┐           │
        │                                        │s05_figs │◄──────────┘
        │                                        │(+maps)  │
        │                                        └─────────┘
        │                                              │
        └──────────────────────────────────────────────┘
                    All stages use src/spatial/ utilities
```

---

## Core Module: `src/spatial/`

### Module Structure

```text
src/spatial/
├── __init__.py              # Public API exports
├── core/                    # Core spatial operations
│   ├── __init__.py
│   ├── io.py                # Spatial data I/O (vector, raster)
│   ├── geometry.py          # Geometry operations (buffer, intersect, etc.)
│   ├── distance.py          # Distance calculations
│   ├── crs.py               # Coordinate reference system handling
│   └── weights.py           # Spatial weights matrices
├── geocoding/               # Address/location resolution
│   ├── __init__.py
│   ├── providers.py         # Geocoding service adapters
│   ├── batch.py             # Batch geocoding with caching
│   └── reverse.py           # Reverse geocoding
├── matching/                # Spatial record linkage
│   ├── __init__.py
│   ├── point_in_polygon.py  # Point-to-area matching
│   ├── nearest.py           # Nearest neighbor matching
│   ├── buffer.py            # Buffer-based matching
│   └── spatial_join.py      # General spatial joins
├── econometrics/            # Spatial econometric methods
│   ├── __init__.py
│   ├── conley.py            # Conley spatial HAC standard errors
│   ├── lag_model.py         # Spatial lag models
│   ├── error_model.py       # Spatial error models
│   ├── durbin.py            # Spatial Durbin model
│   └── diagnostics.py       # Moran's I, spatial autocorrelation tests
└── visualization/           # Geospatial figures
    ├── __init__.py
    ├── choropleth.py        # Choropleth maps
    ├── point_map.py         # Point/bubble maps
    ├── flow_map.py          # Origin-destination flows
    ├── heatmap.py           # Kernel density heatmaps
    └── interactive.py       # Folium/Leaflet interactive maps
```

---

## Core Utilities (`src/spatial/core/`)

### `io.py` — Spatial Data I/O

**Supported Formats**:

| Format | Read | Write | Extension | Use Case |
|--------|------|-------|-----------|----------|
| GeoPackage | Yes | Yes | `.gpkg` | Primary format (recommended) |
| Shapefile | Yes | Yes | `.shp` | Legacy compatibility |
| GeoJSON | Yes | Yes | `.geojson` | Web/API interchange |
| GeoParquet | Yes | Yes | `.geoparquet` | Large datasets, columnar |
| KML/KMZ | Yes | No | `.kml/.kmz` | Google Earth imports |
| GeoTIFF | Yes | Yes | `.tif` | Raster data |
| NetCDF | Yes | No | `.nc` | Climate/environmental data |
| PostGIS | Yes | Yes | connection | Database storage |

**Functions**:

```python
def load_spatial(path: Path, **kwargs) -> gpd.GeoDataFrame:
    """Load any supported spatial format, auto-detecting type."""

def save_spatial(gdf: gpd.GeoDataFrame, path: Path, **kwargs) -> None:
    """Save GeoDataFrame to specified format."""

def load_raster(path: Path, band: int = 1) -> tuple[np.ndarray, dict]:
    """Load raster data with metadata."""

def raster_to_points(raster_path: Path, sample_frac: float = 0.01) -> gpd.GeoDataFrame:
    """Convert raster to point samples."""
```

### `distance.py` — Distance Calculations

**Functions**:

```python
def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance in meters between two points."""

def haversine_matrix(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame = None
) -> np.ndarray:
    """Pairwise distance matrix (meters). If gdf2 is None, self-distances."""

def nearest_neighbor(
    gdf_from: gpd.GeoDataFrame,
    gdf_to: gpd.GeoDataFrame,
    k: int = 1,
    max_distance: float = None
) -> pd.DataFrame:
    """Find k nearest neighbors with distances."""

def distance_to_boundary(
    points: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    signed: bool = True
) -> pd.Series:
    """Distance to nearest boundary edge. Signed = negative if inside."""

def distance_band_weights(
    gdf: gpd.GeoDataFrame,
    threshold: float,
    binary: bool = True
) -> scipy.sparse.csr_matrix:
    """Spatial weights matrix based on distance threshold."""
```

### `crs.py` — Coordinate Reference Systems

**Functions**:

```python
def ensure_crs(
    gdf: gpd.GeoDataFrame,
    target_crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """Ensure GeoDataFrame has specified CRS, reprojecting if needed."""

def to_projected(
    gdf: gpd.GeoDataFrame,
    utm_zone: int = None
) -> gpd.GeoDataFrame:
    """Convert to appropriate UTM projection for accurate distance calculations."""

def estimate_utm_zone(lon: float) -> int:
    """Estimate appropriate UTM zone for a longitude."""
```

### `weights.py` — Spatial Weights Matrices

**Functions**:

```python
def contiguity_weights(
    gdf: gpd.GeoDataFrame,
    method: str = "queen"  # 'queen', 'rook'
) -> libpysal.weights.W:
    """Contiguity-based spatial weights."""

def knn_weights(
    gdf: gpd.GeoDataFrame,
    k: int = 5
) -> libpysal.weights.W:
    """K-nearest neighbor weights."""

def kernel_weights(
    gdf: gpd.GeoDataFrame,
    bandwidth: float,
    kernel: str = "gaussian"  # 'gaussian', 'triangular', 'uniform'
) -> libpysal.weights.W:
    """Kernel-based distance decay weights."""

def row_standardize(W: libpysal.weights.W) -> libpysal.weights.W:
    """Row-standardize a weights matrix."""
```

---

## Geocoding (`src/spatial/geocoding/`)

### Provider Abstraction

**Supported Providers**:

| Provider | API Key Required | Rate Limit | Best For |
|----------|------------------|------------|----------|
| Census Geocoder | No | 10k/batch | US addresses |
| Nominatim (OSM) | No | 1/second | Global, low volume |
| Google Maps | Yes | Pay per use | High accuracy |
| HERE | Yes | Freemium | Commercial use |
| Mapbox | Yes | Freemium | Web integration |

**Functions**:

```python
def geocode_address(
    address: str,
    provider: str = "census",
    **kwargs
) -> dict:
    """Geocode single address. Returns {lat, lon, confidence, components}."""

def geocode_batch(
    addresses: list[str],
    provider: str = "census",
    cache: bool = True,
    cache_dir: Path = None
) -> gpd.GeoDataFrame:
    """Batch geocode with caching. Returns GeoDataFrame with results."""

def reverse_geocode(
    lat: float, lon: float,
    provider: str = "nominatim"
) -> dict:
    """Reverse geocode coordinates to address components."""

def geocode_dataframe(
    df: pd.DataFrame,
    address_col: str = None,
    address_components: dict = None,  # {street: 'addr', city: 'city', ...}
    provider: str = "census"
) -> gpd.GeoDataFrame:
    """Geocode entire DataFrame, adding geometry column."""
```

### Caching Strategy

```python
# Geocode results cached to avoid repeated API calls
# Cache location: data_work/.cache/geocoding/<provider>/<hash>.json
# Cache TTL: configurable, default 30 days
# Failed lookups cached with null geometry to avoid retries
```

---

## Spatial Matching (`src/spatial/matching/`)

**Implements `match_type='spatial'` for `s01_link.py`**

### Match Types

```python
def spatial_join(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    how: str = "inner",  # 'inner', 'left', 'right'
    predicate: str = "intersects"  # 'intersects', 'within', 'contains'
) -> gpd.GeoDataFrame:
    """General spatial join."""

def point_in_polygon(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    polygon_id_col: str
) -> pd.Series:
    """Assign each point to containing polygon."""

def nearest_feature(
    gdf_from: gpd.GeoDataFrame,
    gdf_to: gpd.GeoDataFrame,
    max_distance: float = None,
    return_distance: bool = True
) -> pd.DataFrame:
    """Match to nearest feature with optional distance threshold."""

def buffer_match(
    gdf_from: gpd.GeoDataFrame,
    gdf_to: gpd.GeoDataFrame,
    buffer_distance: float,
    aggregation: str = "count"  # 'count', 'sum', 'mean', 'list'
) -> pd.DataFrame:
    """Match features within buffer distance, with aggregation."""
```

### Integration with `s01_link.py`

```python
# Enhanced LinkConfig for spatial matching
@dataclass
class LinkConfig:
    match_type: str  # 'exact', 'fuzzy', 'spatial'

    # Spatial-specific options (when match_type='spatial')
    spatial_predicate: str = "intersects"  # 'intersects', 'within', 'nearest'
    spatial_buffer: float = None  # Buffer distance in meters
    spatial_max_distance: float = None  # Max distance for nearest match
    spatial_crs: str = "EPSG:4326"  # CRS for distance calculations
```

---

## Spatial Panel Construction (`s02_panel.py` enhancements)

### New Panel Variables

```python
def add_spatial_variables(
    panel: pd.DataFrame,
    geometry: gpd.GeoDataFrame,
    unit_id: str,
    variables: list[str]
) -> pd.DataFrame:
    """
    Add spatial variables to panel.

    Available variables:
    - 'distance_to_treatment': Distance to nearest treated unit
    - 'neighbors_treated': Count of treated neighbors within threshold
    - 'neighbor_outcome_lag': Spatial lag of outcome variable
    - 'distance_to_boundary': Distance to treatment boundary
    - 'cluster_id': Spatial cluster assignment
    """

def create_spatial_lags(
    panel: pd.DataFrame,
    geometry: gpd.GeoDataFrame,
    variables: list[str],
    weights: libpysal.weights.W,
    time_col: str = "period"
) -> pd.DataFrame:
    """Create spatial lags of specified variables for each time period."""

def compute_spillover_exposure(
    panel: pd.DataFrame,
    geometry: gpd.GeoDataFrame,
    treatment_col: str,
    distance_bands: list[float],  # e.g., [1000, 5000, 10000] meters
    decay: str = "binary"  # 'binary', 'linear', 'exponential'
) -> pd.DataFrame:
    """Compute treatment exposure at various distance bands (for spillover analysis)."""
```

---

## Spatial Econometrics (`src/spatial/econometrics/`)

### Conley Standard Errors

```python
def conley_se(
    model_results,  # statsmodels or linearmodels results
    coords: np.ndarray,  # Nx2 array of (lat, lon)
    cutoff: float,  # Distance cutoff in km
    kernel: str = "uniform"  # 'uniform', 'bartlett', 'triangular'
) -> np.ndarray:
    """
    Compute Conley (1999) spatial HAC standard errors.

    Corrects for spatial correlation in residuals within distance cutoff.
    Returns array of corrected standard errors.
    """

def conley_variance(
    residuals: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    cutoff: float,
    kernel: str = "uniform"
) -> np.ndarray:
    """Compute Conley variance-covariance matrix."""
```

### Spatial Regression Models

```python
def spatial_lag_model(
    y: np.ndarray,
    X: np.ndarray,
    W: libpysal.weights.W,
    method: str = "ml"  # 'ml', 'gmm', '2sls'
) -> SpatialRegressionResults:
    """
    Spatial Lag Model: y = rho*Wy + X*beta + epsilon

    Accounts for spatial dependence in outcome variable.
    """

def spatial_error_model(
    y: np.ndarray,
    X: np.ndarray,
    W: libpysal.weights.W,
    method: str = "ml"
) -> SpatialRegressionResults:
    """
    Spatial Error Model: y = X*beta + u, where u = lambda*Wu + epsilon

    Accounts for spatial correlation in error terms.
    """

def spatial_durbin_model(
    y: np.ndarray,
    X: np.ndarray,
    W: libpysal.weights.W
) -> SpatialRegressionResults:
    """
    Spatial Durbin Model: y = rho*Wy + X*beta + WX*theta + epsilon

    Includes spatially lagged dependent variable AND independent variables.
    """

def geographically_weighted_regression(
    y: np.ndarray,
    X: np.ndarray,
    coords: np.ndarray,
    bandwidth: float = None,  # None = optimize via CV
    kernel: str = "gaussian"
) -> GWRResults:
    """
    Geographically Weighted Regression.

    Allows coefficients to vary spatially.
    """
```

### Spatial Diagnostics

```python
def morans_i(
    y: np.ndarray,
    W: libpysal.weights.W,
    permutations: int = 999
) -> dict:
    """
    Global Moran's I test for spatial autocorrelation.
    Returns {I, expected_I, variance, z_score, p_value}.
    """

def local_morans_i(
    y: np.ndarray,
    W: libpysal.weights.W,
    permutations: int = 999
) -> gpd.GeoDataFrame:
    """
    Local Moran's I (LISA) for hot/cold spot detection.
    Returns GeoDataFrame with local I values and cluster classifications.
    """

def lagrange_multiplier_tests(
    ols_results,
    W: libpysal.weights.W
) -> dict:
    """
    LM tests for spatial dependence.
    Returns {LM_lag, LM_error, robust_LM_lag, robust_LM_error} with p-values.
    """

def spatial_autocorrelation_test(
    residuals: np.ndarray,
    W: libpysal.weights.W
) -> dict:
    """Test residuals for spatial autocorrelation."""
```

### Integration with `s03_estimation.py`

```python
# Enhanced EstimationConfig
@dataclass
class EstimationConfig:
    # Existing fields...

    # Spatial options
    spatial_weights: str = None  # Path to weights matrix or 'contiguity', 'knn:5', 'distance:10000'
    spatial_model: str = None  # 'lag', 'error', 'durbin', 'gwr'
    conley_se: bool = False
    conley_cutoff: float = 100  # km
    conley_kernel: str = "uniform"
    coords_cols: tuple = ("latitude", "longitude")
```

---

## Geospatial Visualization (`src/spatial/visualization/`)

### Choropleth Maps

```python
def plot_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str = None,
    cmap: str = "viridis",
    scheme: str = "quantiles",  # 'quantiles', 'equal_interval', 'fisher_jenks', 'natural_breaks'
    k: int = 5,  # Number of classes
    legend: bool = True,
    basemap: str = None,  # 'light', 'dark', 'satellite', 'terrain'
    figsize: tuple = (12, 8),
    output_path: Path = None
) -> plt.Figure:
    """Publication-ready choropleth map."""

def plot_bivariate_choropleth(
    gdf: gpd.GeoDataFrame,
    x_col: str,
    y_col: str,
    title: str = None,
    **kwargs
) -> plt.Figure:
    """Bivariate choropleth (e.g., income vs. education)."""
```

### Point Maps

```python
def plot_point_map(
    gdf: gpd.GeoDataFrame,
    size_col: str = None,
    color_col: str = None,
    title: str = None,
    basemap: str = "light",
    alpha: float = 0.7,
    figsize: tuple = (12, 8),
    output_path: Path = None
) -> plt.Figure:
    """Point/bubble map with optional size and color encoding."""

def plot_kernel_density(
    gdf: gpd.GeoDataFrame,
    bandwidth: float = None,
    title: str = None,
    cmap: str = "Reds",
    **kwargs
) -> plt.Figure:
    """Kernel density heatmap of point distribution."""
```

### Treatment Effect Maps

```python
def plot_treatment_map(
    gdf: gpd.GeoDataFrame,
    treatment_col: str,
    outcome_col: str = None,
    boundary: gpd.GeoDataFrame = None,
    title: str = None,
    **kwargs
) -> plt.Figure:
    """Map showing treatment/control status with optional boundary."""

def plot_rdd_geography(
    gdf: gpd.GeoDataFrame,
    running_var_col: str,
    cutoff: float,
    outcome_col: str,
    title: str = None,
    **kwargs
) -> plt.Figure:
    """Geographic RDD visualization with treatment boundary."""

def plot_spillover_rings(
    gdf: gpd.GeoDataFrame,
    treatment_col: str,
    distances: list[float],
    outcome_col: str = None,
    **kwargs
) -> plt.Figure:
    """Visualize spillover effects at different distance bands."""
```

### Interactive Maps

```python
def create_interactive_map(
    gdf: gpd.GeoDataFrame,
    popup_cols: list[str] = None,
    tooltip_cols: list[str] = None,
    style_col: str = None,
    output_path: Path = None
) -> folium.Map:
    """Create interactive Folium map for exploration."""

def create_comparison_map(
    gdf: gpd.GeoDataFrame,
    col1: str,
    col2: str,
    output_path: Path = None
) -> folium.plugins.DualMap:
    """Side-by-side comparison of two variables."""
```

### Integration with `s05_figures.py`

```python
# New figure types for make_figures command
SPATIAL_FIGURE_TYPES = [
    "choropleth",          # Choropleth of outcome variable
    "treatment_map",       # Treatment/control geography
    "coefficient_map",     # Spatially varying coefficients (GWR)
    "cluster_map",         # LISA hot/cold spots
    "spillover_map",       # Treatment spillover visualization
    "residual_map",        # Spatial distribution of residuals
]
```

---

## Configuration Additions

### New Sections in `src/config.py`

```python
# ============================================================
# GEOSPATIAL SETTINGS
# ============================================================

# Enable/disable spatial features
SPATIAL_ENABLED = True

# Default coordinate reference system
SPATIAL_DEFAULT_CRS = "EPSG:4326"  # WGS84
SPATIAL_PROJECTED_CRS = None  # Auto-detect UTM if None

# Data directories
SPATIAL_DATA_DIR = DATA_WORK_DIR / 'spatial'
SPATIAL_CACHE_DIR = DATA_WORK_DIR / '.cache' / 'spatial'

# Geocoding
GEOCODING_PROVIDER = "census"  # 'census', 'nominatim', 'google', 'here'
GEOCODING_CACHE_TTL_DAYS = 30
GEOCODING_API_KEY = None  # Set via environment variable

# Spatial weights defaults
SPATIAL_WEIGHTS_TYPE = "queen"  # 'queen', 'rook', 'knn', 'distance'
SPATIAL_WEIGHTS_KNN_K = 5
SPATIAL_WEIGHTS_DISTANCE_THRESHOLD = 10000  # meters

# Spatial econometrics
CONLEY_ENABLED = False
CONLEY_CUTOFF_KM = 100
CONLEY_KERNEL = "uniform"  # 'uniform', 'bartlett', 'triangular'

# Visualization
SPATIAL_FIGURE_DPI = 300
SPATIAL_FIGURE_FORMAT = "png"
SPATIAL_BASEMAP = "light"  # 'light', 'dark', 'satellite', 'terrain', None
SPATIAL_CMAP_SEQUENTIAL = "viridis"
SPATIAL_CMAP_DIVERGING = "RdBu_r"
```

---

## CLI Commands

### New Commands

```python
# Geocoding
geocode              # --input, --address-col, --provider, --output
geocode_status       # Show geocoding cache stats

# Spatial operations
spatial_join         # --left, --right, --predicate, --output
spatial_weights      # --input, --type, --output, --k, --threshold
spatial_cluster      # --input, --method, --output

# Spatial diagnostics
spatial_diagnostics  # --input, --variable, --weights
morans_test          # --input, --variable, --weights, --local

# Visualization
map_choropleth       # --input, --variable, --output, --scheme
map_points           # --input, --size, --color, --output
map_treatment        # --input, --treatment, --boundary, --output
map_interactive      # --input, --output (HTML)
```

### Enhanced Existing Commands

```bash
# s01_link with spatial matching
python src/pipeline.py link_records --match-type spatial --predicate within

# s03_estimation with Conley SE
python src/pipeline.py run_estimation --conley-se --conley-cutoff 50

# s05_figures with maps
python src/pipeline.py make_figures --include-maps --basemap satellite
```

---

## Documentation Updates

### New Documentation Files

| File                          | Purpose                            |
|-------------------------------|------------------------------------|
| `doc/GEOSPATIAL_ANALYSIS.md`  | Comprehensive spatial methods guide |
| `doc/SPATIAL_ECONOMETRICS.md` | Conley SE, spatial models reference |
| `doc/GEOCODING_GUIDE.md`      | Geocoding providers and caching    |
| `doc/SPATIAL_VISUALIZATION.md`| Map creation and styling           |

### Updates to Existing Files

| File                     | Changes                                |
|--------------------------|----------------------------------------|
| `doc/README.md`          | Add geospatial section to index        |
| `doc/PIPELINE.md`        | Document spatial enhancements to stages |
| `doc/ARCHITECTURE.md`    | Add spatial module to system diagram   |
| `doc/METHODOLOGY.md`     | Add spatial econometrics section       |
| `doc/DATA_DICTIONARY.md` | Expand spatial variable conventions    |
| `requirements.txt`       | Uncomment and add spatial packages     |
| `CLAUDE.md`              | Add spatial quick reference            |

---

## File Structure

### New Directories and Files

```text
src/spatial/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── io.py
│   ├── geometry.py
│   ├── distance.py
│   ├── crs.py
│   └── weights.py
├── geocoding/
│   ├── __init__.py
│   ├── providers.py
│   ├── batch.py
│   └── reverse.py
├── matching/
│   ├── __init__.py
│   ├── point_in_polygon.py
│   ├── nearest.py
│   ├── buffer.py
│   └── spatial_join.py
├── econometrics/
│   ├── __init__.py
│   ├── conley.py
│   ├── lag_model.py
│   ├── error_model.py
│   ├── durbin.py
│   └── diagnostics.py
└── visualization/
    ├── __init__.py
    ├── choropleth.py
    ├── point_map.py
    ├── flow_map.py
    ├── heatmap.py
    └── interactive.py

data_work/
└── spatial/
    ├── geocoding_cache/     # Cached geocoding results
    ├── weights/             # Saved spatial weights matrices
    └── boundaries/          # Reference boundary files

figures/
├── map_*.png                # Static maps
└── map_*.html               # Interactive maps

tests/
└── test_spatial/
    ├── test_core.py
    ├── test_geocoding.py
    ├── test_matching.py
    ├── test_econometrics.py
    └── test_visualization.py
```

---

## Implementation Phases

### Phase 1: Core Infrastructure

- [ ] Create `src/spatial/` module structure
- [ ] Implement `core/io.py` — spatial data I/O
- [ ] Implement `core/distance.py` — haversine and distance matrices
- [ ] Implement `core/crs.py` — CRS handling
- [ ] Add spatial configuration to `config.py`
- [ ] Uncomment geopandas/shapely/pyproj in `requirements.txt`

### Phase 2: Spatial Matching & Linkage

- [ ] Implement `matching/` submodule
- [ ] Enhance `s01_link.py` with `match_type='spatial'`
- [ ] Implement `geocoding/` submodule with Census provider
- [ ] Add geocoding cache infrastructure
- [ ] Add `geocode` CLI command

### Phase 3: Spatial Weights & Panel

- [ ] Implement `core/weights.py` — spatial weights matrices
- [ ] Enhance `s02_panel.py` with spatial lag variables
- [ ] Add spillover exposure calculations
- [ ] Implement `spatial_weights` CLI command

### Phase 4: Spatial Econometrics

- [ ] Implement `econometrics/conley.py` — Conley standard errors
- [ ] Implement `econometrics/diagnostics.py` — Moran's I, LM tests
- [ ] Enhance `s03_estimation.py` with `--conley-se` option
- [ ] Implement spatial lag/error models
- [ ] Add `spatial_diagnostics` CLI command

### Phase 5: Visualization

- [ ] Implement `visualization/choropleth.py`
- [ ] Implement `visualization/point_map.py`
- [ ] Implement `visualization/interactive.py`
- [ ] Enhance `s05_figures.py` with map generation
- [ ] Add map CLI commands

### Phase 6: Polish & Documentation

- [ ] Complete all documentation
- [ ] Add comprehensive test coverage
- [ ] Create example spatial dataset
- [ ] Performance optimization (spatial indexing)
- [ ] Add remaining geocoding providers

---

## Dependencies

### Required Python Packages

```text
# Core geospatial (uncomment in requirements.txt)
geopandas>=0.14.0          # Spatial DataFrames
shapely>=2.0.0             # Geometry operations
pyproj>=3.5.0              # CRS transformations
fiona>=1.9.0               # Spatial file I/O

# Spatial econometrics
libpysal>=4.7.0            # Spatial weights, Moran's I
esda>=2.4.0                # Exploratory spatial data analysis
spreg>=1.3.0               # Spatial regression models
mgwr>=2.1.0                # Geographically weighted regression

# Raster support (optional)
rasterio>=1.3.0            # Raster I/O
rasterstats>=0.18.0        # Zonal statistics

# Geocoding
geopy>=2.3.0               # Geocoding providers
censusgeocode>=0.5.0       # US Census geocoder

# Visualization
matplotlib>=3.7.0          # Already present
contextily>=1.3.0          # Basemaps
folium>=0.14.0             # Interactive maps
mapclassify>=2.5.0         # Classification schemes
```

---

## Integration with Existing Stages

### s00_ingest.py Enhancements

```python
# Already supports .gpkg via helpers.load_data()
# Add: auto-detection of spatial columns, CRS validation
def ingest_spatial(path: Path) -> gpd.GeoDataFrame:
    """Load spatial data with validation and CRS normalization."""
```

### s01_link.py Enhancements

```python
# Add spatial matching support
if config.match_type == 'spatial':
    from spatial.matching import spatial_join, nearest_feature
    result = spatial_join(left, right, predicate=config.spatial_predicate)
```

### s02_panel.py Enhancements

```python
# Add spatial variable construction
if SPATIAL_ENABLED and 'geometry' in df.columns:
    from spatial.core import create_spatial_lags, compute_spillover_exposure
    df = create_spatial_lags(df, geometry, ['outcome'], weights)
```

### s03_estimation.py Enhancements

```python
# Add Conley SE option
if config.conley_se:
    from spatial.econometrics import conley_se
    se = conley_se(results, coords, cutoff=config.conley_cutoff)
    results.bse = se  # Replace standard errors
```

### s05_figures.py Enhancements

```python
# Add map generation
if SPATIAL_ENABLED and include_maps:
    from spatial.visualization import plot_choropleth, plot_treatment_map
    plot_choropleth(gdf, 'outcome', output_path=figures_dir / 'map_outcome.png')
```

---

## Critical Files to Modify

| File                       | Modification Type                        |
|----------------------------|------------------------------------------|
| `requirements.txt`         | Uncomment spatial packages, add new ones |
| `src/config.py`            | Add SPATIAL_* configuration section      |
| `src/pipeline.py`          | Register new spatial commands            |
| `src/stages/s01_link.py`   | Add spatial matching implementation      |
| `src/stages/s02_panel.py`  | Add spatial variable construction        |
| `src/stages/s03_estimation.py` | Add Conley SE and spatial model options |
| `src/stages/s05_figures.py`| Add map generation                       |
| `src/utils/helpers.py`     | Already has .gpkg support (verify complete) |

---

## Risks & Mitigations

| Risk                        | Mitigation                                        |
|-----------------------------|---------------------------------------------------|
| Large spatial datasets slow | Use spatial indexing (R-tree), chunked processing |
| CRS confusion               | Standardize on WGS84, auto-detect and warn on mismatch |
| Geocoding rate limits       | Aggressive caching, batch processing, provider fallbacks |
| Conley SE computation time  | Pre-compute distance matrix, sparse representation |
| libpysal version conflicts  | Pin versions, test compatibility                  |
| Basemap tile availability   | Cache tiles locally, fallback to no basemap       |

---

## Success Criteria

1. Spatial data loads/saves seamlessly alongside tabular data
2. Researchers can geocode addresses with single command
3. Spatial matching works as `match_type='spatial'` in linkage
4. Conley SE available via `--conley-se` flag on estimation
5. Publication-ready maps generated alongside standard figures
6. Moran's I and spatial diagnostics in QA reports
7. Interactive maps for data exploration

---

## Related Documents

- [QUALITATIVE_MODULE.md](QUALITATIVE_MODULE.md) — Qualitative analysis design
- [../PIPELINE.md](../PIPELINE.md) — Stage reference
- [../METHODOLOGY.md](../METHODOLOGY.md) — Statistical methods
- [../CUSTOMIZATION.md](../CUSTOMIZATION.md) — Existing spatial addon sketch
