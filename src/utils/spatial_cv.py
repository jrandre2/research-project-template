#!/usr/bin/env python3
"""
Spatial Cross-Validation Manager
================================

Implements spatial cross-validation to prevent data leakage from
geographic proximity. Uses GroupKFold with spatially-defined groups
to ensure training and test sets are geographically separated.

Why Use Spatial CV?
-------------------
Standard k-fold cross-validation assumes independence between samples.
In geographic data, nearby observations are often correlated (spatial
autocorrelation). This correlation can leak information from training
to test sets, leading to overly optimistic performance estimates.

Spatial CV creates geographic groups so that entire regions are held
out during testing, providing more honest performance estimates for
models that will be applied to new geographic areas.

Key Features
------------
- Multiple grouping methods (k-means, geographic bands, ZIP digit)
- Contiguity-based grouping for polygon data (requires geopandas)
- Leakage quantification by comparing spatial vs random CV
- Integration with scikit-learn's GroupKFold

Example Usage
-------------
>>> from src.utils.spatial_cv import SpatialCVManager
>>> manager = SpatialCVManager(n_groups=5, method='kmeans')
>>> groups = manager.create_groups_from_coordinates(lats, lons)
>>> results = manager.cross_validate(model, X, y)
>>> print(f"Spatial CV R2: {results['mean']:.3f} +/- {results['std']:.3f}")

>>> # Compare to random CV to quantify leakage
>>> comparison = manager.compare_to_random_cv(model, X, y)
>>> print(f"Estimated leakage: {comparison['leakage']:.3f}")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
import warnings

from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.base import clone

warnings.filterwarnings('ignore')


class SpatialCVManager:
    """
    Manager for spatial cross-validation to prevent geographic data leakage.

    Uses GroupKFold with geographic groups to ensure proper separation
    between training and test sets, preventing spatial autocorrelation
    from inflating performance metrics.

    Parameters
    ----------
    n_groups : int, default=5
        Number of spatial groups for cross-validation.
    method : str, default='kmeans'
        Grouping method. Options:
        - 'kmeans': K-means clustering on coordinates
        - 'balanced_kmeans': K-means with balanced group sizes
        - 'geographic_bands': Latitude-based bands
        - 'longitude_bands': Longitude-based bands
        - 'spatial_blocks': Grid-based spatial blocks
        - 'zip_digit': ZIP code digit-based grouping
        - 'contiguity_queen': Polygon contiguity (requires geopandas)
        - 'contiguity_rook': Edge-only contiguity (requires geopandas)
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    spatial_groups : np.ndarray or None
        Array of group assignments (0 to n_groups-1) for each sample.
    group_kfold : GroupKFold
        Scikit-learn GroupKFold splitter.

    Examples
    --------
    >>> manager = SpatialCVManager(n_groups=5, method='kmeans')
    >>> groups = manager.create_groups_from_coordinates(df['lat'], df['lon'])
    >>> results = manager.cross_validate(Ridge(alpha=1.0), X, y)
    """

    def __init__(
        self,
        n_groups: int = 5,
        method: str = 'kmeans',
        random_state: int = 42,
    ):
        self.n_groups = n_groups
        self.method = method
        self.random_state = random_state
        self.spatial_groups: Optional[np.ndarray] = None
        self.group_kfold = GroupKFold(n_splits=n_groups)

    def create_groups_from_coordinates(
        self,
        latitudes: Union[np.ndarray, pd.Series, List[float]],
        longitudes: Union[np.ndarray, pd.Series, List[float]],
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Create spatial groups from geographic coordinates.

        Parameters
        ----------
        latitudes : array-like
            Array of latitude values.
        longitudes : array-like
            Array of longitude values.
        verbose : bool, default=True
            Whether to print group summary.

        Returns
        -------
        np.ndarray
            Array of group assignments (0 to n_groups-1).

        Raises
        ------
        ValueError
            If method is unknown or requires geopandas.
        """
        latitudes = np.asarray(latitudes)
        longitudes = np.asarray(longitudes)
        coords = np.column_stack([longitudes, latitudes])

        if self.method == 'kmeans':
            kmeans = KMeans(
                n_clusters=self.n_groups,
                random_state=self.random_state,
                n_init=10,
            )
            self.spatial_groups = kmeans.fit_predict(coords)

        elif self.method == 'balanced_kmeans':
            self.spatial_groups = self._balanced_kmeans(coords)

        elif self.method == 'geographic_bands':
            # Create latitude-based bands (horizontal slices)
            quantiles = np.quantile(latitudes, np.linspace(0, 1, self.n_groups + 1))
            self.spatial_groups = np.digitize(latitudes, quantiles) - 1
            self.spatial_groups = np.clip(self.spatial_groups, 0, self.n_groups - 1)

        elif self.method == 'longitude_bands':
            # Create longitude-based bands (vertical slices)
            quantiles = np.quantile(longitudes, np.linspace(0, 1, self.n_groups + 1))
            self.spatial_groups = np.digitize(longitudes, quantiles) - 1
            self.spatial_groups = np.clip(self.spatial_groups, 0, self.n_groups - 1)

        elif self.method == 'spatial_blocks':
            # Create grid-based spatial blocks
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

            n_side = int(np.ceil(np.sqrt(self.n_groups)))
            x_bins = np.linspace(x_min, x_max, n_side + 1)
            y_bins = np.linspace(y_min, y_max, n_side + 1)

            x_groups = np.digitize(coords[:, 0], x_bins) - 1
            y_groups = np.digitize(coords[:, 1], y_bins) - 1
            self.spatial_groups = (x_groups * n_side + y_groups).astype(int)
            self.spatial_groups = self.spatial_groups % self.n_groups

        elif self.method in ('contiguity_queen', 'contiguity_rook'):
            raise ValueError(
                f"Method '{self.method}' requires polygon geometry. "
                "Use create_groups_from_geodata() instead."
            )

        else:
            raise ValueError(f"Unknown coordinate-based method: {self.method}")

        if verbose:
            self._print_group_summary()

        return self.spatial_groups

    def _balanced_kmeans(self, coords: np.ndarray) -> np.ndarray:
        """
        Assign k-means centers with balanced group sizes.

        Uses the Hungarian algorithm to assign points to clusters
        while respecting capacity constraints, resulting in more
        evenly sized groups than standard k-means.
        """
        kmeans = KMeans(
            n_clusters=self.n_groups,
            random_state=self.random_state,
            n_init=10,
        )
        kmeans.fit(coords)
        centers = kmeans.cluster_centers_

        # Calculate balanced capacities
        n_samples = coords.shape[0]
        base = n_samples // self.n_groups
        remainder = n_samples % self.n_groups
        capacities = [
            base + 1 if idx < remainder else base
            for idx in range(self.n_groups)
        ]

        # Expand centers by capacity for assignment
        expanded_centers = []
        expanded_groups = []
        for group_idx, cap in enumerate(capacities):
            for _ in range(cap):
                expanded_centers.append(centers[group_idx])
                expanded_groups.append(group_idx)

        expanded_centers = np.array(expanded_centers)
        distances = np.linalg.norm(
            coords[:, None, :] - expanded_centers[None, :, :], axis=2
        )

        # Use Hungarian algorithm for optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise ImportError(
                "scipy required for balanced_kmeans assignment. "
                "Install with: pip install scipy"
            ) from exc

        row_idx, col_idx = linear_sum_assignment(distances)
        assignments = np.empty(n_samples, dtype=int)
        assignments[row_idx] = np.array(expanded_groups)[col_idx]

        return assignments

    def create_groups_from_zip_codes(
        self,
        zip_codes: Union[List[str], np.ndarray, pd.Series],
        digit_position: int = 3,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Create spatial groups based on ZIP code digits.

        Uses the nth digit of ZIP codes to create geographic groups,
        which provides a simple proxy for geographic location in the US.

        Parameters
        ----------
        zip_codes : array-like
            Array of ZIP codes (as strings or integers).
        digit_position : int, default=3
            Position of digit to use (0-indexed). Default is 3 (4th digit).
        verbose : bool, default=True
            Whether to print group summary.

        Returns
        -------
        np.ndarray
            Array of group assignments.
        """
        zip_strs = pd.Series(zip_codes).astype(str).str.zfill(5)
        digits = zip_strs.str[digit_position].astype(int)

        # Map digits to groups (combine adjacent digits to get ~5 groups)
        group_mapping = digits // 2  # 0-1 -> 0, 2-3 -> 1, etc.
        self.spatial_groups = group_mapping.values

        # Ensure we have the right number of groups
        unique_groups = np.unique(self.spatial_groups)
        if len(unique_groups) > self.n_groups:
            self.spatial_groups = self.spatial_groups % self.n_groups

        if verbose:
            self._print_group_summary()

        return self.spatial_groups

    def create_groups_from_geodata(
        self,
        gdf: Any,  # GeoDataFrame, but avoid type hint to prevent import
        contiguity: str = 'queen',
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Create spatial groups using contiguity-constrained clustering.

        Requires geopandas. Groups are created by clustering polygons
        that share boundaries, resulting in spatially contiguous groups.

        Parameters
        ----------
        gdf : GeoDataFrame
            GeoDataFrame with geometry aligned to the data order.
        contiguity : str, default='queen'
            'queen' (shared vertices/edges) or 'rook' (shared edges only).
        verbose : bool, default=True
            Whether to print group summary.

        Returns
        -------
        np.ndarray
            Array of group assignments.

        Raises
        ------
        ImportError
            If geopandas is not installed.
        ValueError
            If contiguity type is unknown or connectivity matrix fails.
        """
        try:
            import geopandas as gpd  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "geopandas required for contiguity grouping. "
                "Install with: pip install geopandas"
            ) from exc

        contiguity = contiguity.lower()
        if contiguity not in ('queen', 'rook'):
            raise ValueError(f"Unknown contiguity type: {contiguity}")

        connectivity = self._build_contiguity_connectivity(gdf, contiguity)
        if connectivity is None:
            raise ValueError("Failed to build contiguity connectivity matrix")

        # Get centroid coordinates for clustering
        centroids = gdf.geometry.centroid
        coords = np.column_stack([centroids.x.values, centroids.y.values])

        # Agglomerative clustering with connectivity constraint
        cluster = AgglomerativeClustering(
            n_clusters=self.n_groups,
            linkage='ward',
            connectivity=connectivity,
        )
        self.spatial_groups = cluster.fit_predict(coords)

        if verbose:
            self._print_group_summary()

        return self.spatial_groups

    def _build_contiguity_connectivity(
        self,
        gdf: Any,
        contiguity: str,
    ) -> Optional[Any]:
        """
        Build a contiguity-based connectivity matrix.

        Queen contiguity: polygons are neighbors if they share any boundary
        (edge or vertex).

        Rook contiguity: polygons are neighbors only if they share an edge
        (not just a vertex point).
        """
        from scipy.sparse import coo_matrix
        from shapely.geometry import Point, MultiPoint

        geoms = gdf.geometry.reset_index(drop=True)
        sindex = gdf.sindex
        rows: List[int] = []
        cols: List[int] = []

        for i, geom in enumerate(geoms):
            if geom is None or geom.is_empty:
                continue

            candidates = list(sindex.intersection(geom.bounds))
            for j in candidates:
                if j <= i:
                    continue

                other = geoms.iloc[j]
                if other is None or other.is_empty:
                    continue

                # Check boundary intersection
                inter = geom.boundary.intersection(other.boundary)
                if inter.is_empty:
                    continue

                if contiguity == 'rook':
                    # Rook: require edge sharing (intersection must have length > 0)
                    if isinstance(inter, (Point, MultiPoint)):
                        continue
                    if hasattr(inter, 'length') and inter.length == 0:
                        continue

                # Queen: any boundary intersection counts
                rows.extend([i, j])
                cols.extend([j, i])

        if not rows:
            return None

        data = np.ones(len(rows), dtype=int)
        return coo_matrix((data, (rows, cols)), shape=(len(geoms), len(geoms)))

    def _print_group_summary(self) -> None:
        """Print summary of spatial groups."""
        if self.spatial_groups is None:
            return

        unique, counts = np.unique(self.spatial_groups, return_counts=True)
        print(f"   Created {len(unique)} spatial groups:")
        for g, c in zip(unique, counts):
            print(f"      Group {g}: {c} samples")

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate spatial cross-validation splits.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.

        Yields
        ------
        train_idx, test_idx : tuple of np.ndarray
            Indices for training and test sets.

        Raises
        ------
        ValueError
            If spatial groups haven't been created yet.
        """
        if self.spatial_groups is None:
            raise ValueError("Must create spatial groups before splitting")

        for train_idx, test_idx in self.group_kfold.split(
            X, y, groups=self.spatial_groups
        ):
            yield train_idx, test_idx

    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        scale_features: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform spatial cross-validation.

        Parameters
        ----------
        model : sklearn estimator
            Model to evaluate (will be cloned for each fold).
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.
        scale_features : bool, default=True
            Whether to scale features within each fold.

        Returns
        -------
        dict
            Cross-validation results with:
            - 'scores': list of R2 scores per fold
            - 'mean': mean R2 score
            - 'std': standard deviation of R2 scores
            - 'fold_details': list of dicts with per-fold info

        Raises
        ------
        ValueError
            If spatial groups haven't been created yet.
        """
        if self.spatial_groups is None:
            raise ValueError("Must create spatial groups before cross-validation")

        cv_scores: List[float] = []
        fold_details: List[Dict[str, Any]] = []

        for fold, (train_idx, test_idx) in enumerate(self.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Clone and fit model
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)

            # Predict and score
            y_pred = fold_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            cv_scores.append(r2)

            fold_details.append({
                'fold': fold + 1,
                'n_train': len(train_idx),
                'n_test': len(test_idx),
                'r2': r2,
            })

        return {
            'scores': cv_scores,
            'mean': float(np.mean(cv_scores)),
            'std': float(np.std(cv_scores)),
            'fold_details': fold_details,
        }

    def compare_to_random_cv(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        scale_features: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare spatial CV to random CV to quantify data leakage.

        The difference between random CV and spatial CV performance
        estimates the amount of information "leaking" from training
        to test sets due to spatial autocorrelation.

        Parameters
        ----------
        model : sklearn estimator
            Model to evaluate.
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.
        scale_features : bool, default=True
            Whether to scale features.

        Returns
        -------
        dict
            Comparison results with:
            - 'spatial_cv': dict with mean and std
            - 'random_cv': dict with mean and std
            - 'leakage': difference (random - spatial)
            - 'leakage_pct': leakage as percentage of spatial CV mean
        """
        # Spatial CV
        spatial_results = self.cross_validate(model, X, y, scale_features)

        # Random CV (standard k-fold)
        if scale_features:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        random_scores = cross_val_score(
            model, X_scaled, y, cv=self.n_groups, scoring='r2'
        )

        leakage = float(np.mean(random_scores)) - spatial_results['mean']

        return {
            'spatial_cv': {
                'mean': spatial_results['mean'],
                'std': spatial_results['std'],
            },
            'random_cv': {
                'mean': float(np.mean(random_scores)),
                'std': float(np.std(random_scores)),
            },
            'leakage': leakage,
            'leakage_pct': (
                (leakage / spatial_results['mean'] * 100)
                if spatial_results['mean'] != 0
                else np.inf
            ),
        }

    def get_group_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get statistics about the spatial groups.

        Returns
        -------
        dict or None
            Group statistics including counts and balance metrics,
            or None if groups haven't been created.
        """
        if self.spatial_groups is None:
            return None

        unique, counts = np.unique(self.spatial_groups, return_counts=True)

        return {
            'n_groups': len(unique),
            'group_sizes': dict(zip(unique.tolist(), counts.tolist())),
            'min_size': int(counts.min()),
            'max_size': int(counts.max()),
            'mean_size': float(counts.mean()),
            'std_size': float(counts.std()),
            'balance_ratio': float(counts.min() / counts.max()),
        }


def create_spatial_groups_simple(
    df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    n_groups: int = 5,
    method: str = 'kmeans',
) -> np.ndarray:
    """
    Simple function to create spatial groups from a DataFrame.

    Convenience wrapper around SpatialCVManager for quick usage.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing coordinate columns.
    lat_col : str, default='latitude'
        Name of the latitude column.
    lon_col : str, default='longitude'
        Name of the longitude column.
    n_groups : int, default=5
        Target number of groups.
    method : str, default='kmeans'
        Grouping method.

    Returns
    -------
    np.ndarray
        Array of group assignments.

    Examples
    --------
    >>> groups = create_spatial_groups_simple(df, 'lat', 'lon')
    >>> df['spatial_group'] = groups
    """
    manager = SpatialCVManager(n_groups=n_groups, method=method)
    return manager.create_groups_from_coordinates(
        df[lat_col].values,
        df[lon_col].values,
    )


def compare_spatial_vs_random_cv(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    n_groups: int = 5,
    method: str = 'kmeans',
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Quick comparison of spatial vs random CV for geographic data.

    Convenience function that creates spatial groups and compares
    spatial CV to random CV in one call.

    Parameters
    ----------
    model : sklearn estimator
        Model to evaluate.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target array.
    latitudes : np.ndarray
        Latitude coordinates.
    longitudes : np.ndarray
        Longitude coordinates.
    n_groups : int, default=5
        Number of spatial groups.
    method : str, default='kmeans'
        Grouping method.
    verbose : bool, default=True
        Whether to print results.

    Returns
    -------
    dict
        Comparison results (see SpatialCVManager.compare_to_random_cv).

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> results = compare_spatial_vs_random_cv(
    ...     Ridge(alpha=1.0), X, y, df['lat'], df['lon']
    ... )
    >>> print(f"Leakage: {results['leakage']:.3f}")
    """
    manager = SpatialCVManager(n_groups=n_groups, method=method)
    manager.create_groups_from_coordinates(latitudes, longitudes, verbose=False)

    results = manager.compare_to_random_cv(model, X, y)

    if verbose:
        print(f"Random CV:  {results['random_cv']['mean']:.3f} +/- "
              f"{results['random_cv']['std']:.3f}")
        print(f"Spatial CV: {results['spatial_cv']['mean']:.3f} +/- "
              f"{results['spatial_cv']['std']:.3f}")
        print(f"Leakage:    {results['leakage']:+.3f} "
              f"({results['leakage_pct']:.1f}%)")

    return results
