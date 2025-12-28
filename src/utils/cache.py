#!/usr/bin/env python3
"""
Caching utilities for pipeline stages.

This module provides hash-based caching with automatic invalidation
to dramatically speed up iterative development and re-runs.

Usage
-----
    from utils.cache import CacheManager, hash_dataframe

    # Create a cache manager for a stage
    cache = CacheManager('s03_estimation')

    # Cache expensive computations
    result = cache.get_or_compute(
        key='demeaned_baseline',
        compute_fn=lambda: demean_by_fe(df, fe_cols),
        depends_on={'panel': panel_path, 'spec': spec_config}
    )

    # Check cache statistics
    print(cache.stats())
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

# Lazy imports to avoid circular dependencies
_pd = None
_np = None


def _get_pandas():
    """Lazy import pandas."""
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd


def _get_numpy():
    """Lazy import numpy."""
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


# =============================================================================
# CONFIGURATION
# =============================================================================

def _get_cache_dir() -> Path:
    """Get the cache directory from config or use default."""
    try:
        from config import CACHE_DIR
        return CACHE_DIR
    except ImportError:
        return Path(__file__).parent.parent.parent / 'data_work' / '.cache'


def _get_cache_enabled() -> bool:
    """Check if caching is enabled."""
    try:
        from config import CACHE_ENABLED
        return CACHE_ENABLED
    except ImportError:
        return True


def _get_cache_max_age() -> float:
    """Get maximum cache age in hours."""
    try:
        from config import CACHE_MAX_AGE_HOURS
        return CACHE_MAX_AGE_HOURS
    except ImportError:
        return 168  # 1 week


# Global cache settings (loaded from config)
CACHE_ENABLED = _get_cache_enabled()
CACHE_COMPRESSION = False  # Use gzip compression for large caches
CACHE_MAX_AGE_HOURS = _get_cache_max_age()


# =============================================================================
# HASHING UTILITIES
# =============================================================================

def hash_dataframe(df, sample_size: int = 10000) -> str:
    """
    Compute a deterministic hash of a pandas DataFrame.

    Uses a combination of shape, column names, dtypes, and sampled values
    to create a fast but reliable hash.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to hash
    sample_size : int
        Number of rows to sample for value hashing (default: 10000)

    Returns
    -------
    str
        MD5 hash hex digest
    """
    pd = _get_pandas()
    np = _get_numpy()

    hasher = hashlib.md5()

    # Hash shape
    hasher.update(f"shape:{df.shape}".encode())

    # Hash column names and dtypes
    for col in df.columns:
        hasher.update(f"col:{col}:{df[col].dtype}".encode())

    # Hash sampled values for content verification
    if len(df) > 0:
        # Use deterministic sampling
        if len(df) <= sample_size:
            sample = df
        else:
            indices = np.linspace(0, len(df) - 1, sample_size, dtype=int)
            sample = df.iloc[indices]

        # Hash the sample values
        try:
            # Use pandas built-in hash for efficiency
            hash_values = pd.util.hash_pandas_object(sample, index=False)
            hasher.update(hash_values.values.tobytes())
        except (TypeError, ValueError):
            # Fallback for unhashable types
            hasher.update(sample.to_json().encode())

    return hasher.hexdigest()


def hash_file(path: Union[str, Path]) -> str:
    """
    Compute MD5 hash of a file's contents.

    Parameters
    ----------
    path : str or Path
        Path to file

    Returns
    -------
    str
        MD5 hash hex digest
    """
    path = Path(path)
    if not path.exists():
        return f"missing:{path}"

    hasher = hashlib.md5()

    # For large files, hash in chunks
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def hash_config(config: dict) -> str:
    """
    Compute hash of a configuration dictionary.

    Parameters
    ----------
    config : dict
        Configuration dictionary (must be JSON-serializable)

    Returns
    -------
    str
        MD5 hash hex digest
    """
    # Sort keys for deterministic ordering
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()


def hash_dependencies(depends_on: dict) -> str:
    """
    Compute combined hash of multiple dependencies.

    Parameters
    ----------
    depends_on : dict
        Dictionary mapping names to values. Values can be:
        - Path objects (hashed by file content)
        - DataFrames (hashed by content)
        - dicts (hashed as JSON)
        - Other (converted to string and hashed)

    Returns
    -------
    str
        Combined MD5 hash hex digest
    """
    pd = _get_pandas()
    hasher = hashlib.md5()

    for name in sorted(depends_on.keys()):
        value = depends_on[name]

        if isinstance(value, Path):
            dep_hash = hash_file(value)
        elif hasattr(value, 'shape') and hasattr(value, 'columns'):
            # DataFrame-like object
            dep_hash = hash_dataframe(value)
        elif isinstance(value, dict):
            dep_hash = hash_config(value)
        else:
            dep_hash = hashlib.md5(str(value).encode()).hexdigest()

        hasher.update(f"{name}:{dep_hash}".encode())

    return hasher.hexdigest()


# =============================================================================
# CACHE MANAGER
# =============================================================================

class CacheManager:
    """
    Manages cached results for a pipeline stage.

    Provides transparent caching with automatic invalidation based on
    dependency tracking.

    Parameters
    ----------
    stage_name : str
        Name of the stage (used for cache directory)
    cache_dir : Path, optional
        Custom cache directory (default: data_work/.cache)
    enabled : bool, optional
        Whether caching is enabled (default: True)

    Examples
    --------
    >>> cache = CacheManager('s03_estimation')
    >>> result = cache.get_or_compute(
    ...     key='baseline_demeaned',
    ...     compute_fn=lambda: expensive_operation(),
    ...     depends_on={'data': panel_path}
    ... )
    >>> print(cache.stats())
    {'hits': 1, 'misses': 0, 'hit_rate': 100.0}
    """

    def __init__(
        self,
        stage_name: str,
        cache_dir: Optional[Path] = None,
        enabled: bool = True,
    ):
        self.stage_name = stage_name
        self.cache_dir = (cache_dir or _get_cache_dir()) / stage_name
        self.enabled = enabled and CACHE_ENABLED

        # Statistics
        self._hits = 0
        self._misses = 0
        self._saved_time = 0.0

        # Ensure cache directory exists
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str, dep_hash: str) -> Path:
        """Get the cache file path for a key."""
        safe_key = key.replace('/', '_').replace('\\', '_')
        return self.cache_dir / f"{safe_key}_{dep_hash[:12]}.pkl"

    def _get_metadata_path(self, key: str, dep_hash: str) -> Path:
        """Get the metadata file path for a key."""
        safe_key = key.replace('/', '_').replace('\\', '_')
        return self.cache_dir / f"{safe_key}_{dep_hash[:12]}.meta.json"

    def get(
        self,
        key: str,
        depends_on: Optional[dict] = None,
        max_age_hours: Optional[float] = None,
    ) -> tuple[bool, Any]:
        """
        Get a cached value if it exists and is valid.

        Parameters
        ----------
        key : str
            Cache key
        depends_on : dict, optional
            Dependencies to check for invalidation
        max_age_hours : float, optional
            Maximum cache age in hours

        Returns
        -------
        tuple[bool, Any]
            (found, value) - found is True if cache hit, value is the cached data
        """
        if not self.enabled:
            return False, None

        dep_hash = hash_dependencies(depends_on or {})
        cache_path = self._get_cache_path(key, dep_hash)
        meta_path = self._get_metadata_path(key, dep_hash)

        if not cache_path.exists():
            return False, None

        # Check age
        if max_age_hours is None:
            max_age_hours = CACHE_MAX_AGE_HOURS

        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                created = datetime.fromisoformat(meta.get('created', '2000-01-01'))
                age_hours = (datetime.now() - created).total_seconds() / 3600
                if age_hours > max_age_hours:
                    return False, None
            except (json.JSONDecodeError, ValueError):
                pass

        # Load cached value
        try:
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
            self._hits += 1
            return True, value
        except (pickle.PickleError, EOFError, FileNotFoundError):
            return False, None

    def set(
        self,
        key: str,
        value: Any,
        depends_on: Optional[dict] = None,
        compute_time: Optional[float] = None,
    ) -> Path:
        """
        Store a value in the cache.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache (must be picklable)
        depends_on : dict, optional
            Dependencies (stored in metadata for debugging)
        compute_time : float, optional
            Time taken to compute the value

        Returns
        -------
        Path
            Path to the cache file
        """
        if not self.enabled:
            return None

        dep_hash = hash_dependencies(depends_on or {})
        cache_path = self._get_cache_path(key, dep_hash)
        meta_path = self._get_metadata_path(key, dep_hash)

        # Save value
        with open(cache_path, 'wb') as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save metadata
        meta = {
            'key': key,
            'stage': self.stage_name,
            'created': datetime.now().isoformat(),
            'dep_hash': dep_hash,
            'compute_time_sec': compute_time,
            'size_bytes': cache_path.stat().st_size,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        return cache_path

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        depends_on: Optional[dict] = None,
        max_age_hours: Optional[float] = None,
        verbose: bool = True,
    ) -> Any:
        """
        Get cached value or compute and cache it.

        This is the main method for transparent caching.

        Parameters
        ----------
        key : str
            Cache key
        compute_fn : callable
            Function to compute the value if not cached
        depends_on : dict, optional
            Dependencies for cache invalidation
        max_age_hours : float, optional
            Maximum cache age
        verbose : bool
            Print cache hit/miss messages

        Returns
        -------
        Any
            The cached or computed value
        """
        # Try to get from cache
        found, value = self.get(key, depends_on, max_age_hours)

        if found:
            if verbose:
                print(f"    [cache hit] {key}")
            return value

        # Compute and cache
        self._misses += 1
        if verbose:
            print(f"    [cache miss] {key} - computing...")

        start_time = time.time()
        value = compute_fn()
        compute_time = time.time() - start_time

        self._saved_time += compute_time  # Track potential savings
        self.set(key, value, depends_on, compute_time)

        if verbose:
            print(f"    [cached] {key} ({compute_time:.2f}s)")

        return value

    def invalidate(self, key: str, depends_on: Optional[dict] = None) -> bool:
        """
        Invalidate a specific cache entry.

        Parameters
        ----------
        key : str
            Cache key
        depends_on : dict, optional
            Dependencies (needed to find the exact cache file)

        Returns
        -------
        bool
            True if cache entry was found and removed
        """
        if depends_on:
            dep_hash = hash_dependencies(depends_on)
            cache_path = self._get_cache_path(key, dep_hash)
            meta_path = self._get_metadata_path(key, dep_hash)

            removed = False
            if cache_path.exists():
                cache_path.unlink()
                removed = True
            if meta_path.exists():
                meta_path.unlink()

            return removed
        else:
            # Remove all caches matching the key prefix
            pattern = f"{key}_*.pkl"
            removed = False
            for path in self.cache_dir.glob(pattern):
                path.unlink()
                removed = True
                # Also remove metadata
                meta_path = path.with_suffix('.meta.json')
                if meta_path.exists():
                    meta_path.unlink()

            return removed

    def clear(self) -> int:
        """
        Clear all cache entries for this stage.

        Returns
        -------
        int
            Number of cache files removed
        """
        if not self.cache_dir.exists():
            return 0

        count = 0
        for path in self.cache_dir.glob('*.pkl'):
            path.unlink()
            count += 1
            meta_path = path.with_suffix('.meta.json')
            if meta_path.exists():
                meta_path.unlink()

        return count

    def stats(self) -> dict:
        """
        Get cache statistics.

        Returns
        -------
        dict
            Dictionary with hits, misses, hit_rate, and saved_time
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0

        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': round(hit_rate, 1),
            'saved_time_sec': round(self._saved_time, 2),
        }

    def size(self) -> dict:
        """
        Get cache size information.

        Returns
        -------
        dict
            Dictionary with file_count and total_mb
        """
        if not self.cache_dir.exists():
            return {'file_count': 0, 'total_mb': 0.0}

        files = list(self.cache_dir.glob('*.pkl'))
        total_bytes = sum(f.stat().st_size for f in files)

        return {
            'file_count': len(files),
            'total_mb': round(total_bytes / (1024 * 1024), 2),
        }


# =============================================================================
# GLOBAL CACHE OPERATIONS
# =============================================================================

def clear_all_caches() -> dict:
    """
    Clear all caches for all stages.

    Returns
    -------
    dict
        Dictionary mapping stage names to number of files cleared
    """
    cache_root = _get_cache_dir()
    if not cache_root.exists():
        return {}

    results = {}
    for stage_dir in cache_root.iterdir():
        if stage_dir.is_dir():
            count = 0
            for path in stage_dir.glob('*.pkl'):
                path.unlink()
                count += 1
                meta_path = path.with_suffix('.meta.json')
                if meta_path.exists():
                    meta_path.unlink()
            results[stage_dir.name] = count

    return results


def cache_stats_all() -> dict:
    """
    Get cache statistics for all stages.

    Returns
    -------
    dict
        Dictionary mapping stage names to size info
    """
    cache_root = _get_cache_dir()
    if not cache_root.exists():
        return {}

    results = {}
    for stage_dir in cache_root.iterdir():
        if stage_dir.is_dir():
            files = list(stage_dir.glob('*.pkl'))
            total_bytes = sum(f.stat().st_size for f in files)
            results[stage_dir.name] = {
                'file_count': len(files),
                'total_mb': round(total_bytes / (1024 * 1024), 2),
            }

    return results


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == '__main__':
    print("Cache Module Test")
    print("=" * 50)

    # Test hashing
    print("\n1. Testing hash functions...")
    test_config = {'a': 1, 'b': [1, 2, 3], 'c': 'test'}
    h1 = hash_config(test_config)
    h2 = hash_config(test_config)
    print(f"   Config hash (same input): {h1 == h2}")

    # Test cache manager
    print("\n2. Testing CacheManager...")
    cache = CacheManager('test_stage')

    def expensive_fn():
        time.sleep(0.1)
        return {'result': 42}

    # First call (miss)
    result1 = cache.get_or_compute(
        key='test_key',
        compute_fn=expensive_fn,
        depends_on={'config': test_config},
    )

    # Second call (hit)
    result2 = cache.get_or_compute(
        key='test_key',
        compute_fn=expensive_fn,
        depends_on={'config': test_config},
    )

    print(f"   Results match: {result1 == result2}")
    print(f"   Stats: {cache.stats()}")

    # Cleanup
    cache.clear()
    print("\n   Test cache cleared.")
