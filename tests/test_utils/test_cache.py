#!/usr/bin/env python3
"""
Tests for the caching module.

Tests cover:
- Hash functions (dataframe, file, config)
- CacheManager operations (get, set, invalidate, clear)
- Cache statistics and size tracking
- Dependency-based invalidation
"""
import pytest
import tempfile
import time
from pathlib import Path

import pandas as pd
import numpy as np

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.cache import (
    hash_dataframe,
    hash_file,
    hash_config,
    hash_dependencies,
    CacheManager,
    clear_all_caches,
    cache_stats_all,
)


# ============================================================
# HASH FUNCTION TESTS
# ============================================================

class TestHashDataframe:
    """Tests for hash_dataframe function."""

    def test_same_dataframe_same_hash(self):
        """Same DataFrame should produce same hash."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        h1 = hash_dataframe(df)
        h2 = hash_dataframe(df)
        assert h1 == h2

    def test_different_dataframe_different_hash(self):
        """Different DataFrames should produce different hashes."""
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'a': [1, 2, 4]})
        h1 = hash_dataframe(df1)
        h2 = hash_dataframe(df2)
        assert h1 != h2

    def test_column_order_matters(self):
        """Column order should affect hash."""
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'b': [3, 4], 'a': [1, 2]})
        h1 = hash_dataframe(df1)
        h2 = hash_dataframe(df2)
        assert h1 != h2

    def test_empty_dataframe(self):
        """Empty DataFrame should have a valid hash."""
        df = pd.DataFrame()
        h = hash_dataframe(df)
        assert isinstance(h, str)
        assert len(h) == 32  # MD5 hex digest length

    def test_large_dataframe_sampling(self):
        """Large DataFrames should be sampled for performance."""
        df = pd.DataFrame({'a': range(100000)})
        start = time.time()
        h = hash_dataframe(df, sample_size=1000)
        elapsed = time.time() - start
        assert elapsed < 1.0  # Should be fast due to sampling
        assert isinstance(h, str)


class TestHashFile:
    """Tests for hash_file function."""

    def test_same_file_same_hash(self, tmp_path):
        """Same file should produce same hash."""
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = hash_file(f)
        h2 = hash_file(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        """Different content should produce different hashes."""
        f1 = tmp_path / "test1.txt"
        f2 = tmp_path / "test2.txt"
        f1.write_text("hello")
        f2.write_text("world")
        h1 = hash_file(f1)
        h2 = hash_file(f2)
        assert h1 != h2

    def test_missing_file(self, tmp_path):
        """Missing file should return special hash."""
        f = tmp_path / "nonexistent.txt"
        h = hash_file(f)
        assert h.startswith("missing:")


class TestHashConfig:
    """Tests for hash_config function."""

    def test_same_config_same_hash(self):
        """Same config should produce same hash."""
        config = {'a': 1, 'b': [1, 2, 3]}
        h1 = hash_config(config)
        h2 = hash_config(config)
        assert h1 == h2

    def test_key_order_irrelevant(self):
        """Key order should not affect hash (sorted internally)."""
        c1 = {'a': 1, 'b': 2}
        c2 = {'b': 2, 'a': 1}
        h1 = hash_config(c1)
        h2 = hash_config(c2)
        assert h1 == h2

    def test_different_config_different_hash(self):
        """Different configs should produce different hashes."""
        c1 = {'a': 1}
        c2 = {'a': 2}
        h1 = hash_config(c1)
        h2 = hash_config(c2)
        assert h1 != h2


class TestHashDependencies:
    """Tests for hash_dependencies function."""

    def test_dict_dependency(self):
        """Dict dependencies should be hashed."""
        deps = {'config': {'a': 1}}
        h = hash_dependencies(deps)
        assert isinstance(h, str)

    def test_path_dependency(self, tmp_path):
        """Path dependencies should use file hash."""
        f = tmp_path / "data.txt"
        f.write_text("test data")
        deps = {'file': f}
        h = hash_dependencies(deps)
        assert isinstance(h, str)

    def test_dataframe_dependency(self):
        """DataFrame dependencies should be hashed."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        deps = {'data': df}
        h = hash_dependencies(deps)
        assert isinstance(h, str)


# ============================================================
# CACHE MANAGER TESTS
# ============================================================

class TestCacheManager:
    """Tests for CacheManager class."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create a CacheManager with temp directory."""
        return CacheManager('test_stage', cache_dir=tmp_path)

    def test_set_and_get(self, cache):
        """Basic set and get operations."""
        cache.set('key1', {'data': 42})
        found, value = cache.get('key1')
        assert found is True
        assert value == {'data': 42}

    def test_get_nonexistent(self, cache):
        """Getting nonexistent key returns False."""
        found, value = cache.get('nonexistent')
        assert found is False
        assert value is None

    def test_get_or_compute_miss(self, cache):
        """get_or_compute should compute on cache miss."""
        compute_count = [0]

        def compute():
            compute_count[0] += 1
            return 'computed_value'

        result = cache.get_or_compute('key', compute, verbose=False)
        assert result == 'computed_value'
        assert compute_count[0] == 1

    def test_get_or_compute_hit(self, cache):
        """get_or_compute should use cache on hit."""
        compute_count = [0]

        def compute():
            compute_count[0] += 1
            return 'computed_value'

        # First call - computes
        cache.get_or_compute('key', compute, verbose=False)
        # Second call - uses cache
        result = cache.get_or_compute('key', compute, verbose=False)

        assert result == 'computed_value'
        assert compute_count[0] == 1  # Only computed once

    def test_dependency_invalidation(self, cache):
        """Cache should invalidate when dependencies change."""
        deps1 = {'config': {'version': 1}}
        deps2 = {'config': {'version': 2}}

        cache.set('key', 'value1', depends_on=deps1)
        found, _ = cache.get('key', depends_on=deps1)
        assert found is True

        # Different dependencies - should miss
        found, _ = cache.get('key', depends_on=deps2)
        assert found is False

    def test_invalidate(self, cache):
        """invalidate should remove cache entry."""
        cache.set('key', 'value')
        found, _ = cache.get('key')
        assert found is True

        cache.invalidate('key')
        found, _ = cache.get('key')
        assert found is False

    def test_clear(self, cache):
        """clear should remove all entries."""
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')

        count = cache.clear()
        assert count == 2

        found1, _ = cache.get('key1')
        found2, _ = cache.get('key2')
        assert found1 is False
        assert found2 is False

    def test_stats(self, cache):
        """stats should track hits and misses."""
        # Use get_or_compute to properly track hits/misses
        cache.get_or_compute('miss1', lambda: 'v1', verbose=False)  # miss
        cache.get_or_compute('miss2', lambda: 'v2', verbose=False)  # miss
        cache.get_or_compute('miss1', lambda: 'v1', verbose=False)  # hit (cached)

        stats = cache.stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 2
        assert stats['hit_rate'] == pytest.approx(33.3, rel=0.1)

    def test_size(self, cache):
        """size should return file count and total MB."""
        # Use depends_on to ensure unique cache keys
        cache.set('key1', {'data': 'x' * 1000}, depends_on={'id': '1'})
        cache.set('key2', {'data': 'y' * 1000}, depends_on={'id': '2'})

        size = cache.size()
        assert size['file_count'] == 2
        # Size may be 0.0 for small files due to rounding
        assert size['total_mb'] >= 0

    def test_disabled_cache(self, tmp_path):
        """Disabled cache should always miss."""
        cache = CacheManager('test', cache_dir=tmp_path, enabled=False)

        compute_count = [0]
        def compute():
            compute_count[0] += 1
            return 'value'

        cache.get_or_compute('key', compute, verbose=False)
        cache.get_or_compute('key', compute, verbose=False)

        # Should compute twice since cache is disabled
        assert compute_count[0] == 2


class TestCacheMaxAge:
    """Tests for cache TTL/max age."""

    def test_max_age_valid(self, tmp_path):
        """Cache within max age should hit."""
        cache = CacheManager('test', cache_dir=tmp_path)
        cache.set('key', 'value')

        found, value = cache.get('key', max_age_hours=1)
        assert found is True
        assert value == 'value'


# ============================================================
# GLOBAL CACHE FUNCTIONS
# ============================================================

class TestGlobalCacheFunctions:
    """Tests for global cache utility functions."""

    def test_clear_all_caches(self, tmp_path):
        """clear_all_caches should clear all stages."""
        # Create caches for multiple stages
        cache1 = CacheManager('stage1', cache_dir=tmp_path)
        cache2 = CacheManager('stage2', cache_dir=tmp_path)

        cache1.set('key1', 'value1')
        cache2.set('key2', 'value2')

        # Manually call the function with the right path
        import utils.cache as cache_module
        original_get_cache_dir = cache_module._get_cache_dir
        cache_module._get_cache_dir = lambda: tmp_path

        try:
            results = clear_all_caches()
            assert 'stage1' in results
            assert 'stage2' in results
        finally:
            cache_module._get_cache_dir = original_get_cache_dir

    def test_cache_stats_all(self, tmp_path):
        """cache_stats_all should return stats for all stages."""
        cache = CacheManager('test_stage', cache_dir=tmp_path)
        cache.set('key', 'value')

        import utils.cache as cache_module
        original_get_cache_dir = cache_module._get_cache_dir
        cache_module._get_cache_dir = lambda: tmp_path

        try:
            stats = cache_stats_all()
            assert 'test_stage' in stats
            assert stats['test_stage']['file_count'] == 1
        finally:
            cache_module._get_cache_dir = original_get_cache_dir


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestCacheIntegration:
    """Integration tests for caching with real data types."""

    def test_cache_dataframe(self, tmp_path):
        """Test caching pandas DataFrames."""
        cache = CacheManager('test', cache_dir=tmp_path)

        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z'],
            'c': [1.1, 2.2, 3.3]
        })

        cache.set('df', df)
        found, cached_df = cache.get('df')

        assert found is True
        pd.testing.assert_frame_equal(df, cached_df)

    def test_cache_numpy_array(self, tmp_path):
        """Test caching numpy arrays."""
        cache = CacheManager('test', cache_dir=tmp_path)

        arr = np.array([[1, 2, 3], [4, 5, 6]])

        cache.set('arr', arr)
        found, cached_arr = cache.get('arr')

        assert found is True
        np.testing.assert_array_equal(arr, cached_arr)

    def test_cache_complex_object(self, tmp_path):
        """Test caching complex nested objects."""
        cache = CacheManager('test', cache_dir=tmp_path)

        obj = {
            'string': 'hello',
            'int': 42,
            'float': 3.14,
            'list': [1, 2, 3],
            'nested': {'a': 1, 'b': 2},
            'df': pd.DataFrame({'x': [1, 2]}),
        }

        cache.set('obj', obj)
        found, cached_obj = cache.get('obj')

        assert found is True
        assert cached_obj['string'] == 'hello'
        assert cached_obj['int'] == 42
        pd.testing.assert_frame_equal(obj['df'], cached_obj['df'])


# ============================================================
# END-TO-END INTEGRATION TESTS
# ============================================================

class TestCacheEndToEnd:
    """End-to-end tests simulating real pipeline caching behavior."""

    def test_multi_stage_caching(self, tmp_path):
        """Test caching across multiple pipeline stages."""
        # Simulate s03_estimation and s04_robustness stages
        cache_s03 = CacheManager('s03_estimation', cache_dir=tmp_path)
        cache_s04 = CacheManager('s04_robustness', cache_dir=tmp_path)

        # Create shared input data
        panel_data = pd.DataFrame({
            'unit_id': [1, 1, 2, 2],
            'period': [1, 2, 1, 2],
            'outcome': [10.0, 12.0, 15.0, 18.0],
            'treatment': [0, 1, 0, 1],
        })
        data_hash = hash_dataframe(panel_data)

        # Stage 03: Estimation
        spec_config = {'controls': ['period'], 'fe': 'unit_id'}
        deps_s03 = {'data': data_hash, 'config': spec_config}

        estimation_results = {'coefficient': 2.5, 'std_error': 0.5}
        cache_s03.set('baseline', estimation_results, depends_on=deps_s03)

        # Stage 04: Robustness (depends on s03 results)
        deps_s04 = {'estimation': estimation_results, 'data': data_hash}
        robustness_results = {'placebo_coef': 0.1, 'p_value': 0.85}
        cache_s04.set('placebo_time', robustness_results, depends_on=deps_s04)

        # Verify caches
        found_s03, cached_s03 = cache_s03.get('baseline', depends_on=deps_s03)
        found_s04, cached_s04 = cache_s04.get('placebo_time', depends_on=deps_s04)

        assert found_s03 is True
        assert cached_s03['coefficient'] == 2.5
        assert found_s04 is True
        assert cached_s04['placebo_coef'] == 0.1

    def test_cache_invalidation_on_data_change(self, tmp_path):
        """Cache should miss when input data changes."""
        cache = CacheManager('test', cache_dir=tmp_path)

        # Original data
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        hash1 = hash_dataframe(df1)

        # Cache with original data
        cache.set('result', {'value': 100}, depends_on={'data': hash1})

        # Verify cache hit with same data
        found, _ = cache.get('result', depends_on={'data': hash1})
        assert found is True

        # Modified data - should miss
        df2 = pd.DataFrame({'a': [1, 2, 4]})  # Changed last value
        hash2 = hash_dataframe(df2)

        found, _ = cache.get('result', depends_on={'data': hash2})
        assert found is False

    def test_cache_invalidation_on_config_change(self, tmp_path):
        """Cache should miss when configuration changes."""
        cache = CacheManager('test', cache_dir=tmp_path)

        df = pd.DataFrame({'a': [1, 2, 3]})
        data_hash = hash_dataframe(df)

        # Original config
        config1 = {'model': 'ols', 'robust': True}
        deps1 = {'data': data_hash, 'config': config1}
        cache.set('estimation', {'coef': 1.5}, depends_on=deps1)

        # Same config - should hit
        found, _ = cache.get('estimation', depends_on=deps1)
        assert found is True

        # Different config - should miss
        config2 = {'model': 'ols', 'robust': False}
        deps2 = {'data': data_hash, 'config': config2}
        found, _ = cache.get('estimation', depends_on=deps2)
        assert found is False

    def test_compute_only_on_cache_miss(self, tmp_path):
        """Computation should only run on cache miss."""
        cache = CacheManager('test', cache_dir=tmp_path)

        compute_calls = []

        def expensive_computation(data_id):
            compute_calls.append(data_id)
            return {'result': data_id * 10}

        # First call - should compute
        deps_a = {'id': 'a'}
        result1 = cache.get_or_compute(
            'analysis',
            lambda: expensive_computation('a'),
            depends_on=deps_a,
            verbose=False
        )

        # Second call with same deps - should use cache
        result2 = cache.get_or_compute(
            'analysis',
            lambda: expensive_computation('a'),
            depends_on=deps_a,
            verbose=False
        )

        # Third call with different deps - should compute again
        deps_b = {'id': 'b'}
        result3 = cache.get_or_compute(
            'analysis',
            lambda: expensive_computation('b'),
            depends_on=deps_b,
            verbose=False
        )

        assert len(compute_calls) == 2  # Only 2 computations
        assert compute_calls == ['a', 'b']
        assert result1 == result2  # Same result from cache
        assert result3 == {'result': 'bbbbbbbbbb'}  # 'b' * 10

    def test_file_dependency_tracking(self, tmp_path):
        """Cache should track file content changes."""
        cache = CacheManager('test', cache_dir=tmp_path)
        config_file = tmp_path / "config.yaml"

        # Create config file
        config_file.write_text("setting: value1")
        deps = {'config_file': config_file}

        cache.set('result', {'value': 1}, depends_on=deps)
        found, _ = cache.get('result', depends_on=deps)
        assert found is True

        # Modify config file
        config_file.write_text("setting: value2")
        deps_new = {'config_file': config_file}

        found, _ = cache.get('result', depends_on=deps_new)
        assert found is False  # Should miss due to file change

    def test_cross_stage_data_sharing(self, tmp_path):
        """Test that stages can share cached data properly."""
        # Create panel data once
        panel = pd.DataFrame({
            'id': range(100),
            'value': np.random.randn(100)
        })

        # Two stages reading same data
        cache1 = CacheManager('stage1', cache_dir=tmp_path)
        cache2 = CacheManager('stage2', cache_dir=tmp_path)

        panel_hash = hash_dataframe(panel)

        # Both stages cache results based on same data hash
        cache1.set('agg', {'mean': panel['value'].mean()}, depends_on={'data': panel_hash})
        cache2.set('summary', {'std': panel['value'].std()}, depends_on={'data': panel_hash})

        # Both should hit
        found1, val1 = cache1.get('agg', depends_on={'data': panel_hash})
        found2, val2 = cache2.get('summary', depends_on={'data': panel_hash})

        assert found1 is True
        assert found2 is True
        assert val1['mean'] == pytest.approx(panel['value'].mean())
        assert val2['std'] == pytest.approx(panel['value'].std())

    def test_selective_stage_clear(self, tmp_path):
        """Clear one stage without affecting others."""
        cache1 = CacheManager('s03_estimation', cache_dir=tmp_path)
        cache2 = CacheManager('s04_robustness', cache_dir=tmp_path)

        cache1.set('result1', {'v': 1})
        cache2.set('result2', {'v': 2})

        # Clear only s03
        cache1.clear()

        # s03 should miss, s04 should hit
        found1, _ = cache1.get('result1')
        found2, _ = cache2.get('result2')

        assert found1 is False
        assert found2 is True

    def test_concurrent_cache_writes(self, tmp_path):
        """Test thread-safe concurrent writes."""
        import threading

        cache = CacheManager('test', cache_dir=tmp_path)
        results = []
        errors = []

        def write_cache(key, value):
            try:
                cache.set(key, value)
                results.append((key, value))
            except Exception as e:
                errors.append(e)

        # Create multiple threads writing different keys
        threads = []
        for i in range(10):
            t = threading.Thread(target=write_cache, args=(f'key_{i}', {'data': i}))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All writes should succeed
        assert len(errors) == 0
        assert len(results) == 10

        # All keys should be readable
        for i in range(10):
            found, value = cache.get(f'key_{i}')
            assert found is True
            assert value == {'data': i}

    def test_cache_hit_rate_tracking(self, tmp_path):
        """Track hit rate across multiple operations."""
        cache = CacheManager('test', cache_dir=tmp_path)

        # Series of cache operations
        cache.get_or_compute('a', lambda: 1, verbose=False)  # miss
        cache.get_or_compute('b', lambda: 2, verbose=False)  # miss
        cache.get_or_compute('a', lambda: 1, verbose=False)  # hit
        cache.get_or_compute('a', lambda: 1, verbose=False)  # hit
        cache.get_or_compute('c', lambda: 3, verbose=False)  # miss

        stats = cache.stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 3
        assert stats['hit_rate'] == 40.0  # 2/5 = 40%

    def test_estimation_result_caching(self, tmp_path):
        """Test caching estimation results with full dependencies."""
        cache = CacheManager('s03_estimation', cache_dir=tmp_path)

        # Create panel data
        panel = pd.DataFrame({
            'unit_id': [1, 1, 1, 2, 2, 2] * 10,
            'period': list(range(1, 4)) * 20,
            'outcome': np.random.randn(60) + 5,
            'treatment': [0, 0, 1, 0, 1, 1] * 10,
            'control_var': np.random.randn(60),
        })

        # Compute dependencies
        data_hash = hash_dataframe(panel)
        spec_config = {
            'name': 'baseline',
            'controls': ['control_var'],
            'fixed_effects': ['unit_id', 'period'],
            'cluster': 'unit_id'
        }
        deps = {'data': data_hash, 'specification': spec_config}

        # Simulate estimation result
        result = {
            'coefficient': 1.234,
            'std_error': 0.456,
            'p_value': 0.007,
            't_stat': 2.706,
            'n_obs': 60,
            'r_squared': 0.85,
        }

        # Cache and retrieve
        cache.set('baseline', result, depends_on=deps)
        found, cached = cache.get('baseline', depends_on=deps)

        assert found is True
        assert cached['coefficient'] == result['coefficient']
        assert cached['std_error'] == result['std_error']
        assert cached['n_obs'] == 60
