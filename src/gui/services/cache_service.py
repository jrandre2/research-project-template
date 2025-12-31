"""
Cache service for managing pipeline cache.

Provides access to cache statistics and management operations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import DATA_WORK_DIR
from ..models import CacheStats


class CacheService:
    """Service for cache management."""

    def __init__(self):
        self.cache_dir = DATA_WORK_DIR / ".cache"

    def get_stats(self, stage: Optional[str] = None) -> list[CacheStats]:
        """
        Get cache statistics for all stages or a specific stage.

        Parameters
        ----------
        stage : str, optional
            Filter to specific stage.

        Returns
        -------
        list[CacheStats]
            Cache statistics per stage.
        """
        if not self.cache_dir.exists():
            return []

        stats = []

        if stage:
            stage_dir = self.cache_dir / stage
            if stage_dir.exists():
                stats.append(self._get_stage_stats(stage, stage_dir))
        else:
            for stage_dir in sorted(self.cache_dir.iterdir()):
                if stage_dir.is_dir():
                    stats.append(self._get_stage_stats(stage_dir.name, stage_dir))

        return stats

    def _get_stage_stats(self, stage_name: str, stage_dir: Path) -> CacheStats:
        """Get cache stats for a single stage."""
        files = list(stage_dir.glob("*.pkl"))
        total_bytes = sum(f.stat().st_size for f in files if f.exists())

        oldest = None
        newest = None
        for f in files:
            if f.exists():
                mtime = f.stat().st_mtime
                from datetime import datetime
                dt = datetime.fromtimestamp(mtime)
                if oldest is None or dt < oldest:
                    oldest = dt
                if newest is None or dt > newest:
                    newest = dt

        return CacheStats(
            stage=stage_name,
            file_count=len(files),
            total_mb=total_bytes / (1024 * 1024),
            oldest_entry=oldest,
            newest_entry=newest,
        )

    def clear(self, stage: Optional[str] = None) -> int:
        """
        Clear cache for a stage or all stages.

        Parameters
        ----------
        stage : str, optional
            Stage to clear. If None, clears all.

        Returns
        -------
        int
            Number of files deleted.
        """
        if not self.cache_dir.exists():
            return 0

        count = 0

        if stage:
            stage_dir = self.cache_dir / stage
            if stage_dir.exists():
                for f in stage_dir.glob("*"):
                    if f.is_file():
                        f.unlink()
                        count += 1
        else:
            for stage_dir in self.cache_dir.iterdir():
                if stage_dir.is_dir():
                    for f in stage_dir.glob("*"):
                        if f.is_file():
                            f.unlink()
                            count += 1

        return count

    def get_total_stats(self) -> CacheStats:
        """Get aggregate cache stats across all stages."""
        all_stats = self.get_stats()

        return CacheStats(
            stage=None,
            file_count=sum(s.file_count for s in all_stats),
            total_mb=sum(s.total_mb for s in all_stats),
        )


# Singleton instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get the cache service singleton."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
