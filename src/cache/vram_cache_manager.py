"""
VBUS VRAM/RAM Cache Management System
Turns GPU VRAM into L1 cache and system RAM into L2/L3 cache
Manages throughput between memory tiers during ML operations
"""

import os
import sys
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
from enum import Enum
import json
import logging
from pathlib import Path
from collections import OrderedDict
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VBUS.CacheManager")


class CacheTier(Enum):
    """Memory cache tiers"""
    L1_VRAM = 1      # GPU VRAM - fastest
    L2_RAM = 2       # System RAM - fast
    L3_DISK = 3      # Disk storage - persistent


class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"       # Least Recently Used
    LFU = "lfu"       # Least Frequently Used
    FIFO = "fifo"     # First In First Out
    ADAPTIVE = "adaptive"  # ML-based adaptive policy


@dataclass
class CacheEntry:
    """Represents a cached data entry"""
    key: str
    size_bytes: int
    tier: CacheTier
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    data_type: str = "tensor"  # tensor, dataframe, array, model
    pinned: bool = False  # If True, won't be evicted
    checksum: str = ""

    def touch(self):
        """Update access time and count"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0
    evictions: int = 0
    promotions: int = 0
    demotions: int = 0
    total_bytes_transferred: int = 0

    @property
    def l1_hit_rate(self) -> float:
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def overall_hit_rate(self) -> float:
        total_hits = self.l1_hits + self.l2_hits + self.l3_hits
        total_misses = self.l1_misses + self.l2_misses + self.l3_misses
        total = total_hits + total_misses
        return total_hits / total if total > 0 else 0.0


class VRAMManager:
    """Manages GPU VRAM allocation and monitoring"""

    def __init__(self):
        self.gpu_available = False
        self.total_vram_bytes = 0
        self.used_vram_bytes = 0
        self._initialize_gpu()

    def _initialize_gpu(self):
        """Initialize GPU monitoring"""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.total_vram_bytes = mem_info.total
                self.used_vram_bytes = mem_info.used
                self.gpu_available = True
                logger.info(f"GPU initialized. VRAM: {self.total_vram_bytes / (1024**3):.2f} GB")
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}. VRAM caching disabled.")
            self.gpu_available = False

    def get_free_vram(self) -> int:
        """Get available VRAM in bytes"""
        if not self.gpu_available:
            return 0
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem_info.free
        except Exception:
            return 0

    def allocate_vram(self, size_bytes: int) -> bool:
        """Check if VRAM allocation is possible"""
        return self.get_free_vram() >= size_bytes

    def get_utilization(self) -> float:
        """Get VRAM utilization percentage"""
        if not self.gpu_available or self.total_vram_bytes == 0:
            return 0.0
        try:
            import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem_info.used / mem_info.total * 100
        except Exception:
            return 0.0


class RAMManager:
    """Manages system RAM allocation and monitoring"""

    def __init__(self, max_usage_percent: float = 70.0):
        self.max_usage_percent = max_usage_percent
        self._initialize_ram()

    def _initialize_ram(self):
        """Initialize RAM monitoring"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            self.total_ram_bytes = mem.total
            logger.info(f"RAM initialized. Total: {self.total_ram_bytes / (1024**3):.2f} GB")
        except ImportError:
            self.total_ram_bytes = 16 * (1024**3)  # Assume 16GB
            logger.warning("psutil not available. Assuming 16GB RAM.")

    def get_free_ram(self) -> int:
        """Get available RAM in bytes"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            # Respect max usage limit
            max_usable = int(self.total_ram_bytes * self.max_usage_percent / 100)
            currently_used_by_system = mem.total - mem.available
            return max(0, max_usable - currently_used_by_system)
        except ImportError:
            return int(self.total_ram_bytes * 0.5)  # Assume 50% available

    def allocate_ram(self, size_bytes: int) -> bool:
        """Check if RAM allocation is possible"""
        return self.get_free_ram() >= size_bytes

    def get_utilization(self) -> float:
        """Get RAM utilization percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 50.0


class TieredCache:
    """Multi-tier cache with VRAM (L1), RAM (L2), and Disk (L3)"""

    def __init__(
        self,
        l1_max_bytes: int = 0,  # 0 = auto-detect
        l2_max_bytes: int = 0,
        l3_path: str = None,
        policy: CachePolicy = CachePolicy.LRU
    ):
        self.vram_manager = VRAMManager()
        self.ram_manager = RAMManager()

        # Set cache sizes
        self.l1_max_bytes = l1_max_bytes or int(self.vram_manager.total_vram_bytes * 0.6)
        self.l2_max_bytes = l2_max_bytes or int(self.ram_manager.total_ram_bytes * 0.3)
        self.l3_path = Path(l3_path) if l3_path else Path.home() / ".vbus" / "cache"
        self.l3_path.mkdir(parents=True, exist_ok=True)

        self.policy = policy
        self.stats = CacheStatistics()

        # Cache storage (key -> CacheEntry metadata)
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l2_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l3_cache: Dict[str, CacheEntry] = {}

        # Actual data storage (in RAM)
        self._l2_data: Dict[str, Any] = {}

        # Current sizes
        self.l1_current_bytes = 0
        self.l2_current_bytes = 0

        # Threading lock for thread safety
        self._lock = threading.RLock()

        # Background worker for async operations
        self._work_queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._background_worker, daemon=True)
        self._worker_thread.start()

        logger.info(f"TieredCache initialized: L1={self.l1_max_bytes/(1024**3):.2f}GB, "
                   f"L2={self.l2_max_bytes/(1024**3):.2f}GB")

    def _background_worker(self):
        """Background worker for async cache operations"""
        while True:
            try:
                task = self._work_queue.get(timeout=1)
                if task is None:
                    break
                func, args, kwargs = task
                func(*args, **kwargs)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Background worker error: {e}")

    def _compute_checksum(self, data: bytes) -> str:
        """Compute checksum for data integrity"""
        return hashlib.md5(data).hexdigest()

    def _get_data_size(self, data: Any) -> int:
        """Estimate size of data in bytes"""
        try:
            # Try numpy array
            if hasattr(data, 'nbytes'):
                return data.nbytes
            # Try torch tensor
            if hasattr(data, 'element_size') and hasattr(data, 'nelement'):
                return data.element_size() * data.nelement()
            # Try pandas DataFrame
            if hasattr(data, 'memory_usage'):
                return data.memory_usage(deep=True).sum()
            # Fallback to sys.getsizeof
            return sys.getsizeof(data)
        except Exception:
            return 1024  # Default 1KB if can't determine

    def _evict_from_l1(self, required_bytes: int) -> int:
        """Evict entries from L1 (VRAM) to make room"""
        freed = 0
        entries_to_evict = []

        with self._lock:
            if self.policy == CachePolicy.LRU:
                # Evict least recently used
                for key, entry in list(self.l1_cache.items()):
                    if entry.pinned:
                        continue
                    entries_to_evict.append(key)
                    freed += entry.size_bytes
                    if freed >= required_bytes:
                        break
            elif self.policy == CachePolicy.LFU:
                # Evict least frequently used
                sorted_entries = sorted(
                    [(k, e) for k, e in self.l1_cache.items() if not e.pinned],
                    key=lambda x: x[1].access_count
                )
                for key, entry in sorted_entries:
                    entries_to_evict.append(key)
                    freed += entry.size_bytes
                    if freed >= required_bytes:
                        break

            # Perform eviction - demote to L2
            for key in entries_to_evict:
                entry = self.l1_cache.pop(key)
                self.l1_current_bytes -= entry.size_bytes
                # Demote to L2 (data needs to be retrieved from GPU first)
                entry.tier = CacheTier.L2_RAM
                self.l2_cache[key] = entry
                self.stats.demotions += 1
                self.stats.evictions += 1
                logger.debug(f"Evicted {key} from L1 to L2")

        return freed

    def _evict_from_l2(self, required_bytes: int) -> int:
        """Evict entries from L2 (RAM) to L3 (disk)"""
        freed = 0
        entries_to_evict = []

        with self._lock:
            if self.policy == CachePolicy.LRU:
                for key, entry in list(self.l2_cache.items()):
                    if entry.pinned:
                        continue
                    entries_to_evict.append(key)
                    freed += entry.size_bytes
                    if freed >= required_bytes:
                        break

            for key in entries_to_evict:
                entry = self.l2_cache.pop(key)
                self.l2_current_bytes -= entry.size_bytes

                # Save to disk if data exists
                if key in self._l2_data:
                    data = self._l2_data.pop(key)
                    self._save_to_disk(key, data, entry)

                entry.tier = CacheTier.L3_DISK
                self.l3_cache[key] = entry
                self.stats.demotions += 1
                self.stats.evictions += 1
                logger.debug(f"Evicted {key} from L2 to L3")

        return freed

    def _save_to_disk(self, key: str, data: Any, entry: CacheEntry):
        """Save data to disk cache"""
        try:
            import pickle
            file_path = self.l3_path / f"{key}.cache"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            entry.location = str(file_path)
        except Exception as e:
            logger.error(f"Failed to save {key} to disk: {e}")

    def _load_from_disk(self, key: str) -> Any:
        """Load data from disk cache"""
        try:
            import pickle
            file_path = self.l3_path / f"{key}.cache"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load {key} from disk: {e}")
        return None

    def put(
        self,
        key: str,
        data: Any,
        data_type: str = "tensor",
        prefer_tier: CacheTier = CacheTier.L1_VRAM,
        pinned: bool = False
    ) -> bool:
        """
        Store data in the cache.

        Args:
            key: Unique identifier for the data
            data: The data to cache
            data_type: Type of data (tensor, dataframe, array, model)
            prefer_tier: Preferred cache tier
            pinned: If True, entry won't be evicted

        Returns:
            bool: True if successful
        """
        size_bytes = self._get_data_size(data)

        with self._lock:
            # Remove existing entry if present
            self.remove(key)

            entry = CacheEntry(
                key=key,
                size_bytes=size_bytes,
                tier=prefer_tier,
                data_type=data_type,
                pinned=pinned
            )

            # Try to place in preferred tier
            if prefer_tier == CacheTier.L1_VRAM and self.vram_manager.gpu_available:
                if self.l1_current_bytes + size_bytes > self.l1_max_bytes:
                    self._evict_from_l1(size_bytes)

                if self.l1_current_bytes + size_bytes <= self.l1_max_bytes:
                    self.l1_cache[key] = entry
                    self.l1_current_bytes += size_bytes
                    # Note: Actual GPU transfer would happen here
                    # For now, we store in L2 data and track in L1 metadata
                    self._l2_data[key] = data
                    logger.debug(f"Stored {key} in L1 ({size_bytes} bytes)")
                    return True

            # Fallback to L2
            if self.l2_current_bytes + size_bytes > self.l2_max_bytes:
                self._evict_from_l2(size_bytes)

            if self.l2_current_bytes + size_bytes <= self.l2_max_bytes:
                entry.tier = CacheTier.L2_RAM
                self.l2_cache[key] = entry
                self.l2_current_bytes += size_bytes
                self._l2_data[key] = data
                logger.debug(f"Stored {key} in L2 ({size_bytes} bytes)")
                return True

            # Fallback to L3 (disk)
            entry.tier = CacheTier.L3_DISK
            self._save_to_disk(key, data, entry)
            self.l3_cache[key] = entry
            logger.debug(f"Stored {key} in L3 ({size_bytes} bytes)")
            return True

    def get(self, key: str, promote: bool = True) -> Optional[Any]:
        """
        Retrieve data from the cache.

        Args:
            key: Unique identifier
            promote: Whether to promote data to higher tier on access

        Returns:
            The cached data or None if not found
        """
        with self._lock:
            # Check L1
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                entry.touch()
                self.l1_cache.move_to_end(key)  # LRU update
                self.stats.l1_hits += 1
                return self._l2_data.get(key)

            self.stats.l1_misses += 1

            # Check L2
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                entry.touch()
                self.l2_cache.move_to_end(key)
                self.stats.l2_hits += 1

                data = self._l2_data.get(key)

                # Promote to L1 if requested and space available
                if promote and self.vram_manager.gpu_available:
                    if self.l1_current_bytes + entry.size_bytes <= self.l1_max_bytes:
                        self.l2_cache.pop(key)
                        self.l2_current_bytes -= entry.size_bytes
                        entry.tier = CacheTier.L1_VRAM
                        self.l1_cache[key] = entry
                        self.l1_current_bytes += entry.size_bytes
                        self.stats.promotions += 1

                return data

            self.stats.l2_misses += 1

            # Check L3
            if key in self.l3_cache:
                entry = self.l3_cache[key]
                entry.touch()
                self.stats.l3_hits += 1

                data = self._load_from_disk(key)
                if data is not None and promote:
                    # Promote to L2
                    self.l3_cache.pop(key)
                    entry.tier = CacheTier.L2_RAM
                    self._l2_data[key] = data
                    self.l2_cache[key] = entry
                    self.l2_current_bytes += entry.size_bytes
                    self.stats.promotions += 1

                return data

            self.stats.l3_misses += 1
            return None

    def remove(self, key: str) -> bool:
        """Remove an entry from all cache tiers"""
        with self._lock:
            removed = False

            if key in self.l1_cache:
                entry = self.l1_cache.pop(key)
                self.l1_current_bytes -= entry.size_bytes
                removed = True

            if key in self.l2_cache:
                entry = self.l2_cache.pop(key)
                self.l2_current_bytes -= entry.size_bytes
                removed = True

            if key in self._l2_data:
                del self._l2_data[key]

            if key in self.l3_cache:
                entry = self.l3_cache.pop(key)
                file_path = self.l3_path / f"{key}.cache"
                if file_path.exists():
                    file_path.unlink()
                removed = True

            return removed

    def clear(self, tier: Optional[CacheTier] = None):
        """Clear cache (specific tier or all)"""
        with self._lock:
            if tier is None or tier == CacheTier.L1_VRAM:
                self.l1_cache.clear()
                self.l1_current_bytes = 0

            if tier is None or tier == CacheTier.L2_RAM:
                self.l2_cache.clear()
                self._l2_data.clear()
                self.l2_current_bytes = 0

            if tier is None or tier == CacheTier.L3_DISK:
                self.l3_cache.clear()
                # Clear disk cache files
                for f in self.l3_path.glob("*.cache"):
                    f.unlink()

    def get_status(self) -> Dict[str, Any]:
        """Get cache status"""
        return {
            "l1": {
                "used_bytes": self.l1_current_bytes,
                "max_bytes": self.l1_max_bytes,
                "utilization": self.l1_current_bytes / self.l1_max_bytes * 100 if self.l1_max_bytes > 0 else 0,
                "entries": len(self.l1_cache),
                "gpu_available": self.vram_manager.gpu_available
            },
            "l2": {
                "used_bytes": self.l2_current_bytes,
                "max_bytes": self.l2_max_bytes,
                "utilization": self.l2_current_bytes / self.l2_max_bytes * 100 if self.l2_max_bytes > 0 else 0,
                "entries": len(self.l2_cache)
            },
            "l3": {
                "path": str(self.l3_path),
                "entries": len(self.l3_cache)
            },
            "statistics": {
                "l1_hit_rate": f"{self.stats.l1_hit_rate * 100:.2f}%",
                "overall_hit_rate": f"{self.stats.overall_hit_rate * 100:.2f}%",
                "evictions": self.stats.evictions,
                "promotions": self.stats.promotions,
                "demotions": self.stats.demotions
            }
        }


class MLCacheCoordinator:
    """
    Coordinates cache operations specifically for ML workloads.
    Provides intelligent prefetching and memory management.
    """

    def __init__(self, cache: TieredCache):
        self.cache = cache
        self.access_patterns: Dict[str, List[float]] = {}
        self._prefetch_queue = queue.Queue()
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._prefetch_thread.start()

    def _prefetch_worker(self):
        """Background worker for prefetching"""
        while True:
            try:
                keys = self._prefetch_queue.get(timeout=1)
                if keys is None:
                    break
                for key in keys:
                    # Promote from lower tiers
                    self.cache.get(key, promote=True)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Prefetch error: {e}")

    def register_ml_model(self, model_id: str, model: Any, pinned: bool = True):
        """Register an ML model in cache with high priority"""
        self.cache.put(
            key=f"model:{model_id}",
            data=model,
            data_type="model",
            prefer_tier=CacheTier.L1_VRAM,
            pinned=pinned
        )

    def cache_training_batch(self, batch_id: str, data: Any, labels: Any = None):
        """Cache a training batch"""
        self.cache.put(f"batch:{batch_id}:data", data, data_type="tensor")
        if labels is not None:
            self.cache.put(f"batch:{batch_id}:labels", labels, data_type="tensor")

    def prefetch_batches(self, batch_ids: List[str]):
        """Prefetch batches in background"""
        keys = [f"batch:{bid}:data" for bid in batch_ids]
        keys.extend([f"batch:{bid}:labels" for bid in batch_ids])
        self._prefetch_queue.put(keys)

    def cache_inference_input(self, request_id: str, data: Any):
        """Cache inference input for potential reuse"""
        self.cache.put(f"inference:{request_id}", data, data_type="tensor")

    def get_training_batch(self, batch_id: str) -> Tuple[Any, Any]:
        """Retrieve a training batch"""
        data = self.cache.get(f"batch:{batch_id}:data")
        labels = self.cache.get(f"batch:{batch_id}:labels")
        return data, labels

    def get_model(self, model_id: str) -> Any:
        """Retrieve a cached model"""
        return self.cache.get(f"model:{model_id}")


# Global cache instance
_global_cache: Optional[TieredCache] = None


def get_global_cache() -> TieredCache:
    """Get or create the global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = TieredCache()
    return _global_cache


def initialize_cache(
    l1_max_gb: float = 0,
    l2_max_gb: float = 0,
    l3_path: str = None,
    policy: CachePolicy = CachePolicy.LRU
) -> TieredCache:
    """Initialize the global cache with custom settings"""
    global _global_cache
    _global_cache = TieredCache(
        l1_max_bytes=int(l1_max_gb * (1024**3)) if l1_max_gb > 0 else 0,
        l2_max_bytes=int(l2_max_gb * (1024**3)) if l2_max_gb > 0 else 0,
        l3_path=l3_path,
        policy=policy
    )
    return _global_cache


if __name__ == "__main__":
    # Demo usage
    print("=== VBUS Cache Manager Demo ===\n")

    cache = initialize_cache(l1_max_gb=2, l2_max_gb=4)

    # Simulate caching some data
    import random
    test_data = [random.random() for _ in range(1000000)]

    print("Storing test data in cache...")
    cache.put("test_tensor_1", test_data, data_type="tensor")
    cache.put("test_tensor_2", test_data, data_type="tensor", prefer_tier=CacheTier.L2_RAM)

    print("\nRetrieving data...")
    result = cache.get("test_tensor_1")
    print(f"Retrieved: {len(result) if result else 0} elements")

    print("\nCache Status:")
    status = cache.get_status()
    print(json.dumps(status, indent=2))
