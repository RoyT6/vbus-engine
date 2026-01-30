#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              VBUS CORE v1.0                                   ║
║           ViewerDBX Bus - Absolute Dependency & Hierarchical Cache           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  VBUS = ViewerDBX Bus (mandatory system interconnect)                        ║
║                                                                               ║
║  PRINCIPLES:                                                                  ║
║    1. NO FALLBACKS - All paths through VBUS or nothing                       ║
║    2. HIERARCHICAL CACHE - L1 (hot), L2 (warm), L3 (cold)                   ║
║    3. INCREMENTAL RUNS - New arrivals only after initial build              ║
║    4. ACCESS MONITORING - Automatic cache promotion/demotion                 ║
║                                                                               ║
║  VERSION: 1.0.0 | ALGO 95.4 | GPU MANDATORY                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import threading
import pickle
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from enum import Enum, auto
from functools import wraps
from collections import OrderedDict
import weakref

# =============================================================================
# VBUS CONSTANTS
# =============================================================================

ALGO_VERSION = "95.4"
VBUS_VERSION = "1.0.0"
BASE_PATH = Path(r"C:\Users\RoyT6\Downloads")

# Cache configuration (like CPU cache hierarchy)
L1_CACHE_SIZE = 64       # Hot data - most frequently accessed (in-memory)
L2_CACHE_SIZE = 256      # Warm data - recently accessed (in-memory)
L3_CACHE_SIZE = 1024     # Cold data - disk-backed cache
PROMOTION_THRESHOLD = 5   # Access count to promote from L3 -> L2 -> L1
DEMOTION_TTL_SECONDS = 3600  # Time before demotion check (1 hour)

# Enforcement
STRICT_MODE = True        # No fallbacks allowed
AUDIT_ALL_ACCESS = True   # Log every path resolution

# =============================================================================
# CACHE TIERS
# =============================================================================

class CacheTier(Enum):
    """Cache hierarchy tiers (like CPU L1/L2/L3)"""
    L1_HOT = "L1_HOT"      # Fastest - in-memory, most accessed
    L2_WARM = "L2_WARM"    # Fast - in-memory, recently accessed
    L3_COLD = "L3_COLD"    # Slow - disk-backed, infrequently accessed
    UNCACHED = "UNCACHED"  # Not in cache - must fetch


class RunMode(Enum):
    """Pipeline execution modes"""
    FULL_BUILD = "FULL_BUILD"          # All 800k titles (initial)
    INCREMENTAL = "INCREMENTAL"        # New arrivals only
    EQUATION_UPDATE = "EQUATION_UPDATE" # Re-run equations, same data
    VALIDATION_ONLY = "VALIDATION_ONLY" # Validate existing data


# =============================================================================
# CACHE ENTRY
# =============================================================================

@dataclass
class CacheEntry:
    """Entry in the hierarchical cache"""
    key: str
    value: Any
    tier: CacheTier
    access_count: int = 0
    last_access: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    size_bytes: int = 0
    checksum: Optional[str] = None
    component: Optional[str] = None
    category: str = "general"

    def touch(self):
        """Update access time and count"""
        self.access_count += 1
        self.last_access = datetime.now(timezone.utc).isoformat()


@dataclass
class AccessPattern:
    """Tracks access patterns for cache optimization"""
    path_key: str
    total_accesses: int = 0
    accesses_last_hour: int = 0
    accesses_last_day: int = 0
    average_interval_ms: float = 0
    last_access_times: List[float] = field(default_factory=list)
    requesting_components: Set[str] = field(default_factory=set)

    def record_access(self, component: str):
        """Record an access event"""
        now = time.time()
        self.total_accesses += 1
        self.last_access_times.append(now)
        self.requesting_components.add(component)

        # Keep only last 100 access times for interval calculation
        if len(self.last_access_times) > 100:
            self.last_access_times = self.last_access_times[-100:]

        # Calculate average interval
        if len(self.last_access_times) > 1:
            intervals = [self.last_access_times[i] - self.last_access_times[i-1]
                        for i in range(1, len(self.last_access_times))]
            self.average_interval_ms = (sum(intervals) / len(intervals)) * 1000


# =============================================================================
# HIERARCHICAL CACHE MANAGER
# =============================================================================

class HierarchicalCache:
    """
    Three-tier cache with automatic promotion/demotion.

    L1 (Hot):  64 entries  - Most frequently accessed, fastest
    L2 (Warm): 256 entries - Recently accessed
    L3 (Cold): 1024 entries - Disk-backed, least accessed

    Promotion: Access count > threshold moves entry up
    Demotion: TTL expiry without access moves entry down
    """

    def __init__(self):
        self._l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._l2_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._l3_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._access_patterns: Dict[str, AccessPattern] = {}
        self._lock = threading.RLock()

        # Metrics
        self._metrics = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0,
            "promotions": 0,
            "demotions": 0,
            "evictions": 0
        }

        # Load L3 from disk if exists
        self._l3_disk_path = BASE_PATH / ".vbus_cache" / "l3_cache.pkl"
        self._load_l3_from_disk()

    def get(self, key: str, component: str = "unknown") -> Tuple[Optional[Any], CacheTier]:
        """
        Get value from cache, checking L1 -> L2 -> L3.
        Returns (value, tier) or (None, UNCACHED).
        """
        with self._lock:
            # Track access pattern
            if key not in self._access_patterns:
                self._access_patterns[key] = AccessPattern(path_key=key)
            self._access_patterns[key].record_access(component)

            # Check L1 (hot)
            if key in self._l1_cache:
                entry = self._l1_cache[key]
                entry.touch()
                self._l1_cache.move_to_end(key)  # LRU
                self._metrics["l1_hits"] += 1
                return entry.value, CacheTier.L1_HOT

            # Check L2 (warm)
            if key in self._l2_cache:
                entry = self._l2_cache[key]
                entry.touch()
                self._l2_cache.move_to_end(key)
                self._metrics["l2_hits"] += 1

                # Check for promotion to L1
                if entry.access_count >= PROMOTION_THRESHOLD:
                    self._promote(key, entry, CacheTier.L2_WARM, CacheTier.L1_HOT)

                return entry.value, CacheTier.L2_WARM

            # Check L3 (cold)
            if key in self._l3_cache:
                entry = self._l3_cache[key]
                entry.touch()
                self._metrics["l3_hits"] += 1

                # Promote to L2 on access
                self._promote(key, entry, CacheTier.L3_COLD, CacheTier.L2_WARM)

                return entry.value, CacheTier.L3_COLD

            self._metrics["misses"] += 1
            return None, CacheTier.UNCACHED

    def put(self, key: str, value: Any, component: str = "unknown",
            category: str = "general", tier: CacheTier = CacheTier.L2_WARM) -> None:
        """
        Put value in cache at specified tier.
        New entries go to L2 by default, hot data can go directly to L1.
        """
        with self._lock:
            entry = CacheEntry(
                key=key,
                value=value,
                tier=tier,
                component=component,
                category=category,
                size_bytes=self._estimate_size(value)
            )

            if tier == CacheTier.L1_HOT:
                self._put_l1(key, entry)
            elif tier == CacheTier.L2_WARM:
                self._put_l2(key, entry)
            else:
                self._put_l3(key, entry)

    def _put_l1(self, key: str, entry: CacheEntry) -> None:
        """Put entry in L1, evicting if necessary"""
        # Remove from other tiers
        self._l2_cache.pop(key, None)
        self._l3_cache.pop(key, None)

        # Evict oldest if full
        while len(self._l1_cache) >= L1_CACHE_SIZE:
            evicted_key, evicted = self._l1_cache.popitem(last=False)
            # Demote to L2
            evicted.tier = CacheTier.L2_WARM
            self._put_l2(evicted_key, evicted)
            self._metrics["demotions"] += 1

        entry.tier = CacheTier.L1_HOT
        self._l1_cache[key] = entry

    def _put_l2(self, key: str, entry: CacheEntry) -> None:
        """Put entry in L2, evicting if necessary"""
        # Remove from other tiers
        self._l1_cache.pop(key, None)
        self._l3_cache.pop(key, None)

        # Evict oldest if full
        while len(self._l2_cache) >= L2_CACHE_SIZE:
            evicted_key, evicted = self._l2_cache.popitem(last=False)
            # Demote to L3
            evicted.tier = CacheTier.L3_COLD
            self._put_l3(evicted_key, evicted)
            self._metrics["demotions"] += 1

        entry.tier = CacheTier.L2_WARM
        self._l2_cache[key] = entry

    def _put_l3(self, key: str, entry: CacheEntry) -> None:
        """Put entry in L3, evicting if necessary"""
        # Remove from other tiers
        self._l1_cache.pop(key, None)
        self._l2_cache.pop(key, None)

        # Evict oldest if full
        while len(self._l3_cache) >= L3_CACHE_SIZE:
            self._l3_cache.popitem(last=False)
            self._metrics["evictions"] += 1

        entry.tier = CacheTier.L3_COLD
        self._l3_cache[key] = entry

    def _promote(self, key: str, entry: CacheEntry,
                 from_tier: CacheTier, to_tier: CacheTier) -> None:
        """Promote entry to higher cache tier"""
        if to_tier == CacheTier.L1_HOT:
            self._put_l1(key, entry)
        elif to_tier == CacheTier.L2_WARM:
            self._put_l2(key, entry)
        self._metrics["promotions"] += 1

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value"""
        try:
            return len(pickle.dumps(value))
        except:
            return 0

    def _load_l3_from_disk(self) -> None:
        """Load L3 cache from disk"""
        if self._l3_disk_path.exists():
            try:
                with open(self._l3_disk_path, 'rb') as f:
                    self._l3_cache = pickle.load(f)
            except:
                self._l3_cache = OrderedDict()

    def save_l3_to_disk(self) -> None:
        """Persist L3 cache to disk"""
        self._l3_disk_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._l3_disk_path, 'wb') as f:
            pickle.dump(self._l3_cache, f)

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        total_hits = self._metrics["l1_hits"] + self._metrics["l2_hits"] + self._metrics["l3_hits"]
        total_requests = total_hits + self._metrics["misses"]

        return {
            **self._metrics,
            "l1_size": len(self._l1_cache),
            "l2_size": len(self._l2_cache),
            "l3_size": len(self._l3_cache),
            "total_hit_rate": total_hits / total_requests if total_requests > 0 else 0,
            "l1_hit_rate": self._metrics["l1_hits"] / total_requests if total_requests > 0 else 0,
            "hot_keys": self._get_hottest_keys(10)
        }

    def _get_hottest_keys(self, n: int) -> List[Dict[str, Any]]:
        """Get the N most accessed keys"""
        sorted_patterns = sorted(
            self._access_patterns.items(),
            key=lambda x: x[1].total_accesses,
            reverse=True
        )[:n]

        return [
            {
                "key": k,
                "accesses": p.total_accesses,
                "components": list(p.requesting_components),
                "avg_interval_ms": p.average_interval_ms
            }
            for k, p in sorted_patterns
        ]

    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry"""
        with self._lock:
            removed = False
            if key in self._l1_cache:
                del self._l1_cache[key]
                removed = True
            if key in self._l2_cache:
                del self._l2_cache[key]
                removed = True
            if key in self._l3_cache:
                del self._l3_cache[key]
                removed = True
            return removed

    def clear(self) -> None:
        """Clear all cache tiers"""
        with self._lock:
            self._l1_cache.clear()
            self._l2_cache.clear()
            self._l3_cache.clear()
            self._access_patterns.clear()


# =============================================================================
# VBUS ENFORCEMENT
# =============================================================================

class VBUSEnforcementError(Exception):
    """Raised when VBUS enforcement is violated"""
    pass


class VBUSNotInitializedError(Exception):
    """Raised when VBUS is accessed before initialization"""
    pass


def require_vbus(func):
    """
    Decorator that enforces VBUS dependency.
    Functions decorated with this MUST have access to VBUS.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if VBUS is initialized
        if not VBUS._is_initialized():
            raise VBUSNotInitializedError(
                f"VBUS not initialized. Call VBUS.initialize() before using {func.__name__}"
            )
        return func(*args, **kwargs)
    return wrapper


def vbus_component(name: str, role: str, provides: List[str] = None,
                   requires: List[str] = None):
    """
    Class decorator that registers a component with VBUS.
    The component MUST use VBUS for all data access - no fallbacks.
    """
    def decorator(cls):
        # Store component metadata
        cls._vbus_name = name
        cls._vbus_role = role
        cls._vbus_provides = provides or []
        cls._vbus_requires = requires or []

        # Inject VBUS access
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            # Ensure VBUS is available
            self._vbus = VBUS.get_instance()
            if self._vbus is None:
                raise VBUSNotInitializedError(
                    f"VBUS must be initialized before creating {name}"
                )

            # Register this instance
            self._vbus.register_active_component(name, self)

            # Call original init
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init

        # Add VBUS helper methods
        def get_path(self, key: str) -> Path:
            """Get path through VBUS - no fallbacks"""
            return self._vbus.resolve(key, component=name)

        def get_data(self, key: str) -> Any:
            """Get cached data through VBUS"""
            return self._vbus.get_cached(key, component=name)

        def put_data(self, key: str, value: Any, category: str = "general") -> None:
            """Put data in VBUS cache"""
            self._vbus.cache_data(key, value, component=name, category=category)

        def signal(self, target: str, signal_type: str, payload: Dict = None) -> None:
            """Send signal through VBUS"""
            self._vbus.send_signal(name, target, signal_type, payload or {})

        cls.vbus_path = get_path
        cls.vbus_data = get_data
        cls.vbus_put = put_data
        cls.vbus_signal = signal

        return cls

    return decorator


# =============================================================================
# VBUS CORE (SINGLETON)
# =============================================================================

class VBUS:
    """
    ViewerDBX Bus - The mandatory system interconnect.

    All components MUST use VBUS for:
    - Path resolution
    - Data access
    - Inter-component communication
    - Caching

    NO FALLBACKS ALLOWED.
    """

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    @classmethod
    def initialize(cls, run_mode: RunMode = RunMode.INCREMENTAL) -> 'VBUS':
        """Initialize VBUS (must be called before any component)"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls.__new__(cls)
                cls._instance._init_internal(run_mode)
            cls._initialized = True
            return cls._instance

    @classmethod
    def get_instance(cls) -> Optional['VBUS']:
        """Get VBUS instance (None if not initialized)"""
        return cls._instance

    @classmethod
    def _is_initialized(cls) -> bool:
        """Check if VBUS is initialized"""
        return cls._initialized

    def _init_internal(self, run_mode: RunMode):
        """Internal initialization"""
        self.version = VBUS_VERSION
        self.algo_version = ALGO_VERSION
        self.base_path = BASE_PATH
        self.run_mode = run_mode
        self.start_time = datetime.now(timezone.utc)

        # Hierarchical cache
        self._cache = HierarchicalCache()

        # Component registry
        self._components: Dict[str, Dict[str, Any]] = {}
        self._active_instances: Dict[str, weakref.ref] = {}

        # Signal handlers
        self._signal_handlers: Dict[str, List[Callable]] = {}
        self._signal_queue: List[Dict[str, Any]] = []

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []

        # Path registry (static paths that don't change)
        self._path_registry: Dict[str, Path] = {}
        self._register_standard_paths()

        # Incremental run tracking
        self._processed_titles: Set[str] = set()
        self._new_arrivals: Set[str] = set()
        self._last_full_build: Optional[str] = None

        # Load state if exists
        self._load_state()

        # Log initialization
        self._audit("VBUS_INIT", f"VBUS initialized in {run_mode.value} mode", {
            "version": self.version,
            "run_mode": run_mode.value
        })

    def _register_standard_paths(self) -> None:
        """Register standard system paths"""
        # Core databases
        self._path_registry["bfd"] = self._find_latest("BFD_V*.parquet")
        self._path_registry["star"] = self._find_latest("BFD_Star_Schema_V*.parquet")

        # Component directories
        components = [
            "ALGO Engine", "SCHIG", "Abstract Data", "Daily Top 10s",
            "Studios", "Talent", "Money Engine", "Fresh In!", "Credibility",
            "Cloudflare", "Replit", "MAPIE", "Views TRaining Data",
            "Components", "Schema", "Orchestrator", "GPU Enablement", "AUDIT_LOGS"
        ]

        for comp in components:
            key = f"component:{comp.lower().replace(' ', '_')}"
            path = self.base_path / comp
            if path.exists():
                self._path_registry[key] = path

        # Schema
        self._path_registry["schema"] = self.base_path / "Schema" / "SCHEMA_V27.00.json"

        # Bus manifest
        self._path_registry["manifest"] = self.base_path / "bus_manifest.json"

    def _find_latest(self, pattern: str) -> Optional[Path]:
        """Find latest file matching pattern"""
        matches = list(self.base_path.glob(pattern))
        if not matches:
            return None
        matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        return matches[0]

    # =========================================================================
    # PATH RESOLUTION (MANDATORY - NO FALLBACKS)
    # =========================================================================

    def resolve(self, key: str, component: str = "unknown") -> Path:
        """
        Resolve path key to filesystem path.

        THIS IS MANDATORY - NO FALLBACK TO HARDCODED PATHS.

        Raises VBUSEnforcementError if path cannot be resolved.
        """
        if not STRICT_MODE:
            # Non-strict mode (not recommended)
            return self._resolve_internal(key, component)

        # Strict mode - must resolve or fail
        path = self._resolve_internal(key, component)
        if path is None:
            raise VBUSEnforcementError(
                f"VBUS cannot resolve path '{key}'. "
                f"All paths MUST be resolved through VBUS. "
                f"Register the path or check the key."
            )

        return path

    def _resolve_internal(self, key: str, component: str) -> Optional[Path]:
        """Internal path resolution with caching"""
        # Check cache first
        cached, tier = self._cache.get(f"path:{key}", component)
        if cached is not None:
            self._audit("PATH_RESOLVE", f"Cache hit ({tier.value})", {
                "key": key, "component": component, "tier": tier.value
            })
            return cached

        # Check static registry
        if key in self._path_registry:
            path = self._path_registry[key]
            self._cache.put(f"path:{key}", path, component, "path", CacheTier.L2_WARM)
            self._audit("PATH_RESOLVE", "Registry hit", {"key": key, "component": component})
            return path

        # Try to resolve dynamically
        path = self._dynamic_resolve(key)
        if path:
            self._cache.put(f"path:{key}", path, component, "path", CacheTier.L3_COLD)
            self._audit("PATH_RESOLVE", "Dynamic resolve", {"key": key, "component": component})
            return path

        self._audit("PATH_RESOLVE_FAIL", "Cannot resolve path", {"key": key, "component": component})
        return None

    def _dynamic_resolve(self, key: str) -> Optional[Path]:
        """Dynamically resolve path based on key pattern"""
        parts = key.split(":")

        if len(parts) == 2:
            category, name = parts

            if category == "component":
                # Component directory
                for comp_name, comp_path in self._path_registry.items():
                    if comp_name.startswith("component:") and name in comp_name:
                        return comp_path

            elif category == "config":
                # Config file
                for folder in ["Components", "Schema"]:
                    path = self.base_path / folder / f"{name}.json"
                    if path.exists():
                        return path

            elif category == "data":
                # Data file
                for ext in [".parquet", ".json", ".csv"]:
                    path = self.base_path / f"{name}{ext}"
                    if path.exists():
                        return path

        return None

    def register_path(self, key: str, path: Path, hot: bool = False) -> None:
        """Register a path in VBUS"""
        self._path_registry[key] = path
        tier = CacheTier.L1_HOT if hot else CacheTier.L2_WARM
        self._cache.put(f"path:{key}", path, "system", "path", tier)
        self._audit("PATH_REGISTER", f"Registered path", {"key": key, "path": str(path), "hot": hot})

    # =========================================================================
    # DATA CACHING
    # =========================================================================

    def get_cached(self, key: str, component: str = "unknown") -> Optional[Any]:
        """Get cached data"""
        value, tier = self._cache.get(f"data:{key}", component)
        return value

    def cache_data(self, key: str, value: Any, component: str = "unknown",
                   category: str = "general", hot: bool = False) -> None:
        """Cache data"""
        tier = CacheTier.L1_HOT if hot else CacheTier.L2_WARM
        self._cache.put(f"data:{key}", value, component, category, tier)
        self._audit("DATA_CACHE", f"Cached data", {"key": key, "component": component, "hot": hot})

    def invalidate_cache(self, key: str) -> bool:
        """Invalidate cache entry"""
        removed = self._cache.invalidate(f"data:{key}")
        self._cache.invalidate(f"path:{key}")
        self._audit("CACHE_INVALIDATE", f"Invalidated", {"key": key, "removed": removed})
        return removed

    # =========================================================================
    # COMPONENT REGISTRY
    # =========================================================================

    def register_component(self, name: str, role: str,
                          provides: List[str] = None, requires: List[str] = None) -> None:
        """Register a component with VBUS"""
        self._components[name] = {
            "name": name,
            "role": role,
            "provides": provides or [],
            "requires": requires or [],
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "status": "registered"
        }
        self._audit("COMPONENT_REGISTER", f"Registered {name}", {"role": role})

    def register_active_component(self, name: str, instance: Any) -> None:
        """Register an active component instance"""
        self._active_instances[name] = weakref.ref(instance)
        if name in self._components:
            self._components[name]["status"] = "active"
        self._audit("COMPONENT_ACTIVE", f"{name} is active", {})

    def get_component(self, name: str) -> Optional[Any]:
        """Get active component instance"""
        ref = self._active_instances.get(name)
        return ref() if ref else None

    # =========================================================================
    # SIGNAL BUS
    # =========================================================================

    def send_signal(self, source: str, target: str, signal_type: str,
                    payload: Dict[str, Any] = None) -> None:
        """Send signal to component(s)"""
        signal = {
            "source": source,
            "target": target,
            "type": signal_type,
            "payload": payload or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        self._signal_queue.append(signal)
        self._audit("SIGNAL_SENT", f"{source} -> {target}", {"type": signal_type})

        # Dispatch to handlers
        if target == "*":
            # Broadcast
            for handlers in self._signal_handlers.values():
                for handler in handlers:
                    try:
                        handler(signal)
                    except Exception as e:
                        self._audit("SIGNAL_ERROR", str(e), {"signal": signal})
        else:
            # Direct
            handlers = self._signal_handlers.get(target, [])
            for handler in handlers:
                try:
                    handler(signal)
                except Exception as e:
                    self._audit("SIGNAL_ERROR", str(e), {"signal": signal})

    def register_signal_handler(self, component: str, handler: Callable) -> None:
        """Register signal handler for component"""
        if component not in self._signal_handlers:
            self._signal_handlers[component] = []
        self._signal_handlers[component].append(handler)

    def broadcast(self, source: str, signal_type: str, payload: Dict = None) -> None:
        """Broadcast signal to all components"""
        self.send_signal(source, "*", signal_type, payload)

    # =========================================================================
    # INCREMENTAL RUN SUPPORT
    # =========================================================================

    def mark_processed(self, fc_uid: str) -> None:
        """Mark a title as processed"""
        self._processed_titles.add(fc_uid)

    def is_processed(self, fc_uid: str) -> bool:
        """Check if title was processed in a previous run"""
        return fc_uid in self._processed_titles

    def add_new_arrival(self, fc_uid: str) -> None:
        """Add a new arrival to be processed"""
        self._new_arrivals.add(fc_uid)

    def get_titles_to_process(self) -> Set[str]:
        """Get titles that need processing based on run mode"""
        if self.run_mode == RunMode.FULL_BUILD:
            return set()  # All titles (handled by caller)
        elif self.run_mode == RunMode.INCREMENTAL:
            return self._new_arrivals - self._processed_titles
        elif self.run_mode == RunMode.EQUATION_UPDATE:
            return self._processed_titles  # Re-run all processed
        else:
            return set()

    def complete_run(self) -> None:
        """Mark run as complete, update state"""
        if self.run_mode == RunMode.FULL_BUILD:
            self._last_full_build = datetime.now(timezone.utc).isoformat()

        # Move new arrivals to processed
        self._processed_titles.update(self._new_arrivals)
        self._new_arrivals.clear()

        # Save state
        self._save_state()

        # Persist L3 cache
        self._cache.save_l3_to_disk()

        self._audit("RUN_COMPLETE", f"Run complete", {
            "mode": self.run_mode.value,
            "processed_count": len(self._processed_titles)
        })

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    def _save_state(self) -> None:
        """Save VBUS state to disk"""
        state_path = self.base_path / ".vbus_cache" / "vbus_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "version": self.version,
            "last_full_build": self._last_full_build,
            "processed_titles_count": len(self._processed_titles),
            "processed_titles": list(self._processed_titles)[:1000],  # Sample
            "saved_at": datetime.now(timezone.utc).isoformat()
        }

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        # Save full processed set separately
        processed_path = self.base_path / ".vbus_cache" / "processed_titles.pkl"
        with open(processed_path, 'wb') as f:
            pickle.dump(self._processed_titles, f)

    def _load_state(self) -> None:
        """Load VBUS state from disk"""
        state_path = self.base_path / ".vbus_cache" / "vbus_state.json"
        processed_path = self.base_path / ".vbus_cache" / "processed_titles.pkl"

        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                self._last_full_build = state.get("last_full_build")
            except:
                pass

        if processed_path.exists():
            try:
                with open(processed_path, 'rb') as f:
                    self._processed_titles = pickle.load(f)
            except:
                self._processed_titles = set()

    # =========================================================================
    # AUDIT LOGGING
    # =========================================================================

    def _audit(self, event_type: str, description: str, data: Dict[str, Any] = None) -> None:
        """Log audit event"""
        if not AUDIT_ALL_ACCESS and event_type.startswith("PATH_RESOLVE"):
            return  # Skip verbose path logging

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "description": description,
            "data": data or {},
            "machine_generated": True
        }
        self._audit_log.append(entry)

        # Keep bounded
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit entries"""
        return self._audit_log[-limit:]

    # =========================================================================
    # METRICS & STATUS
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get VBUS performance metrics"""
        cache_metrics = self._cache.get_metrics()

        return {
            "vbus_version": self.version,
            "run_mode": self.run_mode.value,
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            "registered_components": len(self._components),
            "active_components": len([r for r in self._active_instances.values() if r() is not None]),
            "registered_paths": len(self._path_registry),
            "processed_titles": len(self._processed_titles),
            "new_arrivals": len(self._new_arrivals),
            "last_full_build": self._last_full_build,
            "cache": cache_metrics,
            "signals_queued": len(self._signal_queue),
            "audit_log_size": len(self._audit_log)
        }

    def print_status(self) -> None:
        """Print VBUS status"""
        metrics = self.get_metrics()
        cache = metrics["cache"]

        print()
        print("╔══════════════════════════════════════════════════════════════════════════╗")
        print("║                           VBUS STATUS                                    ║")
        print(f"║  Version: {self.version:<15} Mode: {self.run_mode.value:<25}     ║")
        print("╠══════════════════════════════════════════════════════════════════════════╣")
        print("║  HIERARCHICAL CACHE                                                      ║")
        print(f"║  L1 (Hot):  {cache['l1_size']:>4}/{L1_CACHE_SIZE:<4} entries    Hit Rate: {cache['l1_hit_rate']*100:>5.1f}%          ║")
        print(f"║  L2 (Warm): {cache['l2_size']:>4}/{L2_CACHE_SIZE:<4} entries    L2 Hits:  {cache['l2_hits']:>6}              ║")
        print(f"║  L3 (Cold): {cache['l3_size']:>4}/{L3_CACHE_SIZE:<4} entries    L3 Hits:  {cache['l3_hits']:>6}              ║")
        print(f"║  Total Hit Rate: {cache['total_hit_rate']*100:>5.1f}%         Promotions: {cache['promotions']:>5}              ║")
        print("╠══════════════════════════════════════════════════════════════════════════╣")
        print("║  HOT DATA (Most Accessed)                                                ║")
        for i, hot in enumerate(cache.get("hot_keys", [])[:5]):
            print(f"║  {i+1}. {hot['key']:<30} ({hot['accesses']} accesses)              ║"[:76] + "║")
        print("╠══════════════════════════════════════════════════════════════════════════╣")
        print("║  INCREMENTAL RUN STATUS                                                  ║")
        print(f"║  Processed Titles: {metrics['processed_titles']:>10}    New Arrivals: {metrics['new_arrivals']:>10}      ║")
        print(f"║  Last Full Build:  {metrics['last_full_build'] or 'Never':<40}         ║"[:76] + "║")
        print("╠══════════════════════════════════════════════════════════════════════════╣")
        print(f"║  Components: {metrics['registered_components']} registered, {metrics['active_components']} active                            ║")
        print(f"║  Registered Paths: {metrics['registered_paths']}                                               ║")
        print("╚══════════════════════════════════════════════════════════════════════════╝")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_vbus() -> VBUS:
    """Get VBUS instance (raises if not initialized)"""
    instance = VBUS.get_instance()
    if instance is None:
        raise VBUSNotInitializedError("VBUS not initialized. Call VBUS.initialize() first.")
    return instance


def init_vbus(mode: RunMode = RunMode.INCREMENTAL) -> VBUS:
    """Initialize VBUS"""
    return VBUS.initialize(mode)


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    # Initialize VBUS in incremental mode
    vbus = init_vbus(RunMode.INCREMENTAL)

    # Print status
    vbus.print_status()

    # Test path resolution
    print("\n[TEST] Path Resolution (MANDATORY - NO FALLBACKS):")
    try:
        bfd = vbus.resolve("bfd", "test")
        print(f"  ✓ BFD: {bfd}")
    except VBUSEnforcementError as e:
        print(f"  ✗ {e}")

    try:
        star = vbus.resolve("star", "test")
        print(f"  ✓ Star Schema: {star}")
    except VBUSEnforcementError as e:
        print(f"  ✗ {e}")

    # Test caching
    print("\n[TEST] Hierarchical Caching:")
    for i in range(10):
        vbus.resolve("bfd", "test")  # Should promote to L1

    metrics = vbus.get_metrics()
    print(f"  Cache Hit Rate: {metrics['cache']['total_hit_rate']*100:.1f}%")
    print(f"  L1 Hits: {metrics['cache']['l1_hits']}")

    # Test data caching
    print("\n[TEST] Data Caching:")
    vbus.cache_data("test_key", {"value": 42}, "test", hot=True)
    result = vbus.get_cached("test_key", "test")
    print(f"  Cached: {result}")

    # Complete run
    vbus.complete_run()

    print("\n✓ VBUS v1.0 operational - NO FALLBACKS ALLOWED")
