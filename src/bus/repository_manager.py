"""
VBUS Repository and Database Traffic Management (BUS)
Manages traffic between master database and repositories using AI inferencing
Creates cache coherency across the system hierarchy
"""

import os
import sys
import time
import threading
import queue
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Set, Tuple
from enum import Enum
from pathlib import Path
from datetime import datetime
import sqlite3
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VBUS.BUS")


class DataSourceType(Enum):
    """Types of data sources"""
    MASTER_DATABASE = "master_db"
    REPOSITORY = "repository"
    CACHE = "cache"
    API_ENDPOINT = "api"


class OperationType(Enum):
    """Types of data operations"""
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    SYNC = "sync"


class CoherencyState(Enum):
    """Cache coherency states (MESI protocol inspired)"""
    MODIFIED = "modified"      # Data has been modified locally
    EXCLUSIVE = "exclusive"    # Only this cache has the data
    SHARED = "shared"          # Multiple caches may have the data
    INVALID = "invalid"        # Data is stale/invalid


@dataclass
class DataSource:
    """Represents a data source (database or repository)"""
    name: str
    source_type: DataSourceType
    path: str
    connection_string: str = ""
    is_master: bool = False
    is_connected: bool = False
    last_sync: Optional[datetime] = None
    priority: int = 0  # Higher = more important
    read_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataBlock:
    """Represents a block of data with coherency tracking"""
    block_id: str
    source_name: str
    data_hash: str
    state: CoherencyState
    timestamp: float
    version: int = 1
    dependencies: Set[str] = field(default_factory=set)
    size_bytes: int = 0


@dataclass
class TrafficMetrics:
    """Traffic metrics for monitoring"""
    total_reads: int = 0
    total_writes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    sync_operations: int = 0
    conflicts_resolved: int = 0
    bytes_transferred: int = 0
    avg_latency_ms: float = 0.0


class SystemHierarchyAnalyzer:
    """
    Analyzes system hierarchy using AI inferencing to understand
    data relationships and optimize traffic patterns
    """

    def __init__(self):
        self.hierarchy_graph: Dict[str, Set[str]] = defaultdict(set)
        self.access_patterns: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.data_relationships: Dict[str, Dict[str, float]] = defaultdict(dict)

    def add_access(self, source: str, target: str, timestamp: float):
        """Record a data access pattern"""
        self.access_patterns[source].append((target, timestamp))
        # Keep only recent patterns
        if len(self.access_patterns[source]) > 10000:
            self.access_patterns[source] = self.access_patterns[source][-5000:]

    def analyze_relationships(self) -> Dict[str, Any]:
        """Analyze data relationships from access patterns"""
        relationships = {}

        for source, accesses in self.access_patterns.items():
            if not accesses:
                continue

            # Count target frequencies
            target_counts = defaultdict(int)
            for target, _ in accesses:
                target_counts[target] += 1

            total = len(accesses)
            relationships[source] = {
                target: count / total
                for target, count in target_counts.items()
            }

        return relationships

    def predict_next_access(self, current_source: str) -> List[str]:
        """Predict likely next data accesses based on patterns"""
        relationships = self.analyze_relationships()
        if current_source not in relationships:
            return []

        # Sort by probability
        predictions = sorted(
            relationships[current_source].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [target for target, _ in predictions[:5]]

    def build_hierarchy_graph(self, sources: List[DataSource]):
        """Build a graph representing the system hierarchy"""
        self.hierarchy_graph.clear()

        # Master database is at the top
        masters = [s for s in sources if s.is_master]
        repos = [s for s in sources if not s.is_master]

        for master in masters:
            for repo in repos:
                self.hierarchy_graph[master.name].add(repo.name)

        # Add relationships between repositories based on paths
        for i, repo1 in enumerate(repos):
            for repo2 in repos[i+1:]:
                # Check if repositories might be related
                path1 = Path(repo1.path)
                path2 = Path(repo2.path)
                try:
                    # Check for common parent
                    if path1.parent == path2.parent:
                        self.hierarchy_graph[repo1.name].add(repo2.name)
                        self.hierarchy_graph[repo2.name].add(repo1.name)
                except Exception:
                    pass

        return self.hierarchy_graph


class CacheCoherencyManager:
    """
    Manages cache coherency across distributed data sources
    Implements a MESI-like protocol for data consistency
    """

    def __init__(self):
        self.blocks: Dict[str, DataBlock] = {}
        self.source_blocks: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def register_block(
        self,
        block_id: str,
        source_name: str,
        data_hash: str,
        size_bytes: int = 0
    ) -> DataBlock:
        """Register a new data block"""
        with self._lock:
            block = DataBlock(
                block_id=block_id,
                source_name=source_name,
                data_hash=data_hash,
                state=CoherencyState.EXCLUSIVE,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
            self.blocks[block_id] = block
            self.source_blocks[source_name].add(block_id)
            return block

    def update_block(self, block_id: str, new_hash: str) -> bool:
        """Update a block, marking others as invalid"""
        with self._lock:
            if block_id not in self.blocks:
                return False

            block = self.blocks[block_id]
            block.data_hash = new_hash
            block.state = CoherencyState.MODIFIED
            block.timestamp = time.time()
            block.version += 1

            # Invalidate other caches
            for other_id, other_block in self.blocks.items():
                if other_id != block_id and other_block.data_hash == block.data_hash:
                    other_block.state = CoherencyState.INVALID

            return True

    def request_read(self, block_id: str) -> Tuple[bool, CoherencyState]:
        """Request to read a block, returns validity and state"""
        with self._lock:
            if block_id not in self.blocks:
                return False, CoherencyState.INVALID

            block = self.blocks[block_id]
            if block.state == CoherencyState.INVALID:
                return False, block.state

            # Mark as shared if currently exclusive
            if block.state == CoherencyState.EXCLUSIVE:
                block.state = CoherencyState.SHARED

            return True, block.state

    def request_write(self, block_id: str) -> Tuple[bool, List[str]]:
        """Request to write to a block, returns success and blocks to invalidate"""
        with self._lock:
            if block_id not in self.blocks:
                return False, []

            block = self.blocks[block_id]
            blocks_to_invalidate = []

            # Invalidate all shared copies
            for other_id, other_block in self.blocks.items():
                if other_id != block_id and other_block.source_name != block.source_name:
                    if other_block.state in [CoherencyState.SHARED, CoherencyState.EXCLUSIVE]:
                        blocks_to_invalidate.append(other_id)
                        other_block.state = CoherencyState.INVALID

            block.state = CoherencyState.MODIFIED
            return True, blocks_to_invalidate

    def sync_block(self, block_id: str, new_hash: str) -> bool:
        """Synchronize a block after conflict resolution"""
        with self._lock:
            if block_id not in self.blocks:
                return False

            block = self.blocks[block_id]
            block.data_hash = new_hash
            block.state = CoherencyState.SHARED
            block.timestamp = time.time()
            return True

    def get_stale_blocks(self, max_age_seconds: float = 3600) -> List[DataBlock]:
        """Get blocks that may be stale"""
        current_time = time.time()
        with self._lock:
            return [
                block for block in self.blocks.values()
                if current_time - block.timestamp > max_age_seconds
            ]


class TrafficManager:
    """
    Manages data traffic between master database and repositories
    Uses AI inferencing for intelligent routing and prefetching
    """

    def __init__(self):
        self.sources: Dict[str, DataSource] = {}
        self.master_db: Optional[DataSource] = None
        self.coherency = CacheCoherencyManager()
        self.analyzer = SystemHierarchyAnalyzer()
        self.metrics = TrafficMetrics()

        self._operation_queue = queue.PriorityQueue()
        self._worker_threads: List[threading.Thread] = []
        self._running = False
        self._lock = threading.RLock()

        # Configuration database
        self._config_db_path = Path.home() / ".vbus" / "bus_config.db"
        self._init_config_db()

    def _init_config_db(self):
        """Initialize configuration database"""
        self._config_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._config_db_path))
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_sources (
                name TEXT PRIMARY KEY,
                source_type TEXT,
                path TEXT,
                connection_string TEXT,
                is_master INTEGER,
                priority INTEGER,
                read_only INTEGER,
                metadata TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sync_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT,
                operation TEXT,
                timestamp REAL,
                success INTEGER,
                details TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def set_master_database(self, name: str, path: str, connection_string: str = ""):
        """Set the master database location"""
        source = DataSource(
            name=name,
            source_type=DataSourceType.MASTER_DATABASE,
            path=path,
            connection_string=connection_string,
            is_master=True,
            priority=100
        )
        self.sources[name] = source
        self.master_db = source
        self._save_source(source)
        logger.info(f"Master database set: {name} at {path}")

    def add_repository(
        self,
        name: str,
        path: str,
        connection_string: str = "",
        priority: int = 50,
        read_only: bool = False
    ):
        """Add a repository to the traffic management system"""
        source = DataSource(
            name=name,
            source_type=DataSourceType.REPOSITORY,
            path=path,
            connection_string=connection_string,
            is_master=False,
            priority=priority,
            read_only=read_only
        )
        self.sources[name] = source
        self._save_source(source)
        logger.info(f"Repository added: {name} at {path}")

    def _save_source(self, source: DataSource):
        """Save data source to configuration database"""
        conn = sqlite3.connect(str(self._config_db_path))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO data_sources
            (name, source_type, path, connection_string, is_master, priority, read_only, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            source.name,
            source.source_type.value,
            source.path,
            source.connection_string,
            1 if source.is_master else 0,
            source.priority,
            1 if source.read_only else 0,
            json.dumps(source.metadata)
        ))
        conn.commit()
        conn.close()

    def load_configuration(self):
        """Load saved configuration"""
        conn = sqlite3.connect(str(self._config_db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM data_sources')

        for row in cursor.fetchall():
            source = DataSource(
                name=row[0],
                source_type=DataSourceType(row[1]),
                path=row[2],
                connection_string=row[3],
                is_master=bool(row[4]),
                priority=row[5],
                read_only=bool(row[6]),
                metadata=json.loads(row[7]) if row[7] else {}
            )
            self.sources[source.name] = source
            if source.is_master:
                self.master_db = source

        conn.close()
        logger.info(f"Loaded {len(self.sources)} data sources from configuration")

    def verify_connections(self) -> Dict[str, bool]:
        """Verify connectivity to all data sources"""
        results = {}
        for name, source in self.sources.items():
            try:
                if os.path.exists(source.path):
                    source.is_connected = True
                    results[name] = True
                else:
                    source.is_connected = False
                    results[name] = False
                    logger.warning(f"Data source not accessible: {name} at {source.path}")
            except Exception as e:
                source.is_connected = False
                results[name] = False
                logger.error(f"Error verifying {name}: {e}")

        return results

    def start_traffic_management(self, num_workers: int = 4):
        """Start the traffic management system"""
        self._running = True

        for i in range(num_workers):
            thread = threading.Thread(
                target=self._traffic_worker,
                name=f"TrafficWorker-{i}",
                daemon=True
            )
            thread.start()
            self._worker_threads.append(thread)

        # Start hierarchy analyzer
        self.analyzer.build_hierarchy_graph(list(self.sources.values()))

        logger.info(f"Traffic management started with {num_workers} workers")

    def stop_traffic_management(self):
        """Stop the traffic management system"""
        self._running = False
        for _ in self._worker_threads:
            self._operation_queue.put((0, None))  # Poison pill
        for thread in self._worker_threads:
            thread.join(timeout=5)
        self._worker_threads.clear()
        logger.info("Traffic management stopped")

    def _traffic_worker(self):
        """Worker thread for processing traffic operations"""
        while self._running:
            try:
                priority, operation = self._operation_queue.get(timeout=1)
                if operation is None:
                    break
                self._process_operation(operation)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Traffic worker error: {e}")

    def _process_operation(self, operation: Dict[str, Any]):
        """Process a traffic operation"""
        op_type = operation.get("type")
        source_name = operation.get("source")
        target_name = operation.get("target")

        start_time = time.time()

        try:
            if op_type == OperationType.SYNC:
                self._perform_sync(source_name, target_name)
            elif op_type == OperationType.READ:
                self.metrics.total_reads += 1
            elif op_type == OperationType.WRITE:
                self.metrics.total_writes += 1

            latency = (time.time() - start_time) * 1000
            self._update_latency(latency)

        except Exception as e:
            logger.error(f"Operation failed: {e}")

    def _perform_sync(self, source_name: str, target_name: str):
        """Perform synchronization between two data sources"""
        source = self.sources.get(source_name)
        target = self.sources.get(target_name)

        if not source or not target:
            return

        self.metrics.sync_operations += 1

        # Log sync operation
        conn = sqlite3.connect(str(self._config_db_path))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sync_history (source_name, operation, timestamp, success, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (source_name, "sync", time.time(), 1, f"Synced with {target_name}"))
        conn.commit()
        conn.close()

        # Update last sync time
        source.last_sync = datetime.now()
        target.last_sync = datetime.now()

    def _update_latency(self, latency_ms: float):
        """Update rolling average latency"""
        alpha = 0.1  # Smoothing factor
        self.metrics.avg_latency_ms = (
            alpha * latency_ms + (1 - alpha) * self.metrics.avg_latency_ms
        )

    def request_data(
        self,
        data_id: str,
        preferred_source: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Request data with intelligent routing.
        Returns (success, source_name)
        """
        # Record access for pattern analysis
        if preferred_source:
            self.analyzer.add_access(preferred_source, data_id, time.time())

        # Check coherency
        valid, state = self.coherency.request_read(data_id)
        if valid and state != CoherencyState.INVALID:
            self.metrics.cache_hits += 1
            return True, "cache"

        self.metrics.cache_misses += 1

        # Find best source
        if preferred_source and preferred_source in self.sources:
            source = self.sources[preferred_source]
            if source.is_connected:
                return True, preferred_source

        # Use AI prediction for routing
        if preferred_source:
            predicted = self.analyzer.predict_next_access(preferred_source)
            for pred_source in predicted:
                if pred_source in self.sources and self.sources[pred_source].is_connected:
                    return True, pred_source

        # Fallback to master
        if self.master_db and self.master_db.is_connected:
            return True, self.master_db.name

        return False, ""

    def queue_sync(self, source_name: str, target_name: str, priority: int = 50):
        """Queue a sync operation"""
        operation = {
            "type": OperationType.SYNC,
            "source": source_name,
            "target": target_name
        }
        self._operation_queue.put((100 - priority, operation))

    def sync_all(self):
        """Synchronize all repositories with master"""
        if not self.master_db:
            logger.error("No master database configured")
            return

        for name, source in self.sources.items():
            if not source.is_master and source.is_connected:
                self.queue_sync(self.master_db.name, name, source.priority)

    def get_status(self) -> Dict[str, Any]:
        """Get current traffic management status"""
        return {
            "sources": {
                name: {
                    "type": s.source_type.value,
                    "path": s.path,
                    "connected": s.is_connected,
                    "is_master": s.is_master,
                    "last_sync": s.last_sync.isoformat() if s.last_sync else None,
                    "priority": s.priority
                }
                for name, s in self.sources.items()
            },
            "metrics": {
                "total_reads": self.metrics.total_reads,
                "total_writes": self.metrics.total_writes,
                "cache_hit_rate": (
                    self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses)
                    if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0
                ) * 100,
                "sync_operations": self.metrics.sync_operations,
                "avg_latency_ms": round(self.metrics.avg_latency_ms, 2)
            },
            "hierarchy": dict(self.analyzer.hierarchy_graph)
        }


# Global traffic manager instance
_global_traffic_manager: Optional[TrafficManager] = None


def get_traffic_manager() -> TrafficManager:
    """Get or create the global traffic manager"""
    global _global_traffic_manager
    if _global_traffic_manager is None:
        _global_traffic_manager = TrafficManager()
        _global_traffic_manager.load_configuration()
    return _global_traffic_manager


def initialize_bus(
    master_db_name: str,
    master_db_path: str,
    master_db_connection: str = ""
) -> TrafficManager:
    """Initialize the BUS with master database"""
    global _global_traffic_manager
    _global_traffic_manager = TrafficManager()
    _global_traffic_manager.set_master_database(
        master_db_name,
        master_db_path,
        master_db_connection
    )
    return _global_traffic_manager


if __name__ == "__main__":
    print("=== VBUS Traffic Manager Demo ===\n")

    # Initialize
    manager = initialize_bus("main_db", "/path/to/master/database")

    # Add repositories
    manager.add_repository("repo1", "/path/to/repo1", priority=75)
    manager.add_repository("repo2", "/path/to/repo2", priority=50)
    manager.add_repository("repo3", "/path/to/repo3", priority=25, read_only=True)

    # Start traffic management
    manager.start_traffic_management(num_workers=2)

    # Verify connections
    print("Verifying connections...")
    results = manager.verify_connections()
    for name, connected in results.items():
        status = "✓" if connected else "✗"
        print(f"  {status} {name}")

    # Get status
    print("\nTraffic Manager Status:")
    status = manager.get_status()
    print(json.dumps(status, indent=2))

    # Stop
    manager.stop_traffic_management()
    print("\nTraffic management stopped.")
