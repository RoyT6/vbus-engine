#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         VBUS MONITOR v1.0                                    ║
║              Real-Time System Visibility & Cache Analytics                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Provides:                                                                    ║
║    - Real-time cache tier monitoring (L1/L2/L3)                              ║
║    - Hot data identification                                                  ║
║    - Component activity tracking                                              ║
║    - Pipeline phase monitoring                                                ║
║    - Failure mode alerts                                                      ║
║                                                                               ║
║  VERSION: 1.0.0 | ALGO 95.4                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import VBUS
from vbus_core import VBUS, get_vbus, CacheTier

# =============================================================================
# MONITOR DATA STRUCTURES
# =============================================================================

@dataclass
class CacheSnapshot:
    """Snapshot of cache state"""
    timestamp: str
    l1_size: int
    l2_size: int
    l3_size: int
    l1_hit_rate: float
    total_hit_rate: float
    promotions: int
    demotions: int
    hot_keys: List[Dict[str, Any]]


@dataclass
class ComponentSnapshot:
    """Snapshot of component state"""
    name: str
    role: str
    status: str
    last_activity: Optional[str]
    cached_items: int


# =============================================================================
# VBUS MONITOR
# =============================================================================

class VBUSMonitor:
    """
    Real-time monitoring for VBUS.

    Tracks:
    - Cache performance across all tiers
    - Hot data (most accessed)
    - Component activity
    - System health
    """

    def __init__(self, vbus: VBUS = None):
        self.vbus = vbus or get_vbus()
        self._history: List[CacheSnapshot] = []
        self._max_history = 100
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

    def get_cache_snapshot(self) -> CacheSnapshot:
        """Get current cache state snapshot"""
        metrics = self.vbus.get_metrics()
        cache = metrics["cache"]

        return CacheSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            l1_size=cache["l1_size"],
            l2_size=cache["l2_size"],
            l3_size=cache["l3_size"],
            l1_hit_rate=cache["l1_hit_rate"],
            total_hit_rate=cache["total_hit_rate"],
            promotions=cache["promotions"],
            demotions=cache["demotions"],
            hot_keys=cache.get("hot_keys", [])
        )

    def get_hot_data_report(self, top_n: int = 20) -> Dict[str, Any]:
        """Get report on hottest data (most accessed)"""
        snapshot = self.get_cache_snapshot()

        # Categorize hot keys
        categories = {
            "paths": [],
            "data": [],
            "config": [],
            "other": []
        }

        for item in snapshot.hot_keys[:top_n]:
            key = item["key"]
            if key.startswith("path:"):
                categories["paths"].append(item)
            elif key.startswith("data:"):
                categories["data"].append(item)
            elif key.startswith("config:"):
                categories["config"].append(item)
            else:
                categories["other"].append(item)

        return {
            "timestamp": snapshot.timestamp,
            "total_hot_items": len(snapshot.hot_keys),
            "l1_utilization": f"{snapshot.l1_size}/64",
            "categories": categories,
            "recommendation": self._get_cache_recommendation(snapshot)
        }

    def _get_cache_recommendation(self, snapshot: CacheSnapshot) -> str:
        """Generate cache optimization recommendation"""
        if snapshot.l1_hit_rate > 0.8:
            return "Excellent: L1 cache is well-tuned for current workload"
        elif snapshot.l1_hit_rate > 0.5:
            return "Good: Consider promoting frequently accessed items to L1"
        elif snapshot.total_hit_rate > 0.7:
            return "Moderate: L2/L3 handling load, review L1 promotion thresholds"
        else:
            return "Poor: High cache misses, review access patterns"

    def get_component_activity(self) -> List[ComponentSnapshot]:
        """Get activity status of all components"""
        components = []
        metrics = self.vbus.get_metrics()

        for name, info in self.vbus._components.items():
            components.append(ComponentSnapshot(
                name=name,
                role=info["role"],
                status=info.get("status", "unknown"),
                last_activity=info.get("last_activity"),
                cached_items=0  # Would track per-component cache usage
            ))

        return components

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        metrics = self.vbus.get_metrics()
        cache = metrics["cache"]

        # Calculate health score (0-100)
        health_score = 0

        # Cache health (40 points)
        if cache["total_hit_rate"] > 0.8:
            health_score += 40
        elif cache["total_hit_rate"] > 0.5:
            health_score += 30
        elif cache["total_hit_rate"] > 0.3:
            health_score += 20
        else:
            health_score += 10

        # Component health (30 points)
        active_ratio = metrics["active_components"] / max(metrics["registered_components"], 1)
        health_score += int(active_ratio * 30)

        # Path resolution health (30 points)
        if metrics["registered_paths"] > 15:
            health_score += 30
        elif metrics["registered_paths"] > 10:
            health_score += 20
        else:
            health_score += 10

        return {
            "health_score": health_score,
            "status": "healthy" if health_score > 70 else "degraded" if health_score > 40 else "critical",
            "cache_hit_rate": f"{cache['total_hit_rate']*100:.1f}%",
            "l1_hit_rate": f"{cache['l1_hit_rate']*100:.1f}%",
            "components": f"{metrics['active_components']}/{metrics['registered_components']}",
            "registered_paths": metrics["registered_paths"],
            "processed_titles": metrics["processed_titles"],
            "uptime_seconds": metrics["uptime_seconds"]
        }

    def print_dashboard(self) -> None:
        """Print monitoring dashboard"""
        snapshot = self.get_cache_snapshot()
        health = self.get_system_health()
        hot_report = self.get_hot_data_report(10)

        print()
        print("╔══════════════════════════════════════════════════════════════════════════╗")
        print("║                         VBUS MONITOR DASHBOARD                           ║")
        print(f"║  Health: {health['status'].upper():<10} Score: {health['health_score']}/100                              ║")
        print("╠══════════════════════════════════════════════════════════════════════════╣")
        print("║  CACHE HIERARCHY STATUS                                                  ║")
        print("╠──────────────────────────────────────────────────────────────────────────╣")
        print(f"║  L1 (Hot):   [{self._bar(snapshot.l1_size, 64)}] {snapshot.l1_size:>3}/64   Hit: {snapshot.l1_hit_rate*100:>5.1f}%  ║")
        print(f"║  L2 (Warm):  [{self._bar(snapshot.l2_size, 256)}] {snapshot.l2_size:>3}/256                    ║")
        print(f"║  L3 (Cold):  [{self._bar(snapshot.l3_size, 1024)}] {snapshot.l3_size:>4}/1024                   ║")
        print(f"║  Total Hit Rate: {snapshot.total_hit_rate*100:>5.1f}%    Promotions: {snapshot.promotions:>4}  Demotions: {snapshot.demotions:>4}  ║")
        print("╠══════════════════════════════════════════════════════════════════════════╣")
        print("║  HOT DATA (Most Accessed)                                                ║")
        print("╠──────────────────────────────────────────────────────────────────────────╣")

        for i, item in enumerate(snapshot.hot_keys[:5]):
            key_display = item['key'][:40] + "..." if len(item['key']) > 40 else item['key']
            print(f"║  {i+1}. {key_display:<45} ({item['accesses']:>5} hits) ║")

        print("╠══════════════════════════════════════════════════════════════════════════╣")
        print(f"║  Recommendation: {hot_report['recommendation'][:54]:<54} ║")
        print("╠══════════════════════════════════════════════════════════════════════════╣")
        print("║  SYSTEM METRICS                                                          ║")
        print(f"║  Components: {health['components']:<10}  Paths: {health['registered_paths']:<5}  Titles: {health['processed_titles']:<10}  ║")
        print(f"║  Uptime: {health['uptime_seconds']:.0f}s                                                         ║"[:76] + "║")
        print("╚══════════════════════════════════════════════════════════════════════════╝")

    def _bar(self, value: int, max_val: int, width: int = 20) -> str:
        """Create a progress bar"""
        filled = int((value / max_val) * width) if max_val > 0 else 0
        return "█" * filled + "░" * (width - filled)

    def start_monitoring(self, interval_seconds: float = 5.0) -> None:
        """Start background monitoring"""
        self._monitoring = True

        def monitor_loop():
            while self._monitoring:
                snapshot = self.get_cache_snapshot()
                self._history.append(snapshot)
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]
                time.sleep(interval_seconds)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        self._monitoring = False

    def get_history(self) -> List[CacheSnapshot]:
        """Get monitoring history"""
        return self._history.copy()

    def export_metrics(self, path: Path = None) -> Path:
        """Export metrics to JSON file"""
        if path is None:
            path = Path("vbus_metrics.json")

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health": self.get_system_health(),
            "cache_snapshot": {
                "l1_size": self.get_cache_snapshot().l1_size,
                "l2_size": self.get_cache_snapshot().l2_size,
                "l3_size": self.get_cache_snapshot().l3_size,
                "hit_rate": self.get_cache_snapshot().total_hit_rate
            },
            "hot_data": self.get_hot_data_report(20),
            "history_length": len(self._history)
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        return path


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def monitor_vbus() -> VBUSMonitor:
    """Create and return a VBUS monitor"""
    return VBUSMonitor()


def print_vbus_status() -> None:
    """Print current VBUS status"""
    monitor = VBUSMonitor()
    monitor.print_dashboard()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from vbus_core import init_vbus, RunMode

    # Initialize VBUS
    vbus = init_vbus(RunMode.INCREMENTAL)

    # Simulate some cache activity
    print("Simulating cache activity...")
    for i in range(20):
        vbus.resolve("bfd", "test")
        vbus.resolve("star", "test")
        if i % 3 == 0:
            vbus.resolve("schema", "test")

    # Create monitor and print dashboard
    monitor = VBUSMonitor(vbus)
    monitor.print_dashboard()

    # Export metrics
    metrics_path = monitor.export_metrics()
    print(f"\nMetrics exported to: {metrics_path}")
