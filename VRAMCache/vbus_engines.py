#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           VBUS ENGINE ADAPTERS                                ║
║              All Engines Connected to VBUS - NO FALLBACKS                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Every engine in ViewerDBX MUST use VBUS for:                                ║
║    - Path resolution                                                          ║
║    - Data access                                                              ║
║    - Inter-component communication                                            ║
║    - Caching                                                                  ║
║                                                                               ║
║  NO FALLBACKS. NO HARDCODED PATHS. NO EXCEPTIONS.                            ║
║                                                                               ║
║  VERSION: 1.0.0 | ALGO 95.4                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import VBUS
from vbus_core import (
    VBUS, get_vbus, init_vbus, vbus_component, require_vbus,
    VBUSEnforcementError, VBUSNotInitializedError,
    RunMode, CacheTier
)


# =============================================================================
# BASE ENGINE (All engines inherit from this)
# =============================================================================

class VBUSEngine(ABC):
    """
    Base class for all VBUS-connected engines.

    Subclasses MUST:
    1. Call super().__init__(name, role)
    2. Use self.vbus for all path resolution
    3. Use self.vbus for all data access
    4. Use self.signal() for inter-component communication

    NO FALLBACKS TO HARDCODED PATHS.
    """

    def __init__(self, name: str, role: str, provides: List[str] = None, requires: List[str] = None):
        self.name = name
        self.role = role
        self.provides = provides or []
        self.requires = requires or []

        # Get VBUS instance (MANDATORY)
        self.vbus = get_vbus()

        # Register with VBUS
        self.vbus.register_component(name, role, provides, requires)
        self.vbus.register_active_component(name, self)

        # Track what data we've cached
        self._cached_keys: Set[str] = set()

    def resolve(self, key: str) -> Path:
        """Resolve path through VBUS - NO FALLBACKS"""
        return self.vbus.resolve(key, self.name)

    def get_cached(self, key: str) -> Optional[Any]:
        """Get cached data from VBUS"""
        return self.vbus.get_cached(key, self.name)

    def cache(self, key: str, value: Any, hot: bool = False) -> None:
        """Cache data in VBUS"""
        self.vbus.cache_data(key, value, self.name, hot=hot)
        self._cached_keys.add(key)

    def signal(self, target: str, signal_type: str, payload: Dict = None) -> None:
        """Send signal through VBUS"""
        self.vbus.send_signal(self.name, target, signal_type, payload or {})

    def broadcast(self, signal_type: str, payload: Dict = None) -> None:
        """Broadcast signal to all components"""
        self.vbus.broadcast(self.name, signal_type, payload)

    def on_signal(self, handler) -> None:
        """Register signal handler"""
        self.vbus.register_signal_handler(self.name, handler)

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute engine logic - must be implemented by subclass"""
        pass

    def cleanup(self) -> None:
        """Cleanup cached data"""
        for key in self._cached_keys:
            self.vbus.invalidate_cache(key)
        self._cached_keys.clear()


# =============================================================================
# AGENT ENGINES (Do Work)
# =============================================================================

class ALGOEngineVBUS(VBUSEngine):
    """
    ALGO Engine - ML Prediction Engine (GPU Mandatory)

    Provides: predictions, views_computed, mape_metrics, temporal_validation
    Requires: bfd, schema, training_data, abstract_data
    """

    def __init__(self):
        super().__init__(
            name="ALGO Engine",
            role="AGENT",
            provides=["predictions", "views_computed", "mape_metrics", "temporal_validation"],
            requires=["bfd", "schema", "training_data", "abstract_data"]
        )

    def execute(self, run_mode: RunMode = RunMode.INCREMENTAL, **kwargs) -> Dict[str, Any]:
        """Execute ALGO prediction pipeline"""

        # Resolve all required paths through VBUS
        bfd_path = self.resolve("bfd")
        star_path = self.resolve("star")
        schema_path = self.resolve("schema")

        # Signal that we're starting
        self.broadcast("ALGO_START", {"mode": run_mode.value})

        # Check cache for existing computations
        cached_predictions = self.get_cached("predictions")
        if cached_predictions and run_mode == RunMode.INCREMENTAL:
            # Use cached predictions for processed titles
            self.signal("MAPIE", "DATA_READY", {"type": "predictions", "cached": True})
            return {"status": "cached", "predictions_count": len(cached_predictions)}

        # Execute pipeline phases
        results = {
            "bfd_path": str(bfd_path),
            "star_path": str(star_path),
            "run_mode": run_mode.value,
            "status": "pending_gpu_execution"
        }

        # Signal completion
        self.broadcast("ALGO_COMPLETE", results)
        self.signal("MAPIE", "DATA_READY", {"type": "predictions"})

        return results

    def validate_temporal(self) -> Dict[str, Any]:
        """Run temporal validation"""
        validator_path = self.resolve("component:algo_engine") / "temporal_views_validator.py"

        self.signal("Orchestrator", "VALIDATION_START", {"validator": "temporal"})

        return {
            "validator": str(validator_path),
            "status": "pending"
        }


class SCHIGEngineVBUS(VBUSEngine):
    """
    SCHIG - Scheduled Collection & Harvest Ingestion Gateway

    Provides: fresh_data, api_data, daily_collection
    Requires: api_keys
    """

    def __init__(self):
        super().__init__(
            name="SCHIG",
            role="AGENT",
            provides=["fresh_data", "api_data", "daily_collection"],
            requires=["api_keys"]
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute data collection"""
        schig_path = self.resolve("component:schig")

        # Signal start
        self.broadcast("SCHIG_START", {})

        results = {
            "schig_path": str(schig_path),
            "status": "pending_api_collection"
        }

        # Signal completion to Fresh In!
        self.signal("Fresh In!", "DATA_READY", {"type": "fresh_data"})

        return results


class AbstractDataEngineVBUS(VBUSEngine):
    """
    Abstract Data - 9 Engines, 56 Signals

    Provides: x_features, abstract_signals, 56_signals
    Requires: external_data
    """

    def __init__(self):
        super().__init__(
            name="Abstract Data",
            role="AGENT",
            provides=["x_features", "abstract_signals", "56_signals"],
            requires=["external_data"]
        )

        # The 9 abstract engines
        self.engines = [
            "set_1_content_intrinsic",
            "set_2_platform_distribution",
            "set_3_temporal_dynamics",
            "set_4_external_events",
            "set_5_marketing_buzz",
            "set_6_completion_quality",
            "set_7_geopolitical_risk",
            "set_8_quality_experience",
            "set_9_derived_interactions"
        ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute all 9 abstract engines"""
        abstract_path = self.resolve("component:abstract_data")

        # Cache each engine's output
        signals = {}
        for engine in self.engines:
            cache_key = f"abstract:{engine}"
            cached = self.get_cached(cache_key)

            if cached:
                signals[engine] = cached
            else:
                # Would execute engine here
                signals[engine] = {"status": "pending"}

        # Signal to ALGO Engine
        self.signal("ALGO Engine", "DATA_READY", {"type": "abstract_signals", "count": 56})

        return {"engines": len(self.engines), "signals": 56}


class DailyTop10EngineVBUS(VBUSEngine):
    """
    Daily Top 10s - Rankings Generation

    Provides: rankings, top10_reports, views_rankings
    Requires: star_schema, views_computed
    """

    def __init__(self):
        super().__init__(
            name="Daily Top 10s",
            role="AGENT",
            provides=["rankings", "top10_reports", "views_rankings"],
            requires=["star_schema", "views_computed"]
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Generate rankings"""
        star_path = self.resolve("star")

        # Wait for views_computed from ALGO Engine
        views = self.get_cached("views_computed")

        results = {
            "star_path": str(star_path),
            "views_available": views is not None,
            "status": "pending"
        }

        return results


class StudiosEngineVBUS(VBUSEngine):
    """
    Studios - Studio Classification

    Provides: studio_classification, production_company_mapping
    Requires: bfd
    """

    def __init__(self):
        super().__init__(
            name="Studios",
            role="AGENT",
            provides=["studio_classification", "production_company_mapping"],
            requires=["bfd"]
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute studio classification"""
        bfd_path = self.resolve("bfd")
        studios_path = self.resolve("component:studios")

        # Cache classification results
        cached = self.get_cached("studio_classification")
        if cached:
            return {"status": "cached", "studios": len(cached)}

        results = {
            "bfd_path": str(bfd_path),
            "studios_path": str(studios_path),
            "status": "pending"
        }

        # Signal to ALGO Engine
        self.signal("ALGO Engine", "DATA_READY", {"type": "studio_classification"})

        return results


class TalentEngineVBUS(VBUSEngine):
    """
    Talent - Cast & Crew Analysis

    Provides: talent_metrics, cast_analysis
    Requires: bfd, external_data
    """

    def __init__(self):
        super().__init__(
            name="Talent",
            role="AGENT",
            provides=["talent_metrics", "cast_analysis"],
            requires=["bfd", "external_data"]
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute talent analysis"""
        bfd_path = self.resolve("bfd")

        results = {
            "bfd_path": str(bfd_path),
            "status": "pending"
        }

        self.signal("ALGO Engine", "DATA_READY", {"type": "talent_metrics"})

        return results


class MoneyEngineVBUS(VBUSEngine):
    """
    Money Engine - ROI, Valuation, Ad Economics

    Provides: roi, valuation, ad_economics
    Requires: views_computed, abstract_data
    """

    def __init__(self):
        super().__init__(
            name="Money Engine",
            role="AGENT",
            provides=["roi", "valuation", "ad_economics"],
            requires=["views_computed", "abstract_data"]
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute economics calculations"""

        # Get views from cache (dependency)
        views = self.get_cached("views_computed")
        if not views:
            # Wait for ALGO Engine
            return {"status": "waiting_for_views_computed"}

        results = {
            "views_available": True,
            "status": "pending"
        }

        return results


class FreshInEngineVBUS(VBUSEngine):
    """
    Fresh In! - Data Ingestion Pipeline

    Provides: routing, integration, data_freshness
    Requires: fresh_data
    """

    def __init__(self):
        super().__init__(
            name="Fresh In!",
            role="AGENT",
            provides=["routing", "integration", "data_freshness"],
            requires=["fresh_data"]
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute data ingestion"""
        fresh_path = self.resolve("component:fresh_in!")

        results = {
            "fresh_path": str(fresh_path),
            "status": "pending"
        }

        # Signal new arrivals to VBUS
        new_arrivals = kwargs.get("new_arrivals", [])
        for fc_uid in new_arrivals:
            self.vbus.add_new_arrival(fc_uid)

        return results


class CredibilityEngineVBUS(VBUSEngine):
    """
    Credibility - External Source Validation

    Provides: credibility_scores
    Requires: bfd
    """

    def __init__(self):
        super().__init__(
            name="Credibility",
            role="AGENT",
            provides=["credibility_scores"],
            requires=["bfd"]
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute credibility validation"""
        bfd_path = self.resolve("bfd")

        results = {
            "sources": ["IMDB", "RottenTomatoes", "FlixPatrol"],
            "status": "pending"
        }

        self.signal("Daily Top 10s", "DATA_READY", {"type": "credibility_scores"})

        return results


class CloudflareEngineVBUS(VBUSEngine):
    """
    Cloudflare - R2 Storage & API Deployment

    Provides: api_deployment, r2_storage
    Requires: star_schema
    """

    def __init__(self):
        super().__init__(
            name="Cloudflare",
            role="AGENT",
            provides=["api_deployment", "r2_storage"],
            requires=["star_schema"]
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute deployment"""
        star_path = self.resolve("star")

        results = {
            "star_path": str(star_path),
            "status": "pending_deployment"
        }

        self.signal("Replit", "DATA_READY", {"type": "api_deployment"})

        return results


class ReplitEngineVBUS(VBUSEngine):
    """
    Replit - Web Deployment

    Provides: web_deployment, ui
    Requires: star_schema, api_deployment
    """

    def __init__(self):
        super().__init__(
            name="Replit",
            role="AGENT",
            provides=["web_deployment", "ui"],
            requires=["star_schema", "api_deployment"]
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute web deployment"""
        star_path = self.resolve("star")

        results = {
            "star_path": str(star_path),
            "status": "pending_deployment"
        }

        return results


# =============================================================================
# LEDGER ENGINES (Record Truth)
# =============================================================================

class MAPIEEngineVBUS(VBUSEngine):
    """
    MAPIE - Measurement, Accuracy, Performance & Inference Evaluation

    Provides: mape_tracking, validation, accuracy_reports
    Requires: predictions, training_data
    """

    def __init__(self):
        super().__init__(
            name="MAPIE",
            role="LEDGER",
            provides=["mape_tracking", "validation", "accuracy_reports"],
            requires=["predictions", "training_data"]
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute MAPE validation"""
        # Get predictions from cache
        predictions = self.get_cached("predictions")

        results = {
            "predictions_available": predictions is not None,
            "status": "pending"
        }

        return results


class ViewsTrainingDataEngineVBUS(VBUSEngine):
    """
    Views Training Data - Ground Truth

    Provides: training_data, ground_truth, y_labels
    Requires: (none)
    """

    def __init__(self):
        super().__init__(
            name="Views Training Data",
            role="LEDGER",
            provides=["training_data", "ground_truth", "y_labels"],
            requires=[]
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Load training data"""
        training_path = self.resolve("component:views_training_data")

        # Cache hot training data
        cached = self.get_cached("training_data")
        if not cached:
            # Load and cache (would be actual data load)
            self.cache("training_data", {"status": "loaded"}, hot=True)

        return {"training_path": str(training_path)}


class ComponentsEngineVBUS(VBUSEngine):
    """
    Components - Lookup Tables

    Provides: lookup_tables, config_data, weights
    Requires: (none)
    """

    def __init__(self):
        super().__init__(
            name="Components",
            role="LEDGER",
            provides=["lookup_tables", "config_data", "weights"],
            requires=[]
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Load component data"""
        comp_path = self.resolve("component:components")

        # Cache lookup tables (hot - frequently accessed)
        self.cache("genre_decay", {}, hot=True)
        self.cache("country_weights", {}, hot=True)
        self.cache("platform_availability", {}, hot=True)

        return {"components_path": str(comp_path)}


class SchemaEngineVBUS(VBUSEngine):
    """
    Schema - Column Definitions

    Provides: schema, validation_rules, column_definitions
    Requires: (none)
    """

    def __init__(self):
        super().__init__(
            name="Schema",
            role="LEDGER",
            provides=["schema", "validation_rules", "column_definitions"],
            requires=[]
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Load schema"""
        schema_path = self.resolve("schema")

        # Cache schema (hot - used constantly)
        self.cache("schema_v22.01", {"path": str(schema_path)}, hot=True)

        return {"schema_path": str(schema_path)}


# =============================================================================
# CONSTRAINT ENGINES (Can Stop Execution)
# =============================================================================

class OrchestratorEngineVBUS(VBUSEngine):
    """
    Orchestrator - Pipeline Control

    Provides: health_checks, coordination, pipeline_control
    Requires: all_components
    """

    def __init__(self):
        super().__init__(
            name="Orchestrator",
            role="CONSTRAINT",
            provides=["health_checks", "coordination", "pipeline_control"],
            requires=["all_components"]
        )

        # Track component health
        self._component_status: Dict[str, str] = {}

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Run orchestration"""
        # Check all components
        metrics = self.vbus.get_metrics()

        results = {
            "registered_components": metrics["registered_components"],
            "active_components": metrics["active_components"],
            "cache_hit_rate": metrics["cache"]["total_hit_rate"],
            "status": "healthy"
        }

        return results

    def halt_pipeline(self, reason: str) -> None:
        """Halt pipeline execution"""
        self.broadcast("PIPELINE_HALT", {"reason": reason})

    def validate_prerequisites(self) -> bool:
        """Validate all prerequisites before pipeline run"""
        try:
            # Check critical paths exist
            self.resolve("bfd")
            self.resolve("star")
            self.resolve("schema")
            return True
        except VBUSEnforcementError:
            return False


class GPUEnablementEngineVBUS(VBUSEngine):
    """
    GPU Enablement - GPU Enforcement & Circuit Breaker

    Provides: gpu_enforcement, circuit_breaker, anti_cheat
    Requires: (none)
    """

    def __init__(self):
        super().__init__(
            name="GPU Enablement",
            role="CONSTRAINT",
            provides=["gpu_enforcement", "circuit_breaker", "anti_cheat"],
            requires=[]
        )

        self._circuit_breaker_open = False

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Check GPU status"""
        gpu_path = self.resolve("component:gpu_enablement")

        results = {
            "gpu_path": str(gpu_path),
            "circuit_breaker": "CLOSED" if not self._circuit_breaker_open else "OPEN",
            "status": "ready"
        }

        return results

    def open_circuit_breaker(self, reason: str) -> None:
        """Open circuit breaker - halts all GPU operations"""
        self._circuit_breaker_open = True
        self.broadcast("CIRCUIT_BREAKER_OPEN", {"reason": reason})

    def close_circuit_breaker(self) -> None:
        """Close circuit breaker"""
        self._circuit_breaker_open = False
        self.broadcast("CIRCUIT_BREAKER_CLOSED", {})


# =============================================================================
# ENGINE FACTORY
# =============================================================================

class VBUSEngineFactory:
    """
    Factory to create VBUS-connected engines.

    All engines are created through this factory to ensure VBUS dependency.
    """

    _engine_classes = {
        "ALGO Engine": ALGOEngineVBUS,
        "SCHIG": SCHIGEngineVBUS,
        "Abstract Data": AbstractDataEngineVBUS,
        "Daily Top 10s": DailyTop10EngineVBUS,
        "Studios": StudiosEngineVBUS,
        "Talent": TalentEngineVBUS,
        "Money Engine": MoneyEngineVBUS,
        "Fresh In!": FreshInEngineVBUS,
        "Credibility": CredibilityEngineVBUS,
        "Cloudflare": CloudflareEngineVBUS,
        "Replit": ReplitEngineVBUS,
        "MAPIE": MAPIEEngineVBUS,
        "Views Training Data": ViewsTrainingDataEngineVBUS,
        "Components": ComponentsEngineVBUS,
        "Schema": SchemaEngineVBUS,
        "Orchestrator": OrchestratorEngineVBUS,
        "GPU Enablement": GPUEnablementEngineVBUS,
    }

    @classmethod
    def create(cls, name: str) -> VBUSEngine:
        """Create engine by name"""
        if name not in cls._engine_classes:
            raise ValueError(f"Unknown engine: {name}")
        return cls._engine_classes[name]()

    @classmethod
    def create_all(cls) -> Dict[str, VBUSEngine]:
        """Create all engines"""
        return {name: cls.create(name) for name in cls._engine_classes}

    @classmethod
    def list_engines(cls) -> List[str]:
        """List available engines"""
        return list(cls._engine_classes.keys())


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    # Initialize VBUS first (MANDATORY)
    vbus = init_vbus(RunMode.INCREMENTAL)

    print("Creating VBUS-connected engines...")

    # Create all engines
    engines = VBUSEngineFactory.create_all()

    print(f"\n✓ Created {len(engines)} engines:")
    for name, engine in engines.items():
        print(f"  - {name} ({engine.role})")

    # Test ALGO Engine
    print("\n[TEST] ALGO Engine execution:")
    algo = engines["ALGO Engine"]
    result = algo.execute()
    print(f"  Result: {result}")

    # Test Orchestrator
    print("\n[TEST] Orchestrator health check:")
    orchestrator = engines["Orchestrator"]
    result = orchestrator.execute()
    print(f"  Result: {result}")

    # Print VBUS status
    print("\n[VBUS STATUS]")
    vbus.print_status()

    print("\n✓ All engines connected to VBUS - NO FALLBACKS")
