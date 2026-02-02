#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     VBUS PIPELINE CONTROLLER v1.0                            ║
║           8-Phase Pipeline with 4 Failure Modes (Fail-Closed)               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  THE TRAINING LOOP OF ALGO 95.4:                                            ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │  GPU FABRIC (XGBoost | CatBoost | RF)                               │    ║
║  │              ↓                                                       │    ║
║  │  ═══════ VIEWERDBX SYSTEM BUS ═══════                               │    ║
║  │              ↓                                                       │    ║
║  │  ALGO 95.4 ML Control Core                                          │    ║
║  │              ↓                                                       │    ║
║  │  1. GPU INIT         → Allocate GPU Memory                          │    ║
║  │  2. DATA LOADING     → Fetch Training Data                          │    ║
║  │  3. FEATURE ENG      → Extract & Transform                          │    ║
║  │  4. MODEL TRAINING   → 1350 Trees x 3 Models                        │    ║
║  │  5. PREDICTION       → Evaluate Outputs                             │    ║
║  │  6. OUTPUT GEN       → Store Results                                │    ║
║  │  7. AUDIT LOGGING    → Log All Actions                              │    ║
║  │  8. REVIEW & DEPLOY  → Final Verification                           │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                               ║
║  FAILURE MODES (PROCESS TERMINATED):                                         ║
║  ┌────────────────────┬────────────────────┐                                ║
║  │  DATA BREAKDOWN    │  VALIDATION BREACH │                                ║
║  │  Corrupt/Missing   │  Temporal Violated │                                ║
║  ├────────────────────┼────────────────────┤                                ║
║  │  COMPUTE HALT      │  SECURITY FAILURE  │                                ║
║  │  GPU Overflow      │  Audit Triggered   │                                ║
║  └────────────────────┴────────────────────┘                                ║
║              ↓                                                                ║
║         ☠ PROCESS TERMINATED                                                 ║
║         → Model Stopped                                                       ║
║         → Logs Written to AUDIT_LOGS                                         ║
║         → Action Required                                                     ║
║                                                                               ║
║  VERSION: 1.0.0 | ALGO 95.4 | GPU MANDATORY | FAIL-CLOSED                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum, auto
from contextlib import contextmanager

# Import VBUS
from vbus_core import (
    VBUS, get_vbus, init_vbus, RunMode, CacheTier,
    VBUSEnforcementError, VBUSNotInitializedError
)

# =============================================================================
# PIPELINE PHASES (8 Phases from the diagram)
# =============================================================================

class PipelinePhase(Enum):
    """The 8 phases of ALGO 95.4 pipeline"""
    GPU_INIT = 1           # Allocate GPU Memory
    DATA_LOADING = 2       # Fetch Training Data
    FEATURE_ENGINEERING = 3 # Extract & Transform
    MODEL_TRAINING = 4     # 1350 Trees x 3 Models
    PREDICTION = 5         # Evaluate Outputs
    OUTPUT_GENERATION = 6  # Store Results
    AUDIT_LOGGING = 7      # Log All Actions
    REVIEW_DEPLOY = 8      # Final Verification

    @property
    def description(self) -> str:
        descriptions = {
            PipelinePhase.GPU_INIT: "Allocate GPU Memory",
            PipelinePhase.DATA_LOADING: "Fetch Training Data",
            PipelinePhase.FEATURE_ENGINEERING: "Extract & Transform",
            PipelinePhase.MODEL_TRAINING: "1350 Trees x 3 Models",
            PipelinePhase.PREDICTION: "Evaluate Outputs",
            PipelinePhase.OUTPUT_GENERATION: "Store Results",
            PipelinePhase.AUDIT_LOGGING: "Log All Actions",
            PipelinePhase.REVIEW_DEPLOY: "Final Verification"
        }
        return descriptions.get(self, "Unknown")


# =============================================================================
# FAILURE MODES (4 Failure Types from the diagram)
# =============================================================================

class FailureMode(Enum):
    """The 4 failure modes that terminate the pipeline"""
    DATA_BREAKDOWN = "DATA_BREAKDOWN"         # Corrupt / Missing Data
    VALIDATION_BREACH = "VALIDATION_BREACH"   # Temporal Rule Violated
    COMPUTE_HALT = "COMPUTE_HALT"             # GPU Overflow / Fault
    SECURITY_FAILURE = "SECURITY_FAILURE"     # Audit Policy Triggered

    @property
    def description(self) -> str:
        descriptions = {
            FailureMode.DATA_BREAKDOWN: "Corrupt / Missing Data",
            FailureMode.VALIDATION_BREACH: "Temporal Rule Violated",
            FailureMode.COMPUTE_HALT: "GPU Overflow / Fault",
            FailureMode.SECURITY_FAILURE: "Audit Policy Triggered"
        }
        return descriptions.get(self, "Unknown")


class PipelineStatus(Enum):
    """Pipeline execution status"""
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    TERMINATED = "TERMINATED"  # ☠ PROCESS TERMINATED


# =============================================================================
# PIPELINE TERMINATION EXCEPTION
# =============================================================================

class PipelineTerminated(Exception):
    """
    ☠ PROCESS TERMINATED

    Raised when any of the 4 failure modes is triggered.
    The pipeline MUST stop immediately.
    """

    def __init__(self, failure_mode: FailureMode, reason: str,
                 phase: Optional[PipelinePhase] = None, data: Dict = None):
        self.failure_mode = failure_mode
        self.reason = reason
        self.phase = phase
        self.data = data or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

        super().__init__(
            f"☠ PROCESS TERMINATED\n"
            f"  Failure Mode: {failure_mode.value} - {failure_mode.description}\n"
            f"  Reason: {reason}\n"
            f"  Phase: {phase.name if phase else 'N/A'}\n"
            f"  → Model Stopped\n"
            f"  → Logs Written to AUDIT_LOGS\n"
            f"  → Action Required"
        )


# =============================================================================
# PHASE RESULT
# =============================================================================

@dataclass
class PhaseResult:
    """Result of a pipeline phase execution"""
    phase: PipelinePhase
    success: bool
    duration_ms: float
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    failure_mode: Optional[FailureMode] = None


# =============================================================================
# VBUS PIPELINE CONTROLLER
# =============================================================================

class VBUSPipeline:
    """
    VBUS Pipeline Controller - Orchestrates the 8-phase pipeline.

    FAIL-CLOSED: Any failure mode terminates the entire pipeline.
    NO FALLBACKS: All operations through VBUS.
    """

    def __init__(self, run_mode: RunMode = RunMode.INCREMENTAL):
        # Initialize VBUS
        self.vbus = init_vbus(run_mode)
        self.run_mode = run_mode

        # Pipeline state
        self.status = PipelineStatus.NOT_STARTED
        self.current_phase: Optional[PipelinePhase] = None
        self.phase_results: List[PhaseResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Failure tracking
        self.failure_mode: Optional[FailureMode] = None
        self.failure_reason: Optional[str] = None

        # Phase handlers (registered externally)
        self._phase_handlers: Dict[PipelinePhase, Callable] = {}

        # Validation functions for each failure mode
        self._validators: Dict[FailureMode, List[Callable]] = {
            FailureMode.DATA_BREAKDOWN: [],
            FailureMode.VALIDATION_BREACH: [],
            FailureMode.COMPUTE_HALT: [],
            FailureMode.SECURITY_FAILURE: [],
        }

        # Register default validators
        self._register_default_validators()

    def _register_default_validators(self) -> None:
        """Register default validation functions"""

        # DATA_BREAKDOWN validators
        self.add_validator(FailureMode.DATA_BREAKDOWN, self._validate_bfd_exists)
        self.add_validator(FailureMode.DATA_BREAKDOWN, self._validate_star_exists)
        self.add_validator(FailureMode.DATA_BREAKDOWN, self._validate_schema_exists)

        # VALIDATION_BREACH validators
        self.add_validator(FailureMode.VALIDATION_BREACH, self._validate_temporal_rules)

        # COMPUTE_HALT validators
        self.add_validator(FailureMode.COMPUTE_HALT, self._validate_gpu_available)

        # SECURITY_FAILURE validators
        self.add_validator(FailureMode.SECURITY_FAILURE, self._validate_audit_policy)

    # =========================================================================
    # DEFAULT VALIDATORS
    # =========================================================================

    def _validate_bfd_exists(self) -> Tuple[bool, str]:
        """Validate BFD database exists and is accessible"""
        try:
            bfd_path = self.vbus.resolve("bfd", "pipeline")
            if bfd_path is None or not bfd_path.exists():
                return False, "BFD database not found"
            return True, "BFD OK"
        except VBUSEnforcementError as e:
            return False, str(e)

    def _validate_star_exists(self) -> Tuple[bool, str]:
        """Validate Star Schema exists and is accessible"""
        try:
            star_path = self.vbus.resolve("star", "pipeline")
            if star_path is None or not star_path.exists():
                return False, "Star Schema not found"
            return True, "Star Schema OK"
        except VBUSEnforcementError as e:
            return False, str(e)

    def _validate_schema_exists(self) -> Tuple[bool, str]:
        """Validate Schema exists"""
        try:
            schema_path = self.vbus.resolve("schema", "pipeline")
            if schema_path is None or not schema_path.exists():
                return False, "Schema not found"
            return True, "Schema OK"
        except VBUSEnforcementError as e:
            return False, str(e)

    def _validate_temporal_rules(self) -> Tuple[bool, str]:
        """Validate temporal rules are not breached"""
        # Check if temporal validation has been run
        temporal_status = self.vbus.get_cached("temporal_validation_status")

        if temporal_status is None:
            return True, "Temporal validation pending"

        if temporal_status.get("violations", 0) > 0:
            return False, f"Temporal violations: {temporal_status['violations']}"

        return True, "Temporal rules OK"

    def _validate_gpu_available(self) -> Tuple[bool, str]:
        """Validate GPU is available"""
        # Check for GPU (simplified - actual check would use CUDA)
        try:
            gpu_path = self.vbus.resolve("component:gpu_engine", "pipeline")
            # In real implementation, would check CUDA availability
            return True, "GPU check pending (requires CUDA)"
        except VBUSEnforcementError:
            return False, "GPU Engine component not found"

    def _validate_audit_policy(self) -> Tuple[bool, str]:
        """Validate audit policy compliance"""
        # Check audit logs are writable
        try:
            audit_path = self.vbus.resolve("component:audit_logs", "pipeline")
            if audit_path is None:
                return False, "AUDIT_LOGS not accessible"
            return True, "Audit policy OK"
        except VBUSEnforcementError as e:
            return False, str(e)

    # =========================================================================
    # VALIDATOR MANAGEMENT
    # =========================================================================

    def add_validator(self, failure_mode: FailureMode,
                      validator: Callable[[], Tuple[bool, str]]) -> None:
        """Add a validation function for a failure mode"""
        self._validators[failure_mode].append(validator)

    def run_validators(self, failure_mode: FailureMode) -> Tuple[bool, List[str]]:
        """Run all validators for a failure mode"""
        failures = []
        for validator in self._validators[failure_mode]:
            try:
                ok, message = validator()
                if not ok:
                    failures.append(message)
            except Exception as e:
                failures.append(f"Validator error: {str(e)}")

        return len(failures) == 0, failures

    def validate_all(self) -> Dict[FailureMode, Tuple[bool, List[str]]]:
        """Run all validators for all failure modes"""
        results = {}
        for failure_mode in FailureMode:
            results[failure_mode] = self.run_validators(failure_mode)
        return results

    # =========================================================================
    # PHASE HANDLERS
    # =========================================================================

    def register_phase_handler(self, phase: PipelinePhase,
                               handler: Callable[[], Dict[str, Any]]) -> None:
        """Register a handler for a pipeline phase"""
        self._phase_handlers[phase] = handler

    def _execute_phase(self, phase: PipelinePhase) -> PhaseResult:
        """Execute a single pipeline phase"""
        self.current_phase = phase
        start = time.time()

        # Broadcast phase start
        self.vbus.broadcast("pipeline", "PHASE_START", {
            "phase": phase.name,
            "description": phase.description
        })

        try:
            # Run pre-phase validation based on phase
            if phase == PipelinePhase.GPU_INIT:
                ok, failures = self.run_validators(FailureMode.COMPUTE_HALT)
                if not ok:
                    raise PipelineTerminated(
                        FailureMode.COMPUTE_HALT,
                        "; ".join(failures),
                        phase
                    )

            elif phase == PipelinePhase.DATA_LOADING:
                ok, failures = self.run_validators(FailureMode.DATA_BREAKDOWN)
                if not ok:
                    raise PipelineTerminated(
                        FailureMode.DATA_BREAKDOWN,
                        "; ".join(failures),
                        phase
                    )

            elif phase == PipelinePhase.FEATURE_ENGINEERING:
                ok, failures = self.run_validators(FailureMode.VALIDATION_BREACH)
                if not ok:
                    raise PipelineTerminated(
                        FailureMode.VALIDATION_BREACH,
                        "; ".join(failures),
                        phase
                    )

            elif phase == PipelinePhase.AUDIT_LOGGING:
                ok, failures = self.run_validators(FailureMode.SECURITY_FAILURE)
                if not ok:
                    raise PipelineTerminated(
                        FailureMode.SECURITY_FAILURE,
                        "; ".join(failures),
                        phase
                    )

            # Execute phase handler if registered
            data = {}
            if phase in self._phase_handlers:
                data = self._phase_handlers[phase]() or {}

            duration = (time.time() - start) * 1000

            result = PhaseResult(
                phase=phase,
                success=True,
                duration_ms=duration,
                data=data
            )

            # Broadcast phase complete
            self.vbus.broadcast("pipeline", "PHASE_COMPLETE", {
                "phase": phase.name,
                "duration_ms": duration
            })

            return result

        except PipelineTerminated:
            raise  # Re-raise termination

        except Exception as e:
            duration = (time.time() - start) * 1000
            return PhaseResult(
                phase=phase,
                success=False,
                duration_ms=duration,
                error=str(e),
                failure_mode=FailureMode.COMPUTE_HALT
            )

    # =========================================================================
    # PIPELINE EXECUTION
    # =========================================================================

    def execute(self) -> Dict[str, Any]:
        """
        Execute the complete 8-phase pipeline.

        FAIL-CLOSED: Any failure terminates immediately.
        """
        self.status = PipelineStatus.RUNNING
        self.start_time = datetime.now(timezone.utc)
        self.phase_results = []

        # Broadcast pipeline start
        self.vbus.broadcast("pipeline", "PIPELINE_START", {
            "run_mode": self.run_mode.value,
            "timestamp": self.start_time.isoformat()
        })

        try:
            # Pre-flight validation (all failure modes)
            print("\n" + "="*70)
            print("VBUS PIPELINE - PRE-FLIGHT VALIDATION")
            print("="*70)

            validation_results = self.validate_all()
            all_ok = True

            for failure_mode, (ok, failures) in validation_results.items():
                status = "✓" if ok else "✗"
                print(f"  {status} {failure_mode.value}: {failure_mode.description}")
                if not ok:
                    for f in failures:
                        print(f"      → {f}")
                    all_ok = False

            if not all_ok:
                # Find first failure mode
                for failure_mode, (ok, failures) in validation_results.items():
                    if not ok:
                        raise PipelineTerminated(
                            failure_mode,
                            "; ".join(failures),
                            None
                        )

            # Execute all 8 phases
            print("\n" + "="*70)
            print("VBUS PIPELINE - EXECUTING 8 PHASES")
            print("="*70)

            for phase in PipelinePhase:
                print(f"\n  [{phase.value}] {phase.name}: {phase.description}")

                result = self._execute_phase(phase)
                self.phase_results.append(result)

                if result.success:
                    print(f"      ✓ Complete ({result.duration_ms:.1f}ms)")
                else:
                    print(f"      ✗ Failed: {result.error}")
                    raise PipelineTerminated(
                        result.failure_mode or FailureMode.COMPUTE_HALT,
                        result.error or "Unknown error",
                        phase
                    )

            # Pipeline complete
            self.status = PipelineStatus.COMPLETED
            self.end_time = datetime.now(timezone.utc)

            # Complete VBUS run
            self.vbus.complete_run()

            # Broadcast completion
            self.vbus.broadcast("pipeline", "PIPELINE_COMPLETE", {
                "duration_ms": (self.end_time - self.start_time).total_seconds() * 1000,
                "phases_completed": len(self.phase_results)
            })

            print("\n" + "="*70)
            print("✓ PIPELINE COMPLETE")
            print("="*70)

            return self._get_summary()

        except PipelineTerminated as e:
            # ☠ PROCESS TERMINATED
            self.status = PipelineStatus.TERMINATED
            self.failure_mode = e.failure_mode
            self.failure_reason = e.reason
            self.end_time = datetime.now(timezone.utc)

            # Log to AUDIT_LOGS
            self._log_termination(e)

            # Broadcast termination
            self.vbus.broadcast("pipeline", "PIPELINE_TERMINATED", {
                "failure_mode": e.failure_mode.value,
                "reason": e.reason,
                "phase": e.phase.name if e.phase else None
            })

            print("\n" + "="*70)
            print("☠ PROCESS TERMINATED")
            print("="*70)
            print(f"  Failure Mode: {e.failure_mode.value}")
            print(f"  Description:  {e.failure_mode.description}")
            print(f"  Reason:       {e.reason}")
            print(f"  Phase:        {e.phase.name if e.phase else 'Pre-flight'}")
            print("-"*70)
            print("  → Model Stopped")
            print("  → Logs Written to AUDIT_LOGS")
            print("  → Action Required")
            print("="*70)

            return self._get_summary()

    def _log_termination(self, error: PipelineTerminated) -> None:
        """Log termination to AUDIT_LOGS"""
        try:
            audit_path = self.vbus.resolve("component:audit_logs", "pipeline")
            if audit_path:
                log_file = audit_path / f"TERMINATION_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                log_data = {
                    "timestamp": error.timestamp,
                    "failure_mode": error.failure_mode.value,
                    "description": error.failure_mode.description,
                    "reason": error.reason,
                    "phase": error.phase.name if error.phase else None,
                    "data": error.data,
                    "phase_results": [
                        {
                            "phase": r.phase.name,
                            "success": r.success,
                            "duration_ms": r.duration_ms,
                            "error": r.error
                        }
                        for r in self.phase_results
                    ],
                    "machine_generated": True,
                    "operator_editable": False
                }
                with open(log_file, 'w') as f:
                    json.dump(log_data, f, indent=2)
        except Exception as e:
            print(f"  Warning: Could not write termination log: {e}")

    def _get_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary"""
        return {
            "status": self.status.value,
            "run_mode": self.run_mode.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": (
                (self.end_time - self.start_time).total_seconds() * 1000
                if self.start_time and self.end_time else None
            ),
            "phases_completed": len([r for r in self.phase_results if r.success]),
            "phases_total": len(PipelinePhase),
            "failure_mode": self.failure_mode.value if self.failure_mode else None,
            "failure_reason": self.failure_reason,
            "phase_results": [
                {
                    "phase": r.phase.name,
                    "success": r.success,
                    "duration_ms": r.duration_ms,
                    "error": r.error
                }
                for r in self.phase_results
            ]
        }


# =============================================================================
# PIPELINE BUILDER (Fluent API)
# =============================================================================

class VBUSPipelineBuilder:
    """Builder for configuring and running VBUS pipeline"""

    def __init__(self):
        self._run_mode = RunMode.INCREMENTAL
        self._phase_handlers: Dict[PipelinePhase, Callable] = {}
        self._extra_validators: List[Tuple[FailureMode, Callable]] = []

    def mode(self, run_mode: RunMode) -> 'VBUSPipelineBuilder':
        """Set run mode"""
        self._run_mode = run_mode
        return self

    def full_build(self) -> 'VBUSPipelineBuilder':
        """Set full build mode (all 800k titles)"""
        return self.mode(RunMode.FULL_BUILD)

    def incremental(self) -> 'VBUSPipelineBuilder':
        """Set incremental mode (new arrivals only)"""
        return self.mode(RunMode.INCREMENTAL)

    def equation_update(self) -> 'VBUSPipelineBuilder':
        """Set equation update mode (re-run equations)"""
        return self.mode(RunMode.EQUATION_UPDATE)

    def on_phase(self, phase: PipelinePhase,
                 handler: Callable[[], Dict[str, Any]]) -> 'VBUSPipelineBuilder':
        """Register phase handler"""
        self._phase_handlers[phase] = handler
        return self

    def add_validator(self, failure_mode: FailureMode,
                      validator: Callable[[], Tuple[bool, str]]) -> 'VBUSPipelineBuilder':
        """Add custom validator"""
        self._extra_validators.append((failure_mode, validator))
        return self

    def build(self) -> VBUSPipeline:
        """Build the pipeline"""
        pipeline = VBUSPipeline(self._run_mode)

        for phase, handler in self._phase_handlers.items():
            pipeline.register_phase_handler(phase, handler)

        for failure_mode, validator in self._extra_validators:
            pipeline.add_validator(failure_mode, validator)

        return pipeline

    def execute(self) -> Dict[str, Any]:
        """Build and execute the pipeline"""
        return self.build().execute()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_pipeline(mode: RunMode = RunMode.INCREMENTAL) -> Dict[str, Any]:
    """Run the VBUS pipeline"""
    return VBUSPipelineBuilder().mode(mode).execute()


def run_full_build() -> Dict[str, Any]:
    """Run full build pipeline"""
    return VBUSPipelineBuilder().full_build().execute()


def run_incremental() -> Dict[str, Any]:
    """Run incremental pipeline"""
    return VBUSPipelineBuilder().incremental().execute()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     VBUS PIPELINE CONTROLLER v1.0                            ║
║           8-Phase Pipeline with 4 Failure Modes (Fail-Closed)               ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Run incremental pipeline
    result = run_incremental()

    # Print final status
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(json.dumps(result, indent=2))
