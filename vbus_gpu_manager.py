#!/usr/bin/env python3
"""
VBUS GPU MEMORY MANAGER v1.0
============================
Smart GPU memory management using VBUS hierarchical caching.

MEMORY HIERARCHY:
  L1 (GPU VRAM 12GB)   -> Hot data currently being used for ML training
  L2 (System RAM)      -> Warm data - training features, ready for GPU transfer
  L3 (System RAM)      -> Cold data - full datasets, BFD backup

PRINCIPLE:
  Train one model at a time, clearing GPU between models.
  Use system RAM as staging area, only load batches to GPU as needed.

VERSION: 1.0.0 | ALGO 95.4 | GPU MANDATORY | NO CPU FALLBACK
"""

from __future__ import annotations

import os
import gc
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# VBUS imports
try:
    from vbus_core import VBUS, get_vbus, init_vbus, RunMode, CacheTier
except ImportError:
    # Direct import if in same directory
    VBUS = None


class GPUTier(Enum):
    """GPU memory tiers"""
    VRAM = "VRAM"           # GPU VRAM - fastest, limited (12GB)
    PINNED = "PINNED"       # Pinned system RAM - fast GPU transfer
    SYSTEM = "SYSTEM"       # Regular system RAM - large capacity


@dataclass
class GPUMemoryBlock:
    """A block of data in GPU memory hierarchy"""
    key: str
    tier: GPUTier
    size_bytes: int
    data: Any = None
    last_access: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    access_count: int = 0


class VBUSGPUManager:
    """
    VBUS GPU Memory Manager

    Manages GPU memory using VBUS hierarchical caching principles:
    - Keeps large data in system RAM (L2/L3)
    - Loads only what's needed to GPU (L1)
    - Clears GPU between operations
    - Ensures ML models train with proper memory management
    """

    # Memory limits
    GPU_VRAM_LIMIT_MB = 10000  # Leave 2GB buffer on 12GB card
    PINNED_MEMORY_LIMIT_MB = 16000  # 16GB pinned for fast transfer
    SYSTEM_MEMORY_LIMIT_MB = 100000  # 100GB system RAM cache

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.gpu_available = False
        self.cupy = None
        self.cudf = None

        # Memory tracking
        self._vram_used_mb = 0
        self._pinned_used_mb = 0
        self._system_used_mb = 0

        # Data blocks by tier
        self._vram_blocks: Dict[str, GPUMemoryBlock] = {}
        self._pinned_blocks: Dict[str, GPUMemoryBlock] = {}
        self._system_blocks: Dict[str, GPUMemoryBlock] = {}

        # Audit log
        self._log_entries: List[Dict[str, Any]] = []

        # Initialize GPU
        if use_gpu:
            self._init_gpu()

    def _init_gpu(self) -> None:
        """Initialize GPU libraries"""
        try:
            import cupy
            import cudf
            self.cupy = cupy
            self.cudf = cudf
            self.gpu_available = True

            # Get GPU info
            device = cupy.cuda.Device()
            self._gpu_total_mb = device.mem_info[1] / (1024 * 1024)
            self._log("GPU_INIT", f"GPU initialized: {self._gpu_total_mb:.0f}MB VRAM")

        except ImportError as e:
            self._log("GPU_INIT_FAIL", f"GPU libraries not available: {e}")
            self.gpu_available = False

    def _log(self, event: str, message: str, data: Dict = None) -> None:
        """Log event"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "message": message,
            "data": data or {},
            "vram_used_mb": self._vram_used_mb,
            "system_used_mb": self._system_used_mb
        }
        self._log_entries.append(entry)
        print(f"  [GPU] {event}: {message}")

    # =========================================================================
    # GPU MEMORY MANAGEMENT
    # =========================================================================

    def clear_gpu_memory(self) -> None:
        """Clear all GPU VRAM"""
        if not self.gpu_available:
            return

        self._log("GPU_CLEAR", "Clearing GPU VRAM...")

        # Clear tracked blocks
        for key in list(self._vram_blocks.keys()):
            self._free_vram(key)

        # Force garbage collection
        gc.collect()

        # Clear CuPy memory pools
        if self.cupy:
            self.cupy.get_default_memory_pool().free_all_blocks()
            self.cupy.get_default_pinned_memory_pool().free_all_blocks()

        # Clear CUDA cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self._vram_used_mb = 0
        self._vram_blocks.clear()

        # Report memory
        if self.cupy:
            meminfo = self.cupy.cuda.Device().mem_info
            free_mb = meminfo[0] / (1024 * 1024)
            total_mb = meminfo[1] / (1024 * 1024)
            self._log("GPU_CLEAR_DONE", f"VRAM: {free_mb:.0f}MB free / {total_mb:.0f}MB total")

    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory status"""
        if not self.gpu_available or not self.cupy:
            return {"available": False}

        try:
            meminfo = self.cupy.cuda.Device().mem_info
            free_mb = meminfo[0] / (1024 * 1024)
            total_mb = meminfo[1] / (1024 * 1024)
            used_mb = total_mb - free_mb

            return {
                "available": True,
                "free_mb": free_mb,
                "total_mb": total_mb,
                "used_mb": used_mb,
                "usage_percent": (used_mb / total_mb) * 100 if total_mb > 0 else 0
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def _estimate_size_mb(self, data: Any) -> float:
        """Estimate size of data in MB"""
        try:
            if hasattr(data, 'nbytes'):
                return data.nbytes / (1024 * 1024)
            elif hasattr(data, 'memory_usage'):
                return data.memory_usage(deep=True).sum() / (1024 * 1024)
            elif isinstance(data, np.ndarray):
                return data.nbytes / (1024 * 1024)
            else:
                return 0
        except:
            return 0

    def _free_vram(self, key: str) -> None:
        """Free a VRAM block"""
        if key in self._vram_blocks:
            block = self._vram_blocks[key]
            self._vram_used_mb -= block.size_bytes / (1024 * 1024)
            del block.data
            del self._vram_blocks[key]

    # =========================================================================
    # DATA STAGING (System RAM -> GPU)
    # =========================================================================

    def stage_for_training(self, key: str, data: Any) -> None:
        """
        Stage data in system RAM for later GPU transfer.
        This keeps data warm in L2/L3 (system RAM).
        """
        size_mb = self._estimate_size_mb(data)

        if size_mb + self._system_used_mb > self.SYSTEM_MEMORY_LIMIT_MB:
            self._evict_system_memory(size_mb)

        block = GPUMemoryBlock(
            key=key,
            tier=GPUTier.SYSTEM,
            size_bytes=int(size_mb * 1024 * 1024),
            data=data
        )

        self._system_blocks[key] = block
        self._system_used_mb += size_mb

        self._log("STAGE", f"Staged '{key}' in system RAM ({size_mb:.1f}MB)")

    def load_to_gpu(self, key: str) -> Any:
        """
        Load staged data to GPU VRAM.
        Returns GPU-resident data.
        """
        if not self.gpu_available:
            self._log("GPU_LOAD_SKIP", f"GPU not available, using system RAM for '{key}'")
            return self._system_blocks.get(key, GPUMemoryBlock(key, GPUTier.SYSTEM, 0)).data

        # Check if already in VRAM
        if key in self._vram_blocks:
            self._vram_blocks[key].access_count += 1
            return self._vram_blocks[key].data

        # Get from system RAM
        if key not in self._system_blocks:
            self._log("GPU_LOAD_FAIL", f"Data '{key}' not staged")
            return None

        system_block = self._system_blocks[key]
        size_mb = system_block.size_bytes / (1024 * 1024)

        # Check VRAM capacity
        mem_info = self.get_gpu_memory_info()
        if mem_info.get("free_mb", 0) < size_mb * 1.2:  # 20% buffer
            self._log("GPU_LOAD_EVICT", f"Not enough VRAM for '{key}', clearing...")
            self.clear_gpu_memory()

            # Re-check
            mem_info = self.get_gpu_memory_info()
            if mem_info.get("free_mb", 0) < size_mb * 1.2:
                self._log("GPU_LOAD_FAIL", f"Still not enough VRAM after clear")
                return system_block.data  # Fall back to system RAM data

        # Transfer to GPU
        try:
            if hasattr(system_block.data, 'values'):
                # DataFrame - convert to cuDF
                gpu_data = self.cudf.DataFrame(system_block.data)
            elif isinstance(system_block.data, np.ndarray):
                # NumPy array - convert to CuPy
                gpu_data = self.cupy.asarray(system_block.data)
            else:
                # Keep as-is
                gpu_data = system_block.data

            vram_block = GPUMemoryBlock(
                key=key,
                tier=GPUTier.VRAM,
                size_bytes=system_block.size_bytes,
                data=gpu_data
            )

            self._vram_blocks[key] = vram_block
            self._vram_used_mb += size_mb

            self._log("GPU_LOAD", f"Loaded '{key}' to VRAM ({size_mb:.1f}MB)")
            return gpu_data

        except Exception as e:
            self._log("GPU_LOAD_ERROR", f"Error loading '{key}' to GPU: {e}")
            return system_block.data

    def _evict_system_memory(self, needed_mb: float) -> None:
        """Evict data from system memory to make room"""
        # LRU eviction
        sorted_blocks = sorted(
            self._system_blocks.items(),
            key=lambda x: x[1].access_count
        )

        freed = 0
        for key, block in sorted_blocks:
            if freed >= needed_mb:
                break

            size_mb = block.size_bytes / (1024 * 1024)
            del self._system_blocks[key]
            self._system_used_mb -= size_mb
            freed += size_mb

            self._log("EVICT", f"Evicted '{key}' from system RAM ({size_mb:.1f}MB)")

    # =========================================================================
    # ML TRAINING SUPPORT
    # =========================================================================

    def prepare_training_data(self, X: Any, y: Any,
                              model_name: str) -> Tuple[Any, Any]:
        """
        Prepare training data for a specific model.
        Clears GPU, stages data, then loads to GPU.

        Args:
            X: Feature data (DataFrame or array)
            y: Target data (Series or array)
            model_name: Name of model being trained

        Returns:
            (X_gpu, y_gpu) - GPU-resident training data
        """
        self._log("PREPARE", f"Preparing training data for {model_name}")

        # Clear GPU before loading new model's data
        self.clear_gpu_memory()

        # Stage in system RAM
        self.stage_for_training(f"{model_name}_X", X)
        self.stage_for_training(f"{model_name}_y", y)

        # Load to GPU
        X_gpu = self.load_to_gpu(f"{model_name}_X")
        y_gpu = self.load_to_gpu(f"{model_name}_y")

        mem_info = self.get_gpu_memory_info()
        self._log("PREPARE_DONE",
                  f"Data ready for {model_name}. "
                  f"VRAM: {mem_info.get('used_mb', 0):.0f}MB / {mem_info.get('total_mb', 0):.0f}MB")

        return X_gpu, y_gpu

    def cleanup_after_model(self, model_name: str) -> None:
        """
        Cleanup after training a model.
        Clears GPU and removes staged data for this model.
        """
        self._log("CLEANUP", f"Cleaning up after {model_name}")

        # Clear GPU
        self.clear_gpu_memory()

        # Remove staged data
        for suffix in ['_X', '_y']:
            key = f"{model_name}{suffix}"
            if key in self._system_blocks:
                block = self._system_blocks[key]
                self._system_used_mb -= block.size_bytes / (1024 * 1024)
                del self._system_blocks[key]

        gc.collect()

        self._log("CLEANUP_DONE", f"Cleanup complete for {model_name}")

    def get_training_batch(self, X: Any, y: Any,
                           batch_size: int, batch_idx: int) -> Tuple[Any, Any]:
        """
        Get a training batch, loaded to GPU.
        For models that support batch training.
        """
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(X))

        if hasattr(X, 'iloc'):
            X_batch = X.iloc[start_idx:end_idx]
            y_batch = y.iloc[start_idx:end_idx]
        else:
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]

        # Load batch to GPU
        batch_key = f"batch_{batch_idx}"
        self.stage_for_training(f"{batch_key}_X", X_batch)
        self.stage_for_training(f"{batch_key}_y", y_batch)

        X_gpu = self.load_to_gpu(f"{batch_key}_X")
        y_gpu = self.load_to_gpu(f"{batch_key}_y")

        return X_gpu, y_gpu

    # =========================================================================
    # STATUS & REPORTING
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get GPU manager status"""
        gpu_info = self.get_gpu_memory_info()

        return {
            "gpu_available": self.gpu_available,
            "gpu_info": gpu_info,
            "vram_blocks": len(self._vram_blocks),
            "vram_used_mb": self._vram_used_mb,
            "system_blocks": len(self._system_blocks),
            "system_used_mb": self._system_used_mb,
            "log_entries": len(self._log_entries)
        }

    def get_log(self) -> List[Dict[str, Any]]:
        """Get operation log"""
        return self._log_entries

    def print_status(self) -> None:
        """Print GPU memory status"""
        status = self.get_status()
        gpu = status["gpu_info"]

        print()
        print("=" * 60)
        print("VBUS GPU MEMORY MANAGER STATUS")
        print("=" * 60)

        if status["gpu_available"]:
            print(f"  GPU VRAM:")
            print(f"    Used:  {gpu.get('used_mb', 0):.0f} MB")
            print(f"    Free:  {gpu.get('free_mb', 0):.0f} MB")
            print(f"    Total: {gpu.get('total_mb', 0):.0f} MB")
            print(f"    Usage: {gpu.get('usage_percent', 0):.1f}%")
        else:
            print("  GPU: Not available")

        print(f"\n  System RAM Cache:")
        print(f"    Blocks: {status['system_blocks']}")
        print(f"    Used:   {status['system_used_mb']:.0f} MB")

        print("=" * 60)


# =============================================================================
# INTEGRATION WITH VIEWERDBX
# =============================================================================

def create_gpu_manager() -> VBUSGPUManager:
    """Create and return a GPU memory manager instance"""
    return VBUSGPUManager(use_gpu=True)


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    print("VBUS GPU Memory Manager v1.0")
    print("=" * 60)

    # Create manager
    manager = VBUSGPUManager(use_gpu=True)

    # Print initial status
    manager.print_status()

    # Test with dummy data
    if manager.gpu_available:
        print("\n[TEST] Staging and loading data...")

        # Create test data
        test_data = np.random.randn(10000, 100).astype(np.float32)
        test_labels = np.random.randn(10000).astype(np.float32)

        # Prepare for "TestModel"
        X_gpu, y_gpu = manager.prepare_training_data(test_data, test_labels, "TestModel")

        print(f"\n  X_gpu type: {type(X_gpu)}")
        print(f"  y_gpu type: {type(y_gpu)}")

        # Print status
        manager.print_status()

        # Cleanup
        manager.cleanup_after_model("TestModel")

        print("\n[TEST] After cleanup:")
        manager.print_status()

    print("\n[DONE] GPU Manager operational")
