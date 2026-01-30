#!/usr/bin/env python3
"""
VBUS ML EXECUTOR v1.0
=====================
GPU-Only ML Training with VBUS Memory Management

ARCHITECTURE:
  System RAM (128GB) ─────┐
       │                  │
       └── VBUS Staging ──┼── GPU VRAM (12GB)
           (L2/L3)        │      (L1)
                          │
                          ▼
                    ML TRAINING
                    ============
                    Phase 1: XGBoost (1350 trees)
                      └── Clear GPU
                    Phase 2: CatBoost (1350 trees)
                      └── Clear GPU
                    Phase 3: cuML RF (1350 trees)
                      └── Clear GPU
                    Phase 4: Ensemble

RULES:
  1. NO CPU FALLBACK - GPU MANDATORY
  2. Train ONE model at a time
  3. Clear GPU between models
  4. Stage data in system RAM via VBUS
  5. Maximum parallelism within each model (1350 trees)

VERSION: 1.0.0 | ALGO 95.4 | GPU MANDATORY
"""

from __future__ import annotations

import os
import gc
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum

# Configuration
TREES_PER_MODEL = 1350
XGBOOST_WEIGHT = 0.52
CATBOOST_WEIGHT = 0.35
CUML_RF_WEIGHT = 0.13


class TrainingPhase(Enum):
    """ML Training phases"""
    STAGING = "STAGING"
    XGBOOST = "XGBOOST"
    CATBOOST = "CATBOOST"
    CUML_RF = "CUML_RF"
    ENSEMBLE = "ENSEMBLE"


@dataclass
class ModelResult:
    """Result from training a single model"""
    name: str
    phase: TrainingPhase
    success: bool
    predictions: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time_ms: float = 0
    trees_trained: int = 0
    gpu_peak_mb: float = 0
    error: Optional[str] = None


@dataclass
class EnsembleResult:
    """Result from ensemble training"""
    success: bool
    predictions: Optional[np.ndarray] = None
    model_results: List[ModelResult] = field(default_factory=list)
    total_time_ms: float = 0
    mape: Optional[float] = None
    r2: Optional[float] = None


class VBUSMLExecutor:
    """
    VBUS ML Executor - GPU-Only Training with Memory Management

    Executes the 3-model ensemble (XGBoost, CatBoost, cuML RF) with:
    - Data staged in system RAM via VBUS caching
    - One model trained at a time on GPU
    - GPU memory cleared between models
    - 1350 trees per model at maximum GPU parallelism
    """

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.gpu_available = False

        # GPU libraries
        self.cupy = None
        self.cudf = None
        self.cuml = None

        # Staged data (in system RAM)
        self._staged_X = None
        self._staged_y = None
        self._staged_features = None

        # Training log
        self._log_entries: List[Dict] = []

        # Initialize
        self._init_gpu()

    def _init_gpu(self) -> None:
        """Initialize GPU - MANDATORY"""
        self._log("INIT", "Initializing GPU for ML training...")

        try:
            import cupy
            import cudf
            from cuml.ensemble import RandomForestRegressor as cuMLRandomForest

            self.cupy = cupy
            self.cudf = cudf
            self.cuml = True
            self.gpu_available = True

            # Get GPU info
            device = cupy.cuda.Device()
            meminfo = device.mem_info
            free_mb = meminfo[0] / (1024 * 1024)
            total_mb = meminfo[1] / (1024 * 1024)

            self._log("INIT", f"GPU ready: {free_mb:.0f}MB free / {total_mb:.0f}MB total")

        except ImportError as e:
            self._log("INIT_ERROR", f"GPU libraries not available: {e}")
            raise RuntimeError("GPU MANDATORY - Cannot proceed without GPU libraries")

    def _log(self, event: str, message: str, data: Dict = None) -> None:
        """Log event"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "message": message,
            "data": data or {}
        }
        self._log_entries.append(entry)
        print(f"  [ML] {event}: {message}")

    def _clear_gpu(self) -> None:
        """Clear GPU memory - called between models"""
        self._log("GPU_CLEAR", "Clearing GPU memory between models...")

        gc.collect()

        if self.cupy:
            self.cupy.get_default_memory_pool().free_all_blocks()
            self.cupy.get_default_pinned_memory_pool().free_all_blocks()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        # Force sync
        if self.cupy:
            self.cupy.cuda.Device().synchronize()

        gc.collect()

        # Report
        if self.cupy:
            meminfo = self.cupy.cuda.Device().mem_info
            free_mb = meminfo[0] / (1024 * 1024)
            total_mb = meminfo[1] / (1024 * 1024)
            self._log("GPU_CLEAR", f"After clear: {free_mb:.0f}MB free / {total_mb:.0f}MB total")

    def _get_gpu_memory_mb(self) -> Tuple[float, float]:
        """Get (free_mb, total_mb) GPU memory"""
        if self.cupy:
            meminfo = self.cupy.cuda.Device().mem_info
            return meminfo[0] / (1024 * 1024), meminfo[1] / (1024 * 1024)
        return 0, 0

    # =========================================================================
    # STAGING - Load data to system RAM
    # =========================================================================

    def stage_training_data(self, X: pd.DataFrame, y: pd.Series,
                            feature_names: List[str]) -> None:
        """
        Stage training data in system RAM.
        This is the L2/L3 cache layer - data stays here until needed by GPU.

        Args:
            X: Feature DataFrame (stays in pandas/numpy)
            y: Target Series (stays in pandas/numpy)
            feature_names: List of feature column names
        """
        self._log("STAGING", f"Staging {len(X):,} rows x {len(feature_names)} features in system RAM")

        # CRITICAL: Clear GPU completely before staging
        self._log("STAGING", "Clearing GPU before staging data...")
        for _ in range(3):
            self._clear_gpu()
            time.sleep(0.5)

        free_mb, total_mb = self._get_gpu_memory_mb()
        self._log("STAGING", f"GPU memory after clear: {free_mb:.0f}MB free / {total_mb:.0f}MB total")

        # Keep as numpy arrays in system RAM (NOT on GPU!)
        self._staged_X = X[feature_names].values.astype(np.float32)
        self._staged_y = y.values.astype(np.float32)
        self._staged_features = feature_names

        size_mb = (self._staged_X.nbytes + self._staged_y.nbytes) / (1024 * 1024)
        self._log("STAGING", f"Staged {size_mb:.1f}MB in system RAM (L2/L3 cache)")

    # =========================================================================
    # MODEL TRAINING - One at a time with GPU clearing
    # =========================================================================

    def _train_xgboost(self) -> ModelResult:
        """Train XGBoost model on GPU"""
        self._log("XGBOOST", f"Training XGBoost ({TREES_PER_MODEL} trees) on GPU...")
        start = time.time()

        try:
            import xgboost as xgb

            # Clear GPU first
            self._clear_gpu()

            # Create DMatrix on GPU
            self._log("XGBOOST", "Creating DMatrix on GPU...")
            dtrain = xgb.DMatrix(self._staged_X, label=self._staged_y)

            # XGBoost params - GPU histogram method
            params = {
                'objective': 'reg:squarederror',
                'tree_method': 'gpu_hist',  # GPU MANDATORY
                'device': 'cuda',
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'verbosity': 1
            }

            # Train
            self._log("XGBOOST", f"Training {TREES_PER_MODEL} trees...")
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=TREES_PER_MODEL,
                verbose_eval=100
            )

            # Predict
            predictions = model.predict(dtrain)

            # Feature importance
            importance = model.get_score(importance_type='gain')

            duration = (time.time() - start) * 1000
            free_mb, total_mb = self._get_gpu_memory_mb()

            self._log("XGBOOST", f"Complete in {duration:.0f}ms. GPU: {total_mb - free_mb:.0f}MB used")

            # Cleanup model from GPU
            del dtrain
            del model
            self._clear_gpu()

            return ModelResult(
                name="XGBoost",
                phase=TrainingPhase.XGBOOST,
                success=True,
                predictions=predictions,
                feature_importance=importance,
                training_time_ms=duration,
                trees_trained=TREES_PER_MODEL,
                gpu_peak_mb=total_mb - free_mb
            )

        except Exception as e:
            self._log("XGBOOST_ERROR", f"Training failed: {e}")
            self._clear_gpu()
            return ModelResult(
                name="XGBoost",
                phase=TrainingPhase.XGBOOST,
                success=False,
                error=str(e),
                training_time_ms=(time.time() - start) * 1000
            )

    def _train_catboost(self) -> ModelResult:
        """Train CatBoost model on GPU"""
        self._log("CATBOOST", f"Training CatBoost ({TREES_PER_MODEL} trees) on GPU...")
        start = time.time()

        try:
            from catboost import CatBoostRegressor

            # Aggressive GPU clear - CRITICAL for CatBoost
            self._log("CATBOOST", "Aggressive GPU memory clear...")
            for _ in range(3):
                self._clear_gpu()
                time.sleep(1)

            free_mb, total_mb = self._get_gpu_memory_mb()
            self._log("CATBOOST", f"GPU memory before training: {free_mb:.0f}MB free / {total_mb:.0f}MB total")

            # CatBoost params - VERY conservative GPU memory usage
            model = CatBoostRegressor(
                iterations=TREES_PER_MODEL,
                depth=6,
                learning_rate=0.03,
                loss_function='RMSE',
                task_type='GPU',  # GPU MANDATORY
                devices='0',
                gpu_ram_part=0.25,  # Use only 25% of GPU RAM
                max_ctr_complexity=1,  # Minimize CTR memory
                boosting_type='Plain',  # Plain is most memory efficient
                bootstrap_type='Bayesian',  # Bayesian works better on GPU
                random_seed=42,
                verbose=100,
                allow_writing_files=False,  # Don't write temp files
            )

            self._log("CATBOOST", f"Training {TREES_PER_MODEL} trees with 25% GPU RAM limit...")

            # Train with staged data
            model.fit(
                self._staged_X,
                self._staged_y,
                verbose=100
            )

            # Predict
            predictions = model.predict(self._staged_X)

            # Feature importance
            importance = dict(zip(
                self._staged_features,
                model.get_feature_importance()
            ))

            duration = (time.time() - start) * 1000
            free_mb, total_mb = self._get_gpu_memory_mb()

            self._log("CATBOOST", f"Complete in {duration:.0f}ms")

            # Cleanup
            del model
            self._clear_gpu()

            return ModelResult(
                name="CatBoost",
                phase=TrainingPhase.CATBOOST,
                success=True,
                predictions=predictions,
                feature_importance=importance,
                training_time_ms=duration,
                trees_trained=TREES_PER_MODEL,
                gpu_peak_mb=total_mb - free_mb
            )

        except Exception as e:
            self._log("CATBOOST_ERROR", f"Training failed: {e}")
            self._clear_gpu()
            return ModelResult(
                name="CatBoost",
                phase=TrainingPhase.CATBOOST,
                success=False,
                error=str(e),
                training_time_ms=(time.time() - start) * 1000
            )

    def _train_cuml_rf(self) -> ModelResult:
        """Train cuML Random Forest on GPU"""
        self._log("CUML_RF", f"Training cuML Random Forest ({TREES_PER_MODEL} trees) on GPU...")
        start = time.time()

        try:
            from cuml.ensemble import RandomForestRegressor as cuMLRF

            # Clear GPU first
            self._clear_gpu()

            # Wait for clear
            time.sleep(1)
            self._clear_gpu()

            # Convert to cuDF/CuPy for cuML
            self._log("CUML_RF", "Converting data to GPU arrays...")
            X_gpu = self.cupy.asarray(self._staged_X)
            y_gpu = self.cupy.asarray(self._staged_y)

            # cuML RF params
            model = cuMLRF(
                n_estimators=TREES_PER_MODEL,
                max_depth=12,
                max_features='sqrt',
                min_samples_leaf=5,
                min_samples_split=10,
                bootstrap=True,
                n_bins=128,
                random_state=42,
                verbose=2
            )

            self._log("CUML_RF", f"Training {TREES_PER_MODEL} trees...")

            # Train
            model.fit(X_gpu, y_gpu)

            # Predict
            predictions = model.predict(X_gpu)
            predictions_np = self.cupy.asnumpy(predictions)

            # Feature importance (cuML RF uses get_feature_importances or similar)
            try:
                if hasattr(model, 'feature_importances_'):
                    fi = model.feature_importances_
                    if hasattr(fi, 'tolist'):
                        fi = fi.tolist()
                    elif hasattr(fi, 'to_numpy'):
                        fi = fi.to_numpy().tolist()
                    else:
                        fi = list(fi)
                    importance = dict(zip(self._staged_features, fi))
                else:
                    importance = {f: 0.0 for f in self._staged_features}
            except Exception as fi_err:
                self._log("CUML_RF", f"Could not get feature importance: {fi_err}")
                importance = {f: 0.0 for f in self._staged_features}

            duration = (time.time() - start) * 1000
            free_mb, total_mb = self._get_gpu_memory_mb()

            self._log("CUML_RF", f"Complete in {duration:.0f}ms")

            # Cleanup
            del X_gpu
            del y_gpu
            del model
            self._clear_gpu()

            return ModelResult(
                name="cuML_RF",
                phase=TrainingPhase.CUML_RF,
                success=True,
                predictions=predictions_np,
                feature_importance=importance,
                training_time_ms=duration,
                trees_trained=TREES_PER_MODEL,
                gpu_peak_mb=total_mb - free_mb
            )

        except Exception as e:
            self._log("CUML_RF_ERROR", f"Training failed: {e}")
            self._clear_gpu()
            return ModelResult(
                name="cuML_RF",
                phase=TrainingPhase.CUML_RF,
                success=False,
                error=str(e),
                training_time_ms=(time.time() - start) * 1000
            )

    # =========================================================================
    # ENSEMBLE EXECUTION
    # =========================================================================

    def train_ensemble(self, X: pd.DataFrame, y: pd.Series,
                       feature_names: List[str]) -> EnsembleResult:
        """
        Train the complete 3-model ensemble.

        Phases:
        1. Stage data in system RAM
        2. Train XGBoost (1350 trees) → Clear GPU
        3. Train CatBoost (1350 trees) → Clear GPU
        4. Train cuML RF (1350 trees) → Clear GPU
        5. Create weighted ensemble

        Args:
            X: Feature DataFrame
            y: Target Series
            feature_names: Feature column names

        Returns:
            EnsembleResult with predictions and metrics
        """
        self._log("ENSEMBLE", "=" * 60)
        self._log("ENSEMBLE", "VBUS ML EXECUTOR - 3-Model GPU Ensemble")
        self._log("ENSEMBLE", "=" * 60)

        total_start = time.time()
        model_results = []

        # Phase 1: Stage data
        self.stage_training_data(X, y, feature_names)

        # Phase 2: XGBoost
        self._log("ENSEMBLE", "-" * 40)
        self._log("ENSEMBLE", "Phase 2: XGBoost Training")
        self._log("ENSEMBLE", "-" * 40)
        xgb_result = self._train_xgboost()
        model_results.append(xgb_result)

        if not xgb_result.success:
            self._log("ENSEMBLE", f"XGBoost failed: {xgb_result.error}")

        # Phase 3: CatBoost
        self._log("ENSEMBLE", "-" * 40)
        self._log("ENSEMBLE", "Phase 3: CatBoost Training")
        self._log("ENSEMBLE", "-" * 40)
        cat_result = self._train_catboost()
        model_results.append(cat_result)

        if not cat_result.success:
            self._log("ENSEMBLE", f"CatBoost failed: {cat_result.error}")

        # Phase 4: cuML RF
        self._log("ENSEMBLE", "-" * 40)
        self._log("ENSEMBLE", "Phase 4: cuML Random Forest Training")
        self._log("ENSEMBLE", "-" * 40)
        rf_result = self._train_cuml_rf()
        model_results.append(rf_result)

        if not rf_result.success:
            self._log("ENSEMBLE", f"cuML RF failed: {rf_result.error}")

        # Phase 5: Ensemble
        self._log("ENSEMBLE", "-" * 40)
        self._log("ENSEMBLE", "Phase 5: Weighted Ensemble")
        self._log("ENSEMBLE", "-" * 40)

        # Create ensemble predictions
        predictions = None
        weights_used = []

        if xgb_result.success and xgb_result.predictions is not None:
            predictions = XGBOOST_WEIGHT * xgb_result.predictions
            weights_used.append(("XGBoost", XGBOOST_WEIGHT))

        if cat_result.success and cat_result.predictions is not None:
            if predictions is None:
                predictions = CATBOOST_WEIGHT * cat_result.predictions
            else:
                predictions += CATBOOST_WEIGHT * cat_result.predictions
            weights_used.append(("CatBoost", CATBOOST_WEIGHT))

        if rf_result.success and rf_result.predictions is not None:
            if predictions is None:
                predictions = CUML_RF_WEIGHT * rf_result.predictions
            else:
                predictions += CUML_RF_WEIGHT * rf_result.predictions
            weights_used.append(("cuML_RF", CUML_RF_WEIGHT))

        # Normalize if not all models succeeded
        if weights_used and predictions is not None:
            total_weight = sum(w for _, w in weights_used)
            if total_weight < 1.0 and total_weight > 0:
                predictions = predictions / total_weight
                self._log("ENSEMBLE", f"Normalized predictions with {len(weights_used)} models (weight: {total_weight:.2f})")

        # Calculate metrics
        mape = None
        r2 = None

        if predictions is not None:
            y_true = self._staged_y
            y_pred = predictions

            # MAPE
            mask = y_true != 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

            # R2
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            self._log("ENSEMBLE", f"Training MAPE: {mape:.2f}%")
            self._log("ENSEMBLE", f"Training R2: {r2:.4f}")

        total_time = (time.time() - total_start) * 1000

        # Summary
        self._log("ENSEMBLE", "=" * 60)
        self._log("ENSEMBLE", "TRAINING COMPLETE")
        self._log("ENSEMBLE", f"  Total Time: {total_time/1000:.1f}s")
        self._log("ENSEMBLE", f"  Models: {len([r for r in model_results if r.success])}/3 successful")
        self._log("ENSEMBLE", f"  Total Trees: {sum(r.trees_trained for r in model_results if r.success)}")
        self._log("ENSEMBLE", "=" * 60)

        # Cleanup staged data
        self._staged_X = None
        self._staged_y = None
        gc.collect()

        return EnsembleResult(
            success=predictions is not None,
            predictions=predictions,
            model_results=model_results,
            total_time_ms=total_time,
            mape=mape,
            r2=r2
        )

    def get_log(self) -> List[Dict]:
        """Get training log"""
        return self._log_entries


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_ml_executor() -> VBUSMLExecutor:
    """Create and return ML executor instance"""
    return VBUSMLExecutor(use_gpu=True)


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    print("VBUS ML Executor v1.0 - Test Mode")
    print("=" * 60)

    # Create executor
    executor = VBUSMLExecutor(use_gpu=True)

    print("\n[TEST] Creating synthetic test data...")

    # Create test data
    n_samples = 10000
    n_features = 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features).astype(np.float32),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = pd.Series(
        np.random.randn(n_samples).astype(np.float32) * 1000000 + 5000000,  # Views-like values
        name="views"
    )

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # Train ensemble
    print("\n[TEST] Training ensemble...")
    result = executor.train_ensemble(X, y, list(X.columns))

    print("\n[RESULT]")
    print(f"  Success: {result.success}")
    print(f"  MAPE: {result.mape:.2f}%" if result.mape else "  MAPE: N/A")
    print(f"  R2: {result.r2:.4f}" if result.r2 else "  R2: N/A")
    print(f"  Total Time: {result.total_time_ms/1000:.1f}s")

    for mr in result.model_results:
        status = "OK" if mr.success else f"FAIL: {mr.error}"
        print(f"  {mr.name}: {status} ({mr.training_time_ms/1000:.1f}s, {mr.trees_trained} trees)")
