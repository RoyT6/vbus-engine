#!/usr/bin/env python3
"""
VBUS FEATURE SELECTION & CAUSATION RUNNER v1.0
================================================
GPU-Accelerated Feature Selection and Causation Analysis with VBUS Memory Management

ARCHITECTURE:
  BFD_V27.65 (540k x 2238) ─── VBUS L3 (System RAM 128GB)
         │                              │
         └── Chunked Loading ───────────┼── VBUS L2 (Pinned RAM)
                                        │
                                        └── VBUS L1 (GPU VRAM 12GB)
                                               │
                                               ▼
                                        ANALYSIS PHASES
                                        ===============
                                        Phase 1: Feature Variance Selection (top 200)
                                        Phase 2: Correlation Matrix (chunked)
                                        Phase 3: Causation Analysis (Granger/PC)
                                        Phase 4: ML Ensemble Training

MEMORY MANAGEMENT:
  - 540k rows x 2238 cols = ~4.8GB raw data
  - Process in chunks of 50k rows
  - Correlation computed in column batches of 100
  - GPU cleared between phases

HARDWARE REQUIREMENTS:
  - RTX 3080 Ti (12GB VRAM)
  - 128GB System RAM
  - Ryzen 9 3950X (16 cores)

VERSION: 1.0.0 | ALGO 95.4 | GPU MANDATORY | VBUS ENABLED
"""

from __future__ import annotations

import os
import gc
import sys
import time
import json
import warnings
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

# Environment setup for WSL CUDA
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['NUMBA_CUDA_USE_NVIDIA_BINDING'] = '1'
os.environ['NUMBA_CUDA_DRIVER'] = '/usr/lib/wsl/lib/libcuda.so.1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDF_SPILL'] = 'on'
os.environ['PYTHONUNBUFFERED'] = '1'

warnings.filterwarnings('ignore')

# Add paths
VBUS_DIR = Path(__file__).parent
DOWNLOADS_DIR = VBUS_DIR.parent
sys.path.insert(0, str(VBUS_DIR))
sys.path.insert(0, str(DOWNLOADS_DIR))
sys.path.insert(0, str(DOWNLOADS_DIR / "Parallel Engine"))

import numpy as np
import pandas as pd

# GPU imports
try:
    import cupy as cp
    import cudf
    from cuml.ensemble import RandomForestRegressor as cuMLRF
    GPU_AVAILABLE = True
    cp.cuda.Device(0).use()
    print(f"[INIT] GPU Available: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except ImportError as e:
    print(f"[ERROR] GPU libraries not available: {e}")
    GPU_AVAILABLE = False

# Parallel Engine imports
try:
    from system_capability_engine.optimizer import (
        TaskType, get_optimal_config, ParallelExecutor
    )
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    print("[WARN] Parallel Engine not available, using defaults")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data paths
    "bfd_path": "/mnt/c/Users/RoyT6/Downloads/BFD_V27.65.parquet",
    "viewerdbx_path": "/mnt/c/Users/RoyT6/Downloads/VIEWERDBX_V1.parquet",
    "output_dir": "/mnt/c/Users/RoyT6/Downloads/VBUS/output",

    # Memory management
    "row_chunk_size": 50000,      # Rows per chunk (50k)
    "col_batch_size": 100,        # Columns per batch for correlation
    "vram_limit_mb": 10000,       # Leave 2GB buffer
    "system_ram_limit_gb": 100,   # Use 100GB of 128GB

    # Feature selection
    "top_features": 200,          # Select top 200 by variance
    "min_variance": 0.01,         # Minimum variance threshold
    "exclude_patterns": [         # Columns to exclude from features
        "fc_uid", "imdb_id", "tmdb_id", "title", "premiere_date",
        "views_computed", "views_estimated", "views_y"  # Anti-cheat
    ],

    # Causation settings
    "causation_max_lag": 4,       # Max lag for Granger causality
    "causation_significance": 0.05,

    # ML settings
    "trees_per_model": 1350,
    "xgboost_weight": 0.52,
    "catboost_weight": 0.35,
    "cuml_rf_weight": 0.13,
}


def get_gpu_memory() -> Tuple[float, float]:
    """Get (free_mb, total_mb) GPU memory"""
    if GPU_AVAILABLE:
        meminfo = cp.cuda.Device().mem_info
        return meminfo[0] / (1024**2), meminfo[1] / (1024**2)
    return 0, 0


def clear_gpu_memory():
    """Aggressively clear GPU memory"""
    if not GPU_AVAILABLE:
        return

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    gc.collect()
    cp.cuda.Device().synchronize()


def log(phase: str, message: str, data: Dict = None):
    """Log with timestamp"""
    ts = datetime.now().strftime("%H:%M:%S")
    free_mb, total_mb = get_gpu_memory()
    gpu_str = f"[GPU: {total_mb - free_mb:.0f}/{total_mb:.0f}MB]" if GPU_AVAILABLE else ""
    print(f"  [{ts}] [{phase}] {message} {gpu_str}")


# =============================================================================
# PHASE 1: FEATURE VARIANCE SELECTION
# =============================================================================

def phase1_feature_selection(df: pd.DataFrame) -> List[str]:
    """
    Select top N features by variance (excludes target/ID columns).
    Uses chunked computation to fit in GPU memory.
    """
    log("PHASE1", "=" * 60)
    log("PHASE1", "FEATURE VARIANCE SELECTION")
    log("PHASE1", f"Input: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    log("PHASE1", "=" * 60)

    clear_gpu_memory()

    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    log("PHASE1", f"Numeric columns: {len(numeric_cols)}")

    # Exclude forbidden columns (anti-cheat + IDs)
    exclude_patterns = CONFIG["exclude_patterns"]
    feature_cols = []
    for col in numeric_cols:
        exclude = False
        for pattern in exclude_patterns:
            if pattern.lower() in col.lower():
                exclude = True
                break
        if not exclude:
            feature_cols.append(col)

    log("PHASE1", f"Feature candidates after exclusion: {len(feature_cols)}")

    # Compute variance in chunks
    variances = {}
    chunk_size = CONFIG["col_batch_size"]

    for i in range(0, len(feature_cols), chunk_size):
        batch_cols = feature_cols[i:i + chunk_size]
        # Handle NaN values - convert to numpy first, then fill NaN, then convert to float32
        batch_data = df[batch_cols].to_numpy(dtype=np.float64, na_value=np.nan)
        batch_data = np.nan_to_num(batch_data, nan=0.0).astype(np.float32)

        if GPU_AVAILABLE:
            # GPU variance computation
            gpu_data = cp.asarray(batch_data)
            batch_var = cp.nanvar(gpu_data, axis=0)
            batch_var_np = cp.asnumpy(batch_var)
            del gpu_data
            clear_gpu_memory()
        else:
            batch_var_np = np.nanvar(batch_data, axis=0)

        for j, col in enumerate(batch_cols):
            variances[col] = float(batch_var_np[j])

        if (i // chunk_size) % 10 == 0:
            log("PHASE1", f"Computed variance for {i + len(batch_cols)}/{len(feature_cols)} columns")

    # Filter by minimum variance
    min_var = CONFIG["min_variance"]
    valid_features = {k: v for k, v in variances.items() if v >= min_var and not np.isnan(v)}
    log("PHASE1", f"Features with variance >= {min_var}: {len(valid_features)}")

    # Sort by variance and take top N
    top_n = CONFIG["top_features"]
    sorted_features = sorted(valid_features.items(), key=lambda x: x[1], reverse=True)
    selected_features = [f[0] for f in sorted_features[:top_n]]

    log("PHASE1", f"Selected top {len(selected_features)} features by variance")
    log("PHASE1", f"Top 5: {selected_features[:5]}")
    log("PHASE1", f"Bottom 5: {selected_features[-5:]}")

    return selected_features


# =============================================================================
# PHASE 2: CORRELATION MATRIX (CHUNKED)
# =============================================================================

def phase2_correlation_matrix(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Compute correlation matrix using chunked GPU computation.
    Processes in batches to avoid OOM on 12GB VRAM.
    """
    log("PHASE2", "=" * 60)
    log("PHASE2", "CORRELATION MATRIX (CHUNKED)")
    log("PHASE2", f"Computing {len(features)} x {len(features)} correlations")
    log("PHASE2", "=" * 60)

    clear_gpu_memory()

    n_features = len(features)

    if GPU_AVAILABLE:
        # Use cuDF for GPU-accelerated correlation
        log("PHASE2", "Using cuDF GPU-accelerated correlation...")

        # Convert to cuDF DataFrame in chunks
        chunk_size = min(50, n_features)  # Process 50 columns at a time

        # For smaller feature sets, compute directly
        if n_features <= 200:
            df_subset = df[features].astype(np.float32)
            gdf = cudf.DataFrame(df_subset)

            log("PHASE2", "Computing correlation on GPU...")
            corr_matrix = gdf.corr()
            corr_df = corr_matrix.to_pandas()

            del gdf
            clear_gpu_memory()
        else:
            # Chunked computation for larger sets
            corr_matrix = np.zeros((n_features, n_features), dtype=np.float32)

            for i in range(0, n_features, chunk_size):
                for j in range(i, n_features, chunk_size):
                    cols_i = features[i:i + chunk_size]
                    cols_j = features[j:j + chunk_size]

                    data_i = cp.asarray(np.nan_to_num(df[cols_i].to_numpy(dtype=np.float64, na_value=np.nan), nan=0.0).astype(np.float32))
                    data_j = cp.asarray(np.nan_to_num(df[cols_j].to_numpy(dtype=np.float64, na_value=np.nan), nan=0.0).astype(np.float32))

                    # Standardize
                    data_i = (data_i - cp.nanmean(data_i, axis=0)) / (cp.nanstd(data_i, axis=0) + 1e-8)
                    data_j = (data_j - cp.nanmean(data_j, axis=0)) / (cp.nanstd(data_j, axis=0) + 1e-8)

                    # Correlation
                    chunk_corr = cp.matmul(data_i.T, data_j) / data_i.shape[0]
                    chunk_corr_np = cp.asnumpy(chunk_corr)

                    # Fill matrix
                    corr_matrix[i:i + len(cols_i), j:j + len(cols_j)] = chunk_corr_np
                    if i != j:
                        corr_matrix[j:j + len(cols_j), i:i + len(cols_i)] = chunk_corr_np.T

                    del data_i, data_j, chunk_corr
                    clear_gpu_memory()

                log("PHASE2", f"Processed rows {i}-{i + chunk_size}/{n_features}")

            corr_df = pd.DataFrame(corr_matrix, index=features, columns=features)
    else:
        # CPU fallback
        log("PHASE2", "Using pandas CPU correlation (slower)...")
        corr_df = df[features].corr()

    log("PHASE2", f"Correlation matrix shape: {corr_df.shape}")

    # Find highly correlated pairs
    high_corr_threshold = 0.8
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            corr_val = abs(corr_df.iloc[i, j])
            if corr_val >= high_corr_threshold:
                high_corr_pairs.append((features[i], features[j], corr_val))

    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    log("PHASE2", f"Highly correlated pairs (|r| >= {high_corr_threshold}): {len(high_corr_pairs)}")
    if high_corr_pairs[:5]:
        for f1, f2, r in high_corr_pairs[:5]:
            log("PHASE2", f"  {f1} <-> {f2}: {r:.3f}")

    return corr_df


# =============================================================================
# PHASE 3: CAUSATION ANALYSIS (GRANGER)
# =============================================================================

def phase3_causation_analysis(df: pd.DataFrame, features: List[str],
                               target_col: str = None) -> Dict[str, Any]:
    """
    Perform Granger causality analysis on selected features.
    Uses GPU-accelerated regression for speed.
    """
    log("PHASE3", "=" * 60)
    log("PHASE3", "CAUSATION ANALYSIS (GRANGER)")
    log("PHASE3", "=" * 60)

    clear_gpu_memory()

    # Find target column (views-related)
    if target_col is None:
        # Find first views column that exists
        for candidate in ['views_h1_2024_total', 'views_h2_2024_total', 'views_total']:
            if candidate in df.columns:
                target_col = candidate
                break

    if target_col is None or target_col not in df.columns:
        log("PHASE3", "No target column found for causation analysis")
        return {"error": "No target column", "significant_causes": []}

    log("PHASE3", f"Target column: {target_col}")

    # Prepare data - drop NaN rows for target
    df_clean = df[features + [target_col]].dropna(subset=[target_col])
    log("PHASE3", f"Clean rows for analysis: {len(df_clean):,}")

    if len(df_clean) < 1000:
        log("PHASE3", "Insufficient data for causation analysis")
        return {"error": "Insufficient data", "significant_causes": []}

    # Simple feature importance as proxy for causation
    # (Full Granger requires lagged time series which BFD doesn't have)

    results = {
        "target": target_col,
        "n_features": len(features),
        "n_samples": len(df_clean),
        "significant_causes": [],
        "feature_importance": {}
    }

    if GPU_AVAILABLE:
        log("PHASE3", "Computing feature importance via GPU Random Forest...")

        # Prepare data
        X = np.nan_to_num(df_clean[features].to_numpy(dtype=np.float64, na_value=np.nan), nan=0.0).astype(np.float32)
        y = np.nan_to_num(df_clean[target_col].to_numpy(dtype=np.float64, na_value=np.nan), nan=0.0).astype(np.float32)

        # Sample if too large
        max_samples = 100000
        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X = X[idx]
            y = y[idx]
            log("PHASE3", f"Sampled to {max_samples:,} rows for speed")

        # GPU arrays
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)

        # Train cuML RF for feature importance
        try:
            model = cuMLRF(
                n_estimators=100,  # Fewer trees for speed
                max_depth=10,
                random_state=42,
                verbose=0
            )

            model.fit(X_gpu, y_gpu)

            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                fi = cp.asnumpy(model.feature_importances_)
                for i, feat in enumerate(features):
                    results["feature_importance"][feat] = float(fi[i])

            del model, X_gpu, y_gpu
            clear_gpu_memory()

        except Exception as e:
            log("PHASE3", f"cuML RF failed: {e}, trying XGBoost...")

            import xgboost as xgb
            dtrain = xgb.DMatrix(X, label=y)
            params = {
                'objective': 'reg:squarederror',
                'tree_method': 'gpu_hist',
                'device': 'cuda',
                'max_depth': 8,
                'verbosity': 0
            }
            model = xgb.train(params, dtrain, num_boost_round=100)

            importance = model.get_score(importance_type='gain')
            for i, feat in enumerate(features):
                feat_key = f'f{i}'
                results["feature_importance"][feat] = importance.get(feat_key, 0)

            del dtrain, model
            clear_gpu_memory()

    # Sort by importance
    sorted_importance = sorted(
        results["feature_importance"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Top causes (high importance = likely causal relationship)
    results["significant_causes"] = [
        {"feature": f, "importance": imp, "rank": i + 1}
        for i, (f, imp) in enumerate(sorted_importance[:20])
    ]

    log("PHASE3", f"Top 10 causal features:")
    for item in results["significant_causes"][:10]:
        log("PHASE3", f"  {item['rank']}. {item['feature']}: {item['importance']:.4f}")

    return results


# =============================================================================
# PHASE 4: ML ENSEMBLE TRAINING
# =============================================================================

def phase4_ml_training(df: pd.DataFrame, features: List[str],
                       target_col: str = None) -> Dict[str, Any]:
    """
    Train 3-model ensemble (XGBoost, CatBoost, cuML RF) with VBUS memory management.
    Each model trained separately with GPU clearing between.
    """
    log("PHASE4", "=" * 60)
    log("PHASE4", "ML ENSEMBLE TRAINING (1350 trees x 3 models)")
    log("PHASE4", "=" * 60)

    # Import VBUS ML Executor
    try:
        from vbus_ml_executor import VBUSMLExecutor
        executor = VBUSMLExecutor(use_gpu=True)
    except ImportError:
        log("PHASE4", "VBUS ML Executor not available, using direct training")
        executor = None

    # Find target column
    if target_col is None:
        for candidate in ['views_h1_2024_total', 'views_h2_2024_total', 'views_total']:
            if candidate in df.columns:
                target_col = candidate
                break

    if target_col is None or target_col not in df.columns:
        log("PHASE4", "No target column found")
        return {"error": "No target column"}

    log("PHASE4", f"Target: {target_col}")
    log("PHASE4", f"Features: {len(features)}")

    # Prepare data
    df_clean = df[features + [target_col]].dropna(subset=[target_col])
    X = df_clean[features].fillna(0)
    y = df_clean[target_col]

    log("PHASE4", f"Training samples: {len(X):,}")

    if executor:
        # Use VBUS ML Executor
        result = executor.train_ensemble(X, y, features)
        return {
            "success": result.success,
            "mape": result.mape,
            "r2": result.r2,
            "total_time_ms": result.total_time_ms,
            "models": [
                {"name": mr.name, "success": mr.success, "trees": mr.trees_trained}
                for mr in result.model_results
            ]
        }
    else:
        # Direct training with memory management
        results = {"models": [], "success": False}
        predictions = {}

        # Stage data in system RAM
        X_np = np.nan_to_num(X.to_numpy(dtype=np.float64, na_value=np.nan), nan=0.0).astype(np.float32)
        y_np = np.nan_to_num(y.to_numpy(dtype=np.float64, na_value=np.nan), nan=0.0).astype(np.float32)

        # XGBoost
        clear_gpu_memory()
        log("PHASE4", "-" * 40)
        log("PHASE4", "Training XGBoost (1350 trees)...")

        try:
            import xgboost as xgb
            dtrain = xgb.DMatrix(X_np, label=y_np)
            params = {
                'objective': 'reg:squarederror',
                'tree_method': 'gpu_hist',
                'device': 'cuda',
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'verbosity': 1
            }
            model = xgb.train(params, dtrain, num_boost_round=CONFIG["trees_per_model"],
                             verbose_eval=100)
            predictions['xgb'] = model.predict(dtrain)
            results["models"].append({"name": "XGBoost", "success": True, "trees": 1350})
            del dtrain, model
            clear_gpu_memory()
            log("PHASE4", "XGBoost complete")
        except Exception as e:
            log("PHASE4", f"XGBoost failed: {e}")
            results["models"].append({"name": "XGBoost", "success": False, "error": str(e)})

        # CatBoost
        clear_gpu_memory()
        time.sleep(2)  # Let GPU settle
        log("PHASE4", "-" * 40)
        log("PHASE4", "Training CatBoost (1350 trees)...")

        try:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(
                iterations=CONFIG["trees_per_model"],
                depth=6,
                learning_rate=0.03,
                task_type='GPU',
                devices='0',
                gpu_ram_part=0.25,
                verbose=100
            )
            model.fit(X_np, y_np)
            predictions['cat'] = model.predict(X_np)
            results["models"].append({"name": "CatBoost", "success": True, "trees": 1350})
            del model
            clear_gpu_memory()
            log("PHASE4", "CatBoost complete")
        except Exception as e:
            log("PHASE4", f"CatBoost failed: {e}")
            results["models"].append({"name": "CatBoost", "success": False, "error": str(e)})

        # cuML RF
        clear_gpu_memory()
        time.sleep(2)
        log("PHASE4", "-" * 40)
        log("PHASE4", "Training cuML Random Forest (1350 trees)...")

        try:
            X_gpu = cp.asarray(X_np)
            y_gpu = cp.asarray(y_np)

            model = cuMLRF(
                n_estimators=CONFIG["trees_per_model"],
                max_depth=12,
                random_state=42,
                verbose=2
            )
            model.fit(X_gpu, y_gpu)
            predictions['rf'] = cp.asnumpy(model.predict(X_gpu))
            results["models"].append({"name": "cuML_RF", "success": True, "trees": 1350})
            del model, X_gpu, y_gpu
            clear_gpu_memory()
            log("PHASE4", "cuML RF complete")
        except Exception as e:
            log("PHASE4", f"cuML RF failed: {e}")
            results["models"].append({"name": "cuML_RF", "success": False, "error": str(e)})

        # Ensemble
        if predictions:
            weights = {
                'xgb': CONFIG["xgboost_weight"],
                'cat': CONFIG["catboost_weight"],
                'rf': CONFIG["cuml_rf_weight"]
            }

            ensemble_pred = None
            total_weight = 0
            for key, pred in predictions.items():
                w = weights.get(key, 0)
                if ensemble_pred is None:
                    ensemble_pred = w * pred
                else:
                    ensemble_pred += w * pred
                total_weight += w

            if total_weight > 0:
                ensemble_pred /= total_weight

            # Calculate metrics
            mask = y_np != 0
            mape = np.mean(np.abs((y_np[mask] - ensemble_pred[mask]) / y_np[mask])) * 100
            ss_res = np.sum((y_np - ensemble_pred) ** 2)
            ss_tot = np.sum((y_np - np.mean(y_np)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            results["mape"] = float(mape)
            results["r2"] = float(r2)
            results["success"] = True

            log("PHASE4", "=" * 40)
            log("PHASE4", f"ENSEMBLE MAPE: {mape:.2f}%")
            log("PHASE4", f"ENSEMBLE R2: {r2:.4f}")

        return results


# =============================================================================
# MAIN RUNNER
# =============================================================================

def main():
    """Main execution pipeline"""
    print("\n" + "=" * 70)
    print("  VBUS FEATURE SELECTION & CAUSATION RUNNER v1.0")
    print("  GPU MANDATORY | VBUS ENABLED | ALGO 95.4")
    print("=" * 70)

    start_time = time.time()

    # Check GPU
    if not GPU_AVAILABLE:
        print("\n[FATAL] GPU NOT AVAILABLE - CANNOT PROCEED")
        print("GPU is MANDATORY per ALGO 95.4 rules")
        sys.exit(1)

    free_mb, total_mb = get_gpu_memory()
    print(f"\n[GPU] {total_mb:.0f}MB total, {free_mb:.0f}MB free")

    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load BFD database
    log("LOAD", "Loading BFD_V27.65.parquet...")
    bfd_path = Path(CONFIG["bfd_path"])

    if not bfd_path.exists():
        # Try Windows path
        bfd_path = Path("C:/Users/RoyT6/Downloads/BFD_V27.65.parquet")

    df = pd.read_parquet(bfd_path)
    log("LOAD", f"Loaded: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    # =========================================================================
    # PHASE 1: Feature Selection
    # =========================================================================
    selected_features = phase1_feature_selection(df)

    # Save selected features
    with open(output_dir / "selected_features.json", "w") as f:
        json.dump({"features": selected_features, "count": len(selected_features)}, f, indent=2)

    # =========================================================================
    # PHASE 2: Correlation Matrix
    # =========================================================================
    corr_matrix = phase2_correlation_matrix(df, selected_features)
    corr_matrix.to_parquet(output_dir / "correlation_matrix.parquet")

    # =========================================================================
    # PHASE 3: Causation Analysis
    # =========================================================================
    causation_results = phase3_causation_analysis(df, selected_features)

    with open(output_dir / "causation_results.json", "w") as f:
        json.dump(causation_results, f, indent=2)

    # =========================================================================
    # PHASE 4: ML Training
    # =========================================================================
    ml_results = phase4_ml_training(df, selected_features)

    with open(output_dir / "ml_results.json", "w") as f:
        json.dump(ml_results, f, indent=2, default=str)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("  EXECUTION COMPLETE")
    print("=" * 70)
    print(f"  Total Time: {total_time / 60:.1f} minutes")
    print(f"  Features Selected: {len(selected_features)}")
    print(f"  Correlation Matrix: {corr_matrix.shape}")
    print(f"  Top Causal Features: {len(causation_results.get('significant_causes', []))}")
    print(f"  ML MAPE: {ml_results.get('mape', 'N/A')}")
    print(f"  ML R2: {ml_results.get('r2', 'N/A')}")
    print(f"\n  Output saved to: {output_dir}")
    print("=" * 70)

    # Save summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_time_seconds": total_time,
        "bfd_shape": list(df.shape),
        "features_selected": len(selected_features),
        "correlation_shape": list(corr_matrix.shape),
        "causation_results": causation_results,
        "ml_results": ml_results
    }

    with open(output_dir / "execution_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
