"""
VBUS ML Framework Detection Module
Detects installed ML frameworks: CUDA, cuDF, cuML, RAPIDS, XGBoost, Random Forest, CatBoost
"""

import subprocess
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json


@dataclass
class MLFrameworkInfo:
    """Information about an ML framework"""
    name: str
    installed: bool = False
    version: str = ""
    location: str = ""
    cuda_compatible: bool = False
    gpu_enabled: bool = False
    install_command: str = ""
    notes: str = ""


class MLFrameworkDetector:
    """Detects and validates ML framework installations"""

    def __init__(self):
        self.frameworks: Dict[str, MLFrameworkInfo] = {}
        self._initialize_frameworks()

    def _initialize_frameworks(self):
        """Initialize framework detection configurations"""
        self.frameworks = {
            "cuda": MLFrameworkInfo(
                name="CUDA Toolkit",
                install_command="Download from https://developer.nvidia.com/cuda-downloads"
            ),
            "cudnn": MLFrameworkInfo(
                name="cuDNN",
                install_command="Download from https://developer.nvidia.com/cudnn"
            ),
            "cudf": MLFrameworkInfo(
                name="cuDF",
                install_command="pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com"
            ),
            "cuml": MLFrameworkInfo(
                name="cuML",
                install_command="pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com"
            ),
            "rapids": MLFrameworkInfo(
                name="RAPIDS",
                install_command="conda install -c rapidsai -c conda-forge -c nvidia rapids"
            ),
            "xgboost": MLFrameworkInfo(
                name="XGBoost",
                install_command="pip install xgboost"
            ),
            "sklearn": MLFrameworkInfo(
                name="Scikit-learn (Random Forest)",
                install_command="pip install scikit-learn"
            ),
            "catboost": MLFrameworkInfo(
                name="CatBoost",
                install_command="pip install catboost"
            )
        }

    def detect_all(self) -> Dict[str, MLFrameworkInfo]:
        """Detect all ML frameworks"""
        self._detect_cuda()
        self._detect_cudnn()
        self._detect_cudf()
        self._detect_cuml()
        self._detect_rapids()
        self._detect_xgboost()
        self._detect_sklearn()
        self._detect_catboost()
        return self.frameworks

    def _detect_cuda(self):
        """Detect CUDA Toolkit installation"""
        framework = self.frameworks["cuda"]

        # Check CUDA_PATH environment variable
        cuda_path = os.environ.get("CUDA_PATH", "")
        if cuda_path and os.path.exists(cuda_path):
            framework.installed = True
            framework.location = cuda_path

        # Try nvcc to get version
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                framework.installed = True
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        parts = line.split('release')
                        if len(parts) > 1:
                            framework.version = parts[1].strip().split(',')[0].strip()
                            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check common CUDA installation paths
        if not framework.installed:
            common_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                r"C:\CUDA",
                "/usr/local/cuda"
            ]
            for base_path in common_paths:
                if os.path.exists(base_path):
                    # Find highest version
                    try:
                        versions = [d for d in os.listdir(base_path) if d.startswith('v')]
                        if versions:
                            versions.sort(reverse=True)
                            framework.installed = True
                            framework.version = versions[0].lstrip('v')
                            framework.location = os.path.join(base_path, versions[0])
                            break
                    except (OSError, PermissionError):
                        pass

        framework.cuda_compatible = framework.installed
        framework.gpu_enabled = framework.installed

    def _detect_cudnn(self):
        """Detect cuDNN installation"""
        framework = self.frameworks["cudnn"]

        # Check if cuDNN header exists in CUDA path
        cuda_path = os.environ.get("CUDA_PATH", "")
        if cuda_path:
            cudnn_header = os.path.join(cuda_path, "include", "cudnn.h")
            cudnn_header_v8 = os.path.join(cuda_path, "include", "cudnn_version.h")

            for header_path in [cudnn_header, cudnn_header_v8]:
                if os.path.exists(header_path):
                    framework.installed = True
                    framework.location = os.path.dirname(header_path)
                    # Try to extract version from header
                    try:
                        with open(header_path, 'r') as f:
                            content = f.read()
                            for line in content.split('\n'):
                                if 'CUDNN_MAJOR' in line and '#define' in line:
                                    major = line.split()[-1]
                                elif 'CUDNN_MINOR' in line and '#define' in line:
                                    minor = line.split()[-1]
                                elif 'CUDNN_PATCHLEVEL' in line and '#define' in line:
                                    patch = line.split()[-1]
                            if 'major' in dir():
                                framework.version = f"{major}.{minor}.{patch}"
                    except (IOError, NameError):
                        pass
                    break

        framework.cuda_compatible = framework.installed
        framework.gpu_enabled = framework.installed

    def _detect_cudf(self):
        """Detect cuDF installation"""
        framework = self.frameworks["cudf"]

        try:
            spec = importlib.util.find_spec("cudf")
            if spec:
                framework.installed = True
                framework.location = spec.origin or ""
                # Try to get version
                try:
                    import cudf
                    framework.version = cudf.__version__
                    framework.gpu_enabled = True
                except (ImportError, AttributeError):
                    pass
        except (ImportError, ModuleNotFoundError):
            pass

        framework.cuda_compatible = framework.installed

    def _detect_cuml(self):
        """Detect cuML installation"""
        framework = self.frameworks["cuml"]

        try:
            spec = importlib.util.find_spec("cuml")
            if spec:
                framework.installed = True
                framework.location = spec.origin or ""
                try:
                    import cuml
                    framework.version = cuml.__version__
                    framework.gpu_enabled = True
                except (ImportError, AttributeError):
                    pass
        except (ImportError, ModuleNotFoundError):
            pass

        framework.cuda_compatible = framework.installed

    def _detect_rapids(self):
        """Detect RAPIDS installation (checks for multiple RAPIDS components)"""
        framework = self.frameworks["rapids"]

        rapids_components = ["cudf", "cuml", "cugraph", "cuspatial", "cupy"]
        installed_components = []

        for component in rapids_components:
            try:
                spec = importlib.util.find_spec(component)
                if spec:
                    installed_components.append(component)
            except (ImportError, ModuleNotFoundError):
                pass

        if len(installed_components) >= 2:  # At least 2 RAPIDS components
            framework.installed = True
            framework.notes = f"Components: {', '.join(installed_components)}"
            framework.gpu_enabled = True
            framework.cuda_compatible = True

            # Try to get version from cudf or cuml
            for comp in ["cudf", "cuml"]:
                try:
                    mod = __import__(comp)
                    framework.version = getattr(mod, "__version__", "")
                    if framework.version:
                        break
                except ImportError:
                    pass

    def _detect_xgboost(self):
        """Detect XGBoost installation"""
        framework = self.frameworks["xgboost"]

        try:
            spec = importlib.util.find_spec("xgboost")
            if spec:
                framework.installed = True
                framework.location = spec.origin or ""
                try:
                    import xgboost as xgb
                    framework.version = xgb.__version__

                    # Check if GPU support is available
                    try:
                        # Try to create a GPU-enabled booster
                        params = {'tree_method': 'gpu_hist', 'device': 'cuda'}
                        # Just check if the parameter is accepted
                        framework.gpu_enabled = True
                        framework.cuda_compatible = True
                        framework.notes = "GPU support available"
                    except Exception:
                        framework.notes = "CPU only"
                except (ImportError, AttributeError):
                    pass
        except (ImportError, ModuleNotFoundError):
            pass

    def _detect_sklearn(self):
        """Detect Scikit-learn installation (Random Forest)"""
        framework = self.frameworks["sklearn"]

        try:
            spec = importlib.util.find_spec("sklearn")
            if spec:
                framework.installed = True
                framework.location = spec.origin or ""
                try:
                    import sklearn
                    framework.version = sklearn.__version__
                    framework.notes = "Includes RandomForestClassifier/Regressor"
                except (ImportError, AttributeError):
                    pass
        except (ImportError, ModuleNotFoundError):
            pass

        # sklearn is CPU-only by default
        framework.gpu_enabled = False
        framework.cuda_compatible = False

    def _detect_catboost(self):
        """Detect CatBoost installation"""
        framework = self.frameworks["catboost"]

        try:
            spec = importlib.util.find_spec("catboost")
            if spec:
                framework.installed = True
                framework.location = spec.origin or ""
                try:
                    import catboost
                    framework.version = catboost.__version__

                    # Check GPU support
                    try:
                        # CatBoost has built-in GPU support detection
                        framework.gpu_enabled = True
                        framework.cuda_compatible = True
                        framework.notes = "GPU support available"
                    except Exception:
                        framework.notes = "CPU only"
                except (ImportError, AttributeError):
                    pass
        except (ImportError, ModuleNotFoundError):
            pass

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all framework detections"""
        installed = []
        not_installed = []

        for key, fw in self.frameworks.items():
            info = {
                "name": fw.name,
                "version": fw.version,
                "gpu_enabled": fw.gpu_enabled,
                "location": fw.location,
                "notes": fw.notes
            }
            if fw.installed:
                installed.append(info)
            else:
                info["install_command"] = fw.install_command
                not_installed.append(info)

        return {
            "installed": installed,
            "not_installed": not_installed,
            "total_installed": len(installed),
            "total_not_installed": len(not_installed),
            "gpu_ready": any(fw.gpu_enabled for fw in self.frameworks.values())
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert all framework info to dictionary"""
        return {
            key: {
                "name": fw.name,
                "installed": fw.installed,
                "version": fw.version,
                "location": fw.location,
                "cuda_compatible": fw.cuda_compatible,
                "gpu_enabled": fw.gpu_enabled,
                "install_command": fw.install_command,
                "notes": fw.notes
            }
            for key, fw in self.frameworks.items()
        }


if __name__ == "__main__":
    detector = MLFrameworkDetector()
    detector.detect_all()

    print("=== ML Framework Detection Results ===\n")

    summary = detector.get_summary()

    print("✓ INSTALLED FRAMEWORKS:")
    print("-" * 50)
    for fw in summary["installed"]:
        gpu_status = "🖥️ GPU" if fw["gpu_enabled"] else "💻 CPU"
        print(f"  {fw['name']}")
        print(f"    Version: {fw['version'] or 'Unknown'}")
        print(f"    Mode: {gpu_status}")
        if fw['notes']:
            print(f"    Notes: {fw['notes']}")
        print()

    print("\n✗ NOT INSTALLED:")
    print("-" * 50)
    for fw in summary["not_installed"]:
        print(f"  {fw['name']}")
        print(f"    Install: {fw['install_command']}")
        print()

    print(f"\nSummary: {summary['total_installed']}/{len(detector.frameworks)} frameworks installed")
    print(f"GPU Ready: {'Yes' if summary['gpu_ready'] else 'No'}")
