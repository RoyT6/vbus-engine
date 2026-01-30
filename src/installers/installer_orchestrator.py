"""
VBUS Installation Orchestrator
Manages the complete installation process including dependencies and components
"""

import os
import sys
import subprocess
import shutil
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from enum import Enum
import urllib.request
import zipfile
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VBUS.Installer")


class InstallStatus(Enum):
    """Installation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class InstallStep:
    """Represents an installation step"""
    name: str
    description: str
    action: Callable
    status: InstallStatus = InstallStatus.PENDING
    progress: int = 0
    error: Optional[str] = None
    required: bool = True
    dependencies: List[str] = field(default_factory=list)


@dataclass
class InstallConfiguration:
    """Complete installation configuration"""
    install_path: str
    vram_cache_enabled: bool = True
    ram_cache_enabled: bool = True
    l1_cache_size_gb: float = 2.0
    l2_cache_size_gb: float = 4.0
    l3_cache_path: str = ""
    master_db_path: str = ""
    repositories: List[str] = field(default_factory=list)
    install_cuda: bool = False
    install_cudf: bool = False
    install_cuml: bool = False
    install_rapids: bool = False
    install_xgboost: bool = True
    install_sklearn: bool = True
    install_catboost: bool = False
    create_shortcuts: bool = True
    add_to_path: bool = True


class MLFrameworkInstaller:
    """Handles installation of ML frameworks"""

    def __init__(self, install_path: Path):
        self.install_path = install_path
        self.venv_path = install_path / "venv"

    def create_virtual_environment(self) -> bool:
        """Create a Python virtual environment for VBUS"""
        try:
            logger.info("Creating virtual environment...")
            subprocess.run(
                [sys.executable, "-m", "venv", str(self.venv_path)],
                check=True,
                capture_output=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create venv: {e}")
            return False

    def get_pip_path(self) -> Path:
        """Get the pip executable path"""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "pip.exe"
        return self.venv_path / "bin" / "pip"

    def get_python_path(self) -> Path:
        """Get the Python executable path"""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "python.exe"
        return self.venv_path / "bin" / "python"

    def pip_install(self, packages: List[str], extra_index: str = None) -> bool:
        """Install packages using pip"""
        pip_path = self.get_pip_path()
        if not pip_path.exists():
            logger.error(f"pip not found at {pip_path}")
            return False

        cmd = [str(pip_path), "install"] + packages
        if extra_index:
            cmd.extend(["--extra-index-url", extra_index])

        try:
            logger.info(f"Installing: {' '.join(packages)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            if result.returncode != 0:
                logger.error(f"pip install failed: {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("pip install timed out")
            return False
        except Exception as e:
            logger.error(f"pip install error: {e}")
            return False

    def install_core_dependencies(self) -> bool:
        """Install core VBUS dependencies"""
        packages = [
            "psutil>=5.9.0",
            "GPUtil>=1.4.0",
            "py-cpuinfo>=9.0.0",
            "PyQt6>=6.4.0",
            "pyyaml>=6.0",
            "rich>=13.0.0",
            "tqdm>=4.65.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0"
        ]
        return self.pip_install(packages)

    def install_nvidia_tools(self) -> bool:
        """Install NVIDIA Python tools"""
        packages = [
            "pynvml>=11.5.0",
            "nvidia-ml-py>=12.535.0"
        ]
        return self.pip_install(packages)

    def install_xgboost(self) -> bool:
        """Install XGBoost"""
        return self.pip_install(["xgboost"])

    def install_sklearn(self) -> bool:
        """Install scikit-learn"""
        return self.pip_install(["scikit-learn"])

    def install_catboost(self) -> bool:
        """Install CatBoost"""
        return self.pip_install(["catboost"])

    def install_cudf(self) -> bool:
        """Install cuDF (GPU DataFrames)"""
        return self.pip_install(
            ["cudf-cu12"],
            extra_index="https://pypi.nvidia.com"
        )

    def install_cuml(self) -> bool:
        """Install cuML (GPU ML)"""
        return self.pip_install(
            ["cuml-cu12"],
            extra_index="https://pypi.nvidia.com"
        )


class InstallationOrchestrator:
    """Orchestrates the complete VBUS installation"""

    def __init__(self, config: InstallConfiguration):
        self.config = config
        self.install_path = Path(config.install_path)
        self.steps: List[InstallStep] = []
        self.current_step: Optional[InstallStep] = None
        self.ml_installer: Optional[MLFrameworkInstaller] = None

        self._callbacks: Dict[str, List[Callable]] = {
            "step_start": [],
            "step_progress": [],
            "step_complete": [],
            "step_error": [],
            "install_complete": []
        }

    def on(self, event: str, callback: Callable):
        """Register event callback"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _emit(self, event: str, *args, **kwargs):
        """Emit event to callbacks"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _build_steps(self):
        """Build the installation steps"""
        self.steps = [
            InstallStep(
                name="validate",
                description="Validating installation configuration",
                action=self._step_validate
            ),
            InstallStep(
                name="create_directories",
                description="Creating installation directories",
                action=self._step_create_directories
            ),
            InstallStep(
                name="create_venv",
                description="Creating Python virtual environment",
                action=self._step_create_venv
            ),
            InstallStep(
                name="install_core",
                description="Installing core dependencies",
                action=self._step_install_core
            ),
            InstallStep(
                name="copy_source",
                description="Copying VBUS source files",
                action=self._step_copy_source
            ),
            InstallStep(
                name="install_nvidia",
                description="Installing NVIDIA tools",
                action=self._step_install_nvidia,
                required=self.config.vram_cache_enabled
            ),
            InstallStep(
                name="install_xgboost",
                description="Installing XGBoost",
                action=self._step_install_xgboost,
                required=self.config.install_xgboost
            ),
            InstallStep(
                name="install_sklearn",
                description="Installing scikit-learn",
                action=self._step_install_sklearn,
                required=self.config.install_sklearn
            ),
            InstallStep(
                name="install_catboost",
                description="Installing CatBoost",
                action=self._step_install_catboost,
                required=self.config.install_catboost
            ),
            InstallStep(
                name="install_cudf",
                description="Installing cuDF",
                action=self._step_install_cudf,
                required=self.config.install_cudf
            ),
            InstallStep(
                name="install_cuml",
                description="Installing cuML",
                action=self._step_install_cuml,
                required=self.config.install_cuml
            ),
            InstallStep(
                name="configure_cache",
                description="Configuring cache system",
                action=self._step_configure_cache
            ),
            InstallStep(
                name="configure_bus",
                description="Configuring BUS system",
                action=self._step_configure_bus
            ),
            InstallStep(
                name="create_config",
                description="Creating configuration files",
                action=self._step_create_config
            ),
            InstallStep(
                name="create_shortcuts",
                description="Creating shortcuts and scripts",
                action=self._step_create_shortcuts,
                required=self.config.create_shortcuts
            ),
            InstallStep(
                name="finalize",
                description="Finalizing installation",
                action=self._step_finalize
            )
        ]

    def install(self) -> bool:
        """Run the complete installation"""
        self._build_steps()
        self.ml_installer = MLFrameworkInstaller(self.install_path)

        total_steps = len([s for s in self.steps if s.required])
        completed_steps = 0

        for step in self.steps:
            if not step.required:
                step.status = InstallStatus.SKIPPED
                continue

            self.current_step = step
            step.status = InstallStatus.IN_PROGRESS

            self._emit("step_start", step.name, step.description)

            try:
                logger.info(f"Starting: {step.description}")
                success = step.action()

                if success:
                    step.status = InstallStatus.COMPLETED
                    step.progress = 100
                    completed_steps += 1
                    overall_progress = int(completed_steps / total_steps * 100)
                    self._emit("step_complete", step.name, overall_progress)
                else:
                    step.status = InstallStatus.FAILED
                    step.error = "Step failed without specific error"
                    self._emit("step_error", step.name, step.error)
                    return False

            except Exception as e:
                step.status = InstallStatus.FAILED
                step.error = str(e)
                logger.error(f"Step failed: {step.name} - {e}")
                self._emit("step_error", step.name, str(e))
                return False

        self._emit("install_complete", True)
        return True

    def _step_validate(self) -> bool:
        """Validate installation configuration"""
        if not self.config.install_path:
            raise ValueError("Installation path not specified")

        # Check disk space
        try:
            import shutil
            parent = Path(self.config.install_path).parent
            if parent.exists():
                free_space = shutil.disk_usage(parent).free
                required_space = 10 * (1024 ** 3)  # 10 GB
                if free_space < required_space:
                    raise ValueError(f"Insufficient disk space. Required: 10GB, Available: {free_space / (1024**3):.1f}GB")
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

        return True

    def _step_create_directories(self) -> bool:
        """Create installation directories"""
        directories = [
            self.install_path,
            self.install_path / "src",
            self.install_path / "src" / "core",
            self.install_path / "src" / "gui",
            self.install_path / "src" / "cache",
            self.install_path / "src" / "bus",
            self.install_path / "src" / "utils",
            self.install_path / "config",
            self.install_path / "logs",
            self.install_path / "data"
        ]

        # Create L3 cache directory
        if self.config.l3_cache_path:
            directories.append(Path(self.config.l3_cache_path))
        else:
            directories.append(Path.home() / ".vbus" / "cache")

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

        return True

    def _step_create_venv(self) -> bool:
        """Create Python virtual environment"""
        return self.ml_installer.create_virtual_environment()

    def _step_install_core(self) -> bool:
        """Install core dependencies"""
        # First upgrade pip
        pip_path = self.ml_installer.get_pip_path()
        try:
            subprocess.run(
                [str(pip_path), "install", "--upgrade", "pip"],
                capture_output=True,
                check=True
            )
        except Exception:
            pass  # Continue even if pip upgrade fails

        return self.ml_installer.install_core_dependencies()

    def _step_copy_source(self) -> bool:
        """Copy VBUS source files"""
        # Get source directory (where this script is)
        source_root = Path(__file__).parent.parent.parent
        dest_src = self.install_path / "src"

        # Copy source files
        for subdir in ["core", "gui", "cache", "bus", "utils", "installers"]:
            src_dir = source_root / "src" / subdir
            dst_dir = dest_src / subdir

            if src_dir.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                for file in src_dir.glob("*.py"):
                    shutil.copy2(file, dst_dir / file.name)
                    logger.debug(f"Copied: {file.name}")

        # Create __init__.py files
        for subdir in ["", "core", "gui", "cache", "bus", "utils", "installers"]:
            init_file = dest_src / subdir / "__init__.py"
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.touch()

        return True

    def _step_install_nvidia(self) -> bool:
        """Install NVIDIA tools"""
        return self.ml_installer.install_nvidia_tools()

    def _step_install_xgboost(self) -> bool:
        """Install XGBoost"""
        return self.ml_installer.install_xgboost()

    def _step_install_sklearn(self) -> bool:
        """Install scikit-learn"""
        return self.ml_installer.install_sklearn()

    def _step_install_catboost(self) -> bool:
        """Install CatBoost"""
        return self.ml_installer.install_catboost()

    def _step_install_cudf(self) -> bool:
        """Install cuDF"""
        return self.ml_installer.install_cudf()

    def _step_install_cuml(self) -> bool:
        """Install cuML"""
        return self.ml_installer.install_cuml()

    def _step_configure_cache(self) -> bool:
        """Configure cache system"""
        cache_config = {
            "vram_enabled": self.config.vram_cache_enabled,
            "ram_enabled": self.config.ram_cache_enabled,
            "l1_size_gb": self.config.l1_cache_size_gb,
            "l2_size_gb": self.config.l2_cache_size_gb,
            "l3_path": self.config.l3_cache_path or str(Path.home() / ".vbus" / "cache"),
            "policy": "lru",
            "auto_eviction": True,
            "prefetch_enabled": True
        }

        config_file = self.install_path / "config" / "cache_config.json"
        with open(config_file, 'w') as f:
            json.dump(cache_config, f, indent=2)

        return True

    def _step_configure_bus(self) -> bool:
        """Configure BUS system"""
        bus_config = {
            "master_db": {
                "name": "master",
                "path": self.config.master_db_path,
                "connection_string": ""
            },
            "repositories": [
                {"name": f"repo_{i}", "path": repo, "priority": 50 - i * 5}
                for i, repo in enumerate(self.config.repositories)
            ],
            "sync": {
                "auto_sync": True,
                "sync_interval_seconds": 300,
                "conflict_resolution": "master_wins"
            },
            "traffic": {
                "max_workers": 4,
                "prefetch_enabled": True,
                "ai_routing": True
            }
        }

        config_file = self.install_path / "config" / "bus_config.json"
        with open(config_file, 'w') as f:
            json.dump(bus_config, f, indent=2)

        return True

    def _step_create_config(self) -> bool:
        """Create main configuration file"""
        main_config = {
            "version": "1.0.0",
            "install_path": str(self.install_path),
            "python_path": str(self.ml_installer.get_python_path()),
            "components": {
                "cache": {
                    "enabled": self.config.vram_cache_enabled or self.config.ram_cache_enabled,
                    "config_file": "config/cache_config.json"
                },
                "bus": {
                    "enabled": bool(self.config.master_db_path),
                    "config_file": "config/bus_config.json"
                }
            },
            "ml_frameworks": {
                "xgboost": self.config.install_xgboost,
                "sklearn": self.config.install_sklearn,
                "catboost": self.config.install_catboost,
                "cudf": self.config.install_cudf,
                "cuml": self.config.install_cuml
            },
            "logging": {
                "level": "INFO",
                "file": "logs/vbus.log",
                "max_size_mb": 10,
                "backup_count": 5
            }
        }

        config_file = self.install_path / "config" / "vbus_config.json"
        with open(config_file, 'w') as f:
            json.dump(main_config, f, indent=2)

        return True

    def _step_create_shortcuts(self) -> bool:
        """Create shortcuts and launch scripts"""
        python_path = self.ml_installer.get_python_path()

        # Create launch script
        if sys.platform == "win32":
            launch_script = self.install_path / "launch_vbus.bat"
            launch_content = f'''@echo off
cd /d "{self.install_path}"
"{python_path}" -m src.gui.installer_gui
'''
        else:
            launch_script = self.install_path / "launch_vbus.sh"
            launch_content = f'''#!/bin/bash
cd "{self.install_path}"
"{python_path}" -m src.gui.installer_gui
'''

        with open(launch_script, 'w') as f:
            f.write(launch_content)

        if sys.platform != "win32":
            launch_script.chmod(0o755)

        # Create CLI script
        if sys.platform == "win32":
            cli_script = self.install_path / "vbus.bat"
            cli_content = f'''@echo off
"{python_path}" -m src.core.vbus_cli %*
'''
        else:
            cli_script = self.install_path / "vbus"
            cli_content = f'''#!/bin/bash
"{python_path}" -m src.core.vbus_cli "$@"
'''

        with open(cli_script, 'w') as f:
            f.write(cli_content)

        if sys.platform != "win32":
            cli_script.chmod(0o755)

        return True

    def _step_finalize(self) -> bool:
        """Finalize installation"""
        # Write installation log
        install_log = {
            "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "1.0.0",
            "config": {
                "install_path": self.config.install_path,
                "vram_cache": self.config.vram_cache_enabled,
                "ram_cache": self.config.ram_cache_enabled
            },
            "steps": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "error": s.error
                }
                for s in self.steps
            ]
        }

        log_file = self.install_path / "logs" / "install_log.json"
        with open(log_file, 'w') as f:
            json.dump(install_log, f, indent=2)

        # Create uninstaller
        self._create_uninstaller()

        logger.info("Installation completed successfully!")
        return True

    def _create_uninstaller(self):
        """Create uninstaller script"""
        install_path = str(self.install_path).replace("\\", "\\\\")

        if sys.platform == "win32":
            uninstall_script = self.install_path / "uninstall.bat"
            uninstall_content = f'''@echo off
echo VBUS Uninstaller
echo ================
echo.
echo This will remove VBUS from your system.
echo Installation path: {install_path}
echo.
set /p confirm="Are you sure you want to uninstall VBUS? (y/n): "
if /i "%confirm%"=="y" (
    echo Removing VBUS...
    rmdir /s /q "{install_path}"
    echo VBUS has been uninstalled.
) else (
    echo Uninstallation cancelled.
)
pause
'''
        else:
            uninstall_script = self.install_path / "uninstall.sh"
            uninstall_content = f'''#!/bin/bash
echo "VBUS Uninstaller"
echo "================"
echo ""
echo "This will remove VBUS from your system."
echo "Installation path: {install_path}"
echo ""
read -p "Are you sure you want to uninstall VBUS? (y/n): " confirm
if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo "Removing VBUS..."
    rm -rf "{install_path}"
    echo "VBUS has been uninstalled."
else
    echo "Uninstallation cancelled."
fi
'''

        with open(uninstall_script, 'w') as f:
            f.write(uninstall_content)

        if sys.platform != "win32":
            uninstall_script.chmod(0o755)

    def get_status(self) -> Dict[str, Any]:
        """Get current installation status"""
        return {
            "steps": [
                {
                    "name": s.name,
                    "description": s.description,
                    "status": s.status.value,
                    "progress": s.progress,
                    "error": s.error,
                    "required": s.required
                }
                for s in self.steps
            ],
            "current_step": self.current_step.name if self.current_step else None,
            "overall_progress": self._calculate_overall_progress()
        }

    def _calculate_overall_progress(self) -> int:
        """Calculate overall installation progress"""
        required_steps = [s for s in self.steps if s.required]
        if not required_steps:
            return 0

        completed = sum(1 for s in required_steps if s.status == InstallStatus.COMPLETED)
        return int(completed / len(required_steps) * 100)


def run_installation(config_dict: Dict[str, Any], progress_callback: Callable = None) -> bool:
    """
    Run installation with the given configuration.

    Args:
        config_dict: Installation configuration dictionary
        progress_callback: Optional callback for progress updates (step_name, progress, message)

    Returns:
        bool: True if installation succeeded
    """
    config = InstallConfiguration(
        install_path=config_dict.get("install_path", ""),
        vram_cache_enabled=config_dict.get("vram_cache_enabled", True),
        ram_cache_enabled=config_dict.get("ram_cache_enabled", True),
        l1_cache_size_gb=config_dict.get("l1_cache_size_gb", 2.0),
        l2_cache_size_gb=config_dict.get("l2_cache_size_gb", 4.0),
        l3_cache_path=config_dict.get("l3_cache_path", ""),
        master_db_path=config_dict.get("master_db_path", ""),
        repositories=config_dict.get("repositories", []),
        install_cuda=config_dict.get("install_cuda", False),
        install_cudf=config_dict.get("install_cudf", False),
        install_cuml=config_dict.get("install_cuml", False),
        install_rapids=config_dict.get("install_rapids", False),
        install_xgboost=config_dict.get("install_xgboost", True),
        install_sklearn=config_dict.get("install_sklearn", True),
        install_catboost=config_dict.get("install_catboost", False),
        create_shortcuts=config_dict.get("create_shortcuts", True),
        add_to_path=config_dict.get("add_to_path", True)
    )

    orchestrator = InstallationOrchestrator(config)

    if progress_callback:
        orchestrator.on("step_start", lambda name, desc: progress_callback(name, 0, f"Starting: {desc}"))
        orchestrator.on("step_complete", lambda name, progress: progress_callback(name, progress, "Complete"))
        orchestrator.on("step_error", lambda name, error: progress_callback(name, -1, f"Error: {error}"))

    return orchestrator.install()


if __name__ == "__main__":
    # Example usage
    test_config = {
        "install_path": str(Path.home() / "VBUS_Test"),
        "vram_cache_enabled": True,
        "ram_cache_enabled": True,
        "l1_cache_size_gb": 2.0,
        "l2_cache_size_gb": 4.0,
        "install_xgboost": True,
        "install_sklearn": True
    }

    def progress(step, progress, message):
        print(f"[{step}] {progress}% - {message}")

    success = run_installation(test_config, progress)
    print(f"\nInstallation {'succeeded' if success else 'failed'}")
