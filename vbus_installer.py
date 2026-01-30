"""
VBUS Standalone Installer Application
A single-file installer that can be compiled to .exe using PyInstaller

Run: pyinstaller --onefile --windowed --icon=vbus.ico --name="VBUS_Installer" vbus_installer.py
"""

import sys
import os
import json
import shutil
import platform
import subprocess
import threading
import time
import importlib.util
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from enum import Enum

# =============================================================================
#  VBUS INSTALLER - CONFIGURATION & CONSTANTS
# =============================================================================

APP_TITLE = "VBUS System Installer | FRAMECORE"
APP_VERSION = "1.0.0"
WINDOW_SIZE = "1000x750"
COMPANY_NAME = "FRAMECORE"

# Required folder structure
VBUS_STRUCTURE = [
    "BUS/CONTROL_PLANE",
    "BUS/INTERCONNECT",
    "BUS/AI_ROUTER",
    "CACHE/L1_VRAM",
    "CACHE/L2_RAM",
    "CACHE/L3_DISK",
    "AUDIT/LOGS",
    "AUDIT/METRICS",
    "DATA/MASTER_DB",
    "DATA/REPOS",
    "CONFIG",
    "SCRIPTS"
]

# ML Libraries to detect
ML_LIBRARIES = [
    ("torch", "PyTorch/CUDA"),
    ("cudf", "RAPIDS cuDF"),
    ("cuml", "RAPIDS cuML"),
    ("xgboost", "XGBoost"),
    ("catboost", "CatBoost"),
    ("sklearn", "Scikit-learn (Random Forest)"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
]

# =============================================================================
#  SYSTEM ANALYZER
# =============================================================================

class SystemAnalyzer:
    """Analyzes system for VBUS compatibility"""

    def __init__(self):
        self.report = {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "gpu_found": False,
            "gpu_name": "Not detected",
            "vram_mb": 0,
            "vram_free_mb": 0,
            "cuda_version": "N/A",
            "driver_version": "N/A",
            "compute_capability": "N/A",
            "ram_total_gb": 0,
            "ram_available_gb": 0,
            "disk_free_gb": 0,
            "libs": {},
            "is_compatible": True,
            "issues": [],
            "warnings": []
        }

    def check_gpu(self) -> bool:
        """Check for NVIDIA GPU using nvidia-smi"""
        try:
            cmd = "nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader,nounits"
            result = subprocess.run(
                cmd.split() if platform.system() != "Windows" else cmd,
                capture_output=True, text=True, timeout=15,
                shell=(platform.system() == "Windows")
            )

            if result.returncode == 0 and result.stdout.strip():
                parts = [p.strip() for p in result.stdout.strip().split(',')]
                if len(parts) >= 5:
                    self.report["gpu_found"] = True
                    self.report["gpu_name"] = parts[0]
                    self.report["vram_mb"] = int(float(parts[1]))
                    self.report["vram_free_mb"] = int(float(parts[2]))
                    self.report["driver_version"] = parts[3]
                    self.report["compute_capability"] = parts[4]

                    # Try to get CUDA version
                    try:
                        nvcc_result = subprocess.run(
                            ["nvcc", "--version"],
                            capture_output=True, text=True, timeout=10
                        )
                        if nvcc_result.returncode == 0:
                            for line in nvcc_result.stdout.split('\n'):
                                if 'release' in line.lower():
                                    cuda_ver = line.split('release')[-1].strip().split(',')[0].strip()
                                    self.report["cuda_version"] = cuda_ver
                                    break
                    except:
                        pass

                    return True
        except Exception as e:
            self.report["issues"].append(f"GPU detection failed: {str(e)}")

        self.report["gpu_found"] = False
        return False

    def check_memory(self):
        """Check system RAM"""
        try:
            import ctypes

            if platform.system() == "Windows":
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

                self.report["ram_total_gb"] = round(stat.ullTotalPhys / (1024**3), 2)
                self.report["ram_available_gb"] = round(stat.ullAvailPhys / (1024**3), 2)
        except Exception as e:
            self.report["warnings"].append(f"RAM detection issue: {str(e)}")

    def check_disk_space(self, path: str = None):
        """Check available disk space"""
        try:
            if path is None:
                path = os.path.expanduser("~")

            if platform.system() == "Windows":
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(path), None, None, ctypes.pointer(free_bytes)
                )
                self.report["disk_free_gb"] = round(free_bytes.value / (1024**3), 2)
            else:
                st = os.statvfs(path)
                self.report["disk_free_gb"] = round((st.f_bavail * st.f_frsize) / (1024**3), 2)
        except Exception as e:
            self.report["warnings"].append(f"Disk space check issue: {str(e)}")

    def check_ml_libs(self):
        """Check for ML libraries"""
        for pkg, name in ML_LIBRARIES:
            try:
                found = importlib.util.find_spec(pkg) is not None
                version = ""
                if found:
                    try:
                        mod = __import__(pkg)
                        version = getattr(mod, "__version__", "")
                    except:
                        pass
                self.report["libs"][name] = {"installed": found, "version": version}
            except:
                self.report["libs"][name] = {"installed": False, "version": ""}

    def check_admin(self) -> bool:
        """Check if running with admin privileges"""
        try:
            if platform.system() == "Windows":
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                return os.geteuid() == 0
        except:
            return False

    def validate_compatibility(self):
        """Validate system compatibility"""
        # Check minimum RAM
        if self.report["ram_total_gb"] < 8:
            self.report["issues"].append(f"Insufficient RAM: {self.report['ram_total_gb']}GB (minimum 8GB)")
            self.report["is_compatible"] = False

        # Check disk space
        if self.report["disk_free_gb"] < 10:
            self.report["issues"].append(f"Insufficient disk space: {self.report['disk_free_gb']}GB (minimum 10GB)")
            self.report["is_compatible"] = False

        # GPU warning (not fatal)
        if not self.report["gpu_found"]:
            self.report["warnings"].append("No NVIDIA GPU detected. L1 VRAM cache will operate in simulation mode.")

        # Check OS
        if platform.system() not in ["Windows", "Linux"]:
            self.report["warnings"].append(f"Untested operating system: {platform.system()}")

    def run_full_scan(self) -> Dict[str, Any]:
        """Run complete system scan"""
        self.check_gpu()
        self.check_memory()
        self.check_disk_space()
        self.check_ml_libs()
        self.validate_compatibility()
        return self.report


# =============================================================================
#  VBUS CODE GENERATOR
# =============================================================================

class VBUSCodeGenerator:
    """Generates VBUS application code and configuration"""

    @staticmethod
    def generate_config(install_path: str, sys_report: Dict, master_db: str, repos: List[str]) -> str:
        """Generate main configuration file"""
        config = {
            "vbus_version": APP_VERSION,
            "generated_at": datetime.now().isoformat(),
            "hardware": {
                "gpu_detected": sys_report["gpu_found"],
                "gpu_name": sys_report["gpu_name"],
                "vram_mb": sys_report["vram_mb"],
                "cuda_version": sys_report["cuda_version"],
                "compute_capability": sys_report["compute_capability"],
                "system_ram_gb": sys_report["ram_total_gb"]
            },
            "cache_hierarchy": {
                "L1_VRAM": {
                    "path": os.path.join(install_path, "CACHE", "L1_VRAM"),
                    "type": "gpu_vram",
                    "enabled": sys_report["gpu_found"],
                    "max_size_mb": int(sys_report["vram_mb"] * 0.6) if sys_report["gpu_found"] else 0,
                    "eviction_policy": "lru"
                },
                "L2_RAM": {
                    "path": os.path.join(install_path, "CACHE", "L2_RAM"),
                    "type": "system_ram",
                    "enabled": True,
                    "max_size_mb": int(sys_report["ram_total_gb"] * 1024 * 0.3),
                    "eviction_policy": "lru"
                },
                "L3_DISK": {
                    "path": os.path.join(install_path, "CACHE", "L3_DISK"),
                    "type": "disk",
                    "enabled": True,
                    "max_size_mb": 10240,  # 10GB default
                    "eviction_policy": "lru"
                }
            },
            "bus_config": {
                "master_db": master_db,
                "repositories": [r for r in repos if r],
                "ai_routing_enabled": True,
                "coherency_protocol": "mesi",
                "sync_interval_seconds": 60
            },
            "ml_frameworks": {
                lib_name: info["installed"]
                for lib_name, info in sys_report.get("libs", {}).items()
            },
            "circuit_breaker": {
                "enabled": True,
                "max_failures": 3,
                "reset_timeout_seconds": 300,
                "fail_closed": True
            },
            "logging": {
                "level": "INFO",
                "path": os.path.join(install_path, "AUDIT", "LOGS"),
                "max_file_size_mb": 10,
                "backup_count": 5
            }
        }
        return json.dumps(config, indent=2)

    @staticmethod
    def generate_daemon_script(install_path: str) -> str:
        """Generate the main VBUS daemon script"""
        return f'''#!/usr/bin/env python3
"""
VBUS Daemon - Cache Hierarchy and Traffic Management System
Generated by VBUS Installer v{APP_VERSION}
"""

import os
import sys
import json
import time
import hashlib
import threading
import logging
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Any, Optional, List
from enum import Enum

# Setup paths
VBUS_ROOT = Path(r"{install_path}")
CONFIG_PATH = VBUS_ROOT / "CONFIG" / "vbus_config.json"
LOG_PATH = VBUS_ROOT / "AUDIT" / "LOGS"

# Configure logging
LOG_PATH.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH / "vbus_daemon.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VBUS")


class CacheTier(Enum):
    L1_VRAM = 1
    L2_RAM = 2
    L3_DISK = 3


class CacheEntry:
    def __init__(self, key: str, size_bytes: int, tier: CacheTier):
        self.key = key
        self.size_bytes = size_bytes
        self.tier = tier
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0

    def touch(self):
        self.last_accessed = time.time()
        self.access_count += 1


class CircuitBreaker:
    """Implements circuit breaker pattern for fault tolerance"""

    def __init__(self, max_failures: int = 3, reset_timeout: int = 300):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN

    def record_failure(self):
        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= self.max_failures:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker OPEN after {{self.failures}} failures")

    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"

    def can_proceed(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.reset_timeout:
                self.state = "HALF-OPEN"
                return True
            return False
        return True  # HALF-OPEN


class CacheManager:
    """Manages the tiered cache hierarchy"""

    def __init__(self, config: Dict):
        self.config = config
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l2_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l2_data: Dict[str, Any] = {{}}
        self.l1_size = 0
        self.l2_size = 0
        self.l1_max = config["cache_hierarchy"]["L1_VRAM"]["max_size_mb"] * 1024 * 1024
        self.l2_max = config["cache_hierarchy"]["L2_RAM"]["max_size_mb"] * 1024 * 1024
        self.l3_path = Path(config["cache_hierarchy"]["L3_DISK"]["path"])
        self.lock = threading.RLock()

        self.stats = {{
            "l1_hits": 0, "l1_misses": 0,
            "l2_hits": 0, "l2_misses": 0,
            "l3_hits": 0, "l3_misses": 0,
            "evictions": 0, "promotions": 0
        }}

        logger.info(f"CacheManager initialized: L1={{self.l1_max//(1024*1024)}}MB, L2={{self.l2_max//(1024*1024)}}MB")

    def put(self, key: str, data: Any, size_bytes: int = 0) -> bool:
        if size_bytes == 0:
            size_bytes = sys.getsizeof(data)

        with self.lock:
            # Try L1 (VRAM simulation)
            if self.config["cache_hierarchy"]["L1_VRAM"]["enabled"]:
                if self.l1_size + size_bytes <= self.l1_max:
                    entry = CacheEntry(key, size_bytes, CacheTier.L1_VRAM)
                    self.l1_cache[key] = entry
                    self.l1_size += size_bytes
                    self.l2_data[key] = data
                    logger.debug(f"Cached {{key}} in L1 ({{size_bytes}} bytes)")
                    return True
                else:
                    self._evict_l1(size_bytes)

            # Try L2 (RAM)
            if self.l2_size + size_bytes <= self.l2_max:
                entry = CacheEntry(key, size_bytes, CacheTier.L2_RAM)
                self.l2_cache[key] = entry
                self.l2_size += size_bytes
                self.l2_data[key] = data
                logger.debug(f"Cached {{key}} in L2 ({{size_bytes}} bytes)")
                return True
            else:
                self._evict_l2(size_bytes)

            # Fallback to L3 (Disk)
            self._save_to_disk(key, data)
            return True

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            # Check L1
            if key in self.l1_cache:
                self.l1_cache[key].touch()
                self.l1_cache.move_to_end(key)
                self.stats["l1_hits"] += 1
                return self.l2_data.get(key)
            self.stats["l1_misses"] += 1

            # Check L2
            if key in self.l2_cache:
                self.l2_cache[key].touch()
                self.l2_cache.move_to_end(key)
                self.stats["l2_hits"] += 1
                return self.l2_data.get(key)
            self.stats["l2_misses"] += 1

            # Check L3 (Disk)
            data = self._load_from_disk(key)
            if data is not None:
                self.stats["l3_hits"] += 1
                # Promote to L2
                self.put(key, data, sys.getsizeof(data))
                self.stats["promotions"] += 1
                return data
            self.stats["l3_misses"] += 1

            return None

    def _evict_l1(self, required_bytes: int):
        while self.l1_size + required_bytes > self.l1_max and self.l1_cache:
            key, entry = self.l1_cache.popitem(last=False)
            self.l1_size -= entry.size_bytes
            # Demote to L2
            entry.tier = CacheTier.L2_RAM
            self.l2_cache[key] = entry
            self.l2_size += entry.size_bytes
            self.stats["evictions"] += 1

    def _evict_l2(self, required_bytes: int):
        while self.l2_size + required_bytes > self.l2_max and self.l2_cache:
            key, entry = self.l2_cache.popitem(last=False)
            self.l2_size -= entry.size_bytes
            if key in self.l2_data:
                self._save_to_disk(key, self.l2_data.pop(key))
            self.stats["evictions"] += 1

    def _save_to_disk(self, key: str, data: Any):
        try:
            import pickle
            file_path = self.l3_path / f"{{hashlib.md5(key.encode()).hexdigest()}}.cache"
            with open(file_path, 'wb') as f:
                pickle.dump({{"key": key, "data": data}}, f)
        except Exception as e:
            logger.error(f"Failed to save {{key}} to disk: {{e}}")

    def _load_from_disk(self, key: str) -> Optional[Any]:
        try:
            import pickle
            file_path = self.l3_path / f"{{hashlib.md5(key.encode()).hexdigest()}}.cache"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    if data.get("key") == key:
                        return data.get("data")
        except Exception as e:
            logger.error(f"Failed to load {{key}} from disk: {{e}}")
        return None

    def get_stats(self) -> Dict:
        total_l1 = self.stats["l1_hits"] + self.stats["l1_misses"]
        total_ops = total_l1 + self.stats["l2_hits"] + self.stats["l2_misses"]
        return {{
            **self.stats,
            "l1_hit_rate": self.stats["l1_hits"] / total_l1 * 100 if total_l1 > 0 else 0,
            "l1_size_mb": self.l1_size / (1024 * 1024),
            "l2_size_mb": self.l2_size / (1024 * 1024)
        }}


class BUSTrafficManager:
    """Manages traffic between master DB and repositories"""

    def __init__(self, config: Dict):
        self.config = config
        self.master_db = config["bus_config"]["master_db"]
        self.repositories = config["bus_config"]["repositories"]
        self.sync_interval = config["bus_config"]["sync_interval_seconds"]
        self.running = False
        self.sync_thread = None
        self.access_log: List[Dict] = []

        logger.info(f"BUS Traffic Manager initialized")
        logger.info(f"  Master DB: {{self.master_db}}")
        logger.info(f"  Repositories: {{len(self.repositories)}}")

    def start(self):
        self.running = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        logger.info("BUS Traffic Manager started")

    def stop(self):
        self.running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5)
        logger.info("BUS Traffic Manager stopped")

    def _sync_loop(self):
        while self.running:
            try:
                self._perform_sync()
            except Exception as e:
                logger.error(f"Sync error: {{e}}")
            time.sleep(self.sync_interval)

    def _perform_sync(self):
        logger.debug("Performing repository sync check...")
        # Check connectivity to master and repos
        if self.master_db and os.path.exists(self.master_db):
            logger.debug(f"Master DB accessible: {{self.master_db}}")

        for repo in self.repositories:
            if repo and os.path.exists(repo):
                logger.debug(f"Repository accessible: {{repo}}")

    def log_access(self, source: str, target: str, operation: str):
        self.access_log.append({{
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "target": target,
            "operation": operation
        }})
        # Keep last 1000 entries
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-500:]


class VBUSDaemon:
    """Main VBUS daemon controller"""

    def __init__(self):
        logger.info("="*60)
        logger.info("VBUS DAEMON STARTING")
        logger.info("="*60)

        # Load config
        self.config = self._load_config()

        # Initialize components
        self.circuit_breaker = CircuitBreaker(
            max_failures=self.config["circuit_breaker"]["max_failures"],
            reset_timeout=self.config["circuit_breaker"]["reset_timeout_seconds"]
        )
        self.cache_manager = CacheManager(self.config)
        self.bus_manager = BUSTrafficManager(self.config)

        self.running = False
        self.start_time = None

    def _load_config(self) -> Dict:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config not found: {{CONFIG_PATH}}")

        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)

        logger.info(f"Configuration loaded from {{CONFIG_PATH}}")
        return config

    def start(self):
        if not self.circuit_breaker.can_proceed():
            logger.error("CIRCUIT BREAKER OPEN - Cannot start")
            return False

        self.running = True
        self.start_time = datetime.now()

        # Start components
        self.bus_manager.start()

        logger.info("VBUS Daemon is now running")
        logger.info(f"  GPU Enabled: {{self.config['hardware']['gpu_detected']}}")
        logger.info(f"  L1 Cache: {{self.config['cache_hierarchy']['L1_VRAM']['max_size_mb']}}MB")
        logger.info(f"  L2 Cache: {{self.config['cache_hierarchy']['L2_RAM']['max_size_mb']}}MB")

        return True

    def stop(self):
        self.running = False
        self.bus_manager.stop()
        logger.info("VBUS Daemon stopped")

    def get_status(self) -> Dict:
        return {{
            "running": self.running,
            "uptime": str(datetime.now() - self.start_time) if self.start_time else "N/A",
            "circuit_breaker": self.circuit_breaker.state,
            "cache_stats": self.cache_manager.get_stats(),
            "bus_connected": len(self.bus_manager.repositories) > 0
        }}

    def run_forever(self):
        if not self.start():
            sys.exit(1)

        try:
            while self.running:
                time.sleep(5)
                # Periodic status log
                status = self.get_status()
                logger.debug(f"Status: CB={{status['circuit_breaker']}}, "
                           f"L1={{status['cache_stats']['l1_size_mb']:.1f}}MB, "
                           f"L2={{status['cache_stats']['l2_size_mb']:.1f}}MB")
        except KeyboardInterrupt:
            logger.info("Shutdown requested...")
        finally:
            self.stop()


def main():
    daemon = VBUSDaemon()
    daemon.run_forever()


if __name__ == "__main__":
    main()
'''

    @staticmethod
    def generate_launcher_bat(install_path: str) -> str:
        """Generate Windows batch launcher"""
        return f'''@echo off
title VBUS System
echo ============================================
echo   VBUS - Virtual Bus Cache System
echo ============================================
echo.
cd /d "{install_path}"
python SCRIPTS\\vbus_daemon.py
if errorlevel 1 (
    echo.
    echo [ERROR] VBUS failed to start
    pause
)
'''

    @staticmethod
    def generate_launcher_ps1(install_path: str) -> str:
        """Generate PowerShell launcher"""
        return f'''# VBUS PowerShell Launcher
$Host.UI.RawUI.WindowTitle = "VBUS System"
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VBUS - Virtual Bus Cache System" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "{install_path}"
python SCRIPTS\\vbus_daemon.py

if ($LASTEXITCODE -ne 0) {{
    Write-Host ""
    Write-Host "[ERROR] VBUS failed to start" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}}
'''


# =============================================================================
#  GUI APPLICATION (TKINTER)
# =============================================================================

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


class VBUSInstallerGUI:
    """Main installer GUI application"""

    # Color scheme - FRAMECORE Blue theme
    COLORS = {
        "bg": "#0a0a1a",
        "bg_secondary": "#12122a",
        "accent": "#1a1a40",
        "highlight": "#3939cc",  # FRAMECORE Blue
        "highlight_light": "#5555ee",
        "text": "#f0f0f0",
        "text_dim": "#8888aa",
        "success": "#00dd77",
        "warning": "#ffbb33",
        "error": "#ff4455",
        "framecore_blue": "#3333cc"
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.configure(bg=self.COLORS["bg"])
        self.root.resizable(True, True)

        # State
        self.current_step = 0
        self.system_report: Optional[Dict] = None
        self.install_path = tk.StringVar(value=str(Path.home() / "VBUS"))
        self.master_db = tk.StringVar()
        self.repos = [tk.StringVar() for _ in range(5)]

        # Configure styles
        self._setup_styles()

        # Build UI
        self._build_ui()

        # Show first step
        self._show_step(0)

    def _setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')

        # Frame styles
        style.configure("Main.TFrame", background=self.COLORS["bg"])
        style.configure("Card.TFrame", background=self.COLORS["bg_secondary"])

        # Label styles
        style.configure("Title.TLabel",
                       background=self.COLORS["bg"],
                       foreground=self.COLORS["highlight"],
                       font=("Segoe UI", 24, "bold"))
        style.configure("Subtitle.TLabel",
                       background=self.COLORS["bg"],
                       foreground=self.COLORS["text_dim"],
                       font=("Segoe UI", 11))
        style.configure("Heading.TLabel",
                       background=self.COLORS["bg_secondary"],
                       foreground=self.COLORS["highlight"],
                       font=("Segoe UI", 13, "bold"))
        style.configure("Body.TLabel",
                       background=self.COLORS["bg_secondary"],
                       foreground=self.COLORS["text"],
                       font=("Segoe UI", 10))
        style.configure("Success.TLabel",
                       background=self.COLORS["bg_secondary"],
                       foreground=self.COLORS["success"],
                       font=("Segoe UI", 10))
        style.configure("Error.TLabel",
                       background=self.COLORS["bg_secondary"],
                       foreground=self.COLORS["error"],
                       font=("Segoe UI", 10))
        style.configure("Warning.TLabel",
                       background=self.COLORS["bg_secondary"],
                       foreground=self.COLORS["warning"],
                       font=("Segoe UI", 10))

        # Button styles
        style.configure("Primary.TButton",
                       background=self.COLORS["framecore_blue"],
                       foreground="#ffffff",
                       font=("Segoe UI", 11, "bold"),
                       padding=(20, 10))
        style.map("Primary.TButton",
                 background=[("active", self.COLORS["highlight_light"])])

        style.configure("Secondary.TButton",
                       background=self.COLORS["accent"],
                       foreground=self.COLORS["text"],
                       font=("Segoe UI", 10),
                       padding=(15, 8))

        # Entry styles
        style.configure("Custom.TEntry",
                       fieldbackground=self.COLORS["accent"],
                       foreground=self.COLORS["text"],
                       insertcolor=self.COLORS["text"])

        # Progressbar
        style.configure("Custom.Horizontal.TProgressbar",
                       background=self.COLORS["highlight"],
                       troughcolor=self.COLORS["accent"])

    def _build_ui(self):
        """Build the main UI structure"""
        # Header with gradient-like effect
        self.header = tk.Frame(self.root, bg=self.COLORS["bg_secondary"], height=90)
        self.header.pack(fill="x")
        self.header.pack_propagate(False)

        # Logo/Brand area
        brand_frame = tk.Frame(self.header, bg=self.COLORS["bg_secondary"])
        brand_frame.pack(side="left", padx=30, pady=15)

        tk.Label(brand_frame, text="FRAMECORE", font=("Segoe UI", 11, "bold"),
                bg=self.COLORS["bg_secondary"], fg=self.COLORS["framecore_blue"]).pack(anchor="w")

        tk.Label(brand_frame, text="VBUS", font=("Segoe UI", 32, "bold"),
                bg=self.COLORS["bg_secondary"], fg=self.COLORS["highlight"]).pack(anchor="w")

        tk.Label(self.header, text=f"v{APP_VERSION} | ALGO 95.4", font=("Segoe UI", 9),
                bg=self.COLORS["bg_secondary"], fg=self.COLORS["text_dim"]).pack(side="left", pady=35)

        # Step indicators
        self.step_frame = tk.Frame(self.header, bg=self.COLORS["bg_secondary"])
        self.step_frame.pack(side="right", padx=30)

        self.step_labels = []
        steps = ["System Check", "Configure", "Install", "Complete"]
        for i, step in enumerate(steps):
            lbl = tk.Label(self.step_frame, text=f"{i+1}. {step}",
                          font=("Segoe UI", 9),
                          bg=self.COLORS["bg_secondary"],
                          fg=self.COLORS["text_dim"])
            lbl.pack(side="left", padx=10)
            self.step_labels.append(lbl)

        # Main content area
        self.content = tk.Frame(self.root, bg=self.COLORS["bg"])
        self.content.pack(fill="both", expand=True, padx=40, pady=30)

        # Footer with navigation
        self.footer = tk.Frame(self.root, bg=self.COLORS["bg_secondary"], height=70)
        self.footer.pack(fill="x", side="bottom")
        self.footer.pack_propagate(False)

        self.back_btn = ttk.Button(self.footer, text="← Back", style="Secondary.TButton",
                                   command=self._go_back)
        self.back_btn.pack(side="left", padx=30, pady=15)

        self.next_btn = ttk.Button(self.footer, text="Next →", style="Primary.TButton",
                                   command=self._go_next)
        self.next_btn.pack(side="right", padx=30, pady=15)

    def _update_step_indicators(self):
        """Update step indicator colors"""
        for i, lbl in enumerate(self.step_labels):
            if i < self.current_step:
                lbl.configure(fg=self.COLORS["success"])
            elif i == self.current_step:
                lbl.configure(fg=self.COLORS["highlight"], font=("Segoe UI", 9, "bold"))
            else:
                lbl.configure(fg=self.COLORS["text_dim"], font=("Segoe UI", 9))

    def _clear_content(self):
        """Clear the content area"""
        for widget in self.content.winfo_children():
            widget.destroy()

    def _show_step(self, step: int):
        """Show the specified step"""
        self.current_step = step
        self._clear_content()
        self._update_step_indicators()

        if step == 0:
            self._show_system_check()
        elif step == 1:
            self._show_configuration()
        elif step == 2:
            self._show_installation()
        elif step == 3:
            self._show_completion()

        # Update navigation buttons
        self.back_btn.configure(state="normal" if step > 0 and step < 3 else "disabled")

        if step == 2:
            self.next_btn.configure(text="Install", state="normal")
        elif step == 3:
            self.next_btn.configure(text="Close", state="normal")
        else:
            self.next_btn.configure(text="Next →", state="normal")

    def _go_back(self):
        if self.current_step > 0:
            self._show_step(self.current_step - 1)

    def _go_next(self):
        if self.current_step == 0:
            if self.system_report is None:
                messagebox.showwarning("Warning", "Please run the system scan first.")
                return
            self._show_step(1)
        elif self.current_step == 1:
            if not self.install_path.get():
                messagebox.showwarning("Warning", "Please specify an installation path.")
                return
            self._show_step(2)
        elif self.current_step == 2:
            self._start_installation()
        elif self.current_step == 3:
            self.root.destroy()

    # -------------------------------------------------------------------------
    #  STEP 0: System Check
    # -------------------------------------------------------------------------

    def _show_system_check(self):
        """Show system compatibility check"""
        # Title
        tk.Label(self.content, text="System Compatibility Check",
                font=("Segoe UI", 20, "bold"),
                bg=self.COLORS["bg"], fg=self.COLORS["highlight"]).pack(anchor="w")

        tk.Label(self.content, text="VBUS will analyze your system for GPU, RAM, and ML framework compatibility.",
                font=("Segoe UI", 11),
                bg=self.COLORS["bg"], fg=self.COLORS["text_dim"]).pack(anchor="w", pady=(5, 20))

        # Results area
        self.results_frame = tk.Frame(self.content, bg=self.COLORS["bg_secondary"])
        self.results_frame.pack(fill="both", expand=True)

        # Log text area
        self.log_text = tk.Text(self.results_frame, height=20,
                               bg=self.COLORS["accent"],
                               fg=self.COLORS["text"],
                               font=("Consolas", 10),
                               relief="flat",
                               insertbackground=self.COLORS["text"])
        self.log_text.pack(fill="both", expand=True, padx=20, pady=20)

        # Configure text tags
        self.log_text.tag_configure("success", foreground=self.COLORS["success"])
        self.log_text.tag_configure("error", foreground=self.COLORS["error"])
        self.log_text.tag_configure("warning", foreground=self.COLORS["warning"])
        self.log_text.tag_configure("highlight", foreground=self.COLORS["highlight"])

        # Scan button
        self.scan_btn = ttk.Button(self.content, text="Run System Scan", style="Primary.TButton",
                                   command=self._run_system_scan)
        self.scan_btn.pack(pady=20)

        # Auto-run scan if not done
        if self.system_report is None:
            self.root.after(500, self._run_system_scan)

    def _run_system_scan(self):
        """Run system scan in background thread"""
        self.scan_btn.configure(state="disabled")
        self.log_text.delete(1.0, tk.END)
        self._log("Starting system analysis...\n", "highlight")

        def scan_thread():
            analyzer = SystemAnalyzer()

            self._log("\n▸ Checking operating system...")
            self._log(f"  OS: {platform.system()} {platform.version()}")
            self._log(f"  Architecture: {platform.machine()}")
            self._log(f"  Python: {platform.python_version()}")

            time.sleep(0.3)

            self._log("\n▸ Checking GPU...")
            analyzer.check_gpu()
            if analyzer.report["gpu_found"]:
                self._log(f"  ✓ GPU Found: {analyzer.report['gpu_name']}", "success")
                self._log(f"    VRAM: {analyzer.report['vram_mb']} MB")
                self._log(f"    CUDA: {analyzer.report['cuda_version']}")
                self._log(f"    Driver: {analyzer.report['driver_version']}")
            else:
                self._log("  ✗ No NVIDIA GPU detected", "warning")
                self._log("    L1 VRAM cache will run in simulation mode", "warning")

            time.sleep(0.3)

            self._log("\n▸ Checking system memory...")
            analyzer.check_memory()
            self._log(f"  Total RAM: {analyzer.report['ram_total_gb']} GB")
            self._log(f"  Available: {analyzer.report['ram_available_gb']} GB")

            time.sleep(0.3)

            self._log("\n▸ Checking disk space...")
            analyzer.check_disk_space()
            self._log(f"  Free space: {analyzer.report['disk_free_gb']} GB")

            time.sleep(0.3)

            self._log("\n▸ Checking ML frameworks...")
            analyzer.check_ml_libs()
            for lib_name, info in analyzer.report["libs"].items():
                if info["installed"]:
                    ver = f" v{info['version']}" if info['version'] else ""
                    self._log(f"  ✓ {lib_name}{ver}", "success")
                else:
                    self._log(f"  ✗ {lib_name}: Not installed", "warning")

            time.sleep(0.3)

            analyzer.validate_compatibility()
            self.system_report = analyzer.report

            self._log("\n" + "="*50)
            if analyzer.report["is_compatible"]:
                self._log("✓ System is compatible with VBUS", "success")
            else:
                self._log("✗ System compatibility issues found:", "error")
                for issue in analyzer.report["issues"]:
                    self._log(f"  • {issue}", "error")

            if analyzer.report["warnings"]:
                self._log("\nWarnings:", "warning")
                for warn in analyzer.report["warnings"]:
                    self._log(f"  • {warn}", "warning")

            self.root.after(0, lambda: self.scan_btn.configure(state="normal", text="Rescan"))

        threading.Thread(target=scan_thread, daemon=True).start()

    def _log(self, message: str, tag: str = None):
        """Add message to log"""
        def update():
            self.log_text.insert(tk.END, message + "\n", tag)
            self.log_text.see(tk.END)
        self.root.after(0, update)

    # -------------------------------------------------------------------------
    #  STEP 1: Configuration
    # -------------------------------------------------------------------------

    def _show_configuration(self):
        """Show configuration page"""
        tk.Label(self.content, text="Installation Configuration",
                font=("Segoe UI", 20, "bold"),
                bg=self.COLORS["bg"], fg=self.COLORS["highlight"]).pack(anchor="w")

        tk.Label(self.content, text="Configure VBUS installation paths and repository locations.",
                font=("Segoe UI", 11),
                bg=self.COLORS["bg"], fg=self.COLORS["text_dim"]).pack(anchor="w", pady=(5, 20))

        # Scrollable frame
        canvas = tk.Canvas(self.content, bg=self.COLORS["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.content, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=self.COLORS["bg"])

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Installation Path
        self._create_path_input(scroll_frame, "Installation Directory",
                               "Where VBUS will be installed:",
                               self.install_path)

        # Master Database
        self._create_path_input(scroll_frame, "Master Database (BUS)",
                               "Path to your master database (optional):",
                               self.master_db)

        # Repositories
        tk.Label(scroll_frame, text="Repository Locations (BUS)",
                font=("Segoe UI", 13, "bold"),
                bg=self.COLORS["bg"], fg=self.COLORS["highlight"]).pack(anchor="w", pady=(20, 5))

        tk.Label(scroll_frame, text="Add paths to repositories for traffic management:",
                font=("Segoe UI", 10),
                bg=self.COLORS["bg"], fg=self.COLORS["text_dim"]).pack(anchor="w")

        for i, repo_var in enumerate(self.repos):
            frame = tk.Frame(scroll_frame, bg=self.COLORS["bg"])
            frame.pack(fill="x", pady=5)

            tk.Label(frame, text=f"Repo {i+1}:", width=8,
                    font=("Segoe UI", 10),
                    bg=self.COLORS["bg"], fg=self.COLORS["text"]).pack(side="left")

            entry = tk.Entry(frame, textvariable=repo_var,
                           font=("Segoe UI", 10),
                           bg=self.COLORS["accent"],
                           fg=self.COLORS["text"],
                           insertbackground=self.COLORS["text"],
                           relief="flat")
            entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

            ttk.Button(frame, text="Browse", style="Secondary.TButton",
                      command=lambda v=repo_var: self._browse_dir(v)).pack(side="right")

    def _create_path_input(self, parent, title: str, subtitle: str, variable: tk.StringVar):
        """Create a path input field"""
        tk.Label(parent, text=title,
                font=("Segoe UI", 13, "bold"),
                bg=self.COLORS["bg"], fg=self.COLORS["highlight"]).pack(anchor="w", pady=(20, 5))

        tk.Label(parent, text=subtitle,
                font=("Segoe UI", 10),
                bg=self.COLORS["bg"], fg=self.COLORS["text_dim"]).pack(anchor="w")

        frame = tk.Frame(parent, bg=self.COLORS["bg"])
        frame.pack(fill="x", pady=5)

        entry = tk.Entry(frame, textvariable=variable,
                        font=("Segoe UI", 10),
                        bg=self.COLORS["accent"],
                        fg=self.COLORS["text"],
                        insertbackground=self.COLORS["text"],
                        relief="flat")
        entry.pack(side="left", fill="x", expand=True, padx=(0, 10), ipady=8)

        ttk.Button(frame, text="Browse", style="Secondary.TButton",
                  command=lambda: self._browse_dir(variable)).pack(side="right")

    def _browse_dir(self, variable: tk.StringVar):
        """Open directory browser"""
        path = filedialog.askdirectory()
        if path:
            variable.set(path)

    # -------------------------------------------------------------------------
    #  STEP 2: Installation
    # -------------------------------------------------------------------------

    def _show_installation(self):
        """Show installation progress"""
        tk.Label(self.content, text="Installing VBUS",
                font=("Segoe UI", 20, "bold"),
                bg=self.COLORS["bg"], fg=self.COLORS["highlight"]).pack(anchor="w")

        tk.Label(self.content, text="Please wait while VBUS is being installed...",
                font=("Segoe UI", 11),
                bg=self.COLORS["bg"], fg=self.COLORS["text_dim"]).pack(anchor="w", pady=(5, 20))

        # Progress bar
        self.progress = ttk.Progressbar(self.content, length=400, mode='determinate',
                                        style="Custom.Horizontal.TProgressbar")
        self.progress.pack(pady=20)

        # Status label
        self.status_label = tk.Label(self.content, text="Preparing installation...",
                                     font=("Segoe UI", 11),
                                     bg=self.COLORS["bg"], fg=self.COLORS["text"])
        self.status_label.pack(pady=10)

        # Log area
        self.install_log = tk.Text(self.content, height=15,
                                   bg=self.COLORS["accent"],
                                   fg=self.COLORS["text"],
                                   font=("Consolas", 9),
                                   relief="flat")
        self.install_log.pack(fill="both", expand=True, pady=20)

        self.install_log.tag_configure("success", foreground=self.COLORS["success"])
        self.install_log.tag_configure("error", foreground=self.COLORS["error"])

        # Disable buttons during install
        self.back_btn.configure(state="disabled")
        self.next_btn.configure(state="disabled")

    def _start_installation(self):
        """Start the installation process"""
        threading.Thread(target=self._perform_installation, daemon=True).start()

    def _perform_installation(self):
        """Perform the actual installation"""
        install_path = self.install_path.get()
        master_db = self.master_db.get()
        repos = [r.get() for r in self.repos if r.get()]

        try:
            steps = [
                ("Creating directory structure...", 10),
                ("Generating configuration...", 30),
                ("Creating VBUS daemon...", 50),
                ("Creating launcher scripts...", 70),
                ("Finalizing installation...", 90),
                ("Complete!", 100)
            ]

            # Step 1: Create directories
            self._update_install_status("Creating directory structure...", 0)
            self._install_log("Creating installation directory...")

            Path(install_path).mkdir(parents=True, exist_ok=True)
            for subdir in VBUS_STRUCTURE:
                (Path(install_path) / subdir).mkdir(parents=True, exist_ok=True)
                self._install_log(f"  Created: {subdir}")

            time.sleep(0.3)

            # Step 2: Generate config
            self._update_install_status("Generating configuration...", 30)
            self._install_log("\nGenerating configuration...")

            config_content = VBUSCodeGenerator.generate_config(
                install_path, self.system_report, master_db, repos
            )
            config_path = Path(install_path) / "CONFIG" / "vbus_config.json"
            config_path.write_text(config_content)
            self._install_log(f"  Config saved: {config_path}")

            time.sleep(0.3)

            # Step 3: Generate daemon
            self._update_install_status("Creating VBUS daemon...", 50)
            self._install_log("\nCreating VBUS daemon script...")

            daemon_content = VBUSCodeGenerator.generate_daemon_script(install_path)
            daemon_path = Path(install_path) / "SCRIPTS" / "vbus_daemon.py"
            daemon_path.write_text(daemon_content)
            self._install_log(f"  Daemon saved: {daemon_path}")

            time.sleep(0.3)

            # Step 4: Create launchers
            self._update_install_status("Creating launcher scripts...", 70)
            self._install_log("\nCreating launcher scripts...")

            # Windows batch file
            bat_content = VBUSCodeGenerator.generate_launcher_bat(install_path)
            bat_path = Path(install_path) / "START_VBUS.bat"
            bat_path.write_text(bat_content)
            self._install_log(f"  Created: START_VBUS.bat")

            # PowerShell script
            ps1_content = VBUSCodeGenerator.generate_launcher_ps1(install_path)
            ps1_path = Path(install_path) / "START_VBUS.ps1"
            ps1_path.write_text(ps1_content)
            self._install_log(f"  Created: START_VBUS.ps1")

            time.sleep(0.3)

            # Step 5: Finalize
            self._update_install_status("Finalizing installation...", 90)
            self._install_log("\nFinalizing installation...")

            # Create installation log
            install_log = {
                "installed_at": datetime.now().isoformat(),
                "version": APP_VERSION,
                "install_path": install_path,
                "system_report": self.system_report,
                "master_db": master_db,
                "repositories": repos
            }
            log_path = Path(install_path) / "AUDIT" / "LOGS" / "install_log.json"
            log_path.write_text(json.dumps(install_log, indent=2))

            time.sleep(0.5)

            # Complete
            self._update_install_status("Installation complete!", 100)
            self._install_log("\n" + "="*50)
            self._install_log("✓ VBUS installed successfully!", "success")
            self._install_log(f"  Location: {install_path}")

            # Move to completion step
            self.root.after(1000, lambda: self._show_step(3))

        except Exception as e:
            self._install_log(f"\n✗ Installation failed: {str(e)}", "error")
            self.root.after(0, lambda: messagebox.showerror("Installation Failed", str(e)))
            self.root.after(0, lambda: self.next_btn.configure(state="normal", text="Retry"))

    def _update_install_status(self, message: str, progress: int):
        """Update installation status"""
        def update():
            self.status_label.configure(text=message)
            self.progress['value'] = progress
        self.root.after(0, update)

    def _install_log(self, message: str, tag: str = None):
        """Add to install log"""
        def update():
            self.install_log.insert(tk.END, message + "\n", tag)
            self.install_log.see(tk.END)
        self.root.after(0, update)

    # -------------------------------------------------------------------------
    #  STEP 3: Completion
    # -------------------------------------------------------------------------

    def _show_completion(self):
        """Show completion page"""
        # Success icon
        tk.Label(self.content, text="✓",
                font=("Segoe UI", 72),
                bg=self.COLORS["bg"], fg=self.COLORS["success"]).pack(pady=20)

        tk.Label(self.content, text="Installation Complete!",
                font=("Segoe UI", 24, "bold"),
                bg=self.COLORS["bg"], fg=self.COLORS["highlight"]).pack()

        tk.Label(self.content, text="VBUS has been successfully installed on your system.",
                font=("Segoe UI", 12),
                bg=self.COLORS["bg"], fg=self.COLORS["text"]).pack(pady=10)

        # Summary frame
        summary_frame = tk.Frame(self.content, bg=self.COLORS["bg_secondary"])
        summary_frame.pack(fill="x", pady=30, padx=50)

        install_path = self.install_path.get()

        summary_text = f"""
Installation Path: {install_path}

To start VBUS, run one of:
  • START_VBUS.bat (Windows)
  • START_VBUS.ps1 (PowerShell)
  • python SCRIPTS/vbus_daemon.py

Cache Configuration:
  • L1 (VRAM): {self.system_report['vram_mb']} MB {'(GPU detected)' if self.system_report['gpu_found'] else '(Simulation mode)'}
  • L2 (RAM): {int(self.system_report['ram_total_gb'] * 1024 * 0.3)} MB
  • L3 (Disk): 10 GB

BUS Configuration:
  • Master DB: {self.master_db.get() or 'Not configured'}
  • Repositories: {len([r for r in self.repos if r.get()])} configured
"""

        tk.Label(summary_frame, text=summary_text,
                font=("Consolas", 10),
                bg=self.COLORS["bg_secondary"],
                fg=self.COLORS["text"],
                justify="left").pack(padx=20, pady=20)

        # Launch button
        def launch_vbus():
            bat_path = Path(install_path) / "START_VBUS.bat"
            if bat_path.exists():
                os.startfile(str(bat_path))

        ttk.Button(self.content, text="Launch VBUS", style="Primary.TButton",
                  command=launch_vbus).pack(pady=10)


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    if not GUI_AVAILABLE:
        print("Error: tkinter is required for the GUI installer")
        print("Please install tkinter or run in console mode")
        sys.exit(1)

    root = tk.Tk()

    # Set icon if available
    try:
        # For compiled exe, icon would be embedded
        pass
    except:
        pass

    app = VBUSInstallerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
