"""
VBUS System Information Module
Collects comprehensive system information for compatibility checking
"""

import platform
import subprocess
import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class GPUInfo:
    """GPU Information container"""
    name: str = ""
    vendor: str = ""
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    driver_version: str = ""
    cuda_version: str = ""
    compute_capability: str = ""
    is_nvidia: bool = False
    device_id: int = 0


@dataclass
class SystemRequirements:
    """Minimum system requirements for VBUS"""
    min_ram_gb: int = 16
    min_vram_gb: int = 4
    min_disk_space_gb: int = 50
    required_os: List[str] = field(default_factory=lambda: ["Windows 10", "Windows 11"])
    min_cuda_version: str = "11.0"
    supported_gpu_vendors: List[str] = field(default_factory=lambda: ["NVIDIA"])


@dataclass
class SystemInfo:
    """Complete system information"""
    os_name: str = ""
    os_version: str = ""
    os_architecture: str = ""
    cpu_name: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    disk_space_gb: float = 0.0
    gpus: List[GPUInfo] = field(default_factory=list)
    python_version: str = ""
    is_admin: bool = False


class SystemInfoCollector:
    """Collects system information for VBUS compatibility checking"""

    def __init__(self):
        self.system_info = SystemInfo()
        self.requirements = SystemRequirements()

    def collect_all(self) -> SystemInfo:
        """Collect all system information"""
        self._collect_os_info()
        self._collect_cpu_info()
        self._collect_memory_info()
        self._collect_disk_info()
        self._collect_gpu_info()
        self._collect_python_info()
        self._check_admin_privileges()
        return self.system_info

    def _collect_os_info(self):
        """Collect operating system information"""
        self.system_info.os_name = platform.system()
        self.system_info.os_version = platform.version()
        self.system_info.os_architecture = platform.machine()

    def _collect_cpu_info(self):
        """Collect CPU information"""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            self.system_info.cpu_name = info.get('brand_raw', 'Unknown')
            self.system_info.cpu_cores = os.cpu_count() or 0
            self.system_info.cpu_threads = info.get('count', os.cpu_count() or 0)
        except ImportError:
            self.system_info.cpu_name = platform.processor()
            self.system_info.cpu_cores = os.cpu_count() or 0
            self.system_info.cpu_threads = os.cpu_count() or 0

    def _collect_memory_info(self):
        """Collect RAM information"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            self.system_info.ram_total_gb = round(mem.total / (1024**3), 2)
            self.system_info.ram_available_gb = round(mem.available / (1024**3), 2)
        except ImportError:
            pass

    def _collect_disk_info(self):
        """Collect disk space information"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            self.system_info.disk_space_gb = round(disk.free / (1024**3), 2)
        except (ImportError, Exception):
            pass

    def _collect_gpu_info(self):
        """Collect GPU information using multiple methods"""
        gpus = []

        # Try NVIDIA SMI first
        gpus = self._collect_nvidia_gpu_info()

        # Fallback to WMI on Windows
        if not gpus and platform.system() == "Windows":
            gpus = self._collect_wmi_gpu_info()

        self.system_info.gpus = gpus

    def _collect_nvidia_gpu_info(self) -> List[GPUInfo]:
        """Collect NVIDIA GPU info using nvidia-smi"""
        gpus = []
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version,compute_cap',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                for idx, line in enumerate(result.stdout.strip().split('\n')):
                    if line:
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 5:
                            gpu = GPUInfo(
                                name=parts[0],
                                vendor="NVIDIA",
                                vram_total_mb=int(float(parts[1])),
                                vram_free_mb=int(float(parts[2])),
                                driver_version=parts[3],
                                compute_capability=parts[4],
                                is_nvidia=True,
                                device_id=idx
                            )
                            gpus.append(gpu)

            # Get CUDA version
            cuda_result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=10
            )

            # Also try nvcc for CUDA toolkit version
            try:
                nvcc_result = subprocess.run(
                    ['nvcc', '--version'], capture_output=True, text=True, timeout=10
                )
                if nvcc_result.returncode == 0:
                    for line in nvcc_result.stdout.split('\n'):
                        if 'release' in line.lower():
                            # Extract version like "release 12.0"
                            parts = line.split('release')
                            if len(parts) > 1:
                                cuda_ver = parts[1].strip().split(',')[0].strip()
                                for gpu in gpus:
                                    gpu.cuda_version = cuda_ver
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

        return gpus

    def _collect_wmi_gpu_info(self) -> List[GPUInfo]:
        """Collect GPU info using WMI on Windows"""
        gpus = []
        try:
            import wmi
            c = wmi.WMI()
            for idx, gpu in enumerate(c.Win32_VideoController()):
                gpu_info = GPUInfo(
                    name=gpu.Name or "Unknown",
                    vendor=gpu.AdapterCompatibility or "Unknown",
                    vram_total_mb=int((gpu.AdapterRAM or 0) / (1024**2)),
                    driver_version=gpu.DriverVersion or "",
                    is_nvidia="nvidia" in (gpu.Name or "").lower(),
                    device_id=idx
                )
                gpus.append(gpu_info)
        except (ImportError, Exception):
            pass

        return gpus

    def _collect_python_info(self):
        """Collect Python version information"""
        self.system_info.python_version = platform.python_version()

    def _check_admin_privileges(self):
        """Check if running with administrator privileges"""
        try:
            if platform.system() == "Windows":
                import ctypes
                self.system_info.is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                self.system_info.is_admin = os.geteuid() == 0
        except (AttributeError, Exception):
            self.system_info.is_admin = False

    def check_compatibility(self) -> Dict[str, Any]:
        """Check system compatibility with VBUS requirements"""
        issues = []
        warnings = []
        passed = []

        # Check OS
        if self.system_info.os_name == "Windows":
            passed.append(f"Operating System: {self.system_info.os_name}")
        else:
            issues.append(f"Unsupported OS: {self.system_info.os_name}. VBUS requires Windows 10/11")

        # Check RAM
        if self.system_info.ram_total_gb >= self.requirements.min_ram_gb:
            passed.append(f"RAM: {self.system_info.ram_total_gb} GB (minimum: {self.requirements.min_ram_gb} GB)")
        else:
            issues.append(f"Insufficient RAM: {self.system_info.ram_total_gb} GB (minimum: {self.requirements.min_ram_gb} GB)")

        # Check GPU
        nvidia_gpus = [g for g in self.system_info.gpus if g.is_nvidia]
        if nvidia_gpus:
            for gpu in nvidia_gpus:
                vram_gb = gpu.vram_total_mb / 1024
                if vram_gb >= self.requirements.min_vram_gb:
                    passed.append(f"GPU: {gpu.name} with {vram_gb:.1f} GB VRAM")
                else:
                    warnings.append(f"GPU {gpu.name} has only {vram_gb:.1f} GB VRAM (recommended: {self.requirements.min_vram_gb} GB)")
        else:
            issues.append("No NVIDIA GPU detected. VBUS requires an NVIDIA GPU with CUDA support")

        # Check disk space
        if self.system_info.disk_space_gb >= self.requirements.min_disk_space_gb:
            passed.append(f"Disk Space: {self.system_info.disk_space_gb:.1f} GB free")
        else:
            warnings.append(f"Low disk space: {self.system_info.disk_space_gb:.1f} GB (recommended: {self.requirements.min_disk_space_gb} GB)")

        # Check admin privileges
        if self.system_info.is_admin:
            passed.append("Administrator privileges: Yes")
        else:
            warnings.append("Not running as administrator. Some features may require elevation")

        is_compatible = len(issues) == 0

        return {
            "compatible": is_compatible,
            "issues": issues,
            "warnings": warnings,
            "passed": passed,
            "system_info": self.system_info
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert system info to dictionary"""
        return {
            "os": {
                "name": self.system_info.os_name,
                "version": self.system_info.os_version,
                "architecture": self.system_info.os_architecture
            },
            "cpu": {
                "name": self.system_info.cpu_name,
                "cores": self.system_info.cpu_cores,
                "threads": self.system_info.cpu_threads
            },
            "memory": {
                "total_gb": self.system_info.ram_total_gb,
                "available_gb": self.system_info.ram_available_gb
            },
            "disk": {
                "free_gb": self.system_info.disk_space_gb
            },
            "gpus": [
                {
                    "name": g.name,
                    "vendor": g.vendor,
                    "vram_total_mb": g.vram_total_mb,
                    "vram_free_mb": g.vram_free_mb,
                    "driver_version": g.driver_version,
                    "cuda_version": g.cuda_version,
                    "compute_capability": g.compute_capability,
                    "is_nvidia": g.is_nvidia
                }
                for g in self.system_info.gpus
            ],
            "python_version": self.system_info.python_version,
            "is_admin": self.system_info.is_admin
        }


if __name__ == "__main__":
    collector = SystemInfoCollector()
    info = collector.collect_all()
    print(json.dumps(collector.to_dict(), indent=2))

    print("\n--- Compatibility Check ---")
    result = collector.check_compatibility()
    print(f"Compatible: {result['compatible']}")
    print("\nPassed:")
    for p in result['passed']:
        print(f"  ✓ {p}")
    print("\nWarnings:")
    for w in result['warnings']:
        print(f"  ⚠ {w}")
    print("\nIssues:")
    for i in result['issues']:
        print(f"  ✗ {i}")
