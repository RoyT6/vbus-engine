# VBUS - Virtual Bus Cache & Traffic Management System

VBUS is a standalone system that provides two core capabilities:

## V (VRAM Cache Management)
- Turns GPU VRAM into L1 cache for ML operations
- Uses system RAM as L2/L3 cache layers
- Intelligently manages throughput during ML training/inference
- Supports CUDA, cuDF, cuML, RAPIDS, XGBoost, Random Forest, CatBoost

## BUS (Repository Traffic Management)
- Manages traffic between master database and repositories
- Uses AI inferencing to understand your system hierarchy
- Creates and maintains cache coherency across sources

---

## Quick Start

### Option 1: Run the Installer (GUI)

```bash
python vbus_installer.py
```

### Option 2: Build Standalone EXE

First, ensure PyInstaller is installed:
```bash
pip install pyinstaller
```

Then build the executable:
```bash
python build_exe.py
```

Or for a quick build:
```bash
python build_exe.py --quick
```

This creates `VBUS_Installer.exe` that can be distributed and run on any Windows machine.

---

## System Requirements

### Minimum
- Windows 10/11 (64-bit)
- 8 GB RAM
- 10 GB free disk space
- Python 3.8+ (for development/build only)

### Recommended
- NVIDIA GPU with CUDA support
- 16+ GB RAM
- 50+ GB free disk space
- CUDA Toolkit installed

### ML Framework Support
VBUS can detect and integrate with:
- CUDA Toolkit
- cuDF (GPU DataFrames)
- cuML (GPU ML algorithms)
- RAPIDS Suite
- XGBoost
- Scikit-learn (Random Forest)
- CatBoost
- PyTorch

---

## Installation Features

The VBUS installer will:
1. Scan your system for compatibility
2. Detect GPU and VRAM availability
3. Check for installed ML frameworks
4. Create the VBUS directory structure
5. Generate configuration based on your system
6. Create the VBUS daemon and launcher scripts

---

## Directory Structure

After installation, VBUS creates:

```
VBUS/
├── BUS/
│   ├── CONTROL_PLANE/    # BUS control logic
│   ├── INTERCONNECT/     # Inter-repo communication
│   └── AI_ROUTER/        # AI-based routing
├── CACHE/
│   ├── L1_VRAM/          # GPU VRAM cache (fastest)
│   ├── L2_RAM/           # System RAM cache (fast)
│   └── L3_DISK/          # Disk cache (persistent)
├── AUDIT/
│   ├── LOGS/             # System logs
│   └── METRICS/          # Performance metrics
├── DATA/
│   ├── MASTER_DB/        # Master database link
│   └── REPOS/            # Repository links
├── CONFIG/
│   └── vbus_config.json  # Main configuration
├── SCRIPTS/
│   └── vbus_daemon.py    # VBUS daemon script
├── START_VBUS.bat        # Windows launcher
└── START_VBUS.ps1        # PowerShell launcher
```

---

## Configuration

### Cache Configuration (vbus_config.json)

```json
{
  "cache_hierarchy": {
    "L1_VRAM": {
      "enabled": true,
      "max_size_mb": 4096,
      "eviction_policy": "lru"
    },
    "L2_RAM": {
      "enabled": true,
      "max_size_mb": 8192,
      "eviction_policy": "lru"
    },
    "L3_DISK": {
      "enabled": true,
      "max_size_mb": 10240,
      "eviction_policy": "lru"
    }
  }
}
```

### BUS Configuration

```json
{
  "bus_config": {
    "master_db": "/path/to/master/database",
    "repositories": [
      "/path/to/repo1",
      "/path/to/repo2"
    ],
    "ai_routing_enabled": true,
    "coherency_protocol": "mesi",
    "sync_interval_seconds": 60
  }
}
```

---

## Running VBUS

After installation, start VBUS using:

**Windows (batch file):**
```bash
START_VBUS.bat
```

**PowerShell:**
```powershell
.\START_VBUS.ps1
```

**Direct Python:**
```bash
python SCRIPTS/vbus_daemon.py
```

---

## Cache Hierarchy

VBUS implements a three-tier cache hierarchy:

| Tier | Location | Speed | Purpose |
|------|----------|-------|---------|
| L1 | GPU VRAM | Fastest | Hot data for ML operations |
| L2 | System RAM | Fast | Warm data, frequently accessed |
| L3 | Disk | Slower | Cold data, persistence |

Data automatically promotes/demotes between tiers based on access patterns using LRU (Least Recently Used) eviction policy.

---

## API Usage (For Developers)

```python
from vbus import CacheManager, BUSTrafficManager

# Initialize cache
cache = CacheManager(config)

# Store data
cache.put("model_weights", weights_tensor)

# Retrieve data (automatically checks L1 → L2 → L3)
data = cache.get("model_weights")

# Get cache statistics
stats = cache.get_stats()
print(f"L1 Hit Rate: {stats['l1_hit_rate']}%")
```

---

## Troubleshooting

### No GPU Detected
- Ensure NVIDIA drivers are installed
- Verify `nvidia-smi` command works
- VBUS will run in simulation mode without GPU

### Installation Fails
- Run as Administrator
- Check disk space (minimum 10GB)
- Ensure Python 3.8+ is installed

### Daemon Won't Start
- Check `AUDIT/LOGS/vbus_daemon.log` for errors
- Verify `CONFIG/vbus_config.json` exists
- Ensure Python is in PATH

---

## License

MIT License - See LICENSE file for details.

---

## Support

For issues and feature requests, please open an issue on the project repository.
