# VBUS - ViewerDBX Bus System
## Session Memory | ALGO 95.66 Compliance | SCHEMA V27.00 | BFD V27.66

---

## P0 CANON: Studio vs Production Company Classification (2026-01-25)

**CRITICAL RULE:** Classification is by PRIMARY FUNCTION, NOT by fixed lookup list.

### Studios (Creative/On-Set)
| Roles | Activities |
|-------|------------|
| Directors, Writers | Filming, Blocking |
| Cinematographers, Cameramen | Location, Direction |
| Set Designers, Crew | Content Creation, Capture |
| Actors, Casting Directors | Casting, Extras |

### Production Companies (Business/Post)
| Roles | Activities |
|-------|------------|
| Lawyers, Finance | Financing, Legal |
| Post-Production, Final Cut | Editing, Mixing |
| Producers, Editors | Distribution, Marketing |
| Music, Sound, Executives | Promotion, Post-Production |

**Ingestion Rules:**
- P0: Classify by FUNCTION, not name matching
- P1: Accept ANY entity meeting criteria (no lookup required)
- P2: Metadata columns OPTIONAL (NULL acceptable)

**Rule File:** `Schema/RULES/STUDIO_VS_PRODUCTION_COMPANY_RULES.json`

---

## SESSION LOG (2026-01-22)

### Accomplished This Session:

1. **Created Professional Windows Installer (install.exe)**
   - Full 8-page installation wizard for Framecore employees
   - Built with tkinter GUI, compiled with PyInstaller
   - Output: `release/install.exe` (17.4 MB)

2. **Installer Pages:**
   - Page 0: Welcome - Overview of VBUS for Cloudflare
   - Page 1: License Agreement - Framecore Inc terms
   - Page 2: System Requirements - Python detection with:
     - Automatic Python/pip detection
     - Version display if found (green status)
     - Warning if missing (red status with instructions)
     - Download link: https://www.python.org/downloads/
     - Recommended version: Python 3.12.x
     - "Add Python to PATH" reminder
     - Recheck Python button
   - Page 3: Installation Location - Path selection with Browse
   - Page 4: API Configuration - Collects credentials:
     - Cloudflare R2 Access Key ID (required)
     - Cloudflare R2 Secret Access Key (required)
     - TMDB API Key (optional)
   - Page 5: Review Installation - Folder tree preview
   - Page 6: Installing - Actual pip install with progress
   - Page 7: Complete - Summary with open folder option

3. **Dependencies Actually Installed via pip:**
   - pyarrow (data file handling)
   - boto3 (Cloudflare R2 connection)
   - requests (HTTP client)
   - pandas (data processing)
   - cudf-cu12 (GPU acceleration - optional)

4. **Folders Created by Installer:**
   - Schema/ - Schema definitions
   - Components/ - VBUS component modules
   - GPU Enablement/ - GPU config and scripts
   - AUDIT_LOGS/ - System audit logs
   - cache/ - Temporary data cache
   - credentials/ - Secure credential storage

5. **Files Installed:**
   - vbus_core.py, vbus_engines.py, vbus_pipeline.py
   - vbus_monitor.py, vbus_gpu_manager.py, vbus_ml_executor.py
   - VBUS.json, SCHEMA.json (V26.00)
   - GPU config files (.circuit_breaker.json, gpu_config.json)
   - cloudflare_config.json, credentials.json
   - launch_vbus.bat

---

## KEY FILES

| File | Purpose |
|------|---------|
| `install.py` | Main installer source (854 lines) |
| `release/install.exe` | Compiled installer executable |
| `installer_venv/` | PyInstaller build environment |
| `SCHEMA.json` | Schema V26.00 (copied from Schema folder) |

---

## BUILD COMMANDS

```bash
# Activate virtual environment and build
cd "C:/Users/RoyT6/Downloads/VBUS"
installer_venv/Scripts/pyinstaller --noconfirm --onefile --windowed \
  --name "install" \
  --add-data "VBUS.json;." \
  --add-data "vbus_core.py;." \
  --add-data "vbus_engines.py;." \
  --add-data "vbus_pipeline.py;." \
  --add-data "vbus_monitor.py;." \
  --add-data "vbus_gpu_manager.py;." \
  --add-data "vbus_ml_executor.py;." \
  --add-data "VBus System.md;." \
  --add-data "Framecore_Blue 2.png;." \
  --add-data "SCHEMA.json;." \
  --add-data "gpu_config.json;." \
  --add-data ".circuit_breaker.json;." \
  --add-data "GPU_VERIFY_QUICK.py;." \
  --add-data "run_gpu.sh;." \
  --hidden-import=PIL \
  --hidden-import=PIL.Image \
  --hidden-import=PIL.ImageTk \
  install.py

# Copy to release folder
cp dist/install.exe release/install.exe
```

---

## TECHNICAL DECISIONS

1. **tkinter Pack Order**: Button bar must be packed BEFORE expanding content frame
2. **PyInstaller Virtual Env**: Required to avoid system "typing package obsolete" error
3. **Python Check**: Installer detects Python/pip before proceeding, shows instructions if missing
4. **Actual pip Install**: Dependencies installed via subprocess pip, not just file copy
5. **Credential Storage**: Saved to credentials/credentials.json in install folder

---

## CLOUDFLARE R2 CONFIGURATION

```python
CLOUDFLARE_DEFAULTS = {
    "r2_endpoint": "https://410d8a36762e8080b6e63e29382c460b.r2.cloudflarestorage.com",
    "bucket": "viewerdbx-backup",
}
```

---

## CONSTRAINTS (ALGO 95.4)

- GPU execution is MANDATORY for VBUS operations
- CPU fallback is FORBIDDEN
- All scripts must use cuDF for parquet I/O
- RTX 3080 Ti (12GB VRAM) available via WSL

---

**Last Updated**: 2026-01-24
**Installer Version**: 1.0.0
**Schema Version**: V28.00

---

## V28.00 Star Hierarchy (2026-01-24)

SCHEMA.json updated to V28.00 with star hierarchy columns:
- `star_1`, `star_2`, `star_3` - Top 3 billed actors
- `supporting_cast` - Remaining cast
- `cast_data` - DEPRECATED
