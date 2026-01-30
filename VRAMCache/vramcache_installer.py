#!/usr/bin/env python3
"""
VBUS Standalone Installer
=========================
GPU VRAM as L1 Cache for ML Workloads

This installer:
1. Detects GPU (NVIDIA/AMD)
2. Asks for master database location
3. Asks for up to 3 contributor repository paths
4. Installs ALL dependencies (cuDF, cuML, XGBoost, CatBoost, etc.)
5. Configures the VBus system
6. Creates launchers

Copyright (c) 2026 Framecore Inc. All rights reserved.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import shutil
import threading
import json
import subprocess
import platform
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_NAME = "VBUS"
APP_VERSION = "1.0.0"
COMPANY_NAME = "Framecore Inc"
COPYRIGHT_YEAR = "2026"

# Default paths
DEFAULT_INSTALL_PATH = os.path.join(os.path.expanduser("~"), "Framecore", "VBUS")
DEFAULT_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".vbus")

# Required Python packages - ACTUALLY INSTALLED
CORE_PACKAGES = [
    ("numpy", "1.24.0", "Numerical computing"),
    ("pandas", "2.0.0", "Data processing"),
    ("pyarrow", "14.0.0", "Parquet file support"),
]

# GPU packages (NVIDIA CUDA)
GPU_PACKAGES = [
    ("cupy-cuda12x", "13.0.0", "GPU array operations"),
]

# RAPIDS packages (require special installation)
RAPIDS_PACKAGES = [
    ("cudf-cu12", "24.02", "GPU DataFrames"),
    ("cuml-cu12", "24.02", "GPU Machine Learning"),
]

# ML Framework packages - ACTUALLY INSTALLED
ML_PACKAGES = [
    ("xgboost", "2.0.0", "Gradient boosting (GPU)"),
    ("catboost", "1.2.0", "Gradient boosting (GPU)"),
    ("lightgbm", "4.0.0", "Light gradient boosting"),
    ("scikit-learn", "1.3.0", "Machine learning utilities"),
]

# Colors
HEADER_BG = "#1a1a2e"
HEADER_FG = "#ffffff"
CONTENT_BG = "#f0f0f0"
TEXT_COLOR = "#333333"
ACCENT_COLOR = "#4338FF"
SUCCESS_COLOR = "#28a745"
WARNING_COLOR = "#ffc107"
ERROR_COLOR = "#dc3545"
BUTTON_BAR_BG = "#e0e0e0"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def resource_path(relative_path):
    """Get path to resource, works for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def detect_gpu():
    """Detect GPU vendor and info"""
    gpu_info = {
        "vendor": "none",
        "name": "Not detected",
        "vram_mb": 0,
        "cuda_available": False,
        "rocm_available": False
    }

    # Try NVIDIA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                parts = lines[0].split(',')
                gpu_info["vendor"] = "nvidia"
                gpu_info["name"] = parts[0].strip()
                gpu_info["vram_mb"] = int(parts[1].strip()) if len(parts) > 1 else 0
                gpu_info["cuda_available"] = True
                return gpu_info
    except:
        pass

    # Try AMD
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and "GPU" in result.stdout:
            gpu_info["vendor"] = "amd"
            gpu_info["name"] = "AMD GPU"
            gpu_info["rocm_available"] = True
            return gpu_info
    except:
        pass

    return gpu_info


def check_python():
    """Check Python and pip installation"""
    info = {"installed": False, "version": None, "pip": False, "path": None}

    try:
        result = subprocess.run(
            [sys.executable, "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            info["installed"] = True
            info["version"] = result.stdout.strip()
            info["path"] = sys.executable
    except:
        pass

    if info["installed"]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True, text=True, timeout=10
            )
            info["pip"] = result.returncode == 0
        except:
            pass

    return info


def is_package_installed(package_name):
    """Check if a Python package is installed"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name.split('[')[0]],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    except:
        return False


def pip_install(package, timeout=600):
    """Install a package via pip"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "--quiet", "--no-cache-dir"],
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Installation timed out"
    except Exception as e:
        return False, str(e)


# ============================================================================
# INSTALLER APPLICATION
# ============================================================================

class VBUSInstaller(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(f"{APP_NAME} Setup - GPU VRAM Cache for ML")
        self.geometry("800x600")
        self.resizable(False, False)
        self.configure(bg=CONTENT_BG)
        self.center_window()

        # Variables
        self.install_path_var = tk.StringVar(value=DEFAULT_INSTALL_PATH)
        self.master_path_var = tk.StringVar()
        self.master_name_var = tk.StringVar(value="MasterDB")
        self.contrib1_path_var = tk.StringVar()
        self.contrib1_name_var = tk.StringVar()
        self.contrib2_path_var = tk.StringVar()
        self.contrib2_name_var = tk.StringVar()
        self.contrib3_path_var = tk.StringVar()
        self.contrib3_name_var = tk.StringVar()
        self.install_rapids_var = tk.BooleanVar(value=True)
        self.install_ml_var = tk.BooleanVar(value=True)
        self.open_folder_var = tk.BooleanVar(value=False)

        # System info
        self.python_info = check_python()
        self.gpu_info = detect_gpu()

        # Pages
        self.pages = []
        self.current_page = 0

        # Build pages
        self.build_welcome_page()          # 0
        self.build_gpu_detection_page()    # 1
        self.build_master_db_page()        # 2
        self.build_contributors_page()     # 3
        self.build_packages_page()         # 4
        self.build_review_page()           # 5
        self.build_progress_page()         # 6
        self.build_complete_page()         # 7

        self.show_page(0)

    def center_window(self):
        self.update_idletasks()
        w, h = 800, 600
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    def show_page(self, idx):
        for p in self.pages:
            p.pack_forget()
        self.pages[idx].pack(fill=tk.BOTH, expand=True)
        self.current_page = idx

    def create_header(self, parent, title, subtitle=None):
        header = tk.Frame(parent, bg=HEADER_BG, height=70)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        tk.Label(header, text=title, font=("Segoe UI", 16, "bold"),
                fg=HEADER_FG, bg=HEADER_BG).pack(side=tk.LEFT, padx=25, pady=15)
        if subtitle:
            tk.Label(header, text=subtitle, font=("Segoe UI", 10),
                    fg="#aaaaaa", bg=HEADER_BG).pack(side=tk.RIGHT, padx=25, pady=20)
        return header

    def create_button_bar(self, parent, buttons):
        btn_bar = tk.Frame(parent, bg=BUTTON_BAR_BG, height=55)
        btn_bar.pack(fill=tk.X, side=tk.BOTTOM)
        btn_bar.pack_propagate(False)
        tk.Frame(parent, bg="#cccccc", height=1).pack(fill=tk.X, side=tk.BOTTOM)

        btn_frame = tk.Frame(btn_bar, bg=BUTTON_BAR_BG)
        btn_frame.pack(side=tk.RIGHT, padx=15, pady=10)

        for text, cmd in buttons:
            ttk.Button(btn_frame, text=text, command=cmd, width=12).pack(side=tk.LEFT, padx=5)

        return btn_bar

    # =========================================================================
    # PAGE 0: WELCOME
    # =========================================================================

    def build_welcome_page(self):
        page = tk.Frame(self, bg=CONTENT_BG)

        self.create_header(page, "Welcome to VBUS Setup")
        self.create_button_bar(page, [
            ("Cancel", self.on_cancel),
            ("Next >", lambda: self.show_page(1))
        ])

        content = tk.Frame(page, bg=CONTENT_BG, padx=40, pady=30)
        content.pack(fill=tk.BOTH, expand=True)

        # Logo/Title
        tk.Label(content, text="VBUS", font=("Segoe UI", 32, "bold"),
                fg=ACCENT_COLOR, bg=CONTENT_BG).pack(pady=15)
        tk.Label(content, text="GPU VRAM as L1 Cache for ML Workloads",
                font=("Segoe UI", 12), fg=TEXT_COLOR, bg=CONTENT_BG).pack()
        tk.Label(content, text=f"Version {APP_VERSION}",
                font=("Segoe UI", 10), fg="#888888", bg=CONTENT_BG).pack(pady=5)

        # Info box
        info_frame = tk.Frame(content, bg="white", relief=tk.GROOVE, bd=1)
        info_frame.pack(fill=tk.X, pady=25, padx=20)

        info_text = """
This wizard will install VBUS with:

   GPU Memory Hierarchy:
     L1 (VRAM)  - Hot data on GPU for fast ML training
     L2 (RAM)   - Warm data ready for GPU transfer
     L3 (Disk)  - Cold data storage

   Data Bus (VBus):
     Master Database    - Your primary data source
     Contributors (3)   - Supplementary data repositories

   ML Frameworks (GPU-accelerated):
     XGBoost, CatBoost, cuML Random Forest, LightGBM

   RAPIDS Ecosystem:
     cuDF (GPU DataFrames), cuML (GPU ML), CuPy (GPU Arrays)
"""
        tk.Label(info_frame, text=info_text, font=("Consolas", 9),
                fg=TEXT_COLOR, bg="white", justify=tk.LEFT).pack(padx=15, pady=15)

        tk.Frame(content, bg=CONTENT_BG).pack(expand=True)
        tk.Label(content, text=f"Copyright (c) {COPYRIGHT_YEAR} {COMPANY_NAME}. All rights reserved.",
                font=("Segoe UI", 8), fg="#999999", bg=CONTENT_BG).pack(side=tk.BOTTOM)

        self.pages.append(page)

    # =========================================================================
    # PAGE 1: GPU DETECTION
    # =========================================================================

    def build_gpu_detection_page(self):
        page = tk.Frame(self, bg=CONTENT_BG)

        self.create_header(page, "GPU Detection", "Step 1 of 6")
        self.create_button_bar(page, [
            ("Cancel", self.on_cancel),
            ("< Back", lambda: self.show_page(0)),
            ("Next >", lambda: self.show_page(2))
        ])

        content = tk.Frame(page, bg=CONTENT_BG, padx=40, pady=25)
        content.pack(fill=tk.BOTH, expand=True)

        tk.Label(content, text="VBUS requires a GPU for optimal performance.",
                font=("Segoe UI", 11), fg=TEXT_COLOR, bg=CONTENT_BG).pack(anchor=tk.W, pady=(0, 20))

        # GPU Status
        if self.gpu_info["vendor"] != "none":
            status_bg = "#d4edda"
            status_fg = "#155724"
            status_text = "GPU DETECTED"
        else:
            status_bg = "#f8d7da"
            status_fg = "#721c24"
            status_text = "NO GPU DETECTED"

        status_frame = tk.Frame(content, bg=status_bg, relief=tk.GROOVE, bd=1)
        status_frame.pack(fill=tk.X, pady=10)

        tk.Label(status_frame, text=status_text, font=("Segoe UI", 14, "bold"),
                fg=status_fg, bg=status_bg).pack(pady=15)

        # GPU Details
        details_frame = tk.LabelFrame(content, text=" GPU Information ",
                                     font=("Segoe UI", 10, "bold"),
                                     bg=CONTENT_BG, fg=TEXT_COLOR, padx=20, pady=15)
        details_frame.pack(fill=tk.X, pady=15)

        if self.gpu_info["vendor"] != "none":
            details = f"""
    Vendor:     {self.gpu_info['vendor'].upper()}
    GPU:        {self.gpu_info['name']}
    VRAM:       {self.gpu_info['vram_mb']:,} MB
    CUDA:       {'Available' if self.gpu_info['cuda_available'] else 'Not available'}
    ROCm:       {'Available' if self.gpu_info['rocm_available'] else 'Not available'}

    Cache Recommendation:
      L1 (VRAM): {int(self.gpu_info['vram_mb'] * 0.8):,} MB (80% of VRAM)
"""
        else:
            details = """
    No GPU detected!

    VBUS works best with:
      - NVIDIA GPU with CUDA 12+
      - AMD GPU with ROCm

    You can continue installation, but GPU features will be limited.
"""
        tk.Label(details_frame, text=details, font=("Consolas", 10),
                fg=TEXT_COLOR, bg=CONTENT_BG, justify=tk.LEFT).pack(anchor=tk.W)

        # Python status
        py_frame = tk.LabelFrame(content, text=" Python Environment ",
                                font=("Segoe UI", 10, "bold"),
                                bg=CONTENT_BG, fg=TEXT_COLOR, padx=20, pady=10)
        py_frame.pack(fill=tk.X, pady=10)

        if self.python_info["installed"] and self.python_info["pip"]:
            py_status = f"    {self.python_info['version']}\n    pip: Available"
            py_color = SUCCESS_COLOR
        else:
            py_status = "    Python or pip not found!"
            py_color = ERROR_COLOR

        tk.Label(py_frame, text=py_status, font=("Consolas", 10),
                fg=py_color, bg=CONTENT_BG, justify=tk.LEFT).pack(anchor=tk.W)

        self.pages.append(page)

    # =========================================================================
    # PAGE 2: MASTER DATABASE
    # =========================================================================

    def build_master_db_page(self):
        page = tk.Frame(self, bg=CONTENT_BG)

        self.create_header(page, "Master Database", "Step 2 of 6")
        self.create_button_bar(page, [
            ("Cancel", self.on_cancel),
            ("< Back", lambda: self.show_page(1)),
            ("Next >", self.validate_master_and_continue)
        ])

        content = tk.Frame(page, bg=CONTENT_BG, padx=40, pady=25)
        content.pack(fill=tk.BOTH, expand=True)

        tk.Label(content, text="Configure your master database location.",
                font=("Segoe UI", 11), fg=TEXT_COLOR, bg=CONTENT_BG).pack(anchor=tk.W, pady=(0, 5))
        tk.Label(content, text="This is your primary data source for ML training.",
                font=("Segoe UI", 10), fg="#666666", bg=CONTENT_BG).pack(anchor=tk.W, pady=(0, 20))

        # Master path
        path_frame = tk.LabelFrame(content, text=" Master Database Path ",
                                  font=("Segoe UI", 10, "bold"),
                                  bg=CONTENT_BG, fg=TEXT_COLOR, padx=20, pady=15)
        path_frame.pack(fill=tk.X, pady=10)

        entry_frame = tk.Frame(path_frame, bg=CONTENT_BG)
        entry_frame.pack(fill=tk.X, pady=5)

        ttk.Entry(entry_frame, textvariable=self.master_path_var,
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(entry_frame, text="Browse...",
                  command=self.browse_master).pack(side=tk.RIGHT, padx=(10, 0))

        # Master name
        tk.Label(path_frame, text="Database Name:", font=("Segoe UI", 10),
                fg=TEXT_COLOR, bg=CONTENT_BG).pack(anchor=tk.W, pady=(15, 5))
        ttk.Entry(path_frame, textvariable=self.master_name_var,
                 font=("Segoe UI", 10), width=40).pack(anchor=tk.W)

        # Info
        info_frame = tk.Frame(content, bg="#e8f4fc", relief=tk.GROOVE, bd=1)
        info_frame.pack(fill=tk.X, pady=20)

        info_text = """
    The master database should contain your primary data files:

      - Parquet files (.parquet)
      - CSV files (.csv)
      - Training data, features, labels

    The VBus will index all data files in this directory for caching.
"""
        tk.Label(info_frame, text=info_text, font=("Segoe UI", 9),
                fg="#0c5460", bg="#e8f4fc", justify=tk.LEFT).pack(padx=15, pady=15)

        self.pages.append(page)

    def browse_master(self):
        path = filedialog.askdirectory(title="Select Master Database Folder")
        if path:
            self.master_path_var.set(path)
            if not self.master_name_var.get() or self.master_name_var.get() == "MasterDB":
                self.master_name_var.set(os.path.basename(path) or "MasterDB")

    def validate_master_and_continue(self):
        path = self.master_path_var.get().strip()
        if not path:
            messagebox.showwarning("Required", "Please specify the master database path.")
            return

        if not os.path.exists(path):
            create = messagebox.askyesno("Path Not Found",
                                        f"The path does not exist:\n{path}\n\nCreate it?")
            if create:
                try:
                    os.makedirs(path, exist_ok=True)
                except Exception as e:
                    messagebox.showerror("Error", f"Could not create directory: {e}")
                    return
            else:
                return

        self.show_page(3)

    # =========================================================================
    # PAGE 3: CONTRIBUTOR REPOSITORIES
    # =========================================================================

    def build_contributors_page(self):
        page = tk.Frame(self, bg=CONTENT_BG)

        self.create_header(page, "Contributor Repositories", "Step 3 of 6")
        self.create_button_bar(page, [
            ("Cancel", self.on_cancel),
            ("< Back", lambda: self.show_page(2)),
            ("Next >", lambda: self.show_page(4))
        ])

        content = tk.Frame(page, bg=CONTENT_BG, padx=40, pady=20)
        content.pack(fill=tk.BOTH, expand=True)

        tk.Label(content, text="Add up to 3 contributor data repositories (optional).",
                font=("Segoe UI", 11), fg=TEXT_COLOR, bg=CONTENT_BG).pack(anchor=tk.W, pady=(0, 5))
        tk.Label(content, text="These provide supplementary data to your ML pipeline.",
                font=("Segoe UI", 10), fg="#666666", bg=CONTENT_BG).pack(anchor=tk.W, pady=(0, 15))

        # Contributor 1
        self._create_contributor_frame(content, "Contributor 1",
                                       self.contrib1_path_var, self.contrib1_name_var,
                                       self.browse_contrib1)

        # Contributor 2
        self._create_contributor_frame(content, "Contributor 2",
                                       self.contrib2_path_var, self.contrib2_name_var,
                                       self.browse_contrib2)

        # Contributor 3
        self._create_contributor_frame(content, "Contributor 3",
                                       self.contrib3_path_var, self.contrib3_name_var,
                                       self.browse_contrib3)

        self.pages.append(page)

    def _create_contributor_frame(self, parent, title, path_var, name_var, browse_cmd):
        frame = tk.LabelFrame(parent, text=f" {title} ",
                             font=("Segoe UI", 9, "bold"),
                             bg=CONTENT_BG, fg=TEXT_COLOR, padx=15, pady=8)
        frame.pack(fill=tk.X, pady=8)

        row1 = tk.Frame(frame, bg=CONTENT_BG)
        row1.pack(fill=tk.X)

        tk.Label(row1, text="Path:", font=("Segoe UI", 9),
                fg=TEXT_COLOR, bg=CONTENT_BG, width=6).pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=path_var, font=("Segoe UI", 9)).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(row1, text="Browse", command=browse_cmd, width=8).pack(side=tk.RIGHT)

        row2 = tk.Frame(frame, bg=CONTENT_BG)
        row2.pack(fill=tk.X, pady=(5, 0))

        tk.Label(row2, text="Name:", font=("Segoe UI", 9),
                fg=TEXT_COLOR, bg=CONTENT_BG, width=6).pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=name_var, font=("Segoe UI", 9), width=30).pack(side=tk.LEFT, padx=5)

    def browse_contrib1(self):
        self._browse_contributor(self.contrib1_path_var, self.contrib1_name_var, "Contributor1")

    def browse_contrib2(self):
        self._browse_contributor(self.contrib2_path_var, self.contrib2_name_var, "Contributor2")

    def browse_contrib3(self):
        self._browse_contributor(self.contrib3_path_var, self.contrib3_name_var, "Contributor3")

    def _browse_contributor(self, path_var, name_var, default_name):
        path = filedialog.askdirectory(title="Select Contributor Folder")
        if path:
            path_var.set(path)
            if not name_var.get():
                name_var.set(os.path.basename(path) or default_name)

    # =========================================================================
    # PAGE 4: PACKAGES SELECTION
    # =========================================================================

    def build_packages_page(self):
        page = tk.Frame(self, bg=CONTENT_BG)

        self.create_header(page, "Package Installation", "Step 4 of 6")
        self.create_button_bar(page, [
            ("Cancel", self.on_cancel),
            ("< Back", lambda: self.show_page(3)),
            ("Next >", self.update_review_and_continue)
        ])

        content = tk.Frame(page, bg=CONTENT_BG, padx=40, pady=20)
        content.pack(fill=tk.BOTH, expand=True)

        tk.Label(content, text="Select packages to install:",
                font=("Segoe UI", 11), fg=TEXT_COLOR, bg=CONTENT_BG).pack(anchor=tk.W, pady=(0, 15))

        # Core packages (always installed)
        core_frame = tk.LabelFrame(content, text=" Core Packages (Required) ",
                                  font=("Segoe UI", 10, "bold"),
                                  bg=CONTENT_BG, fg=TEXT_COLOR, padx=15, pady=10)
        core_frame.pack(fill=tk.X, pady=8)

        core_list = "  " + ", ".join([f"{p[0]} ({p[2]})" for p in CORE_PACKAGES])
        tk.Label(core_frame, text=core_list, font=("Segoe UI", 9),
                fg=TEXT_COLOR, bg=CONTENT_BG, wraplength=650, justify=tk.LEFT).pack(anchor=tk.W)

        # GPU packages
        gpu_frame = tk.LabelFrame(content, text=" GPU Packages ",
                                 font=("Segoe UI", 10, "bold"),
                                 bg=CONTENT_BG, fg=TEXT_COLOR, padx=15, pady=10)
        gpu_frame.pack(fill=tk.X, pady=8)

        ttk.Checkbutton(gpu_frame, text="Install RAPIDS (cuDF, cuML, CuPy) - Requires NVIDIA GPU",
                       variable=self.install_rapids_var).pack(anchor=tk.W)

        rapids_note = "  Note: RAPIDS installation may require conda for full functionality."
        tk.Label(gpu_frame, text=rapids_note, font=("Segoe UI", 8),
                fg="#888888", bg=CONTENT_BG).pack(anchor=tk.W, pady=(5, 0))

        # ML packages
        ml_frame = tk.LabelFrame(content, text=" ML Framework Packages ",
                                font=("Segoe UI", 10, "bold"),
                                bg=CONTENT_BG, fg=TEXT_COLOR, padx=15, pady=10)
        ml_frame.pack(fill=tk.X, pady=8)

        ttk.Checkbutton(ml_frame, text="Install ML Frameworks (XGBoost, CatBoost, LightGBM, scikit-learn)",
                       variable=self.install_ml_var).pack(anchor=tk.W)

        ml_list = "  " + ", ".join([f"{p[0]}" for p in ML_PACKAGES])
        tk.Label(ml_frame, text=ml_list, font=("Segoe UI", 9),
                fg="#666666", bg=CONTENT_BG).pack(anchor=tk.W, pady=(5, 0))

        # Install path
        path_frame = tk.LabelFrame(content, text=" Installation Path ",
                                  font=("Segoe UI", 10, "bold"),
                                  bg=CONTENT_BG, fg=TEXT_COLOR, padx=15, pady=10)
        path_frame.pack(fill=tk.X, pady=8)

        path_row = tk.Frame(path_frame, bg=CONTENT_BG)
        path_row.pack(fill=tk.X)

        ttk.Entry(path_row, textvariable=self.install_path_var,
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_row, text="Browse...",
                  command=self.browse_install_path).pack(side=tk.RIGHT, padx=(10, 0))

        self.pages.append(page)

    def browse_install_path(self):
        path = filedialog.askdirectory(title="Select Installation Folder")
        if path:
            self.install_path_var.set(os.path.join(path, "VBUS"))

    def update_review_and_continue(self):
        self.update_review()
        self.show_page(5)

    # =========================================================================
    # PAGE 5: REVIEW
    # =========================================================================

    def build_review_page(self):
        page = tk.Frame(self, bg=CONTENT_BG)

        self.create_header(page, "Review Installation", "Step 5 of 6")
        self.create_button_bar(page, [
            ("Cancel", self.on_cancel),
            ("< Back", lambda: self.show_page(4)),
            ("Install", self.start_install)
        ])

        content = tk.Frame(page, bg=CONTENT_BG, padx=40, pady=20)
        content.pack(fill=tk.BOTH, expand=True)

        tk.Label(content, text="Review your configuration before installing:",
                font=("Segoe UI", 11), fg=TEXT_COLOR, bg=CONTENT_BG).pack(anchor=tk.W, pady=(0, 15))

        # Review text
        review_frame = tk.Frame(content, bg="white", relief=tk.SUNKEN, bd=1)
        review_frame.pack(fill=tk.BOTH, expand=True)

        self.review_text = tk.Text(review_frame, font=("Consolas", 10),
                                  wrap=tk.NONE, bg="white", fg=TEXT_COLOR,
                                  padx=15, pady=15, state=tk.DISABLED)
        self.review_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(review_frame, command=self.review_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.review_text.config(yscrollcommand=scrollbar.set)

        tk.Label(content, text="Click 'Install' to begin installation. This may take several minutes.",
                font=("Segoe UI", 10), fg=TEXT_COLOR, bg=CONTENT_BG).pack(anchor=tk.W, pady=(15, 0))

        self.pages.append(page)

    def update_review(self):
        self.review_text.config(state=tk.NORMAL)
        self.review_text.delete(1.0, tk.END)

        review = "VRAMCACHE INSTALLATION SUMMARY\n"
        review += "=" * 50 + "\n\n"

        # GPU
        review += f"GPU:  {self.gpu_info['name']}\n"
        if self.gpu_info['vram_mb'] > 0:
            review += f"VRAM: {self.gpu_info['vram_mb']:,} MB\n"
        review += "\n"

        # Master DB
        review += "MASTER DATABASE:\n"
        review += f"  Name: {self.master_name_var.get()}\n"
        review += f"  Path: {self.master_path_var.get()}\n\n"

        # Contributors
        contributors = []
        if self.contrib1_path_var.get():
            contributors.append((self.contrib1_name_var.get() or "Contributor1", self.contrib1_path_var.get()))
        if self.contrib2_path_var.get():
            contributors.append((self.contrib2_name_var.get() or "Contributor2", self.contrib2_path_var.get()))
        if self.contrib3_path_var.get():
            contributors.append((self.contrib3_name_var.get() or "Contributor3", self.contrib3_path_var.get()))

        if contributors:
            review += "CONTRIBUTOR REPOSITORIES:\n"
            for name, path in contributors:
                review += f"  - {name}: {path}\n"
            review += "\n"

        # Packages
        review += "PACKAGES TO INSTALL:\n"
        review += "  Core: " + ", ".join([p[0] for p in CORE_PACKAGES]) + "\n"
        if self.install_rapids_var.get():
            review += "  GPU:  " + ", ".join([p[0] for p in GPU_PACKAGES]) + "\n"
            review += "        " + ", ".join([p[0] for p in RAPIDS_PACKAGES]) + "\n"
        if self.install_ml_var.get():
            review += "  ML:   " + ", ".join([p[0] for p in ML_PACKAGES]) + "\n"
        review += "\n"

        # Install path
        review += f"INSTALL PATH: {self.install_path_var.get()}\n"

        self.review_text.insert(tk.END, review)
        self.review_text.config(state=tk.DISABLED)

    # =========================================================================
    # PAGE 6: PROGRESS
    # =========================================================================

    def build_progress_page(self):
        page = tk.Frame(self, bg=CONTENT_BG)

        self.create_header(page, "Installing", "Step 6 of 6")

        btn_bar = tk.Frame(page, bg=BUTTON_BAR_BG, height=55)
        btn_bar.pack(fill=tk.X, side=tk.BOTTOM)
        btn_bar.pack_propagate(False)
        tk.Frame(page, bg="#cccccc", height=1).pack(fill=tk.X, side=tk.BOTTOM)

        content = tk.Frame(page, bg=CONTENT_BG, padx=40, pady=20)
        content.pack(fill=tk.BOTH, expand=True)

        tk.Label(content, text="Installing VBUS and dependencies...",
                font=("Segoe UI", 11), fg=TEXT_COLOR, bg=CONTENT_BG).pack(anchor=tk.W, pady=(0, 15))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(content, variable=self.progress_var,
                                           maximum=100, length=700, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=10)

        self.status_var = tk.StringVar(value="Preparing...")
        tk.Label(content, textvariable=self.status_var, font=("Segoe UI", 10),
                fg="#666666", bg=CONTENT_BG).pack(anchor=tk.W, pady=5)

        # Log
        log_frame = tk.Frame(content, bg="#1a1a2e", relief=tk.SUNKEN, bd=1)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.log_text = tk.Text(log_frame, font=("Consolas", 9),
                               bg="#1a1a2e", fg="#00ff00",
                               relief=tk.FLAT, padx=10, pady=10, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.pages.append(page)

    # =========================================================================
    # PAGE 7: COMPLETE
    # =========================================================================

    def build_complete_page(self):
        page = tk.Frame(self, bg=CONTENT_BG)

        self.create_header(page, "Installation Complete")
        self.create_button_bar(page, [("Finish", self.on_finish)])

        content = tk.Frame(page, bg=CONTENT_BG, padx=40, pady=20)
        content.pack(fill=tk.BOTH, expand=True)

        tk.Label(content, text="\u2713", font=("Segoe UI", 48),
                fg=SUCCESS_COLOR, bg=CONTENT_BG).pack(pady=15)
        tk.Label(content, text="VBUS has been successfully installed!",
                font=("Segoe UI", 14, "bold"), fg=TEXT_COLOR, bg=CONTENT_BG).pack()

        self.complete_path_label = tk.Label(content, text="",
                                           font=("Segoe UI", 10), fg="#666666", bg=CONTENT_BG)
        self.complete_path_label.pack(pady=10)

        # Usage info
        usage_frame = tk.Frame(content, bg="#e8f4fc", relief=tk.GROOVE, bd=1)
        usage_frame.pack(fill=tk.X, pady=15)

        usage_text = """
    To use VBUS:

      1. Launch: python -m vramcache status
      2. Verify GPU: python -m vramcache verify
      3. In Python:
           from vramcache import VBus, load_config
           bus = VBus(load_config())
           bus.initialize()
"""
        tk.Label(usage_frame, text=usage_text, font=("Consolas", 9),
                fg="#0c5460", bg="#e8f4fc", justify=tk.LEFT).pack(padx=15, pady=15)

        ttk.Checkbutton(content, text="Open installation folder when finished",
                       variable=self.open_folder_var).pack(pady=10)

        tk.Frame(content, bg=CONTENT_BG).pack(expand=True)
        tk.Label(content, text=f"Copyright (c) {COPYRIGHT_YEAR} {COMPANY_NAME}",
                font=("Segoe UI", 8), fg="#999999", bg=CONTENT_BG).pack(side=tk.BOTTOM)

        self.pages.append(page)

    # =========================================================================
    # INSTALLATION
    # =========================================================================

    def log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"> {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.update_idletasks()

    def start_install(self):
        self.show_page(6)
        threading.Thread(target=self.do_install, daemon=True).start()

    def do_install(self):
        import time

        try:
            install_path = Path(self.install_path_var.get())
            config_path = Path(DEFAULT_CONFIG_PATH)

            # Step 1: Create directories
            self.status_var.set("Creating directories...")
            self.progress_var.set(5)
            self.log("Creating installation directories...")

            install_path.mkdir(parents=True, exist_ok=True)
            config_path.mkdir(parents=True, exist_ok=True)
            (config_path / "l3_cache").mkdir(exist_ok=True)
            (config_path / "audit_logs").mkdir(exist_ok=True)

            self.log(f"  Install: {install_path}")
            self.log(f"  Config:  {config_path}")

            # Step 2: Install core packages
            self.status_var.set("Installing core packages...")
            self.progress_var.set(10)
            self.log("")
            self.log("Installing core Python packages...")

            for i, (pkg, ver, desc) in enumerate(CORE_PACKAGES):
                self.log(f"  {pkg} ({desc})...")
                self.status_var.set(f"Installing {pkg}...")
                progress = 10 + (i / len(CORE_PACKAGES)) * 15
                self.progress_var.set(progress)

                if is_package_installed(pkg):
                    self.log(f"    Already installed")
                else:
                    success, error = pip_install(f"{pkg}>={ver}")
                    if success:
                        self.log(f"    Installed")
                    else:
                        self.log(f"    Failed: {error}")

            # Step 3: Install GPU packages
            if self.install_rapids_var.get():
                self.status_var.set("Installing GPU packages...")
                self.progress_var.set(30)
                self.log("")
                self.log("Installing GPU packages (this may take a while)...")

                # CuPy
                for pkg, ver, desc in GPU_PACKAGES:
                    self.log(f"  {pkg} ({desc})...")
                    self.status_var.set(f"Installing {pkg}...")

                    if is_package_installed(pkg):
                        self.log(f"    Already installed")
                    else:
                        success, error = pip_install(f"{pkg}>={ver}", timeout=900)
                        if success:
                            self.log(f"    Installed")
                        else:
                            self.log(f"    Failed (may need conda): {error[:50]}")

                # RAPIDS
                self.log("")
                self.log("Installing RAPIDS packages...")
                for pkg, ver, desc in RAPIDS_PACKAGES:
                    self.log(f"  {pkg} ({desc})...")
                    self.status_var.set(f"Installing {pkg}...")

                    if is_package_installed(pkg.split('-')[0]):
                        self.log(f"    Already installed")
                    else:
                        success, error = pip_install(pkg, timeout=900)
                        if success:
                            self.log(f"    Installed")
                        else:
                            self.log(f"    Note: Install via conda for full support")

            # Step 4: Install ML packages
            if self.install_ml_var.get():
                self.status_var.set("Installing ML packages...")
                self.progress_var.set(50)
                self.log("")
                self.log("Installing ML framework packages...")

                for i, (pkg, ver, desc) in enumerate(ML_PACKAGES):
                    self.log(f"  {pkg} ({desc})...")
                    self.status_var.set(f"Installing {pkg}...")
                    progress = 50 + (i / len(ML_PACKAGES)) * 20
                    self.progress_var.set(progress)

                    if is_package_installed(pkg):
                        self.log(f"    Already installed")
                    else:
                        success, error = pip_install(f"{pkg}>={ver}", timeout=600)
                        if success:
                            self.log(f"    Installed")
                        else:
                            self.log(f"    Failed: {error[:50]}")

            # Step 5: Copy VBUS source files
            self.status_var.set("Installing VBUS module...")
            self.progress_var.set(75)
            self.log("")
            self.log("Installing VBUS module...")

            # Copy source files
            src_dir = Path(resource_path("src/vramcache"))
            if src_dir.exists():
                dest_dir = install_path / "vramcache"
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(src_dir, dest_dir)
                self.log(f"  Copied vramcache module")
            else:
                # Install via pip if bundled
                success, _ = pip_install("vramcache", timeout=120)
                if success:
                    self.log(f"  Installed vramcache package")

            # Step 6: Create configuration
            self.status_var.set("Creating configuration...")
            self.progress_var.set(85)
            self.log("")
            self.log("Creating configuration...")

            # Build config
            config = {
                "version": APP_VERSION,
                "created_at": datetime.now().isoformat(),
                "master_repository": {
                    "name": self.master_name_var.get(),
                    "path": self.master_path_var.get(),
                    "role": "master",
                    "file_patterns": ["*.parquet", "*.csv"],
                    "enabled": True
                },
                "contributor_repositories": [],
                "gpu": {
                    "enabled": self.gpu_info["vendor"] != "none",
                    "backend": "cuda" if self.gpu_info["cuda_available"] else "rocm" if self.gpu_info["rocm_available"] else "cpu",
                    "vram_limit_mb": self.gpu_info["vram_mb"]
                },
                "cache": {
                    "l1_max_mb": int(self.gpu_info["vram_mb"] * 0.8) if self.gpu_info["vram_mb"] > 0 else 4096,
                    "l2_max_mb": 32768,
                    "l3_max_mb": 65536,
                    "l3_disk_path": str(config_path / "l3_cache")
                },
                "audit_log_path": str(config_path / "audit_logs")
            }

            # Add contributors
            if self.contrib1_path_var.get():
                config["contributor_repositories"].append({
                    "name": self.contrib1_name_var.get() or "Contributor1",
                    "path": self.contrib1_path_var.get(),
                    "role": "contributor",
                    "enabled": True
                })
            if self.contrib2_path_var.get():
                config["contributor_repositories"].append({
                    "name": self.contrib2_name_var.get() or "Contributor2",
                    "path": self.contrib2_path_var.get(),
                    "role": "contributor",
                    "enabled": True
                })
            if self.contrib3_path_var.get():
                config["contributor_repositories"].append({
                    "name": self.contrib3_name_var.get() or "Contributor3",
                    "path": self.contrib3_path_var.get(),
                    "role": "contributor",
                    "enabled": True
                })

            # Save config
            config_file = config_path / "config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            self.log(f"  Saved: {config_file}")

            # Step 7: Create launcher scripts
            self.status_var.set("Creating launchers...")
            self.progress_var.set(95)
            self.log("")
            self.log("Creating launcher scripts...")

            # Windows batch file
            bat_content = f'''@echo off
REM VBUS Launcher - Framecore Inc
cd /d "{install_path}"
python -m vramcache %*
'''
            with open(install_path / "vramcache.bat", 'w') as f:
                f.write(bat_content)

            # Unix shell script
            sh_content = f'''#!/bin/bash
# VBUS Launcher - Framecore Inc
cd "{install_path}"
python -m vramcache "$@"
'''
            with open(install_path / "vramcache.sh", 'w') as f:
                f.write(sh_content)

            self.log("  Created vramcache.bat")
            self.log("  Created vramcache.sh")

            # Complete
            self.progress_var.set(100)
            self.status_var.set("Complete!")
            self.log("")
            self.log("=" * 50)
            self.log("INSTALLATION COMPLETE")
            self.log("=" * 50)
            time.sleep(0.5)

            self.complete_path_label.config(text=f"Installed to: {install_path}")
            self.after(0, lambda: self.show_page(7))

        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.after(0, lambda: messagebox.showerror("Error", f"Installation failed:\n{e}"))

    # =========================================================================
    # ACTIONS
    # =========================================================================

    def on_cancel(self):
        if messagebox.askyesno("Cancel", "Are you sure you want to cancel?"):
            self.destroy()

    def on_finish(self):
        if self.open_folder_var.get():
            try:
                if platform.system() == "Windows":
                    os.startfile(self.install_path_var.get())
                else:
                    subprocess.run(["xdg-open", self.install_path_var.get()])
            except:
                pass
        self.destroy()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # DPI awareness on Windows
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    app = VBUSInstaller()

    # Style
    style = ttk.Style()
    try:
        style.theme_use('vista')
    except:
        pass
    style.configure('TButton', font=('Segoe UI', 9), padding=5)
    style.configure('TCheckbutton', background=CONTENT_BG)
    style.configure('TEntry', padding=5)

    app.mainloop()
