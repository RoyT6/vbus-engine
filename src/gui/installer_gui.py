"""
VBUS Installer GUI
PyQt6-based installer with system check, location options, and dependency management
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json
import threading

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QStackedWidget, QFrame,
    QLineEdit, QFileDialog, QCheckBox, QScrollArea, QGroupBox,
    QTextEdit, QMessageBox, QComboBox, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon, QPalette, QColor

from src.utils.system_info import SystemInfoCollector
from src.core.ml_detector import MLFrameworkDetector


@dataclass
class InstallConfig:
    """Installation configuration"""
    install_path: str = ""
    vram_cache_enabled: bool = True
    ram_cache_enabled: bool = True
    l1_cache_size_gb: float = 2.0
    l2_cache_size_gb: float = 4.0
    l3_cache_path: str = ""
    master_db_path: str = ""
    repositories: List[str] = None
    install_cuda: bool = False
    install_cudf: bool = False
    install_cuml: bool = False
    install_rapids: bool = False
    install_xgboost: bool = False
    install_sklearn: bool = True
    install_catboost: bool = False

    def __post_init__(self):
        if self.repositories is None:
            self.repositories = []


class SystemCheckThread(QThread):
    """Background thread for system checks"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)

    def run(self):
        results = {}

        self.progress.emit(10, "Collecting system information...")
        collector = SystemInfoCollector()
        system_info = collector.collect_all()
        results["system_info"] = collector.to_dict()

        self.progress.emit(40, "Checking compatibility...")
        compatibility = collector.check_compatibility()
        results["compatibility"] = compatibility

        self.progress.emit(70, "Detecting ML frameworks...")
        detector = MLFrameworkDetector()
        detector.detect_all()
        results["ml_frameworks"] = detector.to_dict()
        results["ml_summary"] = detector.get_summary()

        self.progress.emit(100, "Complete!")
        self.finished.emit(results)


class ModernFrame(QFrame):
    """Modern styled frame"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            ModernFrame {
                background-color: #2d2d2d;
                border-radius: 10px;
                padding: 15px;
            }
        """)


class WelcomePage(QWidget):
    """Welcome page of the installer"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Title
        title = QLabel("Welcome to VBUS Installer")
        title.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #00d4ff;")
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Virtual Bus - GPU VRAM Cache & Repository Traffic Management")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888888;")
        layout.addWidget(subtitle)

        layout.addSpacing(30)

        # Description frame
        desc_frame = ModernFrame()
        desc_layout = QVBoxLayout(desc_frame)

        desc_text = QLabel("""
<h3>VBUS provides two core capabilities:</h3>

<p><b>1. V (VRAM Cache Management)</b></p>
<ul>
    <li>Turns GPU VRAM into L1 cache for ML operations</li>
    <li>Uses system RAM as L2/L3 cache layers</li>
    <li>Intelligently manages throughput during ML training/inference</li>
    <li>Supports CUDA, cuDF, cuML, RAPIDS, XGBoost, Random Forest, CatBoost</li>
</ul>

<p><b>2. BUS (Repository Traffic Management)</b></p>
<ul>
    <li>Manages traffic between master database and repositories</li>
    <li>Uses AI inferencing to understand your system hierarchy</li>
    <li>Creates and maintains cache coherency across sources</li>
</ul>
        """)
        desc_text.setWordWrap(True)
        desc_text.setStyleSheet("color: #cccccc; font-size: 11pt;")
        desc_layout.addWidget(desc_text)

        layout.addWidget(desc_frame)
        layout.addStretch()

        # Requirements note
        req_label = QLabel("Requirements: Windows 10/11, NVIDIA GPU with CUDA support, 16GB+ RAM")
        req_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        req_label.setStyleSheet("color: #666666; font-size: 9pt;")
        layout.addWidget(req_label)


class SystemCheckPage(QWidget):
    """System compatibility check page"""

    check_complete = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.check_results = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Title
        title = QLabel("System Compatibility Check")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #00d4ff;")
        layout.addWidget(title)

        # Progress area
        self.progress_frame = ModernFrame()
        progress_layout = QVBoxLayout(self.progress_frame)

        self.status_label = QLabel("Click 'Run Check' to analyze your system")
        self.status_label.setStyleSheet("color: #cccccc; font-size: 11pt;")
        progress_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #404040;
                border-radius: 5px;
                text-align: center;
                background-color: #1a1a1a;
            }
            QProgressBar::chunk {
                background-color: #00d4ff;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)

        self.run_check_btn = QPushButton("Run System Check")
        self.run_check_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4ff;
                color: #000000;
                border: none;
                padding: 10px 30px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #33ddff;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #666666;
            }
        """)
        self.run_check_btn.clicked.connect(self._run_check)
        progress_layout.addWidget(self.run_check_btn)

        layout.addWidget(self.progress_frame)

        # Results area
        self.results_tabs = QTabWidget()
        self.results_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #404040;
                border-radius: 5px;
                background-color: #2d2d2d;
            }
            QTabBar::tab {
                background-color: #1a1a1a;
                color: #888888;
                padding: 10px 20px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #2d2d2d;
                color: #00d4ff;
            }
        """)
        self.results_tabs.hide()

        # System Info Tab
        self.system_info_text = QTextEdit()
        self.system_info_text.setReadOnly(True)
        self.system_info_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #cccccc;
                border: none;
                font-family: Consolas, monospace;
                font-size: 10pt;
            }
        """)
        self.results_tabs.addTab(self.system_info_text, "System Info")

        # Compatibility Tab
        self.compat_text = QTextEdit()
        self.compat_text.setReadOnly(True)
        self.compat_text.setStyleSheet(self.system_info_text.styleSheet())
        self.results_tabs.addTab(self.compat_text, "Compatibility")

        # ML Frameworks Tab
        self.ml_text = QTextEdit()
        self.ml_text.setReadOnly(True)
        self.ml_text.setStyleSheet(self.system_info_text.styleSheet())
        self.results_tabs.addTab(self.ml_text, "ML Frameworks")

        layout.addWidget(self.results_tabs)

    def _run_check(self):
        self.run_check_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Running system check...")

        self.check_thread = SystemCheckThread()
        self.check_thread.progress.connect(self._on_progress)
        self.check_thread.finished.connect(self._on_check_complete)
        self.check_thread.start()

    def _on_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def _on_check_complete(self, results: dict):
        self.check_results = results
        self.run_check_btn.setEnabled(True)
        self.results_tabs.show()

        # Populate System Info
        sys_info = results.get("system_info", {})
        sys_text = f"""
=== OPERATING SYSTEM ===
Name: {sys_info.get('os', {}).get('name', 'N/A')}
Version: {sys_info.get('os', {}).get('version', 'N/A')}
Architecture: {sys_info.get('os', {}).get('architecture', 'N/A')}

=== CPU ===
Name: {sys_info.get('cpu', {}).get('name', 'N/A')}
Cores: {sys_info.get('cpu', {}).get('cores', 'N/A')}
Threads: {sys_info.get('cpu', {}).get('threads', 'N/A')}

=== MEMORY ===
Total RAM: {sys_info.get('memory', {}).get('total_gb', 0):.2f} GB
Available: {sys_info.get('memory', {}).get('available_gb', 0):.2f} GB

=== GPU(s) ===
"""
        for i, gpu in enumerate(sys_info.get('gpus', [])):
            sys_text += f"""
GPU {i + 1}: {gpu.get('name', 'Unknown')}
  Vendor: {gpu.get('vendor', 'Unknown')}
  VRAM: {gpu.get('vram_total_mb', 0) / 1024:.2f} GB
  Driver: {gpu.get('driver_version', 'N/A')}
  CUDA: {gpu.get('cuda_version', 'N/A')}
  Compute Capability: {gpu.get('compute_capability', 'N/A')}
"""
        self.system_info_text.setPlainText(sys_text)

        # Populate Compatibility
        compat = results.get("compatibility", {})
        compat_text = f"Compatible: {'✓ YES' if compat.get('compatible') else '✗ NO'}\n\n"

        compat_text += "=== PASSED ===\n"
        for item in compat.get('passed', []):
            compat_text += f"✓ {item}\n"

        compat_text += "\n=== WARNINGS ===\n"
        for item in compat.get('warnings', []):
            compat_text += f"⚠ {item}\n"

        compat_text += "\n=== ISSUES ===\n"
        for item in compat.get('issues', []):
            compat_text += f"✗ {item}\n"

        self.compat_text.setPlainText(compat_text)

        # Populate ML Frameworks
        ml_summary = results.get("ml_summary", {})
        ml_text = f"GPU Ready: {'✓ YES' if ml_summary.get('gpu_ready') else '✗ NO'}\n"
        ml_text += f"Installed: {ml_summary.get('total_installed', 0)}/{ml_summary.get('total_installed', 0) + ml_summary.get('total_not_installed', 0)}\n\n"

        ml_text += "=== INSTALLED ===\n"
        for fw in ml_summary.get('installed', []):
            gpu_status = "🖥️ GPU" if fw.get('gpu_enabled') else "💻 CPU"
            ml_text += f"✓ {fw['name']} v{fw.get('version', 'N/A')} [{gpu_status}]\n"
            if fw.get('notes'):
                ml_text += f"  Notes: {fw['notes']}\n"

        ml_text += "\n=== NOT INSTALLED ===\n"
        for fw in ml_summary.get('not_installed', []):
            ml_text += f"✗ {fw['name']}\n"
            ml_text += f"  Install: {fw.get('install_command', 'N/A')}\n"

        self.ml_text.setPlainText(ml_text)

        self.check_complete.emit(results)


class ConfigurationPage(QWidget):
    """Installation configuration page"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = InstallConfig()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 20, 40, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("Installation Configuration")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #00d4ff;")
        layout.addWidget(title)

        # Scroll area for configuration
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(15)

        # Installation Path
        path_group = QGroupBox("Installation Location")
        path_group.setStyleSheet(self._group_style())
        path_layout = QHBoxLayout(path_group)

        self.install_path_edit = QLineEdit()
        self.install_path_edit.setText(str(Path.home() / "VBUS"))
        self.install_path_edit.setStyleSheet(self._input_style())
        path_layout.addWidget(self.install_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet(self._button_style())
        browse_btn.clicked.connect(self._browse_install_path)
        path_layout.addWidget(browse_btn)

        scroll_layout.addWidget(path_group)

        # Cache Configuration
        cache_group = QGroupBox("Cache Configuration (V)")
        cache_group.setStyleSheet(self._group_style())
        cache_layout = QVBoxLayout(cache_group)

        # VRAM Cache
        vram_layout = QHBoxLayout()
        self.vram_enabled = QCheckBox("Enable VRAM (L1) Cache")
        self.vram_enabled.setChecked(True)
        self.vram_enabled.setStyleSheet("color: #cccccc;")
        vram_layout.addWidget(self.vram_enabled)

        vram_layout.addWidget(QLabel("Size (GB):"))
        self.vram_size = QSpinBox()
        self.vram_size.setRange(1, 48)
        self.vram_size.setValue(2)
        self.vram_size.setStyleSheet(self._input_style())
        vram_layout.addWidget(self.vram_size)
        vram_layout.addStretch()
        cache_layout.addLayout(vram_layout)

        # RAM Cache
        ram_layout = QHBoxLayout()
        self.ram_enabled = QCheckBox("Enable RAM (L2/L3) Cache")
        self.ram_enabled.setChecked(True)
        self.ram_enabled.setStyleSheet("color: #cccccc;")
        ram_layout.addWidget(self.ram_enabled)

        ram_layout.addWidget(QLabel("Size (GB):"))
        self.ram_size = QSpinBox()
        self.ram_size.setRange(1, 128)
        self.ram_size.setValue(4)
        self.ram_size.setStyleSheet(self._input_style())
        ram_layout.addWidget(self.ram_size)
        ram_layout.addStretch()
        cache_layout.addLayout(ram_layout)

        # Disk Cache Path
        disk_layout = QHBoxLayout()
        disk_layout.addWidget(QLabel("Disk Cache (L3) Path:"))
        self.disk_cache_path = QLineEdit()
        self.disk_cache_path.setText(str(Path.home() / ".vbus" / "cache"))
        self.disk_cache_path.setStyleSheet(self._input_style())
        disk_layout.addWidget(self.disk_cache_path)
        disk_browse = QPushButton("Browse...")
        disk_browse.setStyleSheet(self._button_style())
        disk_browse.clicked.connect(self._browse_disk_cache)
        disk_layout.addWidget(disk_browse)
        cache_layout.addLayout(disk_layout)

        scroll_layout.addWidget(cache_group)

        # BUS Configuration
        bus_group = QGroupBox("Repository Traffic Management (BUS)")
        bus_group.setStyleSheet(self._group_style())
        bus_layout = QVBoxLayout(bus_group)

        # Master Database
        master_layout = QHBoxLayout()
        master_layout.addWidget(QLabel("Master Database Path:"))
        self.master_db_path = QLineEdit()
        self.master_db_path.setStyleSheet(self._input_style())
        self.master_db_path.setPlaceholderText("Select your master database location...")
        master_layout.addWidget(self.master_db_path)
        master_browse = QPushButton("Browse...")
        master_browse.setStyleSheet(self._button_style())
        master_browse.clicked.connect(self._browse_master_db)
        master_layout.addWidget(master_browse)
        bus_layout.addLayout(master_layout)

        # Repositories
        bus_layout.addWidget(QLabel("Repository Locations:"))
        self.repo_list = QTextEdit()
        self.repo_list.setPlaceholderText("Enter repository paths, one per line...")
        self.repo_list.setMaximumHeight(100)
        self.repo_list.setStyleSheet(self._input_style())
        bus_layout.addWidget(self.repo_list)

        add_repo_btn = QPushButton("Add Repository...")
        add_repo_btn.setStyleSheet(self._button_style())
        add_repo_btn.clicked.connect(self._add_repository)
        bus_layout.addWidget(add_repo_btn)

        scroll_layout.addWidget(bus_group)

        # ML Framework Installation
        ml_group = QGroupBox("ML Framework Installation")
        ml_group.setStyleSheet(self._group_style())
        ml_layout = QVBoxLayout(ml_group)

        ml_note = QLabel("Select ML frameworks to install (if not already present):")
        ml_note.setStyleSheet("color: #888888;")
        ml_layout.addWidget(ml_note)

        self.ml_checkboxes = {}
        frameworks = [
            ("cuda", "CUDA Toolkit", False),
            ("cudf", "cuDF (GPU DataFrames)", False),
            ("cuml", "cuML (GPU ML algorithms)", False),
            ("rapids", "RAPIDS Suite", False),
            ("xgboost", "XGBoost", True),
            ("sklearn", "Scikit-learn (Random Forest)", True),
            ("catboost", "CatBoost", False)
        ]

        for key, name, default in frameworks:
            cb = QCheckBox(name)
            cb.setChecked(default)
            cb.setStyleSheet("color: #cccccc;")
            self.ml_checkboxes[key] = cb
            ml_layout.addWidget(cb)

        scroll_layout.addWidget(ml_group)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def _group_style(self):
        return """
            QGroupBox {
                font-weight: bold;
                font-size: 11pt;
                color: #00d4ff;
                border: 1px solid #404040;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """

    def _input_style(self):
        return """
            QLineEdit, QSpinBox, QTextEdit {
                background-color: #1a1a1a;
                color: #cccccc;
                border: 1px solid #404040;
                border-radius: 3px;
                padding: 5px;
            }
            QLineEdit:focus, QSpinBox:focus, QTextEdit:focus {
                border-color: #00d4ff;
            }
        """

    def _button_style(self):
        return """
            QPushButton {
                background-color: #404040;
                color: #cccccc;
                border: none;
                padding: 8px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """

    def _browse_install_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Installation Directory")
        if path:
            self.install_path_edit.setText(path)

    def _browse_disk_cache(self):
        path = QFileDialog.getExistingDirectory(self, "Select Disk Cache Directory")
        if path:
            self.disk_cache_path.setText(path)

    def _browse_master_db(self):
        path = QFileDialog.getExistingDirectory(self, "Select Master Database Location")
        if path:
            self.master_db_path.setText(path)

    def _add_repository(self):
        path = QFileDialog.getExistingDirectory(self, "Select Repository Location")
        if path:
            current = self.repo_list.toPlainText()
            if current:
                self.repo_list.setPlainText(current + "\n" + path)
            else:
                self.repo_list.setPlainText(path)

    def get_config(self) -> InstallConfig:
        """Get current configuration"""
        self.config.install_path = self.install_path_edit.text()
        self.config.vram_cache_enabled = self.vram_enabled.isChecked()
        self.config.ram_cache_enabled = self.ram_enabled.isChecked()
        self.config.l1_cache_size_gb = self.vram_size.value()
        self.config.l2_cache_size_gb = self.ram_size.value()
        self.config.l3_cache_path = self.disk_cache_path.text()
        self.config.master_db_path = self.master_db_path.text()

        repo_text = self.repo_list.toPlainText()
        self.config.repositories = [r.strip() for r in repo_text.split('\n') if r.strip()]

        for key, cb in self.ml_checkboxes.items():
            setattr(self.config, f"install_{key}", cb.isChecked())

        return self.config


class InstallationPage(QWidget):
    """Installation progress page"""

    installation_complete = pyqtSignal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Title
        title = QLabel("Installing VBUS")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setStyleSheet("color: #00d4ff;")
        layout.addWidget(title)

        # Progress
        self.progress_frame = ModernFrame()
        progress_layout = QVBoxLayout(self.progress_frame)

        self.overall_label = QLabel("Overall Progress")
        self.overall_label.setStyleSheet("color: #cccccc;")
        progress_layout.addWidget(self.overall_label)

        self.overall_progress = QProgressBar()
        self.overall_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #404040;
                border-radius: 5px;
                text-align: center;
                background-color: #1a1a1a;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #00d4ff;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.overall_progress)

        self.current_task_label = QLabel("Current Task: Waiting...")
        self.current_task_label.setStyleSheet("color: #888888;")
        progress_layout.addWidget(self.current_task_label)

        self.current_progress = QProgressBar()
        self.current_progress.setStyleSheet(self.overall_progress.styleSheet())
        progress_layout.addWidget(self.current_progress)

        layout.addWidget(self.progress_frame)

        # Log output
        log_label = QLabel("Installation Log")
        log_label.setStyleSheet("color: #cccccc;")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff00;
                border: 1px solid #404040;
                border-radius: 5px;
                font-family: Consolas, monospace;
                font-size: 9pt;
            }
        """)
        layout.addWidget(self.log_text)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: #ffffff;
                border: none;
                padding: 10px 30px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
        """)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

    def log(self, message: str, level: str = "info"):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        color = "#00ff00"
        if level == "error":
            color = "#ff4444"
        elif level == "warning":
            color = "#ffaa00"
        elif level == "success":
            color = "#00ff88"

        self.log_text.append(f'<span style="color: #888888;">[{timestamp}]</span> '
                            f'<span style="color: {color};">{message}</span>')

    def set_overall_progress(self, value: int, message: str = ""):
        self.overall_progress.setValue(value)
        if message:
            self.overall_label.setText(f"Overall Progress: {message}")

    def set_current_task(self, task: str, progress: int = 0):
        self.current_task_label.setText(f"Current Task: {task}")
        self.current_progress.setValue(progress)


class CompletionPage(QWidget):
    """Installation completion page"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        layout.addStretch()

        # Icon/Status
        self.status_icon = QLabel("✓")
        self.status_icon.setFont(QFont("Segoe UI", 72))
        self.status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_icon.setStyleSheet("color: #00ff88;")
        layout.addWidget(self.status_icon)

        # Title
        self.title = QLabel("Installation Complete!")
        self.title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet("color: #00d4ff;")
        layout.addWidget(self.title)

        # Message
        self.message = QLabel("VBUS has been successfully installed on your system.")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message.setStyleSheet("color: #cccccc; font-size: 11pt;")
        layout.addWidget(self.message)

        layout.addSpacing(30)

        # Summary frame
        self.summary_frame = ModernFrame()
        summary_layout = QVBoxLayout(self.summary_frame)

        self.summary_text = QLabel()
        self.summary_text.setStyleSheet("color: #cccccc;")
        self.summary_text.setWordWrap(True)
        summary_layout.addWidget(self.summary_text)

        layout.addWidget(self.summary_frame)

        layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.launch_btn = QPushButton("Launch VBUS")
        self.launch_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4ff;
                color: #000000;
                border: none;
                padding: 12px 40px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #33ddff;
            }
        """)
        btn_layout.addWidget(self.launch_btn)

        self.close_btn = QPushButton("Close")
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #404040;
                color: #cccccc;
                border: none;
                padding: 12px 40px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

    def set_success(self, summary: str):
        self.status_icon.setText("✓")
        self.status_icon.setStyleSheet("color: #00ff88;")
        self.title.setText("Installation Complete!")
        self.message.setText("VBUS has been successfully installed on your system.")
        self.summary_text.setText(summary)

    def set_failure(self, error: str):
        self.status_icon.setText("✗")
        self.status_icon.setStyleSheet("color: #ff4444;")
        self.title.setText("Installation Failed")
        self.message.setText("An error occurred during installation.")
        self.summary_text.setText(f"Error: {error}")
        self.launch_btn.hide()


class VBUSInstaller(QMainWindow):
    """Main installer window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VBUS Installer")
        self.setMinimumSize(900, 700)
        self._setup_ui()
        self._apply_dark_theme()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet("background-color: #1a1a1a;")
        header_layout = QHBoxLayout(header)

        logo = QLabel("VBUS")
        logo.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        logo.setStyleSheet("color: #00d4ff;")
        header_layout.addWidget(logo)

        version = QLabel("v1.0.0")
        version.setStyleSheet("color: #666666;")
        header_layout.addWidget(version)
        header_layout.addStretch()

        main_layout.addWidget(header)

        # Stacked pages
        self.pages = QStackedWidget()
        self.pages.setStyleSheet("background-color: #252525;")

        # Create pages
        self.welcome_page = WelcomePage()
        self.system_check_page = SystemCheckPage()
        self.config_page = ConfigurationPage()
        self.install_page = InstallationPage()
        self.complete_page = CompletionPage()

        self.pages.addWidget(self.welcome_page)
        self.pages.addWidget(self.system_check_page)
        self.pages.addWidget(self.config_page)
        self.pages.addWidget(self.install_page)
        self.pages.addWidget(self.complete_page)

        main_layout.addWidget(self.pages)

        # Footer with navigation
        footer = QFrame()
        footer.setFixedHeight(70)
        footer.setStyleSheet("background-color: #1a1a1a;")
        footer_layout = QHBoxLayout(footer)

        # Step indicators
        self.step_labels = []
        steps = ["Welcome", "System Check", "Configure", "Install", "Complete"]
        for i, step in enumerate(steps):
            lbl = QLabel(f"{i + 1}. {step}")
            lbl.setStyleSheet("color: #666666; font-size: 9pt;")
            self.step_labels.append(lbl)
            footer_layout.addWidget(lbl)

        footer_layout.addStretch()

        # Navigation buttons
        self.back_btn = QPushButton("← Back")
        self.back_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #888888;
                border: 1px solid #404040;
                padding: 10px 25px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #333333;
            }
            QPushButton:disabled {
                color: #444444;
                border-color: #333333;
            }
        """)
        self.back_btn.clicked.connect(self._go_back)
        footer_layout.addWidget(self.back_btn)

        self.next_btn = QPushButton("Next →")
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4ff;
                color: #000000;
                border: none;
                padding: 10px 25px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #33ddff;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #666666;
            }
        """)
        self.next_btn.clicked.connect(self._go_next)
        footer_layout.addWidget(self.next_btn)

        main_layout.addWidget(footer)

        # Connect signals
        self.system_check_page.check_complete.connect(self._on_system_check_complete)
        self.install_page.cancel_btn.clicked.connect(self._cancel_installation)
        self.complete_page.close_btn.clicked.connect(self.close)

        # Initial state
        self._update_navigation()

    def _apply_dark_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #252525;
            }
            QLabel {
                color: #cccccc;
            }
            QMessageBox {
                background-color: #2d2d2d;
            }
            QMessageBox QLabel {
                color: #cccccc;
            }
        """)

    def _update_navigation(self):
        """Update navigation buttons and step indicators"""
        current = self.pages.currentIndex()

        # Update step indicators
        for i, lbl in enumerate(self.step_labels):
            if i < current:
                lbl.setStyleSheet("color: #00d4ff; font-size: 9pt;")
            elif i == current:
                lbl.setStyleSheet("color: #ffffff; font-size: 9pt; font-weight: bold;")
            else:
                lbl.setStyleSheet("color: #666666; font-size: 9pt;")

        # Update buttons
        self.back_btn.setEnabled(current > 0 and current < 4)
        self.back_btn.setVisible(current < 4)
        self.next_btn.setVisible(current < 4)

        if current == 3:  # Install page
            self.next_btn.setText("Install")
        else:
            self.next_btn.setText("Next →")

    def _go_back(self):
        current = self.pages.currentIndex()
        if current > 0:
            self.pages.setCurrentIndex(current - 1)
            self._update_navigation()

    def _go_next(self):
        current = self.pages.currentIndex()

        # Validate current page
        if current == 2:  # Config page
            config = self.config_page.get_config()
            if not config.install_path:
                QMessageBox.warning(self, "Validation", "Please specify an installation path.")
                return

        if current == 3:  # Start installation
            self._start_installation()
            return

        if current < self.pages.count() - 1:
            self.pages.setCurrentIndex(current + 1)
            self._update_navigation()

    def _on_system_check_complete(self, results: dict):
        """Handle system check completion"""
        compat = results.get("compatibility", {})
        if not compat.get("compatible"):
            QMessageBox.warning(
                self,
                "Compatibility Warning",
                "Your system may not be fully compatible with VBUS.\n"
                "You can continue, but some features may not work correctly."
            )

    def _start_installation(self):
        """Start the installation process"""
        config = self.config_page.get_config()

        self.next_btn.setEnabled(False)
        self.back_btn.setEnabled(False)

        # Simulate installation (in real implementation, this would do actual work)
        self.install_page.log("Starting VBUS installation...")
        self.install_page.set_overall_progress(0, "Preparing...")

        # Create installation thread
        self.install_thread = threading.Thread(target=self._run_installation, args=(config,))
        self.install_thread.start()

        # Start progress updates
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self._check_installation_progress)
        self.progress_timer.start(500)

    def _run_installation(self, config: InstallConfig):
        """Run the actual installation (in a separate thread)"""
        import time

        steps = [
            ("Creating directories...", 10),
            ("Installing core components...", 30),
            ("Configuring cache system...", 50),
            ("Setting up BUS components...", 70),
            ("Installing ML framework support...", 85),
            ("Finalizing installation...", 95),
            ("Complete!", 100)
        ]

        self.install_progress = 0
        self.install_message = ""
        self.install_error = None
        self.install_complete = False

        try:
            # Create installation directory
            install_path = Path(config.install_path)
            install_path.mkdir(parents=True, exist_ok=True)

            for message, progress in steps:
                self.install_message = message
                self.install_progress = progress
                time.sleep(0.5)  # Simulated work

            # Write config file
            config_file = install_path / "vbus_config.json"
            with open(config_file, 'w') as f:
                json.dump({
                    "install_path": str(install_path),
                    "cache": {
                        "vram_enabled": config.vram_cache_enabled,
                        "ram_enabled": config.ram_cache_enabled,
                        "l1_size_gb": config.l1_cache_size_gb,
                        "l2_size_gb": config.l2_cache_size_gb,
                        "l3_path": config.l3_cache_path
                    },
                    "bus": {
                        "master_db": config.master_db_path,
                        "repositories": config.repositories
                    }
                }, f, indent=2)

            self.install_complete = True

        except Exception as e:
            self.install_error = str(e)
            self.install_complete = True

    def _check_installation_progress(self):
        """Check installation progress and update UI"""
        if hasattr(self, 'install_message'):
            self.install_page.set_overall_progress(self.install_progress, self.install_message)
            self.install_page.log(self.install_message)

        if hasattr(self, 'install_complete') and self.install_complete:
            self.progress_timer.stop()

            if self.install_error:
                self.complete_page.set_failure(self.install_error)
            else:
                config = self.config_page.get_config()
                summary = f"""
Installation Path: {config.install_path}
VRAM Cache: {'Enabled' if config.vram_cache_enabled else 'Disabled'} ({config.l1_cache_size_gb} GB)
RAM Cache: {'Enabled' if config.ram_cache_enabled else 'Disabled'} ({config.l2_cache_size_gb} GB)
Master DB: {config.master_db_path or 'Not configured'}
Repositories: {len(config.repositories)} configured
                """
                self.complete_page.set_success(summary)

            self.pages.setCurrentIndex(4)
            self._update_navigation()

    def _cancel_installation(self):
        """Cancel the installation"""
        reply = QMessageBox.question(
            self,
            "Cancel Installation",
            "Are you sure you want to cancel the installation?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.close()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(37, 37, 37))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(204, 204, 204))
    palette.setColor(QPalette.ColorRole.Base, QColor(26, 26, 26))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(204, 204, 204))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(204, 204, 204))
    palette.setColor(QPalette.ColorRole.Text, QColor(204, 204, 204))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(204, 204, 204))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 212, 255))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    window = VBUSInstaller()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
