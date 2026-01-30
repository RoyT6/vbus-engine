"""
VBUS Installer Build Script
Creates a standalone .exe from the VBUS installer

Requirements:
    pip install pyinstaller

Usage:
    python build_exe.py

This will create:
    dist/VBUS_Installer.exe - Standalone installer executable
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        import PyInstaller
        return True
    except ImportError:
        return False


def install_pyinstaller():
    """Install PyInstaller"""
    print("Installing PyInstaller...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)


def create_spec_file(script_path: Path, output_dir: Path) -> Path:
    """Create a PyInstaller spec file for more control"""
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    [r'{script_path}'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'PIL',
        'cv2',
        'torch',
        'tensorflow',
        'keras',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VBUS_Installer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
    version=r'{output_dir / "version_info.txt"}',
)
'''

    spec_path = output_dir / "VBUS_Installer.spec"
    spec_path.write_text(spec_content)
    return spec_path


def create_version_info(output_dir: Path):
    """Create version info file for Windows exe"""
    version_info = '''# UTF-8
#
# For more details about fixed file info 'ffi' see:
# http://msdn.microsoft.com/en-us/library/ms646997.aspx

VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
    ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'VBUS'),
        StringStruct(u'FileDescription', u'VBUS System Installer'),
        StringStruct(u'FileVersion', u'1.0.0.0'),
        StringStruct(u'InternalName', u'VBUS_Installer'),
        StringStruct(u'LegalCopyright', u'Copyright (c) 2024'),
        StringStruct(u'OriginalFilename', u'VBUS_Installer.exe'),
        StringStruct(u'ProductName', u'VBUS'),
        StringStruct(u'ProductVersion', u'1.0.0.0')])
      ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
    version_path = output_dir / "version_info.txt"
    version_path.write_text(version_info)
    return version_path


def build_exe():
    """Build the executable"""
    print("="*60)
    print("VBUS Installer - EXE Builder")
    print("="*60)
    print()

    # Get paths
    script_dir = Path(__file__).parent
    script_path = script_dir / "vbus_installer.py"
    output_dir = script_dir / "build_output"

    if not script_path.exists():
        print(f"Error: Source file not found: {script_path}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Check/install PyInstaller
    if not check_pyinstaller():
        print("PyInstaller not found.")
        response = input("Install PyInstaller? (y/n): ")
        if response.lower() == 'y':
            install_pyinstaller()
        else:
            print("PyInstaller is required to build the exe.")
            sys.exit(1)

    print(f"\nSource: {script_path}")
    print(f"Output: {output_dir}")
    print()

    # Create version info
    print("Creating version info...")
    create_version_info(output_dir)

    # Create spec file
    print("Creating spec file...")
    spec_path = create_spec_file(script_path, output_dir)

    # Run PyInstaller
    print("\nBuilding executable (this may take a few minutes)...")
    print("-"*60)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "--distpath", str(output_dir / "dist"),
        "--workpath", str(output_dir / "build"),
        "--specpath", str(output_dir),
        str(spec_path)
    ]

    result = subprocess.run(cmd, cwd=str(script_dir))

    if result.returncode == 0:
        exe_path = output_dir / "dist" / "VBUS_Installer.exe"
        if exe_path.exists():
            # Copy to main directory
            final_path = script_dir / "VBUS_Installer.exe"
            shutil.copy2(exe_path, final_path)

            print()
            print("="*60)
            print("BUILD SUCCESSFUL!")
            print("="*60)
            print(f"\nExecutable created: {final_path}")
            print(f"Size: {final_path.stat().st_size / (1024*1024):.1f} MB")
            print("\nYou can now distribute VBUS_Installer.exe")
        else:
            print("\nError: Executable not found in expected location")
            sys.exit(1)
    else:
        print("\nBuild failed!")
        sys.exit(1)


def quick_build():
    """Quick build using simple PyInstaller command"""
    print("="*60)
    print("VBUS Installer - Quick EXE Builder")
    print("="*60)
    print()

    script_dir = Path(__file__).parent
    script_path = script_dir / "vbus_installer.py"

    if not check_pyinstaller():
        print("Installing PyInstaller...")
        install_pyinstaller()

    print("Building executable...")
    print("-"*60)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name", "VBUS_Installer",
        "--clean",
        str(script_path)
    ]

    result = subprocess.run(cmd, cwd=str(script_dir))

    if result.returncode == 0:
        print()
        print("="*60)
        print("BUILD SUCCESSFUL!")
        print("="*60)
        exe_path = script_dir / "dist" / "VBUS_Installer.exe"
        print(f"\nExecutable created: {exe_path}")
    else:
        print("\nBuild failed!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_build()
    else:
        build_exe()
