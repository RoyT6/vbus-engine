# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['install.py'],
    pathex=[],
    binaries=[],
    datas=[('VBUS.json', '.'), ('vbus_core.py', '.'), ('vbus_engines.py', '.'), ('vbus_pipeline.py', '.'), ('vbus_monitor.py', '.'), ('vbus_gpu_manager.py', '.'), ('vbus_ml_executor.py', '.'), ('VBus System.md', '.'), ('Framecore_Blue 2.png', '.')],
    hiddenimports=['PIL', 'PIL.Image', 'PIL.ImageTk'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='VBUS_Setup',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
