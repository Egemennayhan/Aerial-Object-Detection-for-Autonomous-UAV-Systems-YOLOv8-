import os
from PyQt5.QtCore import QLibraryInfo, QCoreApplication

def force_qt_plugins(platform: str = "xcb"):
    # Ensure we use xcb (normal GUI)
    os.environ.setdefault("QT_QPA_PLATFORM", platform)

    # Force Qt plugin paths to PyQt5's own Qt plugins dir
    plugins_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ["QT_PLUGIN_PATH"] = plugins_path
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins_path

    # Also force Qt's internal library path list (extra reliable)
    try:
        QCoreApplication.setLibraryPaths([plugins_path])
    except Exception:
        pass
