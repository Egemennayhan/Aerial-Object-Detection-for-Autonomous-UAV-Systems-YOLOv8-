import os
import sys

# --- Force Qt to use PyQt5's own plugins, NOT OpenCV's bundled Qt plugins ---
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

# Pick xcb (normal GUI)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# Import PyQt5 FIRST to get the correct plugins directory
from PyQt5.QtCore import QLibraryInfo

plugins_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)

# Hard set plugin paths (prevents cv2 from hijacking to .../cv2/qt/plugins)
os.environ["QT_PLUGIN_PATH"] = plugins_path
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins_path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.dashboard.main import main

if __name__ == "__main__":
    main()
