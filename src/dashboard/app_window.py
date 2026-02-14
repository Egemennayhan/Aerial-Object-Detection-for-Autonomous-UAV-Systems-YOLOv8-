from typing import Any, Dict
from PyQt5 import QtCore, QtWidgets

from src.dashboard.control_panel import ControlPanel
from src.dashboard.detection_panel import DetectionPanel
from src.dashboard.metrics_panel import MetricsPanel
from src.dashboard.odometry_panel import OdometryPanel
from src.dashboard.position_panel import PositionPanel


class AppWindow(QtWidgets.QMainWindow):
    """
    Single-window "app-like" dashboard:
    - Center: Detection (video)
    - Left dock: Control
    - Right dock: Metrics
    - Bottom dock: Tabs (Odometry, Position)
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("TEKNOFEST Dashboard")
        self.resize(1700, 950)

        # Center: Detection (stack so we can hide)
        self._central_stack = QtWidgets.QStackedWidget()
        self.detection = DetectionPanel(cfg)
        self._hidden = QtWidgets.QLabel("Detection hidden")
        self._hidden.setAlignment(QtCore.Qt.AlignCenter)
        self._central_stack.addWidget(self.detection)
        self._central_stack.addWidget(self._hidden)
        self.setCentralWidget(self._central_stack)

        # Left dock: Control
        self.control = ControlPanel(cfg)
        self.dock_control = self._dock("Control", self.control, QtCore.Qt.LeftDockWidgetArea)

        # Right dock: Metrics
        self.metrics = MetricsPanel(cfg)
        self.dock_metrics = self._dock("Metrics", self.metrics, QtCore.Qt.RightDockWidgetArea)

        # Odometry dock
        self.odometry = OdometryPanel(cfg)
        self.dock_odometry = self._dock("Odometry", self.odometry, QtCore.Qt.BottomDockWidgetArea)

        # Position dock
        self.position = PositionPanel(cfg)
        self.dock_position = self._dock("Position", self.position, QtCore.Qt.BottomDockWidgetArea)

        # App-like docking behavior
        self.setDockOptions(
            QtWidgets.QMainWindow.AllowTabbedDocks |
            QtWidgets.QMainWindow.AllowNestedDocks |
            QtWidgets.QMainWindow.AnimatedDocks
        )
        self.statusBar().showMessage("Ready")

        # Persist layout
        self._settings = QtCore.QSettings("teknofest", "dashboard")
        self._restore_layout()

    def _dock(self, title: str, widget: QtWidgets.QWidget, area):
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(title)
        dock.setWidget(widget)
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea |
            QtCore.Qt.RightDockWidgetArea |
            QtCore.Qt.BottomDockWidgetArea |
            QtCore.Qt.TopDockWidgetArea
        )
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(area, dock)
        return dock

    def set_panel_visible(self, key: str, visible: bool):
        if key == "det":
            self._central_stack.setCurrentIndex(0 if visible else 1)
        elif key == "met":
            self.dock_metrics.setVisible(visible)
        elif key == "odo":
            self.dock_odometry.setVisible(visible)
        elif key == "pos":
            self.dock_position.setVisible(visible)

    def closeEvent(self, event):
        self._save_layout()
        super().closeEvent(event)

    def _save_layout(self):
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("state", self.saveState())

    def _restore_layout(self):
        g = self._settings.value("geometry")
        s = self._settings.value("state")
        if g is not None:
            self.restoreGeometry(g)
        if s is not None:
            self.restoreState(s)
