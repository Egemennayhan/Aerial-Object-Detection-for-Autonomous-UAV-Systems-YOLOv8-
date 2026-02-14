#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ts="$(date +%Y%m%d_%H%M%S)"
echo "[*] Backup -> src/dashboard/_bak/$ts"
mkdir -p "src/dashboard/_bak/$ts"
cp -f src/dashboard/*.py "src/dashboard/_bak/$ts/" 2>/dev/null || true

echo "[*] Write src/dashboard/app_window.py"
cat > src/dashboard/app_window.py << 'PY'
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

        # Bottom dock: Tabs
        self.tabs_bottom = QtWidgets.QTabWidget()
        self.odometry = OdometryPanel(cfg)
        self.position = PositionPanel(cfg)
        self.tabs_bottom.addTab(self.odometry, "Odometry")
        self.tabs_bottom.addTab(self.position, "Position")

        self.dock_bottom = QtWidgets.QDockWidget("Odometry / Position", self)
        self.dock_bottom.setWidget(self.tabs_bottom)
        self.dock_bottom.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.TopDockWidgetArea)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.dock_bottom)

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
            self.tabs_bottom.tabBar().setTabVisible(0, visible)
            self.dock_bottom.setVisible(self.tabs_bottom.tabBar().isTabVisible(0) or self.tabs_bottom.tabBar().isTabVisible(1))
        elif key == "pos":
            self.tabs_bottom.tabBar().setTabVisible(1, visible)
            self.dock_bottom.setVisible(self.tabs_bottom.tabBar().isTabVisible(0) or self.tabs_bottom.tabBar().isTabVisible(1))

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
PY

echo "[*] Overwrite panels as QWidget (embeddable)"

cat > src/dashboard/detection_panel.py << 'PY'
from typing import Any, Dict
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


class DetectionPanel(QtWidgets.QWidget):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        layout = QtWidgets.QVBoxLayout(self)

        self.video_label = QtWidgets.QLabel("Waiting for frames...")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        layout.addWidget(self.video_label)

        row = QtWidgets.QHBoxLayout()
        layout.addLayout(row)

        self.lbl_fps = QtWidgets.QLabel("FPS: --")
        self.lbl_ms = QtWidgets.QLabel("ms/frame: --")
        self.lbl_landing = QtWidgets.QLabel("Landing: N/A")
        self.lbl_det = QtWidgets.QLabel("Detections: 0")

        row.addWidget(self.lbl_fps)
        row.addWidget(self.lbl_ms)
        row.addWidget(self.lbl_landing)
        row.addWidget(self.lbl_det)

    def update_view(self, pkt):
        frame = pkt.frame_vis
        self.lbl_fps.setText(f"FPS: {pkt.fps:.1f}")
        self.lbl_ms.setText(f"ms/frame: {pkt.ms_per_frame:.1f}")
        self.lbl_landing.setText(f"Landing: {pkt.landing_status}")
        self.lbl_det.setText(f"Detections: {len(pkt.detections)}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )
PY

cat > src/dashboard/metrics_panel.py << 'PY'
from typing import Any, Dict, List
import time
from PyQt5 import QtWidgets
import pyqtgraph as pg


class MetricsPanel(QtWidgets.QWidget):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        layout = QtWidgets.QVBoxLayout(self)

        self.plot = pg.PlotWidget(title="mAP@0.5 (GT yoksa N/A) / Placeholder: ms/frame")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot([], [])
        layout.addWidget(self.plot)

        self.tbl = QtWidgets.QTableWidget(4, 2)
        self.tbl.setHorizontalHeaderLabels(["Class", "IoU"])
        self.tbl.verticalHeader().setVisible(False)
        for i, name in enumerate(["Tasit(0)", "Insan(1)", "UAP(2)", "UAI(3)"]):
            self.tbl.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.tbl.setItem(i, 1, QtWidgets.QTableWidgetItem("N/A"))
        self.tbl.setMaximumHeight(160)
        layout.addWidget(self.tbl)

        self.lbl_speed = QtWidgets.QLabel("ms/frame: --")
        layout.addWidget(self.lbl_speed)

        self.max_points = int(cfg["metrics"]["rolling_points"])
        self.x: List[float] = []
        self.y: List[float] = []
        self.t0 = time.time()

    def update_view(self, pkt):
        self.lbl_speed.setText(f"ms/frame: {pkt.ms_per_frame:.1f}")
        t = time.time() - self.t0
        self.x.append(t)
        self.y.append(float(pkt.ms_per_frame))
        if len(self.x) > self.max_points:
            self.x = self.x[-self.max_points:]
            self.y = self.y[-self.max_points:]
        self.curve.setData(self.x, self.y)
PY

cat > src/dashboard/odometry_panel.py << 'PY'
from typing import Any, Dict
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


class OdometryPanel(QtWidgets.QWidget):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        layout = QtWidgets.QVBoxLayout(self)

        self.flow_label = QtWidgets.QLabel("Optical flow waiting...")
        self.flow_label.setAlignment(QtCore.Qt.AlignCenter)
        self.flow_label.setMinimumSize(640, 360)
        layout.addWidget(self.flow_label)

        self.traj_plot = pg.PlotWidget(title="Top-Down Trajectory (relative)")
        self.traj_plot.showGrid(x=True, y=True, alpha=0.3)
        self.traj_curve = self.traj_plot.plot([], [])
        layout.addWidget(self.traj_plot)

        self.lbl_vel = QtWidgets.QLabel("Velocity (px/frame): --")
        layout.addWidget(self.lbl_vel)

    def update_view(self, pkt):
        if pkt.flow_vis is not None:
            rgb = cv2.cvtColor(pkt.flow_vis, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)
            self.flow_label.setPixmap(
                pix.scaled(self.flow_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            )

        if pkt.traj_2d is not None and len(pkt.traj_2d) > 1:
            self.traj_curve.setData(pkt.traj_2d[:, 0], pkt.traj_2d[:, 1])

        if pkt.vel_2d is not None:
            self.lbl_vel.setText(f"Velocity (px/frame): vx={pkt.vel_2d[0]:.2f}, vy={pkt.vel_2d[1]:.2f}")
PY

cat > src/dashboard/position_panel.py << 'PY'
from typing import Any, Dict, List
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg

try:
    import pyqtgraph.opengl as gl
    HAS_GL = True
except Exception:
    HAS_GL = False


class PositionPanel(QtWidgets.QWidget):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        layout = QtWidgets.QVBoxLayout(self)

        self.plot_xyz = pg.PlotWidget(title="X, Y, Z (estimated) - real-time")
        self.plot_xyz.showGrid(x=True, y=True, alpha=0.3)
        self.curve_x = self.plot_xyz.plot([], [])
        self.curve_y = self.plot_xyz.plot([], [])
        self.curve_z = self.plot_xyz.plot([], [])
        layout.addWidget(self.plot_xyz)

        if HAS_GL:
            self.gl_view = gl.GLViewWidget()
            self.gl_view.setMinimumHeight(260)
            layout.addWidget(self.gl_view)
            self.gl_line = gl.GLLinePlotItem(pos=np.zeros((1, 3)), width=2, antialias=True)
            self.gl_view.addItem(self.gl_line)
        else:
            self.gl_view = None
            self.gl_line = None
            layout.addWidget(QtWidgets.QLabel("3D view unavailable (pyqtgraph.opengl missing)."))

        row = QtWidgets.QHBoxLayout()
        layout.addLayout(row)
        self.lbl_rmse = QtWidgets.QLabel("RMSE: N/A")
        self.lbl_gps = QtWidgets.QLabel("GPS health: NO_GPS")
        row.addWidget(self.lbl_rmse)
        row.addWidget(self.lbl_gps)

        self.max_points = int(cfg["position"]["rolling_points"])
        self.t = 0
        self.ts: List[int] = []
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.zs: List[float] = []
        self.traj3: List[np.ndarray] = []

    def update_view(self, pkt):
        if pkt.pos_xyz is None:
            return

        self.t += 1
        self.ts.append(self.t)
        self.xs.append(float(pkt.pos_xyz[0]))
        self.ys.append(float(pkt.pos_xyz[1]))
        self.zs.append(float(pkt.pos_xyz[2]))

        if len(self.ts) > self.max_points:
            self.ts = self.ts[-self.max_points:]
            self.xs = self.xs[-self.max_points:]
            self.ys = self.ys[-self.max_points:]
            self.zs = self.zs[-self.max_points:]

        self.curve_x.setData(self.ts, self.xs)
        self.curve_y.setData(self.ts, self.ys)
        self.curve_z.setData(self.ts, self.zs)

        self.lbl_gps.setText(f"GPS health: {pkt.gps_health}")
        self.lbl_rmse.setText(f"RMSE: {pkt.rmse:.3f}" if pkt.rmse is not None else "RMSE: N/A")

        self.traj3.append(pkt.pos_xyz.astype(np.float32))
        if len(self.traj3) > 2000:
            self.traj3 = self.traj3[-2000:]

        if self.gl_line is not None:
            pos = np.stack(self.traj3, axis=0)
            self.gl_line.setData(pos=pos)
PY

# Control panel: keep existing logic but convert to QWidget layout root
cat > src/dashboard/control_panel.py << 'PY'
from typing import Any, Dict
import os
from PyQt5 import QtCore, QtWidgets


class ControlPanel(QtWidgets.QWidget):
    sig_model_changed = QtCore.pyqtSignal(str)
    sig_source_changed = QtCore.pyqtSignal(str)
    sig_latency_changed = QtCore.pyqtSignal(bool)

    sig_toggle_detection = QtCore.pyqtSignal(bool)
    sig_toggle_metrics = QtCore.pyqtSignal(bool)
    sig_toggle_odometry = QtCore.pyqtSignal(bool)
    sig_toggle_position = QtCore.pyqtSignal(bool)

    sig_pause = QtCore.pyqtSignal()
    sig_step = QtCore.pyqtSignal()
    sig_screenshot = QtCore.pyqtSignal()
    sig_quit = QtCore.pyqtSignal()

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        layout = QtWidgets.QVBoxLayout(self)

        gb_model = QtWidgets.QGroupBox("Model (YOLO checkpoint)")
        layout.addWidget(gb_model)
        g = QtWidgets.QGridLayout(gb_model)

        self.cmb_model = QtWidgets.QComboBox()
        for p in cfg["ui"]["model_presets"]:
            self.cmb_model.addItem(p)

        cur = cfg["detection"]["model_path"]
        if cur and cur in cfg["ui"]["model_presets"]:
            self.cmb_model.setCurrentText(cur)
        else:
            if cur:
                self.cmb_model.insertItem(0, cur)
                self.cmb_model.setCurrentIndex(0)

        self.btn_browse_model = QtWidgets.QPushButton("Browse...")
        self.btn_reload_model = QtWidgets.QPushButton("Hot Reload")

        g.addWidget(QtWidgets.QLabel("Checkpoint:"), 0, 0)
        g.addWidget(self.cmb_model, 0, 1, 1, 2)
        g.addWidget(self.btn_browse_model, 1, 1)
        g.addWidget(self.btn_reload_model, 1, 2)

        gb_src = QtWidgets.QGroupBox("Video Source")
        layout.addWidget(gb_src)
        s = QtWidgets.QGridLayout(gb_src)

        self.ed_source = QtWidgets.QLineEdit(str(cfg["video"]["source"]))
        self.btn_browse_video = QtWidgets.QPushButton("Browse Video...")
        self.btn_apply_source = QtWidgets.QPushButton("Apply Source")

        s.addWidget(QtWidgets.QLabel("Source (0 / path / rtsp://):"), 0, 0)
        s.addWidget(self.ed_source, 0, 1, 1, 2)
        s.addWidget(self.btn_browse_video, 1, 1)
        s.addWidget(self.btn_apply_source, 1, 2)

        gb_lat = QtWidgets.QGroupBox("RTSP Low-Latency / Buffering")
        layout.addWidget(gb_lat)
        l = QtWidgets.QHBoxLayout(gb_lat)

        self.chk_low_latency = QtWidgets.QCheckBox("Enable low-latency capture (drop frames)")
        self.chk_low_latency.setChecked(bool(cfg["video"]["low_latency"]))
        l.addWidget(self.chk_low_latency)

        gb_win = QtWidgets.QGroupBox("Panels (toggle)")
        layout.addWidget(gb_win)
        t = QtWidgets.QGridLayout(gb_win)

        self.chk_det = QtWidgets.QCheckBox("Detection")
        self.chk_met = QtWidgets.QCheckBox("Metrics")
        self.chk_odo = QtWidgets.QCheckBox("Odometry")
        self.chk_pos = QtWidgets.QCheckBox("Position")

        self.chk_det.setChecked(bool(cfg["ui"]["show_detection"]))
        self.chk_met.setChecked(bool(cfg["ui"]["show_metrics"]))
        self.chk_odo.setChecked(bool(cfg["ui"]["show_odometry"]))
        self.chk_pos.setChecked(bool(cfg["ui"]["show_position"]))

        t.addWidget(self.chk_det, 0, 0)
        t.addWidget(self.chk_met, 0, 1)
        t.addWidget(self.chk_odo, 1, 0)
        t.addWidget(self.chk_pos, 1, 1)

        gb_ctl = QtWidgets.QGroupBox("Playback / Debug")
        layout.addWidget(gb_ctl)
        c = QtWidgets.QHBoxLayout(gb_ctl)

        self.btn_pause = QtWidgets.QPushButton("Pause/Resume (Space)")
        self.btn_step = QtWidgets.QPushButton("Step (N)")
        self.btn_ss = QtWidgets.QPushButton("Screenshot (S)")
        self.btn_quit = QtWidgets.QPushButton("Quit (Q)")

        c.addWidget(self.btn_pause)
        c.addWidget(self.btn_step)
        c.addWidget(self.btn_ss)
        c.addWidget(self.btn_quit)

        self.lbl_status = QtWidgets.QLabel("Status: Ready")
        layout.addWidget(self.lbl_status)
        layout.addStretch(1)

        # wires
        self.btn_browse_model.clicked.connect(self.on_browse_model)
        self.btn_reload_model.clicked.connect(self.on_reload_model)

        self.btn_browse_video.clicked.connect(self.on_browse_video)
        self.btn_apply_source.clicked.connect(self.on_apply_source)

        self.chk_low_latency.toggled.connect(self.on_latency_toggled)

        self.chk_det.toggled.connect(self.sig_toggle_detection.emit)
        self.chk_met.toggled.connect(self.sig_toggle_metrics.emit)
        self.chk_odo.toggled.connect(self.sig_toggle_odometry.emit)
        self.chk_pos.toggled.connect(self.sig_toggle_position.emit)

        self.btn_pause.clicked.connect(self.sig_pause.emit)
        self.btn_step.clicked.connect(self.sig_step.emit)
        self.btn_ss.clicked.connect(self.sig_screenshot.emit)
        self.btn_quit.clicked.connect(self.sig_quit.emit)

    def set_status(self, msg: str):
        self.lbl_status.setText(f"Status: {msg}")

    def on_reload_model(self):
        path = self.cmb_model.currentText().strip()
        if not path:
            self.set_status("Model path empty.")
            return
        self.sig_model_changed.emit(path)
        self.set_status(f"Hot reloading model: {path}")

    def on_browse_model(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select YOLO .pt model", "", "PyTorch Weights (*.pt)")
        if fn:
            if fn not in [self.cmb_model.itemText(i) for i in range(self.cmb_model.count())]:
                self.cmb_model.insertItem(0, fn)
                self.cmb_model.setCurrentIndex(0)
            else:
                self.cmb_model.setCurrentText(fn)
            self.set_status(f"Model file picked: {fn}")

    def on_browse_video(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select video", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)")
        if fn:
            self.ed_source.setText(fn)
            self.set_status(f"Video picked: {os.path.basename(fn)}")

    def on_apply_source(self):
        src = self.ed_source.text().strip()
        if src == "":
            self.set_status("Source empty.")
            return
        self.sig_source_changed.emit(src)
        self.set_status(f"Applying source: {src}")

    def on_latency_toggled(self, enabled: bool):
        self.sig_latency_changed.emit(enabled)
        self.set_status(f"Low-latency: {'ON' if enabled else 'OFF'}")
PY

echo "[*] Patch src/dashboard/main.py to use AppWindow (best-effort)"
python - << 'PY'
from pathlib import Path

p = Path("src/dashboard/main.py")
txt = p.read_text(encoding="utf-8")

# 1) import AppWindow
if "from src.dashboard.app_window import AppWindow" not in txt:
    if "from src.dashboard.control_panel import ControlPanel" in txt:
        txt = txt.replace("from src.dashboard.control_panel import ControlPanel\n",
                          "from src.dashboard.control_panel import ControlPanel\nfrom src.dashboard.app_window import AppWindow\n")

# 2) replace multi-window creation with AppWindow (pattern-based)
needle = "        self.panels: Dict[str, Any] = {}\n        self.create_panels_initial()\n\n        self.ctrl_panel = ControlPanel(cfg)\n        self.ctrl_panel.show()\n"
if needle in txt:
    txt = txt.replace(needle, "        self.window = AppWindow(cfg)\n        self.ctrl_panel = self.window.control\n        self.window.show()\n")

# 3) route updates to window panels (pattern-based)
block = '        if "det" in self.panels:\n            self.panels["det"].update_view(pkt)\n        if "met" in self.panels:\n            self.panels["met"].update_view(pkt)\n        if "odo" in self.panels:\n            self.panels["odo"].update_view(pkt)\n        if "pos" in self.panels:\n            self.panels["pos"].update_view(pkt)\n'
if block in txt:
    txt = txt.replace(block,
                      "        self.window.detection.update_view(pkt)\n"
                      "        self.window.metrics.update_view(pkt)\n"
                      "        self.window.odometry.update_view(pkt)\n"
                      "        self.window.position.update_view(pkt)\n")

# 4) toggle_panel route to AppWindow (safe insert)
if "def toggle_panel(self, key: str, enabled: bool):" in txt and "self.window.set_panel_visible" not in txt:
    txt = txt.replace("def toggle_panel(self, key: str, enabled: bool):",
                      "def toggle_panel(self, key: str, enabled: bool):\n        self.window.set_panel_visible(key, enabled)\n        return\n\n    def _toggle_panel_old(self, key: str, enabled: bool):")

# 5) screenshot: grab app window (pattern-based)
shot = "        pix = self.ctrl_panel.grab()\n        pix.save(f\"{out_dir}/control_{ts}.png\")\n        for name, p in self.panels.items():\n            pix = p.grab()\n            pix.save(f\"{out_dir}/{name}_{ts}.png\")\n"
if shot in txt:
    txt = txt.replace(shot,
                      "        pix = self.window.grab()\n        pix.save(f\"{out_dir}/app_{ts}.png\")\n"
                      "        pix = self.ctrl_panel.grab()\n        pix.save(f\"{out_dir}/control_{ts}.png\")\n")

# 6) quit: close window (pattern-based)
q = "        for p in list(self.panels.values()):\n            p.close()\n        self.ctrl_panel.close()\n"
if q in txt:
    txt = txt.replace(q, "        self.window.close()\n")

p.write_text(txt, encoding="utf-8")
print("main.py patched (pattern-based). If it doesn't compile, we'll fix manually using the backup.")
PY

echo "[*] Done. Run: python scripts/run_dashboard.py"
