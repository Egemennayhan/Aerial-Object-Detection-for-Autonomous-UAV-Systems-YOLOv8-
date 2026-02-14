from typing import Any, Dict
import os
from PyQt5 import QtCore, QtWidgets


class ControlPanel(QtWidgets.QWidget):
    sig_model_changed = QtCore.pyqtSignal(str)
    sig_source_changed = QtCore.pyqtSignal(str)
    sig_latency_changed = QtCore.pyqtSignal(bool)
    sig_speed_mode_changed = QtCore.pyqtSignal(str) # "realtime" or "max"

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

        gb_speed = QtWidgets.QGroupBox("Playback Speed Mode")
        layout.addWidget(gb_speed)
        sp_layout = QtWidgets.QHBoxLayout(gb_speed)
        self.cmb_speed_mode = QtWidgets.QComboBox()
        self.cmb_speed_mode.addItem("Real-Time (Fixed FPS)", "realtime")
        self.cmb_speed_mode.addItem("Max Speed (Processing)", "max")
        sp_layout.addWidget(self.cmb_speed_mode)

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
        self.cmb_speed_mode.currentIndexChanged.connect(self.on_speed_mode_changed)

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

    def on_speed_mode_changed(self, idx: int):
        mode = self.cmb_speed_mode.currentData()
        self.sig_speed_mode_changed.emit(mode)
        self.set_status(f"Speed mode: {self.cmb_speed_mode.currentText()}")
