from typing import Any, Dict
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


class DetectionPanel(QtWidgets.QMainWindow):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("EKRAN 1 - Object Detection")

        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QVBoxLayout(w)

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

        geo = cfg["ui"]["windows"].get("detection")
        if geo:
            self.setGeometry(*geo)

    def update_view(self, pkt):
        frame = pkt.frame_vis
        self.lbl_fps.setText(f"FPS: {pkt.fps:.1f}")
        self.lbl_ms.setText(f"ms/frame: {pkt.ms_per_frame:.1f}")
        self.lbl_landing.setText(f"Landing: {pkt.landing_status}")
        self.lbl_det.setText(f"Detections: {len(pkt.detections)}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))
