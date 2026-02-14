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
        frame = pkt.frame_vis.copy()
        
        # --- Professional HUD Overlay ---
        h, w = frame.shape[:2]
        color = (0, 255, 0) # Green
        
        # Center Crosshair
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), color, 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), color, 1)
        
        # Corner Brackets
        length = 40
        # Top-Left
        cv2.line(frame, (20, 20), (20 + length, 20), color, 2)
        cv2.line(frame, (20, 20), (20, 20 + length), color, 2)
        # Top-Right
        cv2.line(frame, (w - 20, 20), (w - 20 - length, 20), color, 2)
        cv2.line(frame, (w - 20, 20), (w - 20, 20 + length), color, 2)
        # Bottom-Left
        cv2.line(frame, (20, h - 20), (20 + length, h - 20), color, 2)
        cv2.line(frame, (20, h - 20), (20, h - 20 - length), color, 2)
        # Bottom-Right
        cv2.line(frame, (w - 20, h - 20), (w - 20 - length, h - 20), color, 2)
        cv2.line(frame, (w - 20, h - 20), (w - 20, h - 20 - length), color, 2)

        # Telemetry Text Overlay (Top Left)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"MISSION: TEKNOFEST 2026", (40, 50), font, 0.7, color, 2)
        cv2.putText(frame, f"STATUS: {pkt.landing_status}", (40, 80), font, 0.6, color, 1)
        
        # Telemetry Text Overlay (Bottom Left)
        cv2.putText(frame, f"FPS: {pkt.fps:.1f}", (40, h - 50), font, 0.6, color, 1)
        cv2.putText(frame, f"LATENCY: {pkt.ms_per_frame:.1f}ms", (40, h - 80), font, 0.6, color, 1)

        # Update Labels
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
