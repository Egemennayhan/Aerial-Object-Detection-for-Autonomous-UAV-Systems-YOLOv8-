from typing import Any, Dict
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


class OdometryPanel(QtWidgets.QMainWindow):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("EKRAN 3 - Visual Odometry")

        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QVBoxLayout(w)

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

        geo = cfg["ui"]["windows"].get("odometry")
        if geo:
            self.setGeometry(*geo)

    def update_view(self, pkt):
        if pkt.flow_vis is not None:
            rgb = cv2.cvtColor(pkt.flow_vis, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)
            self.flow_label.setPixmap(pix.scaled(
                self.flow_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            ))

        if pkt.traj_2d is not None and len(pkt.traj_2d) > 1:
            x = pkt.traj_2d[:, 0]
            y = pkt.traj_2d[:, 1]
            self.traj_curve.setData(x, y)

        if pkt.vel_2d is not None:
            self.lbl_vel.setText(f"Velocity (px/frame): vx={pkt.vel_2d[0]:.2f}, vy={pkt.vel_2d[1]:.2f}")
