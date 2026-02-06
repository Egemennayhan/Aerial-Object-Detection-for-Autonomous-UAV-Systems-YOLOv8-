from typing import Any, Dict, List
import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg

try:
    import pyqtgraph.opengl as gl
    HAS_GL = True
except Exception:
    HAS_GL = False


class PositionPanel(QtWidgets.QMainWindow):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("EKRAN 4 - Position Tracking")

        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QVBoxLayout(w)

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
        self.lbl_ref = QtWidgets.QLabel("Ref vs Pred: N/A (ref yok)")
        row.addWidget(self.lbl_rmse)
        row.addWidget(self.lbl_gps)
        row.addWidget(self.lbl_ref)

        self.max_points = int(cfg["position"]["rolling_points"])
        self.t = 0
        self.ts: List[int] = []
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.zs: List[float] = []
        self.traj3: List[np.ndarray] = []

        geo = cfg["ui"]["windows"].get("position")
        if geo:
            self.setGeometry(*geo)

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

        if pkt.ref_xyz is not None:
            self.lbl_ref.setText(f"Ref vs Pred: ref={pkt.ref_xyz}, pred={pkt.pos_xyz}")
        else:
            self.lbl_ref.setText("Ref vs Pred: N/A (ref yok)")

        self.traj3.append(pkt.pos_xyz.astype(np.float32))
        if len(self.traj3) > 2000:
            self.traj3 = self.traj3[-2000:]

        if self.gl_line is not None:
            pos = np.stack(self.traj3, axis=0)
            self.gl_line.setData(pos=pos)
