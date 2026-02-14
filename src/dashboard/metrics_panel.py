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
