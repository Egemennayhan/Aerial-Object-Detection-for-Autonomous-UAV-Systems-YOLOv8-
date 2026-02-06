import os
import sys
import time
import queue
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from src.dashboard.qt_bootstrap import force_qt_plugins

from src.dashboard.utils import (
    load_config, ensure_dirs, now_ts, safe_put_latest, FPSCounter,
    VideoSource, CsvLogger, AnomalyMonitor, try_load_yolo_model
)
from src.dashboard.detection_panel import DetectionPanel
from src.dashboard.metrics_panel import MetricsPanel
from src.dashboard.odometry_panel import OdometryPanel
from src.dashboard.position_panel import PositionPanel
from src.dashboard.control_panel import ControlPanel


@dataclass
class Detection:
    cls: int
    conf: float
    xyxy: Tuple[int, int, int, int]


@dataclass
class FramePacket:
    frame_id: int
    t_capture: float
    frame_bgr: np.ndarray


@dataclass
class OutputPacket:
    frame_id: int
    t_capture: float
    t_done: float
    frame_vis: np.ndarray
    detections: List[Detection]
    landing_status: str
    fps: float
    ms_per_frame: float

    flow_vis: Optional[np.ndarray]
    traj_2d: Optional[np.ndarray]
    vel_2d: Optional[np.ndarray]

    pos_xyz: Optional[np.ndarray]
    ref_xyz: Optional[np.ndarray]
    rmse: Optional[float]
    gps_health: str


class UiBus(QtCore.QObject):
    updated = QtCore.pyqtSignal(object)


class PipelineWorker(threading.Thread):
    def __init__(
        self,
        cfg: Dict[str, Any],
        in_q: "queue.Queue[FramePacket]",
        out_q: "queue.Queue[OutputPacket]",
        control: Dict[str, Any],
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.in_q = in_q
        self.out_q = out_q
        self.control = control
        self.stop_event = stop_event

        self.fps_counter = FPSCounter(rolling=30)
        self.model = None
        self.model_path_loaded = None
        self.model_lock = threading.Lock()

        self.prev_gray = None
        self.prev_pts = None
        self.traj = []
        self.pos_xyz = np.zeros(3, dtype=np.float32)

        self.csv_logger = CsvLogger(cfg["logging"]["csv_path"])
        self.anomaly = AnomalyMonitor(
            fps_low=cfg["anomaly"]["fps_low_threshold"],
            rmse_high=cfg["anomaly"]["rmse_high_threshold"],
        )

    def reload_model_if_needed(self):
        desired = self.control.get("model_path")
        if not desired:
            return
        if desired == self.model_path_loaded and self.model is not None:
            return

        with self.model_lock:
            self.model = try_load_yolo_model(desired, device=self.cfg["detection"]["device"])
            self.model_path_loaded = desired

    def run_detection(self, frame_bgr: np.ndarray) -> List[Detection]:
        if self.model is None or not self.cfg["detection"]["enabled"]:
            return []

        with self.model_lock:
            results = self.model.predict(
                source=frame_bgr,
                conf=self.cfg["detection"]["conf_thres"],
                iou=self.cfg["detection"]["iou_thres"],
                verbose=False,
                device=self.cfg["detection"]["device"],
                imgsz=self.cfg["detection"]["imgsz"],
            )

        dets: List[Detection] = []
        if not results:
            return dets

        r0 = results[0]
        if r0.boxes is None:
            return dets

        boxes = r0.boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy().astype(float)
        cls = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            dets.append(Detection(cls=int(k), conf=float(c), xyxy=(int(x1), int(y1), int(x2), int(y2))))
        return dets

    def landing_status(self, dets: List[Detection]) -> str:
        uap = [d for d in dets if d.cls == 2]
        uai = [d for d in dets if d.cls == 3]
        if not uap and not uai:
            return "N/A"

        def tag(arr, name):
            if not arr:
                return None
            m = max(arr, key=lambda d: d.conf)
            if m.conf >= 0.60:
                return f"{name}:SAFE"
            if m.conf >= 0.35:
                return f"{name}:WARN"
            return f"{name}:LOW"

        parts = [p for p in [tag(uap, "UAP"), tag(uai, "UAI")] if p]
        return " | ".join(parts) if parts else "N/A"

    def draw_detections(self, frame_bgr: np.ndarray, dets: List[Detection]) -> np.ndarray:
        out = frame_bgr.copy()
        names = self.cfg["classes"]["names"]
        for d in dets:
            x1, y1, x2, y2 = d.xyxy
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{names.get(str(d.cls), str(d.cls))} {d.conf:.2f}"
            cv2.putText(out, label, (x1, max(15, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return out

    def run_vo(self, frame_bgr: np.ndarray):
        if not self.cfg["odometry"]["enabled"]:
            return None, None, None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.cfg["odometry"]["max_corners"],
                qualityLevel=self.cfg["odometry"]["quality_level"],
                minDistance=self.cfg["odometry"]["min_distance"],
                blockSize=7,
            )
            self.traj = [np.array([0.0, 0.0], dtype=np.float32)]
            return None, np.array(self.traj, dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)

        if self.prev_pts is None or len(self.prev_pts) < self.cfg["odometry"]["min_features"]:
            self.prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=self.cfg["odometry"]["max_corners"],
                qualityLevel=self.cfg["odometry"]["quality_level"],
                minDistance=self.cfg["odometry"]["min_distance"],
                blockSize=7,
            )

        if self.prev_pts is None:
            self.prev_gray = gray
            return None, np.array(self.traj, dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if next_pts is None or status is None:
            self.prev_gray = gray
            return None, np.array(self.traj, dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)

        good_new = next_pts[status.flatten() == 1]
        good_old = self.prev_pts[status.flatten() == 1]

        flow = (good_new - good_old)
        vel = np.mean(flow, axis=0) if len(flow) > 0 else np.array([0.0, 0.0], dtype=np.float32)

        scale = float(self.cfg["odometry"]["traj_scale"])
        last = self.traj[-1]
        newp = last + vel.astype(np.float32) * scale
        self.traj.append(newp)

        vis = frame_bgr.copy()
        for (n, o) in zip(good_new.astype(int), good_old.astype(int)):
            cv2.line(vis, tuple(o), tuple(n), (0, 255, 255), 2)
            cv2.circle(vis, tuple(n), 2, (0, 0, 255), -1)

        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2)

        return vis, np.array(self.traj, dtype=np.float32), vel.astype(np.float32)

    def run_position(self, vel_2d: Optional[np.ndarray], dt: float):
        if vel_2d is not None:
            gain = float(self.cfg["position"]["integration_gain"])
            self.pos_xyz[0] += float(vel_2d[0]) * dt * gain
            self.pos_xyz[1] += float(vel_2d[1]) * dt * gain

        ref = None
        rmse = None
        gps_health = "NO_GPS" if not self.cfg["position"]["use_gps"] else "UNKNOWN"
        return self.pos_xyz.copy(), ref, rmse, gps_health

    def log_csv(self, pkt: OutputPacket):
        if not self.cfg["logging"]["csv_enable"]:
            return
        self.csv_logger.write_row({
            "ts": pkt.t_done,
            "frame_id": pkt.frame_id,
            "fps": pkt.fps,
            "ms_per_frame": pkt.ms_per_frame,
            "num_det": len(pkt.detections),
            "landing_status": pkt.landing_status,
            "pos_x": float(pkt.pos_xyz[0]) if pkt.pos_xyz is not None else float("nan"),
            "pos_y": float(pkt.pos_xyz[1]) if pkt.pos_xyz is not None else float("nan"),
            "pos_z": float(pkt.pos_xyz[2]) if pkt.pos_xyz is not None else float("nan"),
            "rmse": float(pkt.rmse) if pkt.rmse is not None else float("nan"),
            "gps_health": pkt.gps_health,
        })

    def run(self):
        prev_t = time.time()

        while not self.stop_event.is_set():
            self.reload_model_if_needed()

            if self.control.get("paused", False) and not self.control.get("step_once", False):
                time.sleep(0.01)
                continue

            try:
                pkt_in = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            t0 = time.time()
            dt = t0 - prev_t
            prev_t = t0

            dets = self.run_detection(pkt_in.frame_bgr)
            landing = self.landing_status(dets)
            frame_vis = self.draw_detections(pkt_in.frame_bgr, dets)

            flow_vis, traj_2d, vel_2d = self.run_vo(pkt_in.frame_bgr)
            pos_xyz, ref_xyz, rmse, gps_health = self.run_position(vel_2d, max(1e-3, dt))

            fps = self.fps_counter.tick()
            ms = (time.time() - t0) * 1000.0
            t_done = time.time()

            pkt_out = OutputPacket(
                frame_id=pkt_in.frame_id,
                t_capture=pkt_in.t_capture,
                t_done=t_done,
                frame_vis=frame_vis,
                detections=dets,
                landing_status=landing,
                fps=fps,
                ms_per_frame=ms,
                flow_vis=flow_vis,
                traj_2d=traj_2d,
                vel_2d=vel_2d,
                pos_xyz=pos_xyz,
                ref_xyz=ref_xyz,
                rmse=rmse,
                gps_health=gps_health,
            )

            self.anomaly.update(fps=fps, rmse=rmse)
            self.log_csv(pkt_out)

            safe_put_latest(self.out_q, pkt_out)

            if self.control.get("step_once", False):
                self.control["step_once"] = False


class CaptureWorker(threading.Thread):
    def __init__(
        self,
        cfg: Dict[str, Any],
        out_q: "queue.Queue[FramePacket]",
        control: Dict[str, Any],
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.out_q = out_q
        self.control = control
        self.stop_event = stop_event

        self.frame_id = 0
        self.source = None
        self.source_opened = None
        self.low_latency_opened = None

    def open_source_if_needed(self):
        desired_src = self.control.get("video_source")
        desired_lat = bool(self.control.get("low_latency"))

        if self.source is None or desired_src != self.source_opened or desired_lat != self.low_latency_opened:
            if self.source is not None:
                self.source.close()
            self.source = VideoSource(
                source=desired_src,
                width=self.cfg["video"]["width"],
                height=self.cfg["video"]["height"],
                low_latency=desired_lat,
                cfg=self.cfg
            )
            self.source.open()
            self.source_opened = desired_src
            self.low_latency_opened = desired_lat

    def run(self):
        while not self.stop_event.is_set():
            self.open_source_if_needed()

            if self.control.get("paused", False) and not self.control.get("step_once", False):
                time.sleep(0.01)
                continue

            ok, frame = self.source.read()
            if not ok or frame is None:
                if bool(self.cfg["video"]["loop"]):
                    try:
                        self.source.reopen()
                    except Exception:
                        pass
                time.sleep(0.02)
                continue

            self.frame_id += 1
            pkt = FramePacket(frame_id=self.frame_id, t_capture=time.time(), frame_bgr=frame)
            safe_put_latest(self.out_q, pkt)


class OutputPoller(QtCore.QObject):
    def __init__(self, out_q: "queue.Queue[OutputPacket]", bus: UiBus, parent=None):
        super().__init__(parent)
        self.out_q = out_q
        self.bus = bus
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(15)

    def poll(self):
        try:
            pkt = self.out_q.get_nowait()
        except queue.Empty:
            return
        self.bus.updated.emit(pkt)


class DashboardApp(QtWidgets.QApplication):
    def __init__(self, argv, cfg: Dict[str, Any]):
        super().__init__(argv)
        self.cfg = cfg
        ensure_dirs(cfg)

        self.control = {
            "paused": False,
            "step_once": False,
            "model_path": cfg["detection"]["model_path"],
            "video_source": cfg["video"]["source"],
            "low_latency": bool(cfg["video"]["low_latency"]),
        }
        self.stop_event = threading.Event()

        self.cap_q: "queue.Queue[FramePacket]" = queue.Queue(maxsize=2)
        self.out_q: "queue.Queue[OutputPacket]" = queue.Queue(maxsize=2)

        self.bus = UiBus()
        self.bus.updated.connect(self.on_update)

        self.panels: Dict[str, Any] = {}
        self.create_panels_initial()

        self.ctrl_panel = ControlPanel(cfg)
        self.ctrl_panel.show()

        self.ctrl_panel.sig_model_changed.connect(self.on_model_changed)
        self.ctrl_panel.sig_source_changed.connect(self.on_source_changed)
        self.ctrl_panel.sig_latency_changed.connect(self.on_latency_changed)

        self.ctrl_panel.sig_toggle_detection.connect(lambda v: self.toggle_panel("det", v))
        self.ctrl_panel.sig_toggle_metrics.connect(lambda v: self.toggle_panel("met", v))
        self.ctrl_panel.sig_toggle_odometry.connect(lambda v: self.toggle_panel("odo", v))
        self.ctrl_panel.sig_toggle_position.connect(lambda v: self.toggle_panel("pos", v))

        self.ctrl_panel.sig_pause.connect(self.toggle_pause)
        self.ctrl_panel.sig_step.connect(self.step_once)
        self.ctrl_panel.sig_screenshot.connect(self.take_screenshot)
        self.ctrl_panel.sig_quit.connect(self.quit_all)

        self.capture = CaptureWorker(cfg, self.cap_q, self.control, self.stop_event)
        self.pipeline = PipelineWorker(cfg, self.cap_q, self.out_q, self.control, self.stop_event)
        self.capture.start()
        self.pipeline.start()

        self.poller = OutputPoller(self.out_q, self.bus)

        self.installEventFilter(self)

    def create_panels_initial(self):
        if self.cfg["ui"]["show_detection"]:
            self.panels["det"] = DetectionPanel(self.cfg)
            self.panels["det"].show()
        if self.cfg["ui"]["show_metrics"]:
            self.panels["met"] = MetricsPanel(self.cfg)
            self.panels["met"].show()
        if self.cfg["ui"]["show_odometry"]:
            self.panels["odo"] = OdometryPanel(self.cfg)
            self.panels["odo"].show()
        if self.cfg["ui"]["show_position"]:
            self.panels["pos"] = PositionPanel(self.cfg)
            self.panels["pos"].show()

    def toggle_panel(self, key: str, enabled: bool):
        if enabled and key not in self.panels:
            if key == "det":
                self.panels[key] = DetectionPanel(self.cfg)
            elif key == "met":
                self.panels[key] = MetricsPanel(self.cfg)
            elif key == "odo":
                self.panels[key] = OdometryPanel(self.cfg)
            elif key == "pos":
                self.panels[key] = PositionPanel(self.cfg)
            self.panels[key].show()

        if (not enabled) and key in self.panels:
            self.panels[key].close()
            del self.panels[key]

    def on_model_changed(self, path: str):
        self.control["model_path"] = path
        self.ctrl_panel.set_status(f"Model hot reloaded: {path}")

    def on_source_changed(self, src: str):
        self.control["video_source"] = src
        self.ctrl_panel.set_status(f"Source switched: {src}")

    def on_latency_changed(self, enabled: bool):
        self.control["low_latency"] = bool(enabled)
        self.ctrl_panel.set_status(f"Low latency: {'ON' if enabled else 'OFF'}")

    def toggle_pause(self):
        self.control["paused"] = not self.control["paused"]
        self.ctrl_panel.set_status(f"{'Paused' if self.control['paused'] else 'Running'}")

    def step_once(self):
        self.control["step_once"] = True
        self.ctrl_panel.set_status("Step once")

    def take_screenshot(self):
        out_dir = self.cfg["ui"]["screenshot_dir"]
        ts = now_ts()

        pix = self.ctrl_panel.grab()
        pix.save(f"{out_dir}/control_{ts}.png")
        for name, p in self.panels.items():
            pix = p.grab()
            pix.save(f"{out_dir}/{name}_{ts}.png")

        self.ctrl_panel.set_status(f"Screenshot saved: {ts}")

    def on_update(self, pkt: OutputPacket):
        if "det" in self.panels:
            self.panels["det"].update_view(pkt)
        if "met" in self.panels:
            self.panels["met"].update_view(pkt)
        if "odo" in self.panels:
            self.panels["odo"].update_view(pkt)
        if "pos" in self.panels:
            self.panels["pos"].update_view(pkt)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key == QtCore.Qt.Key_Space:
                self.toggle_pause()
                return True
            if key == QtCore.Qt.Key_S:
                self.take_screenshot()
                return True
            if key == QtCore.Qt.Key_Q or key == QtCore.Qt.Key_Escape:
                self.quit_all()
                return True
            if key == QtCore.Qt.Key_N:
                self.step_once()
                return True
        return super().eventFilter(obj, event)

    def quit_all(self):
        self.stop_event.set()
        time.sleep(0.1)
        for p in list(self.panels.values()):
            p.close()
        self.ctrl_panel.close()
        self.quit()


def main():
    force_qt_plugins()
    cfg = load_config("configs/dashboard.yaml")
    app = DashboardApp(sys.argv, cfg)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
