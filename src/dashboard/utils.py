import os
import time
import csv
from typing import Any, Dict, Optional, Tuple
import queue

import cv2
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: Dict[str, Any]):
    os.makedirs(cfg["logging"]["log_dir"], exist_ok=True)
    os.makedirs(cfg["ui"]["screenshot_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["logging"]["csv_path"]), exist_ok=True)


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_put_latest(q: "queue.Queue", item: Any):
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        try:
            _ = q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


class FPSCounter:
    def __init__(self, rolling: int = 30):
        self.rolling = rolling
        self.ts = []

    def tick(self) -> float:
        t = time.time()
        self.ts.append(t)
        if len(self.ts) > self.rolling:
            self.ts = self.ts[-self.rolling:]
        if len(self.ts) < 2:
            return 0.0
        dt = self.ts[-1] - self.ts[0]
        return (len(self.ts) - 1) / dt if dt > 0 else 0.0


def parse_source(src: str):
    s = str(src).strip()
    if s.isdigit():
        return int(s)
    return s


def set_ffmpeg_low_latency_env(cfg: Dict[str, Any], enabled: bool):
    if not enabled:
        return
    opts = cfg["video"].get(
        "ffmpeg_capture_options",
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
    )
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts


class VideoSource:
    def __init__(self, source, width: int, height: int, low_latency: bool, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.source = source
        self.width = width
        self.height = height
        self.low_latency = low_latency
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self):
        set_ffmpeg_low_latency_env(self.cfg, self.low_latency)
        src = parse_source(self.source)
        self.cap = cv2.VideoCapture(src)

        if self.low_latency:
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.cfg["video"].get("buffersize", 1)))
            except Exception:
                pass

        if self.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def reopen(self):
        self.close()
        time.sleep(0.05)
        self.open()

    def read(self) -> Tuple[bool, Optional[Any]]:
        if self.cap is None:
            return False, None

        if self.low_latency and self.cfg["video"].get("drain_grabs", 0) > 0:
            n = int(self.cfg["video"]["drain_grabs"])
            for _ in range(n):
                self.cap.grab()

        return self.cap.read()

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class CsvLogger:
    def __init__(self, path: str):
        self.path = path
        self._fh = None
        self._writer = None
        self._header_written = False

    def _open(self):
        if self._fh is None:
            self._fh = open(self.path, "a", newline="", encoding="utf-8")

    def write_row(self, row: Dict[str, Any]):
        self._open()
        if self._writer is None:
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row.keys()))
        if not self._header_written and os.path.exists(self.path) and os.stat(self.path).st_size == 0:
            self._writer.writeheader()
            self._header_written = True
        self._writer.writerow(row)
        self._fh.flush()


class AnomalyMonitor:
    def __init__(self, fps_low: float = 12.0, rmse_high: float = 10.0):
        self.fps_low = fps_low
        self.rmse_high = rmse_high
        self.last_flags = {"low_fps": False, "high_rmse": False}

    def update(self, fps: float, rmse: Optional[float]):
        self.last_flags["low_fps"] = fps > 0 and fps < self.fps_low
        self.last_flags["high_rmse"] = (rmse is not None and rmse > self.rmse_high)


def try_load_yolo_model(model_path: str, device: str = "cuda"):
    from ultralytics import YOLO
    model = YOLO(model_path)
    return model
