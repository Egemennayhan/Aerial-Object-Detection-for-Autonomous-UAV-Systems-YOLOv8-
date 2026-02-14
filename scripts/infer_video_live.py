import time
import cv2
import torch
from ultralytics import YOLO

# =========================
# PATH CONFIG (SANA ÖZEL)
# =========================
MODEL_PATH = "/home/egemen/teknofest/project/runs/detect/outputs/runs/vehicle_stage1_sanity/weights/best.pt"
VIDEO_PATH = "/mnt/c/Users/egeme/OneDrive/Desktop/TEKNOFEST_YZ_2025/EGEMEN_V2_TEST_SONUCLARI/videoplayback.mp4"

# =========================
# INFERENCE CONFIG
# =========================
CONF = 0.25
IOU = 0.50
IMG_SZ = 960
MAX_DET = 25

SHOW_FPS = True
RESIZE_DISPLAY = None      # örn: (1280, 720)
FRAME_SKIP = 0             # 0 = skip yok

WINDOW_NAME = "TEKNOFEST UAV - Vehicle Detection (q=quit, space=pause)"


def add_safe_globals():
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])


def main():
    add_safe_globals()

    print("Model yükleniyor...")
    model = YOLO(MODEL_PATH)

    print("Video açılıyor...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("❌ Video açılamadı. Path veya codec hatalı.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    paused = False
    frame_id = 0
    last_time = time.time()
    fps_avg = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if FRAME_SKIP > 0 and frame_id % (FRAME_SKIP + 1) != 1:
                continue

            results = model.predict(
                frame,
                imgsz=IMG_SZ,
                conf=CONF,
                iou=IOU,
                max_det=MAX_DET,
                verbose=False
            )

            vis = results[0].plot()

            if RESIZE_DISPLAY:
                vis = cv2.resize(vis, RESIZE_DISPLAY)

            if SHOW_FPS:
                now = time.time()
                fps = 1 / max(now - last_time, 1e-6)
                last_time = now
                fps_avg = fps if fps_avg is None else (0.9 * fps_avg + 0.1 * fps)

                cv2.putText(
                    vis,
                    f"FPS: {fps_avg:.1f}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

            cv2.imshow(WINDOW_NAME, vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print("Bitti.")


if __name__ == "__main__":
    main()
