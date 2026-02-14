import cv2
import time
import torch
from ultralytics import YOLO

# =========================
# PATH CONFIG
# =========================
MODEL_PATH = "/home/egemen/teknofest/project/runs/detect/outputs/runs/vehicle_stage1_sanity/weights/best.pt"
VIDEO_PATH = "/mnt/c/Users/egeme/OneDrive/Desktop/TEKNOFEST_YZ_2025/EGEMEN_V2_TEST_SONUCLARI/videoplayback.mp4"

OUTPUT_VIDEO = "vehicle_inference_output.mp4"

# =========================
# INFERENCE CONFIG
# =========================
CONF = 0.25
IOU = 0.50
IMG_SZ = 960
MAX_DET = 25


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
        raise RuntimeError("❌ Video açılamadı")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    print(f"Video info: {width}x{height} @ {fps:.2f} FPS")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            imgsz=IMG_SZ,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            verbose=False
        )

        annotated = results[0].plot()
        writer.write(annotated)

        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"{frame_count} frame işlendi | avg FPS: {frame_count / elapsed:.2f}")

    cap.release()
    writer.release()

    total_time = time.time() - start_time
    print("Bitti.")
    print(f"Toplam frame: {frame_count}")
    print(f"Ortalama FPS: {frame_count / total_time:.2f}")
    print(f"Çıktı video: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
