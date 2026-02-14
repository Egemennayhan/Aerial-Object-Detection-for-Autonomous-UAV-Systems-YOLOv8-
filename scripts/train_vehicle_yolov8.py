from ultralytics import YOLO

def main():
    # Use a small model first to reduce crash risk on WSL + new GPU stack
    model = YOLO("yolov8n.pt")  # safer than yolov8s.pt for first run

    model.train(
        data="configs/uavdt_vehicle.yaml",
        epochs=5,          # short sanity run
        imgsz=640,
        batch=8,           # INT (not string)
        device=0,
        workers=0,         # WSL stability
        cache=False,       # avoid RAM spikes
        amp=False,         # reduce GPU/driver stress
        pretrained=True,
        name="vehicle_stage1_sanity",
        project="outputs/runs",
    )

if __name__ == "__main__":
    main()

