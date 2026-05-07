from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project="runs",
        name="traffic_sign_detector"
    )

if __name__ == "__main__":
    main()