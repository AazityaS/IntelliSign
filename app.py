import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2

st.title("IntelliSign")
st.write("Traffic Sign Detection System")

# Load YOLO model
model = YOLO("C:/Users/Aaditya/runs/detect/runs/traffic_sign_detector/weights/best.pt")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # YOLO prediction
    results = model(temp_path, conf=0.6, max_det=5)

    img = cv2.imread(temp_path)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):

            if score < 0.6:
                continue

            x1, y1, x2, y2 = map(int, box)

            if (x2 - x1) < 40 or (y2 - y1) < 40:
                continue

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{r.names[int(cls)]} ({score:.2f})"

            cv2.putText(img, label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    st.subheader("Final Output")
    st.image(img, use_container_width=True)