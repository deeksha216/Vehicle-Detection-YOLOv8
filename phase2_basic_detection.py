# phase2_basic_detection.py
# Basic YOLOv8 detection ONLY (all 80 COCO classes). No tracking, no counting.
import cv2
from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")
cap = cv2.VideoCapture("input/traffic.mp4")
if not cap.isOpened():
    raise SystemExit("ERROR: cannot open input/traffic.mp4")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
    for box in model(frame, conf=0.4, verbose=False)[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{model.names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Phase 2 - Basic Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
