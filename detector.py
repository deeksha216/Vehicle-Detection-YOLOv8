# detector.py
# Loads the pretrained YOLOv8 model, detects vehicles and tracks them.

from ultralytics import YOLO

# COCO class IDs used by YOLOv8 -> names we display.
# YOLOv8 pretrained models know 80 COCO classes. We only keep these 4.
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


class VehicleDetector:
    def __init__(self, model_path="models/yolov8n.pt", confidence=0.4):
        self.model = YOLO(model_path)      # pretrained model, no training needed
        self.confidence = confidence

    def detect(self, frame):
        """Phase 3: detection + vehicle filtering only (no IDs).
        Returns a list of dicts: box, class_name, confidence."""
        result = self.model(frame, conf=self.confidence,
                            classes=list(VEHICLE_CLASSES), verbose=False)[0]
        vehicles = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            vehicles.append({
                "box": (x1, y1, x2, y2),
                "class_name": VEHICLE_CLASSES[cls_id],
                "confidence": float(box.conf[0]),
            })
        return vehicles

    def detect_and_track(self, frame):
        """Phase 4: detection + tracking (ByteTrack built into Ultralytics).
        Each vehicle gets a unique 'id' that stays the same across frames."""
        result = self.model.track(frame, persist=True, conf=self.confidence,
                                  classes=list(VEHICLE_CLASSES),
                                  tracker="bytetrack.yaml", verbose=False)[0]
        vehicles = []
        if result.boxes is None or result.boxes.id is None:
            return vehicles          # nothing tracked in this frame
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        confs = result.boxes.conf.cpu().tolist()
        for b, tid, c, conf in zip(boxes, ids, classes, confs):
            vehicles.append({
                "id": tid,
                "box": tuple(int(v) for v in b),
                "class_name": VEHICLE_CLASSES[c],
                "confidence": float(conf),
            })
        return vehicles
