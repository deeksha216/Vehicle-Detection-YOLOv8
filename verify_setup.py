# verify_setup.py - run once to check the installation
import cv2, numpy as np, ultralytics
from ultralytics import YOLO
print("OpenCV      :", cv2.__version__)
print("NumPy       :", np.__version__)
print("Ultralytics :", ultralytics.__version__)
model = YOLO("models/yolov8n.pt")
print("YOLOv8 model loaded! Classes:", len(model.names))
