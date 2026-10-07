# AI-Based Vehicle Detection, Tracking and Traffic Flow Analysis Using YOLOv8

Detects cars, motorcycles, buses and trucks in a traffic video, tracks each with a
unique ID (ByteTrack), counts them when they cross a virtual line (with direction),
and saves a processed video, CSV statistics and a graph.

## Setup (Windows)
```
cd vehicle_detection_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python verify_setup.py
```
(Mac/Linux: `source venv/bin/activate`.) Python 3.10-3.12 recommended.

## Run
1. Put your traffic video at `input/traffic.mp4`
2. `python phase2_basic_detection.py`  (optional: plain detection check)
3. `python main.py`  (full project; press `q` to stop)

Options: `--video`, `--line-pos 0.6` (0=top, 1=bottom), `--conf 0.4`, `--width 960`, `--no-show`

## Files
| File | Purpose |
|---|---|
| main.py | Full pipeline: video in -> detect/track/count -> video + CSV + graph out |
| detector.py | Loads pretrained YOLOv8, filters vehicle classes, tracking (ByteTrack) |
| tracker.py | Remembers each vehicle's previous position |
| counter.py | Counting line, line-crossing, direction, statistics (CSV + graph) |
| phase2_basic_detection.py | Simple detection-only program |
| verify_setup.py | Installation check |
| models/yolov8n.pt | Pretrained YOLOv8 nano weights (COCO) |

## How counting works
Each vehicle's centre point is compared between the previous and current frame. If it
moves from one side of the line to the other, it is counted once. Its ID is stored so it
can never be counted again. Moving down = Entering, moving up = Leaving.
