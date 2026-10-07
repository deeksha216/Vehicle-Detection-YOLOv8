# main.py  -  AI-Based Vehicle Detection, Tracking and Traffic Flow Analysis
# Run:  python main.py
# Optional:  python main.py --video input/mytraffic.mp4 --line-pos 0.6 --no-show

import argparse
import os
import sys
import cv2

from detector import VehicleDetector
from tracker import PositionMemory
from counter import LineCounter, CLASS_ORDER

LABELS = {"Car": "Cars:", "Motorcycle": "Motorcycles:", "Bus": "Buses:", "Truck": "Trucks:"}
COLORS = {"Car": (255, 128, 0), "Motorcycle": (0, 165, 255),
          "Bus": (0, 200, 0), "Truck": (0, 0, 230)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="input/traffic.mp4")
    p.add_argument("--model", default="models/yolov8n.pt")
    p.add_argument("--output", default="output/result.mp4")
    p.add_argument("--csv", default="results/traffic_statistics.csv")
    p.add_argument("--graph", default="results/vehicle_counts.png")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    p.add_argument("--width", type=int, default=960, help="processing width in pixels")
    p.add_argument("--line-pos", type=float, default=0.6,
                   help="counting line height: 0.0 = top, 1.0 = bottom of the frame")
    p.add_argument("--no-show", action="store_true", help="do not open a video window")
    return p.parse_args()


def draw_panel(frame, counter):
    """Phase 8: the TRAFFIC ANALYSIS panel."""
    lines = [("TRAFFIC ANALYSIS", None)]
    lines += [(LABELS[c], counter.counts[c]) for c in CLASS_ORDER]
    lines += [("Total:", counter.total),
              ("Entering:", counter.entering), ("Leaving:", counter.leaving)]
    h = 28 * len(lines) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (270, 10 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    y = 38
    for name, value in lines:
        if value is None:
            cv2.putText(frame, name, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), 2)
        else:
            cv2.putText(frame, name, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)
            cv2.putText(frame, str(value), (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
        y += 28


def main():
    args = parse_args()
    if not os.path.exists(args.video):
        sys.exit(f"ERROR: video not found: {args.video}\n"
                 "Put a traffic video in the input/ folder named traffic.mp4 "
                 "or use --video path/to/video.mp4")

    detector = VehicleDetector(args.model, args.conf)
    memory = PositionMemory()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = min(args.width, src_w)
    height = int(src_h * width / src_w)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    line_y = int(height * args.line_pos)
    counter = LineCounter((0, line_y), (width, line_y))

    frame_no = 0
    print("Processing... press 'q' in the video window to stop early.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        frame = cv2.resize(frame, (width, height))
        time_sec = frame_no / fps

        for v in detector.detect_and_track(frame):
            x1, y1, x2, y2 = v["box"]
            previous, current = memory.update(v["id"], v["box"])
            counted_now = counter.update(v["id"], v["class_name"],
                                         previous, current, time_sec)
            color = COLORS[v["class_name"]]
            thickness = 4 if v["id"] in counter.counted_ids else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f'{v["class_name"]} #{v["id"]}'
            cv2.putText(frame, label, (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, current, 3, (0, 255, 255), -1)

        cv2.line(frame, counter.p1, counter.p2, (0, 0, 255), 3)
        cv2.putText(frame, "COUNTING LINE", (width - 200, line_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        draw_panel(frame, counter)

        writer.write(frame)
        if not args.no_show:
            cv2.imshow("Traffic Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    duration = frame_no / fps
    df = counter.save_statistics(duration, args.csv, args.graph)
    print("\n===== TRAFFIC STATISTICS =====")
    print(df.to_string(index=False))
    print(f"\nProcessed video : {args.output}")
    print(f"CSV statistics  : {args.csv}")
    print(f"Graph           : {args.graph}")


if __name__ == "__main__":
    main()
