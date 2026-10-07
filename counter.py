# counter.py
# Virtual counting line, line-crossing logic, direction and statistics.

import os
import pandas as pd

CLASS_ORDER = ["Car", "Motorcycle", "Bus", "Truck"]


class LineCounter:
    def __init__(self, p1, p2):
        self.p1 = p1                      # line start (x, y)
        self.p2 = p2                      # line end   (x, y)
        self.counted_ids = set()          # IDs already counted -> no double counting
        self.counts = {c: 0 for c in CLASS_ORDER}
        self.entering = 0                 # moved from above the line to below it
        self.leaving = 0                  # moved from below the line to above it
        self.events = []                  # (time_in_seconds, class, direction)

    def _side(self, point):
        """Which side of the line a point is on: >0 one side, <0 the other side."""
        (x1, y1), (x2, y2) = self.p1, self.p2
        return (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)

    def _within_line(self, point):
        return min(self.p1[0], self.p2[0]) <= point[0] <= max(self.p1[0], self.p2[0])

    def update(self, vehicle_id, class_name, previous, current, time_sec):
        """Call once per vehicle per frame. Returns True if it was just counted."""
        if previous is None or vehicle_id in self.counted_ids:
            return False
        before, after = self._side(previous), self._side(current)
        crossed = before * after < 0 and self._within_line(current)
        if not crossed:
            return False

        self.counted_ids.add(vehicle_id)          # never count this ID again
        self.counts[class_name] += 1
        direction = "Entering" if before < 0 else "Leaving"
        if direction == "Entering":
            self.entering += 1
        else:
            self.leaving += 1
        self.events.append((time_sec, class_name, direction))
        return True

    @property
    def total(self):
        return sum(self.counts.values())

    def save_statistics(self, duration_sec, csv_path, graph_path):
        minutes = max(duration_sec / 60.0, 1e-9)
        rows = [("Total Vehicles", self.total)]
        rows += [(c, self.counts[c]) for c in CLASS_ORDER]
        rows += [("Vehicles Entering", self.entering),
                 ("Vehicles Leaving", self.leaving),
                 ("Video Duration (seconds)", round(duration_sec, 1)),
                 ("Vehicles per Minute", round(self.total / minutes, 2))]
        df = pd.DataFrame(rows, columns=["Metric", "Value"], dtype=object)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        df.to_csv(csv_path, index=False)

        # Simple bar graph of vehicle types
        import matplotlib
        matplotlib.use("Agg")                     # no window needed
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        bars = plt.bar(CLASS_ORDER, [self.counts[c] for c in CLASS_ORDER],
                       color=["#2e86de", "#e67e22", "#27ae60", "#c0392b"])
        plt.bar_label(bars)
        plt.title("Vehicle Count by Type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(graph_path, dpi=120)
        plt.close()
        return df
