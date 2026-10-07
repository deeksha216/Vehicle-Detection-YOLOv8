# tracker.py
# Remembers where each tracked vehicle was in the previous frame.
# (The tracking itself - giving IDs - is done by ByteTrack inside detector.py.)


class PositionMemory:
    def __init__(self):
        self.last_position = {}     # vehicle id -> (x, y) centre in previous frame

    @staticmethod
    def centre(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, vehicle_id, box):
        """Return (previous_centre, current_centre). previous is None for a new vehicle."""
        current = self.centre(box)
        previous = self.last_position.get(vehicle_id)
        self.last_position[vehicle_id] = current
        return previous, current
