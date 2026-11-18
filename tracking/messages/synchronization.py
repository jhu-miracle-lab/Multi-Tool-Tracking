"""Bounded timestamp matching, independent of ROS for deterministic tests."""
from collections import deque
import math


def stamp_seconds(message):
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class StampedHistory:
    def __init__(self, capacity=100):
        self.samples = deque(maxlen=capacity)

    def add(self, stamp, value):
        if math.isfinite(stamp):
            self.samples.append((stamp, value))

    def nearest(self, stamp, tolerance):
        if not self.samples or not math.isfinite(stamp):
            return None
        ts, value = min(self.samples, key=lambda item: abs(item[0] - stamp))
        return value if abs(ts - stamp) <= tolerance else None


class JointSynchronizer:
    """ROS-independent timestamp matching with per-arm failure isolation.

    add() accepts positions, not ROS messages. snapshot() returns canonical
    seven-value vectors; missing, stale or malformed arms are omitted.
    """
    def __init__(self, arm_names, tolerance=0.03, capacity=100):
        import threading
        if not math.isfinite(tolerance) or tolerance <= 0 or capacity < 1:
            raise ValueError("Expected positive finite tolerance and capacity")
        self.tolerance = tolerance
        self.histories = {name: [StampedHistory(capacity), StampedHistory(capacity)] for name in arm_names}
        self._lock = threading.Lock()

    def add(self, arm_name, slot, stamp, positions):
        if slot not in (0, 1):
            raise ValueError("Joint slot must be 0 (arm) or 1 (jaw)")
        with self._lock:
            self.histories[arm_name][slot].add(stamp, tuple(positions))

    def snapshot(self, stamp):
        from tracking.messages.kinematics import validated_joint_angles
        result = {}
        with self._lock:
            for name, histories in self.histories.items():
                joints, jaw = [history.nearest(stamp, self.tolerance) for history in histories]
                if joints is None or jaw is None:
                    continue
                try:
                    result[name] = validated_joint_angles(joints, jaw)
                except (TypeError, ValueError):
                    continue
        return result
