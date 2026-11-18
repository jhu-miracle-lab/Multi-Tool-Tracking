"""Replaceable raw-image sources. Rectification belongs to the caller boundary."""
from pathlib import Path
import cv2


class DirectoryStereoImages:
    def __init__(self, left_directory, right_directory):
        self.left_directory = Path(left_directory)
        self.right_directory = Path(right_directory)

    def read(self, frame_idx):
        def load(directory):
            path = directory / f"{frame_idx}.png"
            return cv2.imread(str(path)) if path.is_file() else None
        return load(self.left_directory), load(self.right_directory)
