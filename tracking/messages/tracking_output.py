"""Checked video/image output shared by offline CLI and ROS entry points."""
from pathlib import Path
import cv2


class TrackingOutput:
    def __init__(self, data_dir, method, out_dir="", *, ros=False, display=False,
                 size=(640, 480), fps=30.0):
        suffix = "tracking_output_offline_ros2" if ros else "tracking_output_offline"
        self.root = (Path(out_dir).expanduser() if out_dir
                     else Path.cwd() / method / suffix).resolve()
        self.frames_dir = self.root / "frames"
        self.video_path = self.root / (Path(data_dir).name + "_tracking.mp4")
        self.display = display
        self.size, self.fps = size, fps
        self.writer = None
        self.count = 0

    def __enter__(self):
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
        if not self.writer.isOpened():
            self.writer.release()
            raise OSError(f"Cannot open video output: {self.video_path}")
        return self

    def write(self, frame_idx, image):
        if (image.shape[1], image.shape[0]) != self.size:
            raise ValueError(f"Unexpected output image size: {image.shape}")
        path = self.frames_dir / f"{frame_idx:06d}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Cannot save image: {path}")
        self.writer.write(image)
        self.count += 1

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.writer.release()
        finally:
            if self.display:
                cv2.destroyAllWindows()
        if exc_type is not None:
            return False
        if self.count == 0:
            raise RuntimeError(f"No frames written to {self.video_path}")
        # VideoWriter.write has no success return value. Check finalized output
        # before callers report success; this verifies readability, not every frame.
        capture = cv2.VideoCapture(str(self.video_path))
        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise OSError(f"Saved video is not readable: {self.video_path}")
        finally:
            capture.release()
        return False
