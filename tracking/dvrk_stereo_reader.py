"""dvrk_display GStreamer unix-socket stereo frame reader.

Ported from dlc_live_rt.py (Multi_tool_tracking) -- pure GStreamer + numpy,
no ROS2/rclpy dependency, so it lives under tracking/ (reused by both
offline/standalone tooling and the ROS2 node
ros2_node/dvrk_stereo_publisher.py) rather than under ros2_node/.
"""

import getpass
import os
import queue
import threading
import time

import numpy as np


def default_socket_path() -> str:
    return f"/tmp/dvrk_display_stereo_{getpass.getuser()}.sock"


class StereoUnixFdReader:
    """Reads video frames from a GStreamer unixfdsrc socket, delivering the
    FULL side-by-side stereo frame (no `videocrop`) -- left/right eyes are
    split out of it by the caller (a cheap array slice, no copy) instead of
    by two separate GStreamer pipelines.

    This is deliberate, not just simpler: two independent pipelines each
    pulling from the same unix socket may not even both be able to connect
    (depending on whether the upstream dvrk_display socket supports
    multiple readers), and even if they can, each pipeline's pull happens
    on its own schedule -- there's no guarantee the two "simultaneous"
    reads land on the same underlying video frame, which defeats the point
    of stereo (triangulation, left/right face disambiguation) needing a
    genuinely synchronized pair. Reading one full frame and slicing it
    guarantees left and right always come from the exact same capture
    instant, for free.
    """

    def __init__(self, socket_path: str):
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst, GstApp  # noqa: F401 -- triggers registration

        self._Gst = Gst
        Gst.init(None)

        pipeline_str = (
            f"unixfdsrc socket-path={socket_path} "
            "! videoconvert n-threads=4 "
            "! video/x-raw,format=BGR "
            "! appsink name=appsink drop=true max-buffers=1 sync=false"
        )
        print(f"[StereoUnixFdReader] Pipeline: {pipeline_str}")
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._sink = self._pipeline.get_by_name("appsink")
        self._sink.set_property("emit-signals", False)

        self._width = None
        self._height = None
        self._pipeline.set_state(Gst.State.PLAYING)

    def read(self):
        """Returns (True, full_bgr_frame) or (False, None). full_bgr_frame
        is the whole side-by-side pair -- split it yourself:
        left = frame[:, :frame.shape[1]//2], right = frame[:, frame.shape[1]//2:]."""
        Gst = self._Gst
        sample = self._sink.try_pull_sample(Gst.SECOND * 2)
        if sample is None:
            return False, None

        buf = sample.get_buffer()
        caps = sample.get_caps()
        st = caps.get_structure(0)
        w = st.get_value("width")
        h = st.get_value("height")
        self._width, self._height = w, h

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return False, None
        data = np.frombuffer(mapinfo.data, dtype=np.uint8).copy()
        buf.unmap(mapinfo)
        return True, data.reshape((h, w, 3))

    @property
    def frame_size(self):
        return self._width, self._height

    def release(self):
        self._pipeline.set_state(self._Gst.State.NULL)


class ThreadedReader:
    """
    Wraps any reader with a background thread so frame capture overlaps
    with GPU inference/publishing. The queue holds at most 1 frame --
    always the latest.

    Frames are timestamped at capture (time.perf_counter()) -- feed
    t_capture into a downstream latency-sensitive consumer (e.g. a
    particle filter) so the measurement update is applied against the
    state at capture time, not at read()-return time.
    """

    def __init__(self, reader):
        self._reader = reader
        self._q = queue.Queue(maxsize=1)
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stopped:
            ok, frame = self._reader.read()
            t_capture = time.perf_counter()
            if not ok:
                self._q.put((False, None, t_capture))
                return
            # Drop stale frame if the consumer hasn't consumed it yet
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put((True, frame, t_capture))

    def read(self):
        return self._q.get(timeout=5.0)

    def release(self):
        self._stopped = True
        self._reader.release()


def open_threaded_stereo_reader(socket_path: str = None):
    """Opens a StereoUnixFdReader (wrapped in a ThreadedReader) on the
    dvrk_display stereo socket, warms it up, and returns
    (reader, full_w, full_h).

    Raises FileNotFoundError if the socket doesn't exist (dvrk_display's
    stereo node isn't running), RuntimeError if the frame size can't be
    determined after warm-up.
    """
    socket_path = socket_path or default_socket_path()
    if not os.path.exists(socket_path):
        raise FileNotFoundError(
            f"Unix socket not found: {socket_path}\n"
            "Make sure the stereo display node is running:\n"
            "  ros2 run dvrk_display stereo -c stereo_display_decklink_hdmi_gooviz.json"
        )
    raw_reader = StereoUnixFdReader(socket_path)
    reader = ThreadedReader(raw_reader)

    print("[StereoUnixFdReader] Warming up unix socket reader …")
    full_w = full_h = None
    for _ in range(30):
        ok, _, _ = reader.read()
        full_w, full_h = raw_reader.frame_size
        if ok and full_w is not None:
            break

    if full_w is None:
        raise RuntimeError("Could not determine frame size from unix socket")

    print(
        f"[StereoUnixFdReader] Stereo frame from pipeline: {full_w}×{full_h} "
        f"(left+right, {full_w // 2}×{full_h} each)"
    )
    return reader, full_w, full_h
