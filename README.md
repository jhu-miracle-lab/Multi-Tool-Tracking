# Multi-Tool Tracking

Stereo tracking of two dVRK tools using keypoints, shaft edges, robot kinematics,
and a shared particle filter. **FPRT** fixes the eligible primary to the first
arm (normally PSM1); **APRT** allows either arm to become primary. Both use
reliability gating and cumulative relative tracking.

Default pipeline: **raw stereo images → ELSED shaft edges + keypoints → joint PF**.

## Code structure

| Directory | Responsibility |
| --- | --- |
| `tracking/messages/` | ROS synchronization, offline data loading, execution, and outputs |
| `tracking/features/` | Keypoint extraction, per-view state, ROI preparation, and shaft detection |
| `tracking/pf/` | Robot geometry, observation likelihoods, PF updates, and relative tracking |
| `tracking/common/` | Stereo calibration, transforms, shared data structures, and visualization |
| `FPRT/`, `APRT/` | Method settings and offline/ROS entry points |
| `Parameter/` | Stereo calibration YAML and `LND.json` tool geometry |
| `launch/` | Tracker launch and combined dVRK camera/tracker launch |
| `resource/` | ROS package-index marker |



## Python installation (offline)

Install Conda first. On Ubuntu/Linux, run from the downloaded project directory:

```bash
sudo apt update
sudo apt install build-essential pkg-config libopencv-dev pybind11-dev
conda create -n multi-tool-tracking python=3.10 pip -y
conda activate multi-tool-tracking
conda env config vars set PYTHONNOUSERSITE=1 PYTHONPATH=""
conda deactivate
conda activate multi-tool-tracking
which python
python -m pip --version
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
python FPRT/offline_tracking.py --help
```

The environment saves both isolation settings: activation disables user-site
packages and clears inherited Python paths (including ROS); deactivation restores
the previous values (originally unset variables may become empty strings, which
disable these Python overrides). Configure once, then use `conda activate multi-tool-tracking`
in new terminals. Re-activation is required immediately after saving the settings.

## ROS 2 installation

Install ROS 2 Humble first. Use its system Python in a separate terminal;
the Conda installation above is not required for this route. Place this package
under `~/tracking_ws/src/`, then run:

```bash
# If a Conda environment is active, deactivate it before loading ROS.
conda deactivate
# Repeat conda deactivate if another environment (including base) remains active.
source /opt/ros/humble/setup.bash
sudo apt install python3-colcon-common-extensions python3-rosdep \
  python3-dev python3-setuptools python3-wheel build-essential pkg-config pybind11-dev
# Run sudo rosdep init once if rosdep has never been initialized.
rosdep update
cd ~/tracking_ws
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
/usr/bin/python3 -m colcon build --packages-select surgical_tool_tracking_ros2
source install/setup.bash
ros2 pkg executables surgical_tool_tracking_ros2
```

Skip `conda deactivate` if Conda is not installed. Source ROS and the workspace
in each new terminal; do not clear `PYTHONPATH` after sourcing ROS. Rebuild after
changing installed launch files, assets, or native code.

## Resources to supply

- **Hand-eye YAML:** use offline `--handeye` or ROS `handeye`. Offline mode also
  recognizes `<dataset>/pnp_handeye_manual.yaml`. Missing files raise an error.
- **Offline dataset:** images, joint JSON, and cached keypoint CSVs. No online
  inference environment is required for cached CSV tracking.
- **Online inference:** compatible PyTorch/DLCLive dependencies and DEKR weights
  with the identity head. Set `dekr_model_path` or `DEKR_LIVE_MODEL_PATH`.
  These dependencies, weights, and GPU drivers are not installed automatically.
- **Direct dVRK capture:** GStreamer/PyGObject and the `unixfdsrc` plugin;
  check with `gst-inspect-1.0 unixfdsrc`.

`Parameter/stereo_calibration.yaml` and `LND.json` must match the hardware.

## Offline running

Run from the project directory. Expected dataset layout:

```text
dataset/
├── pnp_handeye_manual.yaml
└── regular/
    ├── image/
    │   ├── left_resized/    # 0.png, 1.png, ... and the two CSVs below
    │   └── right_resized/   # Matching image IDs and CSV names
    └── kinematic/
        ├── PSM1/           # 0.json, 1.json, ...
        └── PSM2/           # 0.json, 1.json, ...
```

CSV names: `image_predictions_DEKR_shuffle10_best140_tool1.csv` and
`image_predictions_DEKR_shuffle10_best140_tool2.csv`. Each uses DLC headers with
`x`, `y`, and `likelihood` columns. Joint JSON format is defined in
`tracking/messages/kinematics.py`. Image/JSON filenames use unpadded frame IDs;
matching left/right IDs must refer to the same recording time.

```bash
python FPRT/offline_tracking.py --data /path/to/dataset --no_display
python APRT/offline_tracking.py --data /path/to/dataset --no_display \
  --handeye /path/to/handeye.yaml --n_particles 500 \
  --out_dir ./results/aprt
```

Offline defaults are 500 particles and up to 50 warmup frames. Select the
optional Canny/Hough backend with `--shaft_backend canny_hough`. Historical line
selection remains enabled; failed detections may retain the previous display
line, but do not reuse it as a fresh PF edge observation. Optional dataset
files `shaft_params_psm1.json` and `shaft_params_psm2.json` override defaults in
`tracking/features/shaft_config.py`; legacy `canny_params_<arm>.json` is supported.

## ROS running

With existing raw stereo image publishers:

```bash
export DEKR_LIVE_MODEL_PATH=/path/to/model_ident.pt
ros2 launch surgical_tool_tracking_ros2 tracking.launch.py \
  handeye:=/path/to/handeye.yaml \
  image_topic:=/camera/left/image_raw \
  right_image_topic:=/camera/right/image_raw \
  allow_primary_switch:=false
```

Set `allow_primary_switch:=true` for APRT. Launch defaults are 300 particles and
50 warmup frames. Both launch files default to ELSED; use
`shaft_backend:=canny_hough` to switch. Direct nodes accept
`-p shaft_backend:=canny_hough`. For additional parameters, including headless operation:

```bash
ros2 run surgical_tool_tracking_ros2 aprt_tracking_node --ros-args \
  -p handeye:=/path/to/handeye.yaml -p dekr_model_path:=/path/to/model_ident.pt \
  -p warmup_frames:=50 -p no_display:=true
```

Use `fprt_tracking_node` for FPRT. Joint inputs are `/PSM1/measured_js`,
`/PSM1/jaw/measured_js`, and equivalent PSM2 topics. Timestamps must share a clock;
`stereo_sync_slop` defaults to 15 ms and `joint_sync_slop` to 30 ms.

To start dVRK capture and tracking together, use `live_dvrk.launch.py` with
`handeye`, `dekr_model_path`, and optional `socket_path`. Its image topic arguments
are named `left_topic` and `right_topic`.

For offline ROS publishing:

```bash
ros2 run surgical_tool_tracking_ros2 aprt_offline_node --ros-args \
  -p data:=/path/to/dataset -p handeye:=/path/to/handeye.yaml \
  -p out_dir:=/path/to/results -p no_display:=true
```

Use `fprt_offline_node` for FPRT. ROS outputs are `/tracking/psm1/pose`,
`/tracking/psm2/pose`, and `/tracking/visualization`.

## Stereo and output conventions

- Entry points resize raw stereo images to 640 x 480.   Keypoints remain in raw-image coordinates;
  their projection model includes lens distortion.
- The supplied calibration requires `swap_cameras=True`: left uses K2/D2 and
  right uses K1/D1. YAML translation is converted from millimetres to metres.
- Each arm has independent left/right edge-detection histories. Both cameras
  constrain the same PF; saved videos show only the left image. Tracking, ROS poses, and visualization
  use the raw left camera frame.
- Offline output defaults to `./METHOD/tracking_output_offline/` (ROS:
  `tracking_output_offline_ros2/`). It contains a 30 FPS MP4 and frame PNGs.
  Choose separate `out_dir` paths to avoid overwriting previous runs.
