# Surgical Tool Tracking (ROS2)

Markerless 6-DoF tracking of da Vinci PSM surgical tool tips from stereo
endoscope video, fusing robot forward kinematics with 2-D keypoint
detections (DeepLabCut / DEKR) and Canny+Hough shaft-edge detection inside
a particle filter. Implements the Lumped-Error observation model of
Richter et al. 2022, *"Robotic Tool Tracking Under Partially Visible
Kinematic Chain"*.

The pipeline runs either as a set of ROS2 nodes (live camera + joint-state
topics) or as standalone offline scripts over a pre-recorded dataset
(images + kinematic logs + DLC/DEKR prediction CSVs).

## Contents

- [Feature overview](#feature-overview)
- [Class architecture](#class-architecture)
- [Directory layout](#directory-layout)
- [Usage](#usage)

## Feature overview

### 1. Forward kinematics & pose math (`core/`)
Camera- and ROS2-independent math shared by everything else:
- DH-parameter forward kinematics for a PSM-style tool, including gripper
  jaw geometry (`core/RobotLink.py`).
- A generic particle filter with both a scalar (loop-based) API and a
  vectorised predict/update API, stratified resampling, and an SO(3)-aware
  weighted-mean pose estimate (`core/ParticleFilter.py`).
- Pose/rotation conversions (axis-angle ↔ rotation matrix, pose ↔
  homogeneous transform) and an HSV colour-segmentation helper
  (`core/utils.py`, `core/probability_functions.py`).
- Full stereo camera calibration loading + undistort/rectify (used for
  offline calibration/rectification work, independent of the lighter-weight
  `SimpleProjector` used by the live tracker) (`core/StereoCamera.py`).

### 2. Per-arm tracking pipeline (`tracking/`)
The core tracking logic, reused by both the offline script and the ROS2
nodes:
- Loads per-arm runtime configuration (kinematics directory, hand-eye
  calibration file, process noise, keypoint label remap) via
  `build_arm_configs()`.
- Reads joint angles either from recorded JSON logs (offline) or live
  `sensor_msgs/JointState` messages (ROS2) (`tracking/kinematics.py`).
- Loads pre-computed DeepLabCut keypoint predictions from CSV
  (`tracking/dlc_csv.py`) or runs DLC inference live on GPU
  (`tracking/dlc_runner.py`).
- Resolves keypoint identity ambiguity every frame — which face (left/
  right, front/back) is visible, and which of the two yaw-tip keypoints is
  which — via stereo triangulation with a face-normal fallback, plus a 2×2
  optimal match for the yaw tips (`tracking/label_assignment.py`).
- Detects the tool shaft's two edges in the image with a shared
  Canny+HoughLines detector, angle-restricted by the FK-predicted shaft
  direction (`tracking/canny_hough.py`).
- Fuses keypoint and shaft-edge observations into one vectorised
  per-particle log-likelihood (the Lumped-Error model)
  (`tracking/observation.py`, `tracking/shaft_edge.py`).
- Orchestrates all of the above per arm, per frame, in `ArmTracker`
  (`tracking/arm_tracker.py`).
- A standalone CLI (`tracking/offline_tracking.py`) runs the full pipeline
  over a recorded dataset without ROS2.

### 3. ROS2 nodes (`ros2_node/`, `launch/`)
- `tracking_node.py`: live dual-arm tracking from a camera topic and
  per-arm joint-state topics; publishes tracked pose and an annotated
  visualization image.
- `offline_node.py`: the same offline dataset pipeline as
  `tracking/offline_tracking.py`, wrapped as a ROS2 node that publishes
  results as topics (useful for replaying a dataset through RViz/rosbag
  tooling).
- `launch/tracking.launch.py`: launch file for `tracking_node` with
  configurable arms, particle count, and topic names.

### 4. Shaft-edge detector tuning (`edge_detection/`)
Standalone tools for developing the Canny+Hough shaft-edge detector used
by `tracking/canny_hough.py`:
- `test_canny_hough.py`: interactive OpenCV trackbar tuner; step through
  dataset frames and save tuned parameters to `canny_params_{arm}.json`,
  which `ArmTracker`/`offline_tracking.py` pick up automatically.
- `render_canny_edges_video.py`: non-interactive batch renderer that
  produces a detection-result video from a saved parameter file, for one
  or both arms.

### 5. Keypoint model utilities (`keypoints/`)
Utilities around the DEKR/DLC keypoint models, independent of the
tracking runtime:
- `generate_dekr4_predictions.py`: batch-runs a trained dual-tool DEKR
  model over a dataset's image folders and writes per-tool prediction CSVs.
- `split_dlc_csv_by_tool.py`: splits a multi-animal DLC CSV into separate
  single-tool CSVs.
- `visualize_dekr.py`: overlays DEKR keypoint predictions and skeleton
  lines on the source frames and renders an MP4.

### 6. Shared parameters (`Parameter/`)
Robot DH parameters (`LND.json`) and stereo camera intrinsics
(`intrinsics.npz`) used by every pipeline above.

## Class architecture

```
core/
  RobotLink            Forward kinematics: joint angles -> 3-D point/shaft/
                        skeleton features in the robot base frame.
  ParticleFilter        Generic particle filter (scalar + vectorised API).
  StereoCamera          Stereo calibration load + undistort/rectify + project.

tracking/
  SimpleProjector        Lightweight camera-frame -> pixel projector (.npz
  (stereo.py)             intrinsics), with/without lens distortion.
  DLCRunner               Wraps a DeepLabCut PyTorch pose model for live
  (dlc_runner.py)         single-frame inference.
  ArmTracker              Ties one arm's RobotLink + ParticleFilter +
  (arm_tracker.py)        label assignment + shaft detection + observation
                          model together. Public API: update(), warmup(),
                          draw().

ros2_node/
  DualArmTrackingNode     ROS2 Node: camera/joint-state subscriptions ->
  (tracking_node.py)      DLCRunner + ArmTracker per arm -> pose/vis publish.
  OfflineTrackingNode     ROS2 Node: same as tracking/offline_tracking.py,
  (offline_node.py)       publishing results as topics instead of only
                          writing to disk.

edge_detection/
  ArmState                Per-arm loaded config/DLC/FK state used by the
  (render_canny_edges_video.py)  batch shaft-edge-detection video renderer.
```

`ArmTracker` is the composition root: it owns a `RobotLink` (kinematics),
a `ParticleFilter` (state estimation), a `SimpleProjector` (projection),
and calls into the free functions of `label_assignment.py`,
`canny_hough.py`, and `observation.py`/`shaft_edge.py` each frame. Both
`DualArmTrackingNode` and `tracking/offline_tracking.py`'s `main()`
construct one `ArmTracker` per configured arm and drive it frame by frame
— the only difference is where the frame/joint-angle data comes from
(a ROS2 topic vs. a file on disk).

## Directory layout

```
core/            Camera- and ROS2-independent math (kinematics, PF, stereo calib).
tracking/        Tracking pipeline: config, detection fusion, ArmTracker, offline CLI.
ros2_node/       ROS2 node wrappers around the tracking pipeline.
launch/          ROS2 launch files.
edge_detection/  Interactive tuner + batch renderer for the shaft-edge detector.
keypoints/       DEKR/DLC prediction generation, splitting, and visualization tools.
Parameter/       Shared robot DH parameters and stereo intrinsics.
```

## Usage

### Offline (no ROS2)

```bash
python3 tracking/offline_tracking.py \
    --data /path/to/dataset \
    [--handeye /path/to/pnp_handeye.yaml] \
    [--n_particles 500] \
    [--no_display]
```

Expects a dataset laid out as `regular/image/{left,right}_resized/` (PNG
frames + DLC prediction CSVs) and `regular/kinematic/{PSM1,PSM2}/` (per-frame
joint-angle JSON). Writes an annotated video and per-frame PNGs to
`tracking_output_offline/`.

### ROS2

```bash
ros2 launch surgical_tool_tracking_ros2 tracking.launch.py \
    arms:="['psm1', 'psm2']" n_particles:=300
```

Subscribes to `/camera/left/image_raw` and each arm's
`measured_js`/`jaw/measured_js` topics; publishes
`/tracking/{arm}/pose` and `/tracking/visualization`.

### Tuning the shaft-edge detector

```bash
python3 edge_detection/test_canny_hough.py --data /path/to/dataset --arm psm1
```

Adjust sliders interactively, press `p` to save (or `q` to save and quit)
to `{data}/canny_params_{arm}.json`, which `ArmTracker` loads automatically
on the next run.
