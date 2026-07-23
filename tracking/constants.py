"""Shared numeric constants for all tracking pipelines."""

import numpy as np

LIKELIHOOD_THRESHOLD = 0.05
LIKELIHOOD_THRESH_OPT = 0.05

DLC_W, DLC_H = 640, 480
CAL_W, CAL_H = 1920, 1080

MIN_TRI_DIST_MM = 3.0

# --- Lumped Error observation model
LUMPED_GAMMA_M = 0.05  # point association cost weight [1/px^2]
LUMPED_GAMMA_RHO = 0.002  # edge association cost weight, rho term [1/px]
LUMPED_GAMMA_PHI = 30.0  # edge association cost weight, theta term [1/rad]
LUMPED_CMAX_PT = 6.5  # point: ~p90 residual (11.2px) treated as missed
LUMPED_CMAX_EDGE = 2.2  # edge: ~p90 residual (364px rho / 2.93deg theta) treated as missed
LUMPED_N_POINTS = 7  # total point slots: yaw_1, yaw_2, roll_1, roll_2, pitch_1, pitch_2, pitch_3

# --- delta_T mode ---
# state = [dtx, dty, dtz, drx, dry, drz]; zero vector = identity = FK exactly correct
DELTA_PROCESS_STD = np.array([5e-4, 5e-4, 5e-4, 5e-3, 5e-3, 5e-3])

# --- Canny+Hough shaft edge PF observation ---
SHAFT_EDGE_EMA_ALPHA = 0.35  # weight of new detection in EMA smoothing (0=frozen,1=raw)

# Live shaft detection always goes through tracking/canny_hough.py; this
# flag is kept for compatibility with ArmTracker's detection branch.
SHAFT_USE_HOUGH = True

# --- DEKR multi-animal live model (tracking/dekr_live_runner.py) ---
# Bodypart order the DEKR model's keypoint head was trained with.
DEKR_BODYPARTS = ["yaw_1", "yaw_2", "roll_1", "roll_2", "pitch_1", "pitch_2", "pitch_3"]
DEKR_PCUTOFF = 0.5
# Fixed index<->arm mapping for DekrLiveRunner's identity-head-resolved
# output (tracking/dekr_live_runner.py's assign_tool_identities(): index 0
# = tool1, index 1 = tool2) -- a training-data convention, not derived at
# runtime. This is the ONLY place instance-to-arm identity is decided in
# the live pipeline.
DEKR_ARM_ORDER = ["psm1", "psm2"]
