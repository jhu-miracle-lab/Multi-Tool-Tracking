"""Shared numeric constants for all tracking pipelines."""

import numpy as np

LIKELIHOOD_THRESHOLD = 0.05
LIKELIHOOD_THRESH_OPT = 0.05

DLC_W, DLC_H = 640, 480
CAL_W, CAL_H = 1920, 1080

MIN_TRI_DIST_MM = 3.0

# --- Lumped Error observation model
LUMPED_GAMMA_M = 0.05  # point association cost weight [1/px^2]
LUMPED_GAMMA_PHI = 30.0  # edge association cost weight, theta term [1/rad]
LUMPED_CMAX_PT = 6.5  # point: ~p90 residual (11.2px) treated as missed
# LUMPED_GAMMA_RHO / LUMPED_CMAX_EDGE, retuned 2026-07-29 after fixing a real
# bug in tracking/shaft_edge.py's predict_shaft_edges_hough_batch(): it was
# missing the focal-length/principal-point conversion (Richter et al. 2022,
# "Robotic Tool Tracking Under Partially Visible Kinematic Chain", Appendix C
# Eq. 50) from the cylinder tangent-plane's normalized-camera-ray line to a
# real pixel Hough line -- it silently computed theta/rho as if f_x=f_y=1,
# c_x=c_y=0, leaving `rho` off from true pixel rho by roughly f_x (~1192 for
# this project's camera, not simply invertible as a rescale since c_x,c_y
# were also missing). The old values below (0.002 / 2.2, implying a design
# p90 of "364px rho / 2.93deg theta") were tuned against that wrong-scale
# rho, not real pixels.
#
# After the fix, real predicted-edge-vs-detected-edge residuals (straight/
# swap-matched, the same quantity this cost scores every frame) were
# measured against dataset 20260718_115614 (independent scheme, 1086
# frames, both arms, n=3762 edge samples): p90 rho = 11.76px, p90 theta =
# 1.52deg. An initial retune (0.06 / 1.5) preserved the original design's
# cost-budget split between rho and theta at this new p90 -- but swept
# against real keypoint-reprojection error (median/p90 px vs. real DEKR
# detections, seeded np.random for reproducible comparisons -- see
# analysis/error_analysis.py), that value measurably regressed psm2's p90
# (17.70px) vs. weaker rho weights (~14.3-14.9px across 0.0-0.02); gamma_rho
# turned out to be the dominant lever (raising cmax_edge alone, up to 6.0,
# had no further effect at gamma_rho=0.06 -- the cost rarely reaches cmax in
# practice). This means the "preserve the old cost-budget ratio" heuristic
# gave the shaft-edge term too much influence relative to how noisy this
# dataset's real Canny+Hough detections actually are: 0.0 (edge term fully
# off) performs about as well as any nonzero weight tested. Settled on a
# small nonzero LUMPED_GAMMA_RHO=0.01 (matches or slightly beats 0.0 on both
# arms' medians without regressing p90) rather than disabling the term
# outright, keeping LUMPED_CMAX_EDGE=1.5 (its exact value barely matters at
# this gamma_rho, since the cap is rarely approached).
LUMPED_GAMMA_RHO = 0.01  # edge association cost weight, rho term [1/px]
LUMPED_CMAX_EDGE = 1.5  # edge: cap rarely reached at this gamma_rho; see derivation above
LUMPED_N_POINTS = 7  # total point slots: yaw_1, yaw_2, roll_1, roll_2, pitch_1, pitch_2, pitch_3

# --- delta_T mode ---
# state = [dtx, dty, dtz, drx, dry, drz]; zero vector = identity = FK exactly correct
DELTA_PROCESS_STD = np.array([5e-4, 5e-4, 5e-4, 5e-3, 5e-3, 5e-3])

# --- Joint (dual-arm) optimization mode ---
# state = [dtx, dty, dtz, drx, dry, drz], shared by both arms via a fixed
# known b1_T_b2 base-to-base transform (see tracking/joint_tracker.py).
# 2x DELTA_PROCESS_STD -- swept (0.5x-5x) against real keypoint-reprojection
# error (median/p90 px vs. real DEKR detections, independent of any
# b1_T_b2/GT-anchor assumption) on dataset 20260718_115614: 2x gave the best
# psm1 p90 and a modest median improvement on both arms over 1x, while >=3x
# started making both arms measurably worse (particle cloud too diffuse).
# This does NOT close the gap to independent per-arm tracking on that
# dataset -- the dominant error source there is b1_T_b2 itself being off by
# ~85mm/3.5deg from an independently-solved cross-check (see
# pf_simulation/compute_b1_t_b2_from_independent_pnp.py); re-tune this
# constant if/when b1_T_b2 is re-calibrated, since the right process-noise
# scale depends on how much slack the filter needs to compensate for a
# possibly-imperfect b1_T_b2.
JOINT_DELTA_PROCESS_STD = np.array([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])

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
