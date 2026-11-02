"""Fixed tracker configuration; no environment-variable feature switches.

Method selection is passed explicitly to CombinedArmTracker by each entry.
These numerical defaults are shared by FPRT and APRT.
"""

SWITCH_RESET_COVARIANCE_SCALE = 1.0
OVERLAP_BETA = 3.0
PRIOR_SHAFT_ROI = True
