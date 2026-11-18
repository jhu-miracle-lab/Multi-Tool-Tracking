"""Shared shaft geometry and isolated detector defaults.

Dataset shaft_params_<arm>.json may contain common, canny_hough and elsed
sections. Legacy flat canny_params files remain accepted.
"""
COMMON_SHAFT_PARAMS = {
    "psm1": dict(min_w=38, max_w=60, angle_tol=18, par_tol=4, roi_pad=140, edge_margin=0),
    "psm2": dict(min_w=43, max_w=63, angle_tol=36, par_tol=5, roi_pad=79, edge_margin=30),
}
CANNY_HOUGH_PARAMS = {
    "psm1": dict(blur=2, canny_lo=49, canny_hi=92, do_skel=False, hough_thr=57, temporal_tol=29),
    "psm2": dict(blur=2, canny_lo=25, canny_hi=103, do_skel=False, hough_thr=34, temporal_tol=15),
}
ELSED_NATIVE_PARAMS = dict(
    sigma=1, gradientThreshold=30, minLineLen=15, lineFitErrThreshold=0.2,
    pxToSegmentDistTh=1.5, validationTh=0.15, validate=True, treatJunctions=True,
)
ELSED_PARAMS = dict(
    **ELSED_NATIVE_PARAMS, max_candidates=40, long_edge_weight=0.25,
    temporal_weight=0.5, temporal_cap_px=40, association_angle_weight=50,
)


def resolve_shaft_params(name, backend, overrides=None):
    """Produce an independent flat runtime config containing only active keys."""
    if backend not in ("elsed", "canny_hough"):
        raise ValueError(f"Unknown shaft backend: {backend}")
    if name not in COMMON_SHAFT_PARAMS and overrides is None:
        return None
    common = COMMON_SHAFT_PARAMS.get(name, {})
    specific = ELSED_PARAMS if backend == "elsed" else CANNY_HOUGH_PARAMS.get(name, {})
    config = {**common, **specific}
    overrides = overrides or {}
    # Recognize shared keys even for custom arm names.
    common_keys = set(COMMON_SHAFT_PARAMS["psm1"])
    backend_keys = set(ELSED_PARAMS if backend == "elsed" else CANNY_HOUGH_PARAMS["psm1"])
    config.update({k: v for k, v in overrides.items() if k in common_keys | backend_keys})
    for section, allowed in (("common", common_keys), (backend, backend_keys)):
        values = overrides.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Shaft parameter section {section} must be an object")
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown {section} shaft parameters: {sorted(unknown)}")
        config.update(values)
    return config
