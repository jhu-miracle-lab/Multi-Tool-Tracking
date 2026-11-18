# FPRT Entry Points

`config.py` explicitly sets `ALLOW_PRIMARY_SWITCH = False`. The three entry scripts call the shared implementation in the project root instead of switching algorithm configurations through symbolic links.

Run from `project_code/`:

```bash
python FPRT/offline_tracking.py --data /path/to/dataset --no_display
python FPRT/ros_tracking.py
python FPRT/ros_offline.py --ros-args -p data:=/path/to/dataset
```

See the [shared project README](../README.md) for configuration, algorithms, and interfaces.

FPRT allows only the first arm (usually PSM1) to be primary, with reliability gating still enabled.
