# APRT Entry Points

`config.py` explicitly sets `ALLOW_PRIMARY_SWITCH = True`. The three entry scripts call the shared implementation in the project root instead of switching algorithm configurations through symbolic links.

Run from `project_code/`:

```bash
python APRT/offline_tracking.py --data /path/to/dataset --no_display
python APRT/ros_tracking.py
python APRT/ros_offline.py --ros-args -p data:=/path/to/dataset
```

See the [shared project README](../README.md) for configuration, algorithms, and interfaces.

APRT allows either arm to be primary, using the same reliability thresholds and hysteresis rules.
