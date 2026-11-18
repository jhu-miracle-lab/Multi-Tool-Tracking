"""Locate calibration assets in source checkouts and installed packages."""
import os
from pathlib import Path
import sys

PACKAGE_NAME = "surgical_tool_tracking_ros2"


def _has_assets(path):
    return all((path / "Parameter" / name).is_file()
               for name in ("stereo_calibration.yaml", "LND.json"))


def resolve_asset_root():
    """Return a complete resource directory without requiring ROS for pip use."""
    override = os.environ.get("STT_SRC_DIR")
    if override:
        path = Path(override).expanduser().resolve()
        if not _has_assets(path):
            raise FileNotFoundError(f"STT_SRC_DIR has no complete Parameter directory: {path}")
        return path

    source_root = Path(__file__).resolve().parents[2]
    if _has_assets(source_root):
        return source_root

    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
    except ImportError:
        pass
    else:
        try:
            candidates.append(Path(get_package_share_directory(PACKAGE_NAME)))
        except PackageNotFoundError:
            pass

    candidates.append(Path(sys.prefix) / "share" / PACKAGE_NAME)
    # Covers pip --target, user installs, and ROS prefix/lib/pythonX/site-packages.
    candidates.extend(parent / "share" / PACKAGE_NAME
                      for parent in (source_root, *source_root.parents))
    checked = []
    for path in candidates:
        if path in checked:
            continue
        checked.append(path)
        if _has_assets(path):
            return path
    paths = "\n  ".join(str(path) for path in [source_root, *checked])
    raise FileNotFoundError(
        "Cannot locate stereo_calibration.yaml and LND.json. "
        f"Set STT_SRC_DIR to their project/resource root. Checked:\n  {paths}")
