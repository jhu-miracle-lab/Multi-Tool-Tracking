"""Run FPRT using the shared implementation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from FPRT.config import ALLOW_PRIMARY_SWITCH


def main():
    from ros2_node.offline_node import main as run
    return run(allow_primary_switch=ALLOW_PRIMARY_SWITCH)


if __name__ == "__main__":
    main()
