"""Run APRT using the shared implementation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from APRT.config import ALLOW_PRIMARY_SWITCH


def main():
    from tracking.messages.offline_tracking import main as run
    return run(allow_primary_switch=ALLOW_PRIMARY_SWITCH)


if __name__ == "__main__":
    main()
