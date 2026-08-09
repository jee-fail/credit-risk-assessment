"""
DEPRECATED wrapper — use scripts/retarget_threshold.py instead.

  .\\.venv\\Scripts\\python.exe scripts\\retarget_threshold.py --strategy max_accuracy
  .\\.venv\\Scripts\\python.exe scripts\\retarget_threshold.py --strategy middle_ground   # production
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

print(
    "NOTE: retarget_threshold_max_accuracy.py is deprecated. "
    "Forwarding to scripts/retarget_threshold.py --strategy max_accuracy",
    file=sys.stderr,
)
script = Path(__file__).with_name("retarget_threshold.py")
sys.argv = [str(script), "--strategy", "max_accuracy", *sys.argv[1:]]
runpy.run_path(str(script), run_name="__main__")
