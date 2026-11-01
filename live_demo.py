"""Live demo entry point.

    python live_demo.py            # interactive scenario demo (default)
    python live_demo.py benchmark  # long-horizon endsem benchmark
"""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    if len(sys.argv) > 1 and sys.argv[1].lower() == "benchmark":
        runpy.run_path(str(here / "benchmark.py"), run_name="__main__")
    else:
        runpy.run_path(str(here / "demo.py"), run_name="__main__")
