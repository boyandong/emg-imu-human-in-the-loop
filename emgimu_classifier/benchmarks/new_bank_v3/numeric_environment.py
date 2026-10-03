"""Small, non-identifying numerical-runtime fingerprint for V3 native runs."""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import joblib
import numpy
import scipy
import sklearn
import threadpoolctl


CONTROL_VARS = ("PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def fingerprint() -> dict:
    pools = []
    for row in threadpoolctl.threadpool_info():
        pools.append({key: row.get(key) for key in
                      ("user_api", "internal_api", "prefix", "version", "num_threads")})
    pools.sort(key=lambda row: (str(row["user_api"]), str(row["internal_api"]),
                                str(row["prefix"]), str(row["version"])))
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "packages": {
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
            "threadpoolctl": threadpoolctl.__version__,
        },
        "threadpools": pools,
        "thread_controls": {key: os.environ.get(key) for key in CONTROL_VARS},
    }


def verify_saved(root: Path | None = None) -> list[str]:
    folder = root or Path(__file__).resolve().parent
    current = fingerprint()
    mismatches = []
    for name in ("SPEC_F2C_GRAB_RESULTS.json", "SPEC_F2C_WEARING_RESULTS.json",
                 "SPEC_F2C_TRANSFER_RESULTS.json"):
        recorded = json.loads((folder / name).read_text(encoding="utf-8"))
        if recorded.get("numeric_environment") != current:
            mismatches.append(name)
    return mismatches


if __name__ == "__main__":
    mismatches = verify_saved()
    if mismatches:
        print("Numerical environment differs from saved run: " + ", ".join(mismatches),
              file=sys.stderr)
        raise SystemExit(1)
    print("Numerical environment matches all three saved V3 F2c runs")
