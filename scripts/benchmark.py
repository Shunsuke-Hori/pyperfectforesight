"""Benchmark pyperfectforesight against Dynare 6.2 on the RBC model.

Both solvers are timed on the same model and shock:
  - RBC: resource constraint + Euler equation + auxiliary equation for z(+1)
  - Parameters: alpha=0.5, sigma=0.5, delta=0.02, beta=1/1.05
  - Shock: z = 1.2 in period 1, z = 1 thereafter
  - Horizons: T = 50, 100, 200, 500, 1000

The Python model explicitly includes the auxiliary variable for z(+1) that
Dynare adds automatically ("Substitution of exo leads"), so both solvers
operate on the same 3-equation, 3-variable system.

Only solver time is measured (process_model / perfect_foresight_setup excluded).

Usage
-----
  python scripts/benchmark.py                # Python only
  python scripts/benchmark.py --dynare       # Python + Dynare (requires MATLAB)
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyperfectforesight import v, process_model, solve_perfect_foresight

# ---------------------------------------------------------------------------
# Model: exact match to benchmark_dynare.mod (including Dynare's auxiliary
# variable substitution for z(+1)).
#
# Dynare automatically replaces z(+1) in the Euler equation with an auxiliary
# endogenous variable zl and appends the equation  zl(0) = z(+1).  We do the
# same here so both solvers operate on an identical 3×3 system.
# ---------------------------------------------------------------------------
ALPHA = 0.5
SIGMA = 0.5
DELTA = 0.02
BETA  = 1 / 1.05
Z_SS  = 1.0

EQ_RESOURCE = (
    v("c", 0) + v("k", 0)
    - v("z", 0) * v("k", -1) ** ALPHA
    - (1 - DELTA) * v("k", -1)
)
EQ_EULER = (
    v("c", 0) ** (-SIGMA)
    - BETA * (ALPHA * v("zl", 0) * v("k", 0) ** (ALPHA - 1) + 1 - DELTA)
    * v("c", 1) ** (-SIGMA)
)
EQ_AUX = v("zl", 0) - v("z", 1)   # zl(t) = z(t+1): mirrors Dynare's exo-lead substitution

VARS_DYN = ["c", "k", "zl"]
VARS_EXO = ["z"]

K_SS = ((1 / BETA - (1 - DELTA)) / (Z_SS * ALPHA)) ** (1 / (ALPHA - 1))
C_SS = Z_SS * K_SS ** ALPHA - DELTA * K_SS
SS   = np.array([C_SS, K_SS, Z_SS])

T_VALUES = [50, 100, 200, 500, 1000]
N_REPS   = 20


# ---------------------------------------------------------------------------
# Python benchmark
# ---------------------------------------------------------------------------
def benchmark_python():
    print("Compiling model (process_model)...", flush=True)
    t0 = time.perf_counter()
    model_funcs = process_model([EQ_RESOURCE, EQ_EULER, EQ_AUX], VARS_DYN, vars_exo=VARS_EXO)
    compile_ms = (time.perf_counter() - t0) * 1000
    print(f"  process_model: {compile_ms:.1f} ms  (one-time cost, not included below)\n")

    results = {}
    for T in T_VALUES:
        exog      = np.ones((T, 1)) * Z_SS
        exog[0, 0] = 1.2
        k_neg1    = np.array([K_SS])  # only k is a stock variable

        times = []
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            sol = solve_perfect_foresight(
                T, {}, SS, model_funcs, VARS_DYN,
                exog_path=exog,
                initial_state=k_neg1,
                stock_var_indices=[1],  # k is at index 1 in ["c", "k", "zl"]
                homotopy_fallback=False,
            )
            times.append(time.perf_counter() - t0)

        assert sol.success, f"Solver failed at T={T}: {sol.message}"
        med_ms = np.median(times) * 1000
        results[T] = med_ms
        print(f"  T={T:5d}: {med_ms:8.2f} ms  (median of {N_REPS} runs)")

    return results


# ---------------------------------------------------------------------------
# Dynare benchmark (via MATLAB subprocess)
# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MOD_FILE    = os.path.join(SCRIPTS_DIR, "benchmark_dynare.mod")
CSV_OUT     = os.path.join(SCRIPTS_DIR, "benchmark_dynare_times.csv")

_DEFAULT_DYNARE_PATH = r"C:\dynare\6.2\matlab"


def _resolve_dynare_path(cli_arg):
    """Return the Dynare MATLAB path, preferring CLI arg > env var > default."""
    if cli_arg:
        return cli_arg
    env = os.environ.get("DYNARE_PATH")
    if env:
        return env
    return _DEFAULT_DYNARE_PATH


def benchmark_dynare(dynare_path):
    if not os.path.exists(MOD_FILE):
        raise FileNotFoundError(f"Dynare .mod file not found: {MOD_FILE}")
    if not os.path.isdir(dynare_path):
        raise FileNotFoundError(
            f"Dynare MATLAB path not found: {dynare_path}\n"
            "Set it with --dynare-path or the DYNARE_PATH environment variable."
        )

    # Escape single quotes for MATLAB string literals (MATLAB uses '' inside '…').
    def _mesc(p):
        return p.replace("'", "''")

    matlab_cmd = (
        f"addpath('{_mesc(dynare_path)}'); "
        f"cd('{_mesc(SCRIPTS_DIR)}'); "
        f"dynare benchmark_dynare nointeractive nolog; "
        f"exit;"
    )
    print("Running Dynare benchmark via MATLAB (this may take a minute)...", flush=True)
    result = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True, text=True, timeout=600,
    )
    # Echo any Dynare fprintf output
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("Warning"):
            print(" ", stripped)

    if result.returncode != 0:
        print("MATLAB stderr:", result.stderr[-3000:], file=sys.stderr)
        raise RuntimeError("MATLAB/Dynare benchmark failed.")

    if not os.path.exists(CSV_OUT):
        raise FileNotFoundError(f"Expected output CSV not found: {CSV_OUT}")

    data = np.loadtxt(CSV_OUT, delimiter=",")
    return {int(row[0]): row[1] * 1000 for row in data}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_table(py_results, dynare_results=None):
    header = f"{'T':>6}  {'Python (ms)':>12}"
    if dynare_results:
        header += f"  {'Dynare (ms)':>12}  {'Speedup':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for T in T_VALUES:
        py_ms  = py_results.get(T)
        row    = f"{T:6d}  {py_ms:12.2f}" if py_ms is not None else f"{T:6d}  {'N/A':>12}"
        if dynare_results:
            dyn_ms = dynare_results.get(T)
            if dyn_ms is not None and py_ms is not None:
                row += f"  {dyn_ms:12.2f}  {dyn_ms/py_ms:7.1f}x"
            else:
                row += f"  {'N/A':>12}  {'N/A':>8}"
        print(row)
    print("=" * len(header))
    note_py  = f"{N_REPS} runs"
    note_dyn = "10 runs" if dynare_results else ""
    print(f"Median solve time — Python: {note_py}" +
          (f", Dynare: {note_dyn}" if note_dyn else "") + ".")
    print("Solver only (excludes process_model / perfect_foresight_setup).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dynare", action="store_true",
                        help="Also run Dynare benchmark (requires MATLAB with Dynare 6.2)")
    parser.add_argument("--dynare-path", default=None,
                        help="Path to the Dynare MATLAB directory "
                             "(default: DYNARE_PATH env var, then C:\\dynare\\6.2\\matlab)")
    args = parser.parse_args()

    print("=== pyperfectforesight benchmark ===\n")
    py_results = benchmark_python()

    dynare_results = None
    if args.dynare:
        print("\n=== Dynare 6.2 benchmark ===\n")
        dynare_results = benchmark_dynare(_resolve_dynare_path(args.dynare_path))

    print_table(py_results, dynare_results)
