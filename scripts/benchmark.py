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

Results dict format: {T: (median_ms, q25_ms, q75_ms)}.
When loaded from an old 2-column CSV, q25/q75 are None.

Usage
-----
  python scripts/benchmark.py                # Python only
  python scripts/benchmark.py --dynare       # Python + Dynare (requires MATLAB)
  python scripts/benchmark.py --plot         # Python + load saved Dynare CSV + plot
  python scripts/benchmark.py --dynare --plot  # run both + plot
"""

import argparse
import datetime
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
    """Return {T: (median_ms, q25_ms, q75_ms)} for each T in T_VALUES."""
    print("Compiling model (process_model)...", flush=True)
    t0 = time.perf_counter()
    model_funcs = process_model([EQ_RESOURCE, EQ_EULER, EQ_AUX], VARS_DYN, vars_exo=VARS_EXO)
    compile_ms = (time.perf_counter() - t0) * 1000
    print(f"  process_model: {compile_ms:.1f} ms  (one-time cost, not included below)\n")

    results = {}
    for T in T_VALUES:
        exog       = np.ones((T, 1)) * Z_SS
        exog[0, 0] = 1.2
        k_neg1     = np.array([K_SS])  # only k is a stock variable

        times = []
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            sol = solve_perfect_foresight(
                T, {}, model_funcs, VARS_DYN,
                exog_path=exog,
                initial_state=k_neg1,
                stock_var_indices=[1],  # k is at index 1 in ["c", "k", "zl"]
                endval=SS,
                homotopy_fallback=False,
            )
            elapsed = time.perf_counter() - t0
            if not sol.success:
                raise RuntimeError(f"Solver failed at T={T}: {sol.message}")
            times.append(elapsed * 1000)

        times_arr = np.array(times)
        med = float(np.median(times_arr))
        q25 = float(np.percentile(times_arr, 25))
        q75 = float(np.percentile(times_arr, 75))
        results[T] = (med, q25, q75)
        print(f"  T={T:5d}: {med:7.2f} ms  [IQR {q25:.2f}–{q75:.2f}]  (median of {N_REPS} runs)")

    return results


# ---------------------------------------------------------------------------
# Dynare benchmark (via MATLAB subprocess)
# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MOD_FILE    = os.path.join(SCRIPTS_DIR, "benchmark_dynare.mod")
CSV_OUT     = os.path.join(SCRIPTS_DIR, "benchmark_dynare_times.csv")

_DEFAULT_DYNARE_PATH = r"C:\dynare\6.2\matlab"


def _read_dynare_nreps():
    """Parse N_reps from benchmark_dynare.mod so Python and MATLAB stay in sync."""
    import re
    try:
        with open(MOD_FILE) as fh:
            for line in fh:
                m = re.match(r"\s*N_reps\s*=\s*(\d+)\s*;", line)
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    return None


def _resolve_dynare_path(cli_arg):
    """Return the Dynare MATLAB path, preferring CLI arg > env var > default."""
    if cli_arg:
        return cli_arg
    env = os.environ.get("DYNARE_PATH")
    if env:
        return env
    return _DEFAULT_DYNARE_PATH


def _parse_dynare_csv(path):
    """
    Parse a Dynare benchmark CSV and return {T: (median_ms, q25_ms, q75_ms)}.

    Accepts both the new 4-column format [T, median_s, q25_s, q75_s] written
    by the updated benchmark_dynare.mod and the old 2-column format [T, median_s].
    When only 2 columns are present, q25/q75 are returned as None.
    """
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:          # single-row file
        data = data.reshape(1, -1)
    n_cols = data.shape[1]
    if n_cols == 4:
        return {int(row[0]): (row[1] * 1000, row[2] * 1000, row[3] * 1000)
                for row in data}
    if n_cols == 2:
        return {int(row[0]): (row[1] * 1000, None, None) for row in data}
    raise ValueError(
        f"Unexpected column count {n_cols} in Dynare CSV {path!r}; "
        f"expected 2 (legacy) or 4 columns. Detected shape: {data.shape}."
    )


def load_dynare_csv():
    """
    Load a previously saved Dynare timing CSV.

    Returns
    -------
    results : dict  {T: (median_ms, q25_ms, q75_ms)}
    measured_on : str  ISO-8601 UTC file-modification timestamp
    """
    mtime = os.path.getmtime(CSV_OUT)
    measured_on = datetime.datetime.fromtimestamp(
        mtime, tz=datetime.timezone.utc
    ).isoformat(timespec="minutes")
    results = _parse_dynare_csv(CSV_OUT)
    return results, measured_on


def benchmark_dynare(dynare_path):
    """Run Dynare via MATLAB and return {T: (median_ms, q25_ms, q75_ms)}."""
    if not os.path.exists(MOD_FILE):
        raise FileNotFoundError(f"Dynare .mod file not found: {MOD_FILE}")
    if not os.path.isdir(dynare_path):
        raise FileNotFoundError(
            f"Dynare MATLAB path not found: {dynare_path}\n"
            "Set it with --dynare-path or the DYNARE_PATH environment variable."
        )

    def _mesc(p):
        return p.replace("'", "''")

    matlab_cmd = (
        f"addpath('{_mesc(dynare_path)}'); "
        f"cd('{_mesc(SCRIPTS_DIR)}'); "
        f"dynare benchmark_dynare nointeractive nolog; "
        f"exit;"
    )
    # Remove any stale CSV so a failed MATLAB run cannot silently reuse old data.
    if os.path.exists(CSV_OUT):
        os.remove(CSV_OUT)

    print("Running Dynare benchmark via MATLAB (this may take a minute)...", flush=True)
    result = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True, text=True, timeout=600,
    )
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("Warning"):
            print(" ", stripped)

    if result.returncode != 0:
        print("MATLAB stderr:", result.stderr[-3000:], file=sys.stderr)
        raise RuntimeError("MATLAB/Dynare benchmark failed.")

    if not os.path.exists(CSV_OUT):
        raise FileNotFoundError(f"Expected output CSV not found: {CSV_OUT}")

    return _parse_dynare_csv(CSV_OUT)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_table(py_results, dynare_results=None, dynare_nreps=N_REPS):
    header = f"{'T':>6}  {'Python (ms)':>12}"
    if dynare_results is not None:
        header += f"  {'Dynare (ms)':>12}  {'Speedup':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for T in T_VALUES:
        py_entry = py_results.get(T)
        py_ms    = py_entry[0] if py_entry is not None else None
        row      = f"{T:6d}  {py_ms:12.2f}" if py_ms is not None else f"{T:6d}  {'N/A':>12}"
        if dynare_results is not None:
            dyn_entry = dynare_results.get(T)
            dyn_ms    = dyn_entry[0] if dyn_entry is not None else None
            if dyn_ms is not None and py_ms is not None:
                row += f"  {dyn_ms:12.2f}  {dyn_ms / py_ms:7.1f}x"
            else:
                row += f"  {'N/A':>12}  {'N/A':>8}"
        print(row)
    print("=" * len(header))
    dyn_reps_str = (f", Dynare: {dynare_nreps} runs" if dynare_nreps is not None
                    else ", Dynare: runs unknown (loaded from CSV)")
    print(f"Median solve time — Python: {N_REPS} runs" +
          (dyn_reps_str if dynare_results is not None else "") + ".")
    print("Solver only (excludes process_model / perfect_foresight_setup).")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_benchmark(py_results, dynare_results, out_path,
                   dynare_source="live", dynare_measured_on=None, dynare_nreps=N_REPS,
                   show=True):
    """
    Parameters
    ----------
    dynare_source : "live" | "csv"
        Whether Dynare timings came from a fresh MATLAB run or a saved CSV.
    dynare_measured_on : str or None
        Human-readable timestamp for when the CSV was recorded.
    show : bool
        Call plt.show() after saving. Set False for headless/CI environments.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print(
            "Error: plotting requires matplotlib. "
            "Install it with: pip install matplotlib  (or pip install -e '.[dev]')",
            file=sys.stderr,
        )
        raise SystemExit(1)

    Ts = [T for T in T_VALUES if T in py_results and T in dynare_results]
    if not Ts:
        raise ValueError(
            "No overlapping horizons between py_results and dynare_results. "
            f"Python has T={list(py_results)}, Dynare has T={list(dynare_results)}."
        )

    py_med  = [py_results[T][0]     for T in Ts]
    py_q25  = [py_results[T][1]     for T in Ts]
    py_q75  = [py_results[T][2]     for T in Ts]
    dyn_med = [dynare_results[T][0] for T in Ts]
    dyn_q25 = [dynare_results[T][1] for T in Ts]
    dyn_q75 = [dynare_results[T][2] for T in Ts]
    speedup = [d / p for d, p in zip(dyn_med, py_med)]

    has_py_iqr  = all(v is not None for v in py_q25)
    has_dyn_iqr = all(v is not None for v in dyn_q25)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle(
        "Perfect foresight solver: pyperfectforesight vs Dynare 6.2\n"
        "RBC model (3 variables) — median solve time, solver only",
        fontsize=11,
    )

    # --- Panel 1: solve time (log–log) with IQR error bars ------------------
    eb_kw = dict(fmt="o-", linewidth=2, markersize=6, capsize=3, capthick=1.2)

    if has_py_iqr:
        py_err = (
            [m - q25 for m, q25 in zip(py_med, py_q25)],
            [q75 - m for m, q75 in zip(py_med, py_q75)],
        )
        ax1.errorbar(Ts, py_med, yerr=py_err, color="#1f77b4",
                     label="pyperfectforesight (Python)", **eb_kw)
    else:
        ax1.plot(Ts, py_med, "o-", color="#1f77b4", linewidth=2, markersize=6,
                 label="pyperfectforesight (Python)")

    dyn_eb_kw = dict(fmt="s--", linewidth=2, markersize=6, capsize=3, capthick=1.2)
    if has_dyn_iqr:
        dyn_err = (
            [m - q25 for m, q25 in zip(dyn_med, dyn_q25)],
            [q75 - m for m, q75 in zip(dyn_med, dyn_q75)],
        )
        ax1.errorbar(Ts, dyn_med, yerr=dyn_err, color="#ff7f0e",
                     label="Dynare 6.2 (MATLAB)", **dyn_eb_kw)
    else:
        ax1.plot(Ts, dyn_med, "s--", color="#ff7f0e", linewidth=2, markersize=6,
                 label="Dynare 6.2 (MATLAB)")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Simulation horizon T", fontsize=11)
    ax1.set_ylabel("Median solve time (ms)", fontsize=11)
    ax1.set_title("Solve time vs horizon (log–log)", fontsize=11)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.set_xticks(Ts)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    for T, py, dyn in zip(Ts, py_med, dyn_med):
        ax1.annotate(f"{py:.1f}", (T, py),   textcoords="offset points",
                     xytext=(-4, 6),   fontsize=7.5, color="#1f77b4")
        ax1.annotate(f"{dyn:.1f}", (T, dyn), textcoords="offset points",
                     xytext=(-4, -13), fontsize=7.5, color="#ff7f0e")

    # --- Panel 2: speedup ratio (bar chart) ---------------------------------
    bar_x = range(len(Ts))
    bars  = ax2.bar(bar_x, speedup, color="#2ca02c", alpha=0.82, edgecolor="white")
    ax2.set_xticks(list(bar_x))
    ax2.set_xticklabels([str(T) for T in Ts])
    ax2.set_xlabel("Simulation horizon T", fontsize=11)
    ax2.set_ylabel("Speedup  (Dynare time / Python time)", fontsize=11)
    ax2.set_title("Speedup of pyperfectforesight over Dynare", fontsize=11)
    ax2.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax2.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    # Y-axis: keep baseline at zero to avoid exaggerating differences
    margin = (max(speedup) - min(speedup)) * 0.5
    ax2.set_ylim(bottom=0, top=max(speedup) + margin + 3)
    for bar, val in zip(bars, speedup):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + margin * 0.1,
            f"{val:.1f}x",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # --- Footnote -----------------------------------------------------------
    dyn_reps_str = (f"{dynare_nreps} runs" if dynare_nreps is not None
                    else "runs unknown")
    note_parts = [
        f"Python: {N_REPS} runs each T.",
        f"Dynare: {dyn_reps_str} each T.",
    ]
    if dynare_source == "csv":
        ts_str = f" measured {dynare_measured_on}" if dynare_measured_on else ""
        note_parts.append(f"Dynare timings from saved CSV{ts_str}; Python timings are fresh.")
    note = "  ".join(note_parts)
    fig.subplots_adjust(bottom=0.15)
    fig.text(0.5, 0.02, note, ha="center", fontsize=8, color="#555555", style="italic")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    if show:
        plt.show()


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
    parser.add_argument("--plot", action="store_true",
                        help="Produce a comparison plot (requires Dynare results, "
                             "either from --dynare or a saved benchmark_dynare_times.csv)")
    parser.add_argument("--plot-out", default=None,
                        help="Output path for the plot PNG "
                             "(default: docs/benchmark_plot.png)")
    parser.add_argument("--no-show", action="store_true",
                        help="Save the plot without calling plt.show() "
                             "(useful in headless/CI environments)")
    args = parser.parse_args()

    print("=== pyperfectforesight benchmark ===\n")
    py_results = benchmark_python()

    dynare_results     = None
    dynare_source      = "live"
    dynare_measured_on = None
    dynare_nreps       = None   # set from .mod file for live runs; unknown for CSV

    if args.dynare:
        print("\n=== Dynare 6.2 benchmark ===\n")
        dynare_nreps   = _read_dynare_nreps()   # reads N_reps from .mod file
        dynare_results = benchmark_dynare(_resolve_dynare_path(args.dynare_path))
    elif args.plot and os.path.exists(CSV_OUT):
        print(f"\nLoading saved Dynare timings from {CSV_OUT}")
        dynare_results, dynare_measured_on = load_dynare_csv()
        dynare_source = "csv"
        # Rep count is not stored in the CSV — report as unknown.
        dynare_nreps = None
        for T, entry in sorted(dynare_results.items()):
            med, q25, q75 = entry
            iqr_str = f"  [IQR {q25:.2f}–{q75:.2f}]" if q25 is not None else ""
            print(f"  T={T:5d}: {med:8.2f} ms{iqr_str}")
        print(f"  (measured {dynare_measured_on})")

    print_table(py_results, dynare_results, dynare_nreps=dynare_nreps)

    if args.plot:
        if dynare_results is None:
            print("\nError: --plot was requested but no Dynare results are available. "
                  "Pass --dynare or ensure benchmark_dynare_times.csv exists.",
                  file=sys.stderr)
            sys.exit(1)
        else:
            out = args.plot_out or os.path.join(
                os.path.dirname(SCRIPTS_DIR), "docs", "benchmark_plot.png"
            )
            plot_benchmark(py_results, dynare_results, out,
                           dynare_source=dynare_source,
                           dynare_measured_on=dynare_measured_on,
                           dynare_nreps=dynare_nreps,
                           show=not args.no_show)
