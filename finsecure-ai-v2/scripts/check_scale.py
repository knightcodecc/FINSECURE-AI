"""
check_scale.py — Power-Law Multi-Size Scale Benchmark & Resource Extrapolation
=================================================================================
Measures execution time and peak RSS RAM (via psutil including child processes)
for graph_features.py across 4 dataset sizes (100k, 200k, 300k, 600k rows).
Fits a power law y = a * x^b (printing fitted exponent b) for time and RAM,
and extrapolates resource requirements for full 6.36M-row PaySim dataset.
"""

import os
import sys
import time
from pathlib import Path
import psutil
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph_features import attach_graph_features_expanding


def get_peak_ram_mb() -> float:
    """Measure current process + children RSS RAM in MB."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            mem += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return mem / (1024 * 1024)


def benchmark_graph_size(df: pd.DataFrame, n_rows: int) -> tuple[float, float]:
    # Tile dataset if requested size exceeds loaded dataframe length
    if len(df) < n_rows:
        reps = (n_rows // len(df)) + 1
        sample_df = pd.concat([df] * reps, ignore_index=True).iloc[:n_rows].copy()
    else:
        sample_df = df.iloc[:n_rows].copy()

    ram_before = get_peak_ram_mb()
    t0 = time.time()

    _ = attach_graph_features_expanding(sample_df)

    duration = time.time() - t0
    ram_after = get_peak_ram_mb()
    peak_ram = max(ram_before, ram_after)

    return duration, peak_ram


def main():
    print("==========================================================================")
    print("      FinSecure AI v2 — GRAPH FEATURES POWER-LAW SCALE BENCHMARK      ")
    print("==========================================================================")

    data_path = Path("data/paysim_synth.csv")
    if not data_path.exists():
        print("Generating synthetic dataset first...")
        from scripts.make_synthetic_paysim import generate_synthetic_paysim
        df_full = generate_synthetic_paysim()
    else:
        df_full = pd.read_csv(data_path)

    sizes = [100_000, 200_000, 300_000, 600_000]
    times = []
    rams = []

    print(f"\nBenchmarking graph_features.py across sizes {sizes} using psutil peak RAM...")
    print(f"{'Size (Rows)':<15} | {'Time (seconds)':<15} | {'Peak RSS RAM (MB)':<20}")
    print("-" * 55)

    for n in sizes:
        dur, ram = benchmark_graph_size(df_full, n)
        times.append(dur)
        rams.append(ram)
        print(f"{n:<15,} | {dur:<15.2f} | {ram:<20.2f}")

    # Fit Power Law: log(y) = log(a) + b * log(x)
    x_arr = np.array(sizes, dtype=float)
    y_time = np.array(times, dtype=float)
    y_ram = np.array(rams, dtype=float)

    # Time power law fit
    poly_time = np.polyfit(np.log(x_arr), np.log(y_time), 1)
    b_time = poly_time[0]  # Exponent for time
    log_a_time = poly_time[1]

    # RAM power law fit
    poly_ram = np.polyfit(np.log(x_arr), np.log(y_ram), 1)
    b_ram = poly_ram[0]    # Exponent for RAM
    log_a_ram = poly_ram[1]

    # Extrapolate to 6,362,620 rows using fitted power law
    full_rows = 6_362_620
    extrap_time_sec = np.exp(log_a_time) * (full_rows ** b_time)
    extrap_time_min = extrap_time_sec / 60.0

    extrap_ram_mb = np.exp(log_a_ram) * (full_rows ** b_ram)
    extrap_ram_gb = extrap_ram_mb / 1024.0

    pass_time = extrap_time_min <= 30.0
    pass_ram = extrap_ram_gb <= 10.0

    print("\n--------------------------------------------------------------------------")
    print(f"FITTED POWER LAW PROFILE (y = a * x^b):")
    print(f"  - Execution Time Power Exponent (b_time) : {b_time:.4f}")
    print(f"  - Peak RAM Power Exponent (b_ram)       : {b_ram:.4f}")

    print("\n==========================================================================")
    print("      EXTRAPOLATED RESOURCE ESTIMATES FOR FULL PAYSIM (6.36M ROWS)      ")
    print("==========================================================================")
    print(f"  - Extrapolated Execution Time : {extrap_time_min:.2f} minutes (Target <= 30.0 mins) -> {'PASS' if pass_time else 'FAIL'}")
    print(f"  - Extrapolated Peak RSS RAM   : {extrap_ram_gb:.2f} GB (Target <= 10.0 GB)       -> {'PASS' if pass_ram else 'FAIL'}")
    print(f"  - Overall Scale Target Status : {'PASS' if (pass_time and pass_ram) else 'FAIL'}")
    print("==========================================================================\n")


if __name__ == "__main__":
    main()
