#!/usr/bin/env python3
"""
Tensora Benchmark Engine (Upgraded)

Features:
  - Command-line interface
  - Multi-run experiment batching
  - Baseline vs Tension solver comparison
  - FLOP, time, and tour-length statistics
  - Optional plotting (matplotlib)
  - JSON logging to results/
"""

import time
import json
import argparse
from pathlib import Path
import numpy as np

from flops_counter import GLOBAL_FLOPS
from tsp_problem import TSPProblem
from baseline_solver import BaselineSolver
from tension_solver import TensionSolver
from spring_tension import SpringTension

# -----------------------------------------------------------
# Core experiment for one run
# -----------------------------------------------------------
def run_single(num_cities, steps, tension_k, seed):
    np.random.seed(seed)

    problem = TSPProblem.random_instance(num_cities)

    # -------- Baseline --------
    GLOBAL_FLOPS.reset()
    baseline = BaselineSolver(problem, steps=steps, step_size=0.002)
    t0 = time.time()
    sol_b = baseline.solve()
    t1 = time.time()

    length_b = problem.tour_length(sol_b)
    flops_b = GLOBAL_FLOPS.get()
    time_b = t1 - t0

    # -------- Tension --------
    GLOBAL_FLOPS.reset()
    field = SpringTension(k=tension_k)
    tens = TensionSolver(problem, tension_field=field, steps=steps, step_size=0.002)

    t0 = time.time()
    sol_t = tens.solve()
    t1 = time.time()

    length_t = problem.tour_length(sol_t)
    flops_t = GLOBAL_FLOPS.get()
    time_t = t1 - t0

    return {
        "baseline": {"length": float(length_b), "flops": flops_b, "time": time_b},
        "tension": {"length": float(length_t), "flops": flops_t, "time": time_t},
    }


# -----------------------------------------------------------
# Pretty-print helpers
# -----------------------------------------------------------
def bold(s): return f"\033[1m{s}\033[0m"
def green(s): return f"\033[92m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def cyan(s): return f"\033[96m{s}\033[0m"


def print_summary(results):
    print("\n" + bold("=== Tensora Benchmark Summary ===") + "\n")

    def stat(arr):
        return np.mean(arr), np.std(arr)

    b_len = [r["baseline"]["length"] for r in results]
    t_len = [r["tension"]["length"] for r in results]
    b_fl  = [r["baseline"]["flops"] for r in results]
    t_fl  = [r["tension"]["flops"] for r in results]
    b_tm  = [r["baseline"]["time"] for r in results]
    t_tm  = [r["tension"]["time"] for r in results]

    print(bold("Tour Length:"))
    print(f"  Baseline: mean={np.mean(b_len):.3f} std={np.std(b_len):.3f}")
    print(f"  Tension : mean={np.mean(t_len):.3f} std={np.std(t_len):.3f}\n")

    print(bold("FLOPs:"))
    print(f"  Baseline: mean={np.mean(b_fl):.1f} std={np.std(b_fl):.1f}")
    print(f"  Tension : mean={np.mean(t_fl):.1f} std={np.std(t_fl):.1f}\n")

    print(bold("Runtime (seconds):"))
    print(f"  Baseline: mean={np.mean(b_tm):.3f} std={np.std(b_tm):.3f}")
    print(f"  Tension : mean={np.mean(t_tm):.3f} std={np.std(t_tm):.3f}\n")


# -----------------------------------------------------------
# Plot
# -----------------------------------------------------------
def plot_results(results):
    import matplotlib.pyplot as plt

    b_len = [r["baseline"]["length"] for r in results]
    t_len = [r["tension"]["length"] for r in results]
    b_fl  = [r["baseline"]["flops"] for r in results]
    t_fl  = [r["tension"]["flops"] for r in results]

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(b_fl, b_len, label="Baseline", marker="o")
    plt.scatter(t_fl, t_len, label="Tension", marker="x")

    plt.xlabel("FLOPs")
    plt.ylabel("Tour Length")
    plt.title("Tensora: FLOPs vs. Tour Quality")
    plt.legend()

    Path("results").mkdir(exist_ok=True)
    out = "results/plot_flops_vs_length.png"
    plt.savefig(out, dpi=150)
    print(green(f"\nPlot saved: {out}\n"))


# -----------------------------------------------------------
# Main + CLI
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tensora Benchmark Engine")
    parser.add_argument("--cities", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--tension-k", type=float, default=0.1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    print(cyan("\n=== Running Tensora Benchmarks ==="))
    print(f"Cities:     {args.cities}")
    print(f"Steps:      {args.steps}")
    print(f"Tension k:  {args.tension_k}")
    print(f"Runs:       {args.runs}\n")

    results = []
    for i in range(args.runs):
        print(yellow(f"Run {i+1}/{args.runs}..."))
        out = run_single(args.cities, args.steps, args.tension_k, seed=i)
        results.append(out)

    print_summary(results)

    if args.plot:
        plot_results(results)

    if args.save:
        Path("results").mkdir(exist_ok=True)
        ts = int(time.time())
        path = Path("results") / f"batch_{ts}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(green(f"Results saved to {path}\n"))


if __name__ == "__main__":
    main()
