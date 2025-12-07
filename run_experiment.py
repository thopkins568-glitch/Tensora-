#!/usr/bin/env python3
"""
Unified experiment runner for Tensora.
Runs:
  - Baseline solver
  - Tension-based solver
Collects:
  - FLOPs
  - wall-clock time
  - final objective value
  - configuration used
Outputs:
  - printed summary
  - JSON logged to results/<timestamp>.json
"""

import time
import json
import os
from pathlib import Path

import numpy as np

from flops_counter import GLOBAL_FLOPS

from tsp_problem import TSPProblem
from baseline_solver import BaselineSolver
from tension_solver import TensionSolver
from spring_tension import SpringTension


def run_experiment(
    num_cities=32,
    steps=5000,
    tension_k=0.1,
    seed=0
):
    """Run baseline and tension solvers on a random TSP instance."""
    np.random.seed(seed)

    # -------------------------------
    # Generate TSP instance
    # -------------------------------
    problem = TSPProblem.random_instance(num_cities)

    # -------------------------------
    # Baseline Solver
    # -------------------------------
    GLOBAL_FLOPS.reset()
    baseline = BaselineSolver(
        problem=problem,
        steps=steps,
        step_size=0.002
    )

    t0 = time.time()
    solution_baseline = baseline.solve()
    t1 = time.time()

    flops_baseline = GLOBAL_FLOPS.get()
    length_baseline = problem.tour_length(solution_baseline)
    time_baseline = t1 - t0

    # -------------------------------
    # Tension Solver
    # -------------------------------
    GLOBAL_FLOPS.reset()
    field = SpringTension(k=tension_k)

    tension = TensionSolver(
        problem=problem,
        tension_field=field,
        steps=steps,
        step_size=0.002
    )

    t0 = time.time()
    solution_tension = tension.solve()
    t1 = time.time()

    flops_tension = GLOBAL_FLOPS.get()
    length_tension = problem.tour_length(solution_tension)
    time_tension = t1 - t0

    # -------------------------------
    # Package results
    # -------------------------------
    result = {
        "config": {
            "num_cities": num_cities,
            "steps": steps,
            "tension_k": tension_k,
            "seed": seed,
        },
        "baseline": {
            "length": float(length_baseline),
            "flops": int(flops_baseline),
            "time_sec": time_baseline,
        },
        "tension": {
            "length": float(length_tension),
            "flops": int(flops_tension),
            "time_sec": time_tension,
        }
    }

    return result


def save_result(result):
    """Save experiment result to results/<timestamp>.json"""
    Path("results").mkdir(exist_ok=True)
    ts = int(time.time())
    outfile = Path("results") / f"experiment_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(result, f, indent=2)
    return outfile


def main():
    print("=== Tensora Experiment Runner ===")

    result = run_experiment(
        num_cities=32,
        steps=3000,
        tension_k=0.1,
        seed=0
    )

    print("\n=== RESULTS ===")
    print(json.dumps(result, indent=2))

    path = save_result(result)
    print(f"\nSaved to: {path}\n")


if __name__ == "__main__":
    main()
