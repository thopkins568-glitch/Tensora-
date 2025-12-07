# baseline_solver.py
"""
Baseline solver used for all Tensora benchmarks.

This solver:
  - Maintains a single current state vector
  - Computes its energy via problem.energy()
  - Proposes a small random move via problem.propose()
  - Accepts if energy improves
  - Counts FLOPs via the GLOBAL_FLOPS instrumented utilities

This gives a universal, minimal baseline algorithm.
"""

import numpy as np
from flops_counter import GLOBAL_FLOPS
from solver import Solver


class BaselineSolver(Solver):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        self.x = None
        self.current_energy = None

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------
    def initialize(self):
        # Get initial state from the problem
        self.x = self.problem.initial_state()

        # Compute starting energy
        self.current_energy = self.problem.energy(self.x)

    # ---------------------------------------------------------
    # One solver iteration
    # ---------------------------------------------------------
    def step(self):
        """
        Returns:
          - best energy after this step
          - done (bool)
        """

        # Propose a new state
        x_new = self.problem.propose(self.x)

        # Compute energy
        e_new = self.problem.energy(x_new)

        # Accept if better
        if e_new < self.current_energy:
            self.x = x_new
            self.current_energy = e_new

        # Problem may define its own convergence rule
        done = self.problem.converged(self.current_energy)

        return float(self.current_energy), done
