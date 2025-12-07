# tension_solver.py
"""
Tension-augmented solver for Tensora.

This class mirrors BaselineSolver exactly, with one change:
it applies a tension field update before evaluating proposals.
"""

import numpy as np
from flops_counter import GLOBAL_FLOPS
from solver import Solver


class TensionSolver(Solver):
    def __init__(self, problem, field):
        super().__init__()
        self.problem = problem
        self.field = field
        self.x = None
        self.current_energy = None

    def initialize(self):
        self.x = self.problem.initial_state()
        self.current_energy = self.problem.energy(self.x)

    def step(self):
        """
        Returns:
          - best energy after this step
          - done (bool)
        """

        # Apply tension displacement
        tension_dx = self.field.force(self.x)

        # Baseline proposal
        base_x = self.problem.propose(self.x)

        # Combine both effects (count vector adds)
        dim = len(self.x) if hasattr(self.x, "__len__") else 1
        GLOBAL_FLOPS.count_add(dim)

        x_new = base_x + tension_dx

        # Compute energy
        e_new = self.problem.energy(x_new)

        # Accept if improved
        if e_new < self.current_energy:
            self.x = x_new
            self.current_energy = e_new

        # Same convergence condition as baseline
        done = self.problem.converged(self.current_energy)

        return float(self.current_energy), done
