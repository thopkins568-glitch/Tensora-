# tsp_baseline.py
"""
Baseline TSP problem definition for Tensora.

Implements:
  - initial_state()
  - energy(x)
  - propose(x)
  - converged(e)

This version keeps everything extremely minimal and deterministic so that
FLOP counting is clean, controlled, and comparable across solvers.
"""

import numpy as np
from flops_counter import GLOBAL_FLOPS


class TSP:
    def __init__(self, coords):
        """
        coords: (N, 2) array of city coordinates
        """
        self.coords = np.array(coords, dtype=float)
        self.N = len(coords)

        # Set a fixed convergence threshold (optional, simple)
        self.convergence_eps = 1e-6

    # ---------------------------------------------------------
    #  State Representation
    # ---------------------------------------------------------
    # x is a permutation of N cities (integer vector)
    # ---------------------------------------------------------

    def initial_state(self):
        GLOBAL_FLOPS.count_add(self.N)   # trivial cost for constructing array
        return np.arange(self.N)

    def energy(self, x):
        """
        Tour length.

        Computes:
          sum over i of distance(x[i], x[i+1])
        """
        coords = self.coords[x]
        diffs = coords[1:] - coords[:-1]

        # FLOPs for subtract + square + add + sqrt per pair
        # diff = (dx, dy) → dx^2 + dy^2 → sqrt
        GLOBAL_FLOPS.count_add(2 * (self.N - 1))     # two subtractions
        GLOBAL_FLOPS.count_mul(2 * (self.N - 1))     # dx*dx, dy*dy
        GLOBAL_FLOPS.count_add((self.N - 1))         # dx2 + dy2
        GLOBAL_FLOPS.count_sqrt((self.N - 1))

        segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        return float(segment_lengths.sum())

    def propose(self, x):
        """
        Returns a slightly modified permutation by swapping two indices.

        This is intentionally simple (O(1) FLOPs).
        """
        x_new = x.copy()

        # Pick two positions to swap
        i, j = np.random.randint(0, self.N, size=2)

        # FLOPs for swap: 3 assignments (count as adds)
        GLOBAL_FLOPS.count_add(3)

        tmp = x_new[i]
        x_new[i] = x_new[j]
        x_new[j] = tmp

        return x_new

    def converged(self, e):
        """
        Stops when energy stabilizes.
        Very loose convergence rule.
        """
        return e < self.convergence_eps
