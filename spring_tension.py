"""
Spring-based tension field for Tensora.

Implements a simple Hooke-like coupling that pulls the state vector
toward its own centroid. This is a minimal but effective form of
search-space contraction.

Mathematically:

    dx_i = k * (mean(x) - x_i)

This reduces variance and encourages coherence in the search trajectory.

All FLOPs are counted explicitly.
"""

import numpy as np
from tensora.core.flops_counter import GLOBAL_FLOPS


class SpringTensionField:
    def __init__(self, strength: float = 0.1):
        """
        Args:
            strength (float): coupling coefficient k
        """
        self.k = float(strength)

    def force(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the tension displacement vector.

        Steps:
          1. Compute centroid m = mean(x)
          2. Compute displacement dx = k * (m - x)

        FLOP accounting:
            - mean: N additions + 1 division → counted as 2N FLOPs
            - subtract: N FLOPs
            - multiply: N FLOPs
            Total ≈ 4N FLOPs
        """

        # Number of dimensions
        N = x.shape[0] if hasattr(x, "shape") else len(x)

        # 1. Compute mean
        m = np.mean(x)
        GLOBAL_FLOPS.count_add(N)      # N additions in sum
        GLOBAL_FLOPS.count_add(1)      # division → cost ≈ 1 FLOP

        # 2. m - x   (vector subtract)
        GLOBAL_FLOPS.count_add(N)

        delta = m - x

        # 3. k * delta  (vector scale)
        GLOBAL_FLOPS.count_add(N)

        return self.k * delta
