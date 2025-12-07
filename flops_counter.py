# flops_counter.py
"""
Explicit, auditable FLOP counter for Tensora.

This version avoids monkey-patching NumPy (too global and dangerous).
Instead, we count FLOPs explicitly inside solver code.

Provides:
  - FlopCounter class
  - GLOBAL_FLOPS: shared counter instance
  - helper functions: add_ops(), mul_ops(), dot_ops(), norm_ops(), etc.
"""

from __future__ import annotations

class FlopCounter:
    """
    Minimal, explicit FLOP counter.
    Tracks floating-point operations used inside solvers.

    Supported ops:
      - basic arithmetic (add, mul, sub, div)
      - dot products
      - vector norms
      - matrix-vector multiplies
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all counters to zero."""
        self.total = 0
        self.adds = 0
        self.muls = 0
        self.subs = 0
        self.divs = 0
        self.dots = 0
        self.norms = 0
        self.matvecs = 0

    def count_add(self, n=1):
        """Count n addition FLOPs."""
        self.adds += n
        self.total += n

    def count_mul(self, n=1):
        """Count n multiplication FLOPs."""
        self.muls += n
        self.total += n

    def count_sub(self, n=1):
        """Count n subtraction FLOPs."""
        self.subs += n
        self.total += n

    def count_div(self, n=1):
        """Count n division FLOPs."""
        self.divs += n
        self.total += n

    def count_dot(self, n):
        """
        Dot product of length n:
          ~ n multiplications + (n - 1) additions ≈ 2n - 1 FLOPs.
        """
        if n > 0:
            self.muls += n
            self.adds += (n - 1)
            self.total += (2 * n - 1)
            self.dots += 1

    def count_norm(self, n):
        """
        L2 norm:
          square each element (n multiplications),
          sum them (n-1 additions),
          sqrt approximation (1 mul + 1 add).
        """
        if n > 0:
            ops = n + (n - 1) + 2
            self.muls += (n + 1)
            self.adds += (n - 1)
            self.norms += 1
            self.total += ops

    def count_matvec(self, rows, cols):
        """
        Matrix-vector multiply cost:
          rows dot products, each of length cols.
        """
        for _ in range(rows):
            self.count_dot(cols)
        self.matvecs += 1

    def to_dict(self):
        """
        Return a dictionary with breakouts of counts.
        """
        return {
            "total": self.total,
            "adds": self.adds,
            "muls": self.muls,
            "subs": self.subs,
            "divs": self.divs,
            "dots": self.dots,
            "norms": self.norms,
            "matvecs": self.matvecs,
        }

    def __repr__(self):
        return f"<FLOPS total={self.total}>"

# ---------------------------------------------------------
# Global shared instance (used everywhere in Tensora)
# ---------------------------------------------------------
GLOBAL_FLOPS = FlopCounter()

# Convenience forwarding functions
def add_ops(n): GLOBAL_FLOPS.count_add(n)
def mul_ops(n): GLOBAL_FLOPS.count_mul(n)
def sub_ops(n): GLOBAL_FLOPS.count_sub(n)
def div_ops(n): GLOBAL_FLOPS.count_div(n)
def dot_ops(n): GLOBAL_FLOPS.count_dot(n)
def norm_ops(n): GLOBAL_FLOPS.count_norm(n)
def matvec_ops(r, c): GLOBAL_FLOPS.count_matvec(r, c)
