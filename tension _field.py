# tensora/fields/spring_tension.py
"""
Spring-tension field (Hookean coupling) for Tensora.

Provides a simple, auditable tension field implementation:
- class SpringTension(k=0.1, weight_fn=None)
  - force(x) -> displacement to apply to each state vector
  - energy(x) -> scalar coupling energy (useful for diagnostics)
  - set_strength(k) / get_strength()

Counting FLOPs:
This module attempts conservative, easily-auditable FLOP accounting using
the repository global FLOP counter. It will try common global names and
fall back gracefully if none are present (useful during early development).

API expectations for `x`:
- x is a numpy array of shape (n, d) for population-style states
- For single-state problems x may also be a 1-D array (treated as (1, d))
"""

from __future__ import annotations
from typing import Callable, Optional
import numpy as np

# Robust global flop accessor (works with several naming variants used while bootstrapping)
def _get_global_flops():
    """
    Try to find a global FLOP counter object in common module names/attributes:
      - flops_counter.GLOBAL_FLOPS
      - flop_counter.GLOBAL_FLOPS
      - from tensora.core.flop_counter import GLOBAL_FLOPS
    The object is expected to expose either:
      - add(n) or count_add(n) or count(n)
      - snapshot()/get()/flops attribute for inspection (not needed here)
    Returns the object or None if not found.
    """
    import importlib
    candidates = [
        ("flops_counter", "GLOBAL_FLOPS"),
        ("flop_counter", "GLOBAL_FLOPS"),
        ("tensora.core.flop_counter", "GLOBAL_FLOPS"),
        ("tensora.core.flops_counter", "GLOBAL_FLOPS"),
    ]
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
            val = getattr(mod, attr, None)
            if val is not None:
                return val
        except Exception:
            continue
    # last resort: look for any GLOBAL_FLOPS in globals() of already-loaded modules
    for m in list(importlib.sys.modules.values()):
        try:
            if hasattr(m, "GLOBAL_FLOPS"):
                return getattr(m, "GLOBAL_FLOPS")
        except Exception:
            continue
    return None


def _add_flops(n: int):
    """Add n FLOPs to the global counter if available (silent no-op otherwise)."""
    if not n:
        return
    gf = _get_global_flops()
    if gf is None:
        return
    # try common method names
    for name in ("add", "count_add", "count", "inc", "add_flops"):
        fn = getattr(gf, name, None)
        if callable(fn):
            try:
                fn(int(n))
                return
            except Exception:
                # try next
                continue
    # fallback: try attribute
    if hasattr(gf, "flops"):
        try:
            gf.flops = int(getattr(gf, "flops", 0)) + int(n)
            return
        except Exception:
            pass
    # if nothing worked, give up quietly (development mode)


class SpringTension:
    """
    Hookean spring tension field operating on a set of state vectors.

    The coupling for each state i is:
        F_i = k * sum_j w(i,j) * (x_j - x_i)

    For the default implementation we use uniform global coupling weights:
        w(i,j) = 1 / n   (i != j)
    which simplifies to:
        F_i = k * (mean_x - x_i)

    Parameters
    ----------
    k : float
        Tension strength (multiplier).
    weight_fn : Optional[Callable[[np.ndarray], np.ndarray]]
        Optional custom weight function that accepts the state array x (n,d)
        and returns an (n,n) matrix of weights. If provided, the implementation
        will use it (and count the FLOPs for its invocation as a black-box).
    """

    def __init__(self, k: float = 0.1, weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.k = float(k)
        self.weight_fn = weight_fn

    def set_strength(self, k: float):
        self.k = float(k)

    def get_strength(self) -> float:
        return float(self.k)

    def force(self, x: np.ndarray) -> np.ndarray:
        """
        Compute per-state displacement produced by the tension field.

        Args:
            x: numpy array shaped (n, d) or (d,) for single state.
        Returns:
            numpy array of same shape as x with displacement to add.
        FLOP accounting:
            - If using default mean-coupling:
                mean: for n*d values -> (n*d - 1) adds + n*d divides ≈ n*d ops
                subtract mean - x: n*d ops
                scale by k: n*d ops
              we conservatively count 3 * n * d FLOPs
            - If weight_fn provided, we treat the weight_fn call as black-box:
                we add a small heuristic cost (n^2) to account for weight computation,
                then compute weighted sum at cost ~ n^2 * d.
        """
        # Normalize input shape
        arr = np.asarray(x)
        original_shape = arr.shape
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]  # shape -> (1, d)

        n, d = arr.shape

        # Default uniform mean-based coupling
        if self.weight_fn is None:
            # compute mean across rows (per-dim)
            # FLOPs: sum across n items for each dim ~ n*d adds, + division by n ~ n*d ops -> ~2*n*d
            # subtract & scale: (n*d) + (n*d) -> 2*n*d
            # total conservative count: 4 * n * d
            _add_flops(int(4 * n * d))

            mean = np.mean(arr, axis=0)          # numpy does the arithmetic (we accounted above)
            # compute coupling: k * (mean - x)
            coupling = self.k * (mean - arr)     # vectorized
            return coupling.reshape(original_shape)

        # Custom weight function: compute weight matrix W = weight_fn(x)
        # Expect W shape (n, n). Then F = k * (W @ x - (sum_j W_ij) * x_i)
        # We'll compute W @ x (n x d matrix multiply)
        # FLOP accounting:
        #  - weight_fn cost: heuristic n*n
        #  - matmul: n * n * d * 2 FLOPs (mult + add)
        #  - minor extra ops: n*d
        _add_flops(int(n * n) + int(2 * n * n * d) + int(n * d))

        W = self.weight_fn(arr)  # Black-box; user must ensure correctness
        # Ensure W is numpy array
        W = np.asarray(W)
        # W @ x -> shape (n, d)
        Wx = W.dot(arr)
        # row sums s_i = sum_j W_ij
        s = np.sum(W, axis=1, keepdims=True)
        # broadcast: s * x -> (n,d)
        coupling = self.k * (Wx - s * arr)
        return coupling.reshape(original_shape)

    def energy(self, x: np.ndarray) -> float:
        """
        Compute scalar coupling energy for diagnostics:
            E = 0.5 * k * sum_i || mean_x - x_i ||^2   [for uniform mean-based weights]

        FLOP accounting (uniform mean):
            - mean: ~ n*d ops (already counted similarly in force)
            - subtraction & square: n*d ops to subtract, n*d ops to square
            - summation: n*d adds
            - multiply by 0.5*k: n*d ops (or 1 op after sum)
          conservative count: 5 * n * d
        """
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        n, d = arr.shape

        if self.weight_fn is None:
            _add_flops(int(5 * n * d))
            mean = np.mean(arr, axis=0)
            dif = mean - arr
            energy = 0.5 * self.k * float(np.sum(dif * dif))
            return energy

        # For weighted case: E = 0.5 * sum_i sum_j W_ij ||x_j - x_i||^2 * (k)
        # FLOP heuristic: n^2 * d for pairwise diffs + n^2 for squares/sums
        _add_flops(int(n * n * d + n * n))
        W = self.weight_fn(arr)
        # compute pairwise squared distances (inefficient but explicit for clarity)
        # Note: this is O(n^2 * d)
        diffs = arr[:, None, :] - arr[None, :, :]        # (n, n, d)
        sq = np.sum(diffs * diffs, axis=-1)             # (n, n)
        energy = 0.5 * self.k * float(np.sum(W * sq))
        return energy
```0
