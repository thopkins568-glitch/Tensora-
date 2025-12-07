# tsp_problem.py
"""
TSP problem definition for Tensora.

Provides:
 - TSPProblem class with:
    - initial_state() -> initial tour (numpy int array)
    - energy(tour) -> float (tour length), counts FLOPs
    - propose(tour) -> neighbor tour (2-opt style), counts FLOPs
    - converged(energy) -> bool, simple tolerance-based check
 - internal helpers: compute_distance_matrix, path_length (with FLOP accounting)

This version is robust to several possible FLOP-counter module names / APIs
and uses the precise FLOP formula for the vectorized distance matrix computation:
  For n points in d dims: per ordered pair FLOPs = d (sub) + d (square) + (d-1) (adds) + 1 (sqrt)
  => per-pair = 3*d - 1
Total for full n x n matrix = n^2 * (3*d - 1)
"""

from __future__ import annotations
import numpy as np
from typing import Optional

# Try to locate a GLOBAL_FLOPS object under common names/locations.
# This makes the problem module resilient to variations in how the FLOP counter was implemented.
def _locate_global_flops():
    candidates = [
        ("tensora.core.flop_counter", "GLOBAL_FLOPS"),
        ("flop_counter", "GLOBAL_FLOPS"),
        ("flops_counter", "GLOBAL_FLOPS"),
        ("tensora.core.flops_counter", "GLOBAL_FLOPS"),
        ("tensora.core.flop_counter", "FLOPS"),
        ("flop_counter", "FLOPS"),
    ]
    for module_name, attr in candidates:
        try:
            mod = __import__(module_name, fromlist=[attr])
            obj = getattr(mod, attr, None)
            if obj is not None:
                return obj
        except Exception:
            continue
    # As a last resort, check global namespace if the user injected a GLOBAL_FLOPS
    try:
        from globals import GLOBAL_FLOPS as gf  # unlikely, but harmless
        return gf
    except Exception:
        pass
    return None


GLOBAL_FLOPS = _locate_global_flops()


def add_flops(n: int):
    """
    Add n FLOPs to the global counter in a robust way.
    Tries a number of common method/attribute names.
    If no counter found, silently no-op (safe during early development).
    """
    if n is None or n == 0:
        return
    if GLOBAL_FLOPS is None:
        # no counter available; silently ignore to avoid crashing in dev
        return

    # prefer direct well-known methods
    for method in ("add", "count_add", "count", "inc", "increment", "add_flops", "count_flops", "record"):
        fn = getattr(GLOBAL_FLOPS, method, None)
        if callable(fn):
            try:
                fn(int(n))
                return
            except TypeError:
                # some implementations might expose a different signature; keep trying
                pass

    # try common attribute-based accumulation
    if hasattr(GLOBAL_FLOPS, "flops"):
        try:
            # support both numeric and property-like attribute
            current = getattr(GLOBAL_FLOPS, "flops")
            try:
                setattr(GLOBAL_FLOPS, "flops", int(current) + int(n))
                return
            except Exception:
                # if setting fails, continue to other fallbacks
                pass
        except Exception:
            pass

    # try snapshot/add fallback
    if hasattr(GLOBAL_FLOPS, "snapshot") and hasattr(GLOBAL_FLOPS, "add"):
        try:
            GLOBAL_FLOPS.add(int(n))
            return
        except Exception:
            pass

    # give up quietly (development mode)
    return


# ---- Utilities with FLOP accounting ----
def compute_distance_matrix(points: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise Euclidean distance matrix and account FLOPs.

    Vectorized computation does:
      diff = points[:, None, :] - points[None, :, :]   # n^2 * d subtractions
      diff * diff                                       # n^2 * d multiplications (squares)
      sum over last axis (d-1 adds per pair)            # n^2 * (d-1) adds
      sqrt per pair                                      # n^2 * 1

    Total FLOPs per ordered pair = 3*d - 1
    Total FLOPs for full matrix = n^2 * (3*d - 1)
    """
    points = np.asarray(points)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array of shape (n, d)")
    n, d = points.shape
    per_pair = 3 * d - 1
    total_flops = int(n * n * per_pair)
    add_flops(total_flops)

    # compute distances (vectorized)
    diff = points[:, None, :] - points[None, :, :]  # shape (n, n, d)
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return dist


def path_length(order: np.ndarray, dist: np.ndarray) -> float:
    """
    Compute the total length of a TSP tour given a (full) distance matrix.

    We count 1 FLOP per addition when summing the tour length. Array lookups are
    memory ops and do not count as FLOPs.
    """
    order = np.asarray(order, dtype=int)
    m = order.size
    # m additions to accumulate the tour length (one per edge)
    add_flops(int(m))
    total = 0.0
    for i in range(m):
        j = (i + 1) % m
        total += float(dist[order[i], order[j]])
    return float(total)


# ---- Problem class ----
class TSPProblem:
    def __init__(
        self,
        n_cities: int = 32,
        seed: Optional[int] = None,
        coord_scale: float = 1.0,
        tol: float = 1e-9,
        precompute_distance_matrix: bool = True,
    ):
        """
        Args:
            n_cities: number of cities
            seed: RNG seed for reproducibility
            coord_scale: scale of coordinates
            tol: convergence tolerance for energy (used in `converged`)
            precompute_distance_matrix: if True, compute distance matrix at init (counts FLOPs once)
        """
        if seed is not None:
            np.random.seed(int(seed))
        self.n = int(n_cities)
        self.coords = np.random.uniform(0.0, coord_scale, size=(self.n, 2))
        self.tol = float(tol)
        self.known_best: Optional[float] = None

        if precompute_distance_matrix:
            # distance computation counts FLOPs (one-time cost)
            self.dist = compute_distance_matrix(self.coords)
        else:
            self.dist = None

    # --- interface used by solvers ---
    def initial_state(self) -> np.ndarray:
        """Return an initial tour ordering (numpy array of ints)."""
        order = np.arange(self.n, dtype=int)
        # small FLOP count for creating/copying the vector (heuristic)
        add_flops(int(self.n))
        return order

    def energy(self, order: np.ndarray) -> float:
        """Return the tour length (float). Counts FLOPs via path_length."""
        if self.dist is None:
            # compute on the fly (counts FLOPs each time)
            dist = compute_distance_matrix(self.coords)
        else:
            dist = self.dist
        return path_length(order, dist)

    def propose(self, order: np.ndarray) -> np.ndarray:
        """
        Propose a neighbor by performing a random 2-opt reversal on a copy.

        FLOP accounting (heuristic):
          - copying the array: n operations
          - reversing a slice of length L: ~L operations
          (random index generation cost is not included as FLOPs)
        """
        order = np.asarray(order, dtype=int)
        n = order.size
        i = int(np.random.randint(0, n))
        j = int(np.random.randint(0, n))

        if i == j:
            # rotate by one to avoid no-op
            proposed = np.roll(order, 1)
            add_flops(1)
            return proposed

        if i > j:
            i, j = j, i

        proposed = order.copy()
        add_flops(int(n))  # copy cost heuristic

        slice_len = max(1, j - i)
        add_flops(int(slice_len))  # reversal cost heuristic

        # perform in-place reversal of slice
        proposed[i:j] = proposed[i:j][::-1]
        return proposed

    def converged(self, current_energy: float, best_so_far: Optional[float] = None, iter_no: Optional[int] = None) -> bool:
        """
        Convergence test:
          - If known_best provided and current_energy <= known_best + tol -> converged
          - If best_so_far provided and improvement is smaller than tol (absolute or relative) -> converged
          - Otherwise: not converged
        """
        if self.known_best is not None:
            if current_energy <= self.known_best + self.tol:
                return True

        if best_so_far is not None:
            if abs(current_energy - best_so_far) <= max(self.tol, self.tol * abs(best_so_far)):
                return True

        return False


# ---- Quick self-test runner (manual use) ----
if __name__ == "__main__":
    p = TSPProblem(n_cities=16, seed=0)
    tour = p.initial_state()
    e = p.energy(tour)
    print("Initial energy (tour length):", e)
    prop = p.propose(tour)
    e2 = p.energy(prop)
    print("Proposed energy:", e2)
