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
"""

import numpy as np
from flops_counter import GLOBAL_FLOPS


# ---- FLOP helper (adapts to whichever API your GLOBAL_FLOPS exposes) ----
def add_flops(n: int):
    """Robustly add n FLOPs to GLOBAL_FLOPS regardless of method name."""
    if n is None or n == 0:
        return
    # try common names in order of likelihood
    for name in ("add", "count_add", "count", "inc", "count_flops", "add_flops"):
        fn = getattr(GLOBAL_FLOPS, name, None)
        if callable(fn):
            try:
                fn(int(n))
                return
            except TypeError:
                # maybe the API expects no args (unlikely) — ignore
                pass
    # fallback: try 'flops' attribute
    if hasattr(GLOBAL_FLOPS, "flops"):
        try:
            GLOBAL_FLOPS.flops = int(GLOBAL_FLOPS.flops) + int(n)
            return
        except Exception:
            pass
    # last-resort: try snapshot/add via add if present
    if hasattr(GLOBAL_FLOPS, "snapshot") and hasattr(GLOBAL_FLOPS, "add"):
        try:
            GLOBAL_FLOPS.add(int(n))
            return
        except Exception:
            pass
    # If nothing worked, silently ignore (safe fallback during development)
    return


# ---- Utilities with FLOP accounting ----
def compute_distance_matrix(points: np.ndarray):
    """
    Compute full pairwise Euclidean distance matrix and account FLOPs.

    For n points in d dims:
      - For each pair: d subtractions, d squares, (d-1) adds, 1 sqrt
      -> FLOPs per pair ≈ (3*d - 1)  (as discussed)
      -> total ~ n^2 * (3*d - 1)
    We'll count the symmetric matrix (all pairs) for simplicity.
    """
    n, d = points.shape
    # cost estimate
    per_pair = 3 * d - 1
    total_flops = n * n * per_pair
    add_flops(total_flops)

    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return dist


def path_length(order: np.ndarray, dist: np.ndarray) -> float:
    """
    Compute the total length of a TSP tour given a (full) distance matrix.
    We count 1 FLOP per add (summing the lengths). Lookups are not FLOPs.
    """
    m = order.size
    # m additions
    add_flops(m)
    total = 0.0
    for i in range(m):
        j = (i + 1) % m
        total += dist[order[i], order[j]]
    return float(total)


# ---- Problem class ----
class TSPProblem:
    def __init__(self, n_cities: int = 32, seed: int = None, coord_scale: float = 1.0, tol: float = 1e-9):
        """
        Args:
            n_cities: number of cities
            seed: RNG seed for reproducibility
            coord_scale: scale of coordinates
            tol: convergence tolerance for energy (used in `converged`)
        """
        if seed is not None:
            np.random.seed(seed)
        self.n = int(n_cities)
        self.coords = np.random.uniform(0.0, coord_scale, size=(self.n, 2))
        # precompute distance matrix (counts FLOPs inside)
        self.dist = compute_distance_matrix(self.coords)
        self.tol = float(tol)
        # optional known optimum (None by default)
        self.known_best = None

    # --- interface used by solvers ---
    def initial_state(self) -> np.ndarray:
        """Return an initial tour ordering (numpy array of ints)."""
        # a simple initial tour: 0..n-1
        order = np.arange(self.n, dtype=int)
        # small FLOP count for copying / creation (negligible but counted)
        add_flops(self.n)
        return order

    def energy(self, order: np.ndarray) -> float:
        """Return the tour length (float). Counts FLOPs via path_length."""
        return path_length(order, self.dist)

    def propose(self, order: np.ndarray) -> np.ndarray:
        """
        Propose a neighbor by performing a random 2-opt reversal on a copy.

        FLOP accounting:
          - selecting indices (random ints): negligible (random generator cost)
          - copying array of length n: count n
          - reversing slice: count approx slice length
        """
        n = order.size
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i == j:
            # small modification: rotate by 1 to avoid no-op (counts 1 flop)
            proposed = np.roll(order, 1)
            add_flops(1)
            return proposed

        # ensure i < j
        if i > j:
            i, j = j, i

        # copy (account for copying cost)
        proposed = order.copy()
        add_flops(n)  # copy cost heuristic

        # perform 2-opt reversal on slice [i:j]
        slice_len = max(1, j - i)
        # reversing: slice_len/2 swaps ~ slice_len ops (rough)
        add_flops(slice_len)
        proposed[i:j] = proposed[i:j][::-1]
        return proposed

    def converged(self, current_energy: float, best_so_far: float = None, iter_no: int = None) -> bool:
        """
        Simple convergence test:
         - If known_best provided and current_energy <= known_best + tol -> converged
         - If absolute improvement small enough (compared to best_so_far) -> converged
         - Otherwise keep running. Most solvers will use this in conjunction with max-iterations.
        """
        if self.known_best is not None:
            if current_energy <= self.known_best + self.tol:
                return True

        if best_so_far is not None:
            # relative improvement check
            if abs(current_energy - best_so_far) <= max(self.tol, self.tol * abs(best_so_far)):
                return True

        return False


# ---- Quick self-test runner (manual use) ----
if __name__ == "__main__":
    # quick sanity check (not a unit test)
    p = TSPProblem(n_cities=16, seed=0)
    tour = p.initial_state()
    e = p.energy(tour)
    print("Initial energy (tour length):", e)
    prop = p.propose(tour)
    e2 = p.energy(prop)
    print("Proposed energy:", e2)
