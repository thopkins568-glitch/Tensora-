# tensora/core/core.py
"""Core utilities for Tensora.

Exposes:
 - FLOPS: the shared global flop counter (from flop_counter.py)
 - ExperimentResult: canonical experiment result dataclass
 - ConvergenceChecker: reusable early-stopping utility (abs/rel/plateau)
 - set_seed(seed): deterministic RNG seeding helper
 - Timer: a tiny context manager for wall-time measurement
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import numpy as np
from typing import Optional, List, Dict, Any

# Import the flop counter implementation (file: tensora/core/flop_counter.py)
# This file is expected to define: FlopCounter and GLOBAL_FLOPS
try:
    # relative import (works when package is installed or run as package)
    from .flop_counter import FlopCounter, GLOBAL_FLOPS
except Exception:  # fallback to top-level import if executed differently
    from flop_counter import FlopCounter, GLOBAL_FLOPS  # type: ignore

# canonical name used across the repository
FLOPS: FlopCounter = GLOBAL_FLOPS


# -------------------------
# Experiment result object
# -------------------------
@dataclass
class ExperimentResult:
    problem: str
    solver: str
    config: Dict[str, Any]
    baseline_best: Optional[float] = None
    baseline_flops: Optional[int] = None
    baseline_iters: Optional[int] = None
    tension_best: Optional[float] = None
    tension_flops: Optional[int] = None
    tension_iters: Optional[int] = None
    baseline_path: Optional[List[float]] = None
    tension_path: Optional[List[float]] = None
    converged: Optional[Dict[str, bool]] = None
    wall_time_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -------------------------
# Convergence checker
# -------------------------
class ConvergenceChecker:
    """
    Utility to detect convergence.

    Stopping conditions (any satisfied):
      - absolute change < eps_abs
      - relative change < eps_rel
      - plateau: no improvement for `plateau_patience` iterations

    Usage:
        checker = ConvergenceChecker(eps_abs=1e-6, eps_rel=1e-4, plateau_patience=20)
        for it in range(max_iters):
            ...
            done = checker.update(current_best)
            if done: break
    """

    def __init__(
        self,
        eps_abs: float = 1e-6,
        eps_rel: float = 1e-4,
        plateau_patience: int = 20,
        min_iters: int = 0,
    ):
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.plateau_patience = int(plateau_patience)
        self.min_iters = int(min_iters)

        self._best: Optional[float] = None
        self._iters_since_improve: int = 0
        self._total_iters: int = 0

    def reset(self):
        self._best = None
        self._iters_since_improve = 0
        self._total_iters = 0

    def update(self, current_value: float) -> bool:
        """Call each iteration with the current best objective. Returns True if should stop."""

        self._total_iters += 1
        if self._best is None:
            self._best = float(current_value)
            self._iters_since_improve = 0
            return False

        cur = float(current_value)
        abs_change = abs(self._best - cur)
        rel_change = abs_change / (abs(self._best) + 1e-12)

        improved = False
        if cur < self._best:
            improved = True
            self._best = cur
            self._iters_since_improve = 0
        else:
            self._iters_since_improve += 1

        # require minimum iterations before allowing early stop
        if self._total_iters <= self.min_iters:
            return False

        if abs_change < self.eps_abs:
            return True
        if rel_change < self.eps_rel:
            return True
        if self._iters_since_improve >= self.plateau_patience:
            return True

        return False

    @property
    def best(self) -> Optional[float]:
        return self._best

    @property
    def iters(self) -> int:
        return self._total_iters


# -------------------------
# RNG seeding helper
# -------------------------
def set_seed(seed: Optional[int]):
    """Set seeds for NumPy (and any other RNGs we add later)."""
    if seed is None:
        return
    np.random.seed(int(seed))


# -------------------------
# Timer context manager
# -------------------------
class Timer:
    """Simple wall-clock timer as a context manager."""

    def __init__(self):
        self.start: Optional[float] = None
        self.end: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start = time.time()
        self.end = None
        self.elapsed = None
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end = time.time()
        self.elapsed = self.end - (self.start or self.end)

    def seconds(self) -> float:
        if self.elapsed is None:
            if self.start is None:
                return 0.0
            return time.time() - self.start
        return float(self.elapsed)
