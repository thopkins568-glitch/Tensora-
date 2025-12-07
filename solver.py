# solver.py
"""
Abstract base Solver class for Tensora.

Every solver must implement:
  - initialize()
  - step()

The base class handles:
  - timing (start_time, end_time)
  - FLOP counting: resets + final report
  - run() loop with max steps
"""

import time
from flops_counter import GLOBAL_FLOPS

class Solver:
    """
    Base class. Subclasses must implement:
      initialize(self)
      step(self)
    """

    def __init__(self):
        self.initialized = False
        self.history = []
        self.last_result = None

    # ---------------------------------------------------------
    # Abstract methods
    # ---------------------------------------------------------
    def initialize(self):
        """
        Prepare internal solver state.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def step(self):
        """
        Perform one solver iteration.
        Must be implemented by subclasses.
        
        Must return:
          - result value (float or any meaningful metric)
          - done (bool) indicating convergence or completion
        """
        raise NotImplementedError

    # ---------------------------------------------------------
    # Execution loop
    # ---------------------------------------------------------
    def run(self, max_steps=1000):
        """
        Run solver loop up to max_steps or until done=True.
        Returns a dictionary with:
          - result: final result value
          - steps: number of iterations
          - time_sec: wall time
          - flops: total FLOPs consumed
        """

        # Reset FLOP counter for this run
        GLOBAL_FLOPS.reset()

        # Initialize solver
        if not self.initialized:
            self.initialize()
            self.initialized = True

        start = time.time()

        result = None
        done = False
        steps = 0

        for i in range(max_steps):
            result, done = self.step()
            self.history.append(result)
            steps += 1

            if done:
                break

        end = time.time()

        self.last_result = result

        return {
            "result": result,
            "steps": steps,
            "time_sec": end - start,
            "flops": GLOBAL_FLOPS.total,
        }
