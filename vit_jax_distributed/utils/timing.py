"""Wall-clock timing utilities for training loops."""

import time

import numpy as np


class Timer:
    """Context manager that records elapsed wall time.

    Usage::

        with Timer() as t:
            do_work()
        print(t.elapsed)
    """

    def __init__(self):
        self._start = None
        self._elapsed = None

    def __enter__(self):
        self._start = time.perf_counter()
        self._elapsed = None
        return self

    def __exit__(self, *exc):
        self._elapsed = time.perf_counter() - self._start
        return False

    @property
    def elapsed(self):
        """Elapsed time in seconds. Available after the context exits."""
        if self._elapsed is None:
            if self._start is not None:
                return time.perf_counter() - self._start
            return 0.0
        return self._elapsed


class StepTimer:
    """Accumulates per-step wall-clock times and computes statistics."""

    def __init__(self):
        self._times = []
        self._last_tick = None

    def tick(self):
        """Call once per step. The interval between consecutive ticks is
        recorded as the step time."""
        now = time.perf_counter()
        if self._last_tick is not None:
            self._times.append(now - self._last_tick)
        self._last_tick = now

    def reset(self):
        """Clear all recorded step times."""
        self._times.clear()
        self._last_tick = None

    def get_stats(self):
        """Return a dict with mean, std, min, max, p50, p95, p99 of step
        times (in seconds). Returns an empty dict if no intervals have been
        recorded."""
        if not self._times:
            return {}
        arr = np.array(self._times)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def get_throughput(self, batch_size):
        """Return images per second based on the mean step time.

        Args:
            batch_size: Number of images processed per step.

        Returns:
            Throughput in images/sec, or 0.0 if no data is available.
        """
        stats = self.get_stats()
        if not stats or stats["mean"] == 0.0:
            return 0.0
        return batch_size / stats["mean"]
