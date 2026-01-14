"""Quick benchmark suite for iterative development.

This module provides fast benchmarks for rapid iteration during optimization work.
Use benchmark_core.py for comprehensive baseline measurements.
"""

import time
from dataclasses import dataclass

import numpy as np

from propagator.core import (
    FUEL_SYSTEM_LEGACY,
    BoundaryConditions,
    Propagator,
    PropagatorOutOfBoundsError,
)


@dataclass
class QuickBenchResult:
    """Results from a quick benchmark run."""

    name: str
    total_time: float
    steps_executed: int
    time_per_step: float

    def __str__(self) -> str:
        return (
            f"{self.name:30s}: {self.total_time:6.3f}s total, "
            f"{self.time_per_step:.5f}s/step, {self.steps_executed:5d} steps"
        )


def _warmup_jit() -> None:
    """Quick JIT warmup."""
    veg = np.full((50, 50), 5, dtype=np.int32)
    dem = np.zeros((50, 50), dtype=np.float32)
    sim = Propagator(
        dem=dem,
        veg=veg,
        realizations=2,
        fuels=FUEL_SYSTEM_LEGACY,
        do_spotting=False,
        out_of_bounds_mode="ignore",
    )
    bc = BoundaryConditions(
        time=0,
        ignitions=[(25, 25)],
        wind_speed=20.0,
        wind_dir=90.0,
        moisture=0.0,
    )
    sim.set_boundary_conditions(bc)
    for _ in range(5):
        if sim.next_time() is None:
            break
        sim.step()


def quick_benchmark(
    name: str,
    grid_size: tuple[int, int],
    realizations: int,
    sim_duration: int,
) -> QuickBenchResult:
    """Run a quick benchmark with given parameters."""
    rows, cols = grid_size
    veg = np.full((rows, cols), 5, dtype=np.int32)
    dem = np.zeros((rows, cols), dtype=np.float32)

    sim = Propagator(
        dem=dem,
        veg=veg,
        realizations=realizations,
        fuels=FUEL_SYSTEM_LEGACY,
        do_spotting=False,
        out_of_bounds_mode="ignore",
    )

    center_x, center_y = rows // 2, cols // 2
    bc = BoundaryConditions(
        time=0,
        ignitions=[(center_x, center_y)],
        wind_speed=25.0,
        wind_dir=90.0,
        moisture=0.0,
    )
    sim.set_boundary_conditions(bc)

    start = time.perf_counter()
    steps = 0

    while sim.time < sim_duration:
        if sim.next_time() is None:
            break
        try:
            sim.step()
            steps += 1
        except PropagatorOutOfBoundsError:
            break

    total_time = time.perf_counter() - start

    return QuickBenchResult(
        name=name,
        total_time=total_time,
        steps_executed=steps,
        time_per_step=total_time / steps if steps > 0 else 0,
    )


def run_quick_suite() -> dict[str, QuickBenchResult]:
    """Run fast benchmark suite for development iteration.

    Returns
    -------
    dict[str, QuickBenchResult]
        Benchmark results keyed by test name
    """
    print("=" * 80)
    print("QUICK BENCHMARK SUITE (for iterative development)")
    print("=" * 80)

    _warmup_jit()

    results = {}

    # Small baseline
    results["small_base"] = quick_benchmark(
        "Small baseline", (500, 500), 10, 1800
    )

    # Medium scale
    results["medium"] = quick_benchmark("Medium scale", (1000, 1000), 20, 3600)

    # Realizations scaling
    results["many_real"] = quick_benchmark(
        "Many realizations", (800, 800), 50, 3600
    )

    # Large grid
    results["large_grid"] = quick_benchmark(
        "Large grid", (2000, 2000), 10, 3600
    )

    # Print results
    print("\nResults:")
    print("-" * 80)
    for result in results.values():
        print(result)

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    run_quick_suite()
