"""Performance benchmarks for the Propagator simulator.

This module provides benchmarks for the core simulation components
to identify performance bottlenecks and track optimization progress.
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
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    grid_size: tuple[int, int]
    realizations: int
    total_time: float
    steps_executed: int
    time_per_step: float
    final_burned_cells: int
    peak_memory_mb: float | None = None

    def __str__(self) -> str:
        return (
            f"{self.name}\n"
            f"  Grid: {self.grid_size[0]}x{self.grid_size[1]}, "
            f"Realizations: {self.realizations}\n"
            f"  Total time: {self.total_time:.3f}s, "
            f"Steps: {self.steps_executed}, "
            f"Time/step: {self.time_per_step:.4f}s\n"
            f"  Burned cells: {self.final_burned_cells}"
        )


def _warmup_jit(
    grid_size: tuple[int, int] = (100, 100),
    realizations: int = 2,
    spotting: bool = False,
) -> None:
    """Warm up Numba JIT compilation before benchmarking.

    This runs a small simulation to trigger JIT compilation of all
    numba-decorated functions so that benchmarks measure actual
    execution time, not compilation time.

    Parameters
    ----------
    grid_size : tuple[int, int]
        Small grid for warm-up
    realizations : int
        Minimal realizations for warm-up
    spotting : bool
        Enable spotting if needed
    """
    rows, cols = grid_size
    veg = np.full((rows, cols), 5, dtype=np.int32)
    dem = np.zeros((rows, cols), dtype=np.float32)

    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=realizations,
        fuels=FUEL_SYSTEM_LEGACY,
        do_spotting=spotting,
        out_of_bounds_mode="ignore",
    )

    center_x, center_y = rows // 2, cols // 2
    boundary_condition = BoundaryConditions(
        time=0,
        ignitions=[(center_x, center_y)],
        wind_speed=20.0,
        wind_dir=90.0,
        moisture=0.0,
    )
    simulator.set_boundary_conditions(boundary_condition)

    # Run a few steps to trigger JIT compilation
    for _ in range(10):
        if simulator.next_time() is None:
            break
        simulator.step()


def benchmark_basic_spread(
    grid_size: tuple[int, int] = (500, 500),
    realizations: int = 10,
    sim_duration: int = 3600,
    wind_speed: float = 20.0,
    spotting: bool = False,
    skip_warmup: bool = False,
) -> BenchmarkResult:
    """Benchmark basic fire spread scenario.

    Parameters
    ----------
    grid_size : tuple[int, int]
        Dimensions of simulation grid
    realizations : int
        Number of stochastic realizations
    sim_duration : int
        Simulation duration in seconds
    wind_speed : float
        Wind speed in km/h
    spotting : bool
        Enable fire spotting
    skip_warmup : bool
        Skip JIT warmup (for internal use when already warmed up)

    Returns
    -------
    BenchmarkResult
        Timing and performance metrics
    """
    # Warm up JIT compiler first
    if not skip_warmup:
        _warmup_jit(spotting=spotting)

    rows, cols = grid_size
    veg = np.full((rows, cols), 5, dtype=np.int32)
    dem = np.zeros((rows, cols), dtype=np.float32)

    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=realizations,
        fuels=FUEL_SYSTEM_LEGACY,
        do_spotting=spotting,
        out_of_bounds_mode="ignore",
    )

    center_x, center_y = rows // 2, cols // 2
    boundary_condition = BoundaryConditions(
        time=0,
        ignitions=[(center_x, center_y)],
        wind_speed=wind_speed,
        wind_dir=90.0,
        moisture=0.0,
    )
    simulator.set_boundary_conditions(boundary_condition)

    start_time = time.perf_counter()
    steps = 0
    last_log_time = start_time

    while simulator.time < sim_duration:
        next_time = simulator.next_time()
        if next_time is None:
            break
        try:
            simulator.step()
            steps += 1

            # Log progress every 2 seconds
            current_time = time.perf_counter()
            if current_time - last_log_time > 2.0:
                print(
                    f"    Progress: step {steps}, sim_time={simulator.time}/{sim_duration}, elapsed={current_time - start_time:.1f}s",
                    flush=True,
                )
                last_log_time = current_time
        except PropagatorOutOfBoundsError:
            break

    end_time = time.perf_counter()
    total_time = end_time - start_time

    fire_prob = simulator.compute_fire_probability()
    burned_cells = int(np.sum(fire_prob > 0))

    return BenchmarkResult(
        name=f"basic_spread_{'spotting' if spotting else 'no_spotting'}",
        grid_size=grid_size,
        realizations=realizations,
        total_time=total_time,
        steps_executed=steps,
        time_per_step=total_time / steps if steps > 0 else 0,
        final_burned_cells=burned_cells,
    )


def benchmark_multiple_realizations(
    grid_size: tuple[int, int] = (300, 300),
    realization_counts: list[int] = [1, 5, 10, 20],
    sim_duration: int = 1800,
) -> list[BenchmarkResult]:
    """Benchmark scaling with number of realizations.

    Parameters
    ----------
    grid_size : tuple[int, int]
        Grid dimensions
    realization_counts : list[int]
        Realization counts to benchmark
    sim_duration : int
        Simulation duration in seconds

    Returns
    -------
    list[BenchmarkResult]
        Results for each realization count
    """
    # Warm up JIT once for all runs
    _warmup_jit(spotting=False)

    results = []
    for n_real in realization_counts:
        result = benchmark_basic_spread(
            grid_size=grid_size,
            realizations=n_real,
            sim_duration=sim_duration,
            spotting=False,
            skip_warmup=True,
        )
        results.append(result)
    return results


def benchmark_grid_sizes(
    sizes: list[tuple[int, int]] = [
        (100, 100),
        (300, 300),
        (500, 500),
        (1000, 1000),
    ],
    realizations: int = 10,
    sim_duration: int = 1800,
) -> list[BenchmarkResult]:
    """Benchmark scaling with grid size.

    Parameters
    ----------
    sizes : list[tuple[int, int]]
        Grid sizes to benchmark
    realizations : int
        Number of realizations
    sim_duration : int
        Simulation duration in seconds

    Returns
    -------
    list[BenchmarkResult]
        Results for each grid size
    """
    # Warm up JIT once for all runs
    _warmup_jit(spotting=False)

    results = []
    for size in sizes:
        result = benchmark_basic_spread(
            grid_size=size,
            realizations=realizations,
            sim_duration=sim_duration,
            spotting=False,
            skip_warmup=True,
        )
        results.append(result)
    return results


def benchmark_heterogeneous_fuels(
    grid_size: tuple[int, int] = (2000, 2000),
    realizations: int = 50,
    sim_duration: int = 7200,
) -> BenchmarkResult:
    """Benchmark with heterogeneous fuel distribution.

    Creates a realistic patchy fuel distribution.
    """
    _warmup_jit(spotting=False)

    rows, cols = grid_size
    # Create patchy fuel distribution (more realistic)
    veg = np.random.choice([2, 3, 4, 5, 6, 7], size=(rows, cols)).astype(
        np.int32
    )
    # Add some no-fuel patches
    no_fuel_mask = np.random.random((rows, cols)) < 0.15
    veg[no_fuel_mask] = 0

    dem = np.zeros((rows, cols), dtype=np.float32)

    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=realizations,
        fuels=FUEL_SYSTEM_LEGACY,
        do_spotting=False,
        out_of_bounds_mode="ignore",
    )

    center_x, center_y = rows // 2, cols // 2
    boundary_condition = BoundaryConditions(
        time=0,
        ignitions=[(center_x, center_y)],
        wind_speed=25.0,
        wind_dir=90.0,
        moisture=0.1,
    )
    simulator.set_boundary_conditions(boundary_condition)

    start_time = time.perf_counter()
    steps = 0
    last_log_time = start_time

    while simulator.time < sim_duration:
        next_time = simulator.next_time()
        if next_time is None:
            break
        try:
            simulator.step()
            steps += 1

            # Log progress every 2 seconds
            current_time = time.perf_counter()
            if current_time - last_log_time > 2.0:
                print(
                    f"    Progress: step {steps}, sim_time={simulator.time}/{sim_duration}, elapsed={current_time - start_time:.1f}s",
                    flush=True,
                )
                last_log_time = current_time
        except PropagatorOutOfBoundsError:
            break

    end_time = time.perf_counter()
    total_time = end_time - start_time

    fire_prob = simulator.compute_fire_probability()
    burned_cells = int(np.sum(fire_prob > 0))

    return BenchmarkResult(
        name="heterogeneous_fuels",
        grid_size=grid_size,
        realizations=realizations,
        total_time=total_time,
        steps_executed=steps,
        time_per_step=total_time / steps if steps > 0 else 0,
        final_burned_cells=burned_cells,
    )


def benchmark_with_terrain(
    grid_size: tuple[int, int] = (2000, 2000),
    realizations: int = 50,
    sim_duration: int = 7200,
) -> BenchmarkResult:
    """Benchmark with realistic terrain (slope effects)."""
    _warmup_jit(spotting=False)

    rows, cols = grid_size
    veg = np.full((rows, cols), 5, dtype=np.int32)

    # Create realistic terrain with slopes
    x = np.linspace(0, 10, cols)
    y = np.linspace(0, 10, rows)
    X, Y = np.meshgrid(x, y)
    dem = (
        100 * np.sin(X / 2) * np.cos(Y / 2)
        + 50 * np.sin(X / 3) * np.sin(Y / 4)
        + 200
    ).astype(np.float32)

    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=realizations,
        fuels=FUEL_SYSTEM_LEGACY,
        do_spotting=False,
        out_of_bounds_mode="ignore",
    )

    center_x, center_y = rows // 2, cols // 2
    boundary_condition = BoundaryConditions(
        time=0,
        ignitions=[(center_x, center_y)],
        wind_speed=30.0,
        wind_dir=45.0,
        moisture=0.05,
    )
    simulator.set_boundary_conditions(boundary_condition)

    start_time = time.perf_counter()
    steps = 0
    last_log_time = start_time

    while simulator.time < sim_duration:
        next_time = simulator.next_time()
        if next_time is None:
            break
        try:
            simulator.step()
            steps += 1

            # Log progress every 2 seconds
            current_time = time.perf_counter()
            if current_time - last_log_time > 2.0:
                print(
                    f"    Progress: step {steps}, sim_time={simulator.time}/{sim_duration}, elapsed={current_time - start_time:.1f}s",
                    flush=True,
                )
                last_log_time = current_time
        except PropagatorOutOfBoundsError:
            break

    end_time = time.perf_counter()
    total_time = end_time - start_time

    fire_prob = simulator.compute_fire_probability()
    burned_cells = int(np.sum(fire_prob > 0))

    return BenchmarkResult(
        name="with_terrain",
        grid_size=grid_size,
        realizations=realizations,
        total_time=total_time,
        steps_executed=steps,
        time_per_step=total_time / steps if steps > 0 else 0,
        final_burned_cells=burned_cells,
    )


def benchmark_variable_wind(
    grid_size: tuple[int, int] = (2000, 2000),
    realizations: int = 50,
    sim_duration: int = 7200,
) -> BenchmarkResult:
    """Benchmark with spatially variable wind field."""
    _warmup_jit(spotting=False)

    rows, cols = grid_size
    veg = np.full((rows, cols), 5, dtype=np.int32)
    dem = np.zeros((rows, cols), dtype=np.float32)

    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=realizations,
        fuels=FUEL_SYSTEM_LEGACY,
        do_spotting=False,
        out_of_bounds_mode="ignore",
    )

    # Create spatially variable wind
    x = np.linspace(0, 2 * np.pi, cols)
    y = np.linspace(0, 2 * np.pi, rows)
    X, Y = np.meshgrid(x, y)
    wind_speed = (20 + 10 * np.sin(X) * np.cos(Y)).astype(np.float32)
    wind_dir = (90 + 30 * np.cos(X / 2) * np.sin(Y / 2)).astype(np.float32)

    center_x, center_y = rows // 2, cols // 2
    boundary_condition = BoundaryConditions(
        time=0,
        ignitions=[(center_x, center_y)],
        wind_speed=wind_speed,
        wind_dir=wind_dir,
        moisture=0.08,
    )
    simulator.set_boundary_conditions(boundary_condition)

    start_time = time.perf_counter()
    steps = 0
    last_log_time = start_time

    while simulator.time < sim_duration:
        next_time = simulator.next_time()
        if next_time is None:
            break
        try:
            simulator.step()
            steps += 1

            # Log progress every 2 seconds
            current_time = time.perf_counter()
            if current_time - last_log_time > 2.0:
                print(
                    f"    Progress: step {steps}, sim_time={simulator.time}/{sim_duration}, elapsed={current_time - start_time:.1f}s",
                    flush=True,
                )
                last_log_time = current_time
        except PropagatorOutOfBoundsError:
            break

    end_time = time.perf_counter()
    total_time = end_time - start_time

    fire_prob = simulator.compute_fire_probability()
    burned_cells = int(np.sum(fire_prob > 0))

    return BenchmarkResult(
        name="variable_wind",
        grid_size=grid_size,
        realizations=realizations,
        total_time=total_time,
        steps_executed=steps,
        time_per_step=total_time / steps if steps > 0 else 0,
        final_burned_cells=burned_cells,
    )


def profile_step_components(
    grid_size: tuple[int, int] = (500, 500),
    realizations: int = 10,
    num_steps: int = 10,
) -> dict[str, float]:
    """Profile time spent in different components of a step.

    Parameters
    ----------
    grid_size : tuple[int, int]
        Grid dimensions
    realizations : int
        Number of realizations
    num_steps : int
        Number of steps to profile

    Returns
    -------
    dict[str, float]
        Time spent in each component
    """
    # Warm up JIT compiler first
    _warmup_jit(spotting=False)

    rows, cols = grid_size
    veg = np.full((rows, cols), 5, dtype=np.int32)
    dem = np.zeros((rows, cols), dtype=np.float32)

    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=realizations,
        fuels=FUEL_SYSTEM_LEGACY,
        do_spotting=False,
        out_of_bounds_mode="ignore",
    )

    center_x, center_y = rows // 2, cols // 2
    boundary_condition = BoundaryConditions(
        time=0,
        ignitions=[(center_x, center_y)],
        wind_speed=20.0,
        wind_dir=90.0,
        moisture=0.0,
    )
    simulator.set_boundary_conditions(boundary_condition)

    # Profile components - just use step() since it's now refactored
    timings = {
        "step": 0.0,
        "total": 0.0,
    }

    for _ in range(num_steps):
        start = time.perf_counter()

        if simulator.next_time() is None:
            break

        # Time the step
        t0 = time.perf_counter()
        simulator.step()
        timings["step"] += time.perf_counter() - t0

        timings["total"] += time.perf_counter() - start

    # Average over steps
    for key in timings:
        timings[key] /= num_steps

    return timings


def run_benchmark_suite() -> None:
    """Run comprehensive benchmark suite (10x scale) and print results."""
    print("=" * 80)
    print("PROPAGATOR PERFORMANCE BENCHMARK SUITE (10X SCALE)")
    print("=" * 80)
    print("\nNote: Benchmarks exclude JIT compilation time via warm-up runs")
    print(
        "Warning: Large grids/realizations - this will take significant time!"
    )
    print("=" * 80)

    print("\n1. LARGE SCALE - Basic spread (10x baseline)")
    print("-" * 80)
    result = benchmark_basic_spread(
        grid_size=(5000, 5000),
        realizations=100,
        sim_duration=36000,
        skip_warmup=False,
    )
    print(result)

    print("\n2. SCALING - Realizations (large grid)")
    print("-" * 80)
    results = benchmark_multiple_realizations(
        grid_size=(3000, 3000),
        realization_counts=[10, 25, 50, 100],
        sim_duration=18000,
    )
    for r in results:
        print(
            f"  {r.realizations:3d} realizations: "
            f"{r.total_time:.3f}s total, "
            f"{r.time_per_step:.4f}s/step, "
            f"{r.steps_executed} steps"
        )

    print("\n3. SCALING - Grid size (many realizations)")
    print("-" * 80)
    results = benchmark_grid_sizes(
        sizes=[(1000, 1000), (3000, 3000), (5000, 5000)],
        realizations=50,
        sim_duration=18000,
    )
    for r in results:
        print(
            f"  {r.grid_size[0]:4d}x{r.grid_size[1]:4d}: "
            f"{r.total_time:.3f}s total, "
            f"{r.time_per_step:.4f}s/step, "
            f"{r.steps_executed} steps"
        )

    print("\n4. HETEROGENEOUS - Patchy fuel distribution")
    print("-" * 80)
    result = benchmark_heterogeneous_fuels(
        grid_size=(2000, 2000),
        realizations=50,
        sim_duration=7200,
    )
    print(result)

    print("\n5. TERRAIN - Complex topography with slopes")
    print("-" * 80)
    result = benchmark_with_terrain(
        grid_size=(2000, 2000),
        realizations=50,
        sim_duration=7200,
    )
    print(result)

    print("\n6. VARIABLE WIND - Spatially varying wind field")
    print("-" * 80)
    result = benchmark_variable_wind(
        grid_size=(2000, 2000),
        realizations=50,
        sim_duration=7200,
    )
    print(result)

    print("\n7. COMPONENT PROFILING - Large scale (average per step)")
    print("-" * 80)
    timings = profile_step_components(
        grid_size=(5000, 5000),
        realizations=50,
        num_steps=50,
    )
    for component, t in sorted(timings.items(), key=lambda x: -x[1]):
        pct = (t / timings["total"] * 100) if timings["total"] > 0 else 0
        print(f"  {component:20s}: {t:.4f}s ({pct:5.1f}%)")

    print("\n8. SPOTTING - Impact at scale")
    print("-" * 80)
    # Warm up both variants
    _warmup_jit(spotting=False)
    _warmup_jit(spotting=True)

    no_spotting = benchmark_basic_spread(
        grid_size=(2000, 2000),
        realizations=50,
        sim_duration=7200,
        spotting=False,
        skip_warmup=True,
    )
    with_spotting = benchmark_basic_spread(
        grid_size=(2000, 2000),
        realizations=50,
        sim_duration=7200,
        spotting=True,
        skip_warmup=True,
    )
    print(f"  Without spotting: {no_spotting.total_time:.3f}s")
    print(f"  With spotting:    {with_spotting.total_time:.3f}s")
    overhead = (with_spotting.total_time / no_spotting.total_time - 1) * 100
    print(f"  Overhead:         {overhead:.1f}%")

    print("\n" + "=" * 80)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark_suite()
