"""Benchmark with large domain and 100 realizations."""

import time

import numpy as np

from propagator.core import (
    FUEL_SYSTEM_LEGACY,
    BoundaryConditions,
    Propagator,
)


def benchmark_large_domain():
    """Run a benchmark with a large domain (5000x5000) and 100 realizations."""
    print("=" * 80)
    print("LARGE DOMAIN BENCHMARK: 5000x5000 grid, 100 realizations")
    print("=" * 80)

    # Create large domain
    domain_size = 5000
    print(f"\nSetting up {domain_size}x{domain_size} domain...")
    setup_start = time.perf_counter()

    veg = np.full((domain_size, domain_size), 5, dtype=np.int32)
    dem = np.zeros((domain_size, domain_size), dtype=np.float32)

    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=100,
        fuels=FUEL_SYSTEM_LEGACY,
        do_spotting=False,
        out_of_bounds_mode="ignore",
    )

    center_x, center_y = domain_size // 2, domain_size // 2
    boundary_condition = BoundaryConditions(
        time=0,
        ignitions=[(center_x, center_y)],
        wind_speed=30.0,
        wind_dir=90.0,
        moisture=0.0,
    )
    simulator.set_boundary_conditions(boundary_condition)

    setup_time = time.perf_counter() - setup_start
    print(f"Setup completed in {setup_time:.2f}s")

    # Warmup to trigger JIT compilation
    print("\nWarming up (JIT compilation)...")
    warmup_start = time.perf_counter()
    # for _ in range(5):
    #     if simulator.next_time() is None:
    #         break
    simulator.step(seconds=60)

    warmup_time = time.perf_counter() - warmup_start
    print(f"Warmup completed in {warmup_time:.2f}s at t={simulator.time}s")

    # Run benchmark
    print("\nRunning benchmark...")
    print(f"Initial state: time={simulator.time}s")

    benchmark_start = time.perf_counter()
    step_count = 0
    # 6 hours of simulation time
    target_time = 12 * 3600.0

    while simulator.time < target_time:
        next_time = simulator.next_time()
        if next_time is None:
            print("No more events in queue")
            break
        simulator.step(seconds=3600)
        step_count += 1

        # Progress reporting every 100 steps
        # if step_count % 1000 == 0:
        elapsed = time.perf_counter() - benchmark_start
        print(
            f"  Step {step_count}: sim_time={simulator.time:.1f}s, "
            f"real_time={elapsed:.2f}s"
        )

    benchmark_time = time.perf_counter() - benchmark_start

    # Results summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Domain size:          {domain_size}x{domain_size} cells")
    print(f"Total cells:          {domain_size * domain_size:,}")
    print("Realizations:         100")
    print(f"Steps executed:       {step_count}")
    print(f"Simulation time:      {simulator.time:.1f}s")
    print(f"Benchmark duration:   {benchmark_time:.2f}s")
    if step_count == 0:
        print("Steps per second:     n/a (no steps executed)")
        print("Sim time per step:    n/a (no steps executed)")
        print("Real time per step:   n/a (no steps executed)")
    else:
        print(f"Steps per second:     {step_count / benchmark_time:.2f}")
        print(f"Sim time per step:    {simulator.time / step_count:.2f}s")
        print(
            f"Real time per step:   {benchmark_time / step_count * 1000:.2f}ms"
        )

    # Memory estimate
    memory_mb = (
        domain_size * domain_size * 100 * 4 / (1024 * 1024)
    )  # 4 bytes per float32
    print(f"Est. state memory:    {memory_mb:.1f} MB")

    print("=" * 80)

    return {
        "domain_size": domain_size,
        "realizations": 100,
        "steps": step_count,
        "sim_time": simulator.time,
        "benchmark_time": benchmark_time,
        "steps_per_second": (
            step_count / benchmark_time if step_count > 0 else 0.0
        ),
    }


if __name__ == "__main__":
    results = benchmark_large_domain()
