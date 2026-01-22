"""Detailed profiling with pyinstrument for the propagator simulator."""

import numpy as np
from pyinstrument import Profiler

from propagator.core import (
    FUEL_SYSTEM_LEGACY,
    BoundaryConditions,
    Propagator,
)


def profile_simulation():
    """Run a simulation with detailed profiling."""
    print("Setting up simulation...")
    veg = np.full((1000, 1000), 5, dtype=np.int32)
    dem = np.zeros((1000, 1000), dtype=np.float32)

    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=100,  # Test with many realizations
        fuels=FUEL_SYSTEM_LEGACY,
        do_spotting=False,
        out_of_bounds_mode="ignore",
    )

    center_x, center_y = veg.shape[0] // 2, veg.shape[1] // 2
    boundary_condition = BoundaryConditions(
        time=0,
        ignitions=[(center_x, center_y)],
        wind_speed=30.0,
        wind_dir=90.0,
        moisture=0.0,
    )
    simulator.set_boundary_conditions(boundary_condition)

    # Warmup to exclude JIT
    for _ in range(10):
        if simulator.next_time() is None:
            break
        simulator.step(seconds=3600)

    print(f"Warmup complete, starting profiler at {simulator.time}s...")
    profiler = Profiler()
    profiler.start()

    # Run simulation for limited time

    while simulator.time < 86400:
        next_time = simulator.next_time()
        if next_time is None:
            break
        simulator.step(seconds=3600)

    profiler.stop()

    print(f"\nSimulation completed: time={simulator.time}s")
    print("\n" + "=" * 80)
    print("PROFILE RESULTS")
    print("=" * 80)
    profiler.print()


if __name__ == "__main__":
    profile_simulation()
