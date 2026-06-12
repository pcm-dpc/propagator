"""Large-domain benchmark based on the example runner.

Saves periodic outputs as NPZ files for later comparison.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import numpy as np

from propagator.core import (  # type: ignore
    FUEL_SYSTEM_LEGACY,
    BoundaryConditions,
    Propagator,
    PropagatorOutOfBoundsError,
)


@dataclass
class LargeDomainResult:
    name: str
    grid_size: tuple[int, int]
    realizations: int
    sim_duration: int
    step_window: int
    output_interval: int
    total_time: float
    steps_executed: int
    outputs_saved: int
    output_dir: Path


def _save_snapshot(
    output_dir: Path,
    time_seconds: int,
    simulator: Propagator,
) -> None:
    output = simulator.get_output()
    stats = output.stats
    np.savez(
        output_dir / f"snapshot_{time_seconds}.npz",
        time=np.array(time_seconds, dtype=np.int64),
        fire_probability=output.fire_probability,
        ros_mean=output.ros_mean,
        ros_max=output.ros_max,
        fli_mean=output.fli_mean,
        fli_max=output.fli_max,
        stats_n_active=np.array(stats.n_active, dtype=np.int64),
        stats_area_mean=np.array(stats.area_mean, dtype=np.float64),
        stats_area_50=np.array(stats.area_50, dtype=np.float64),
        stats_area_75=np.array(stats.area_75, dtype=np.float64),
        stats_area_90=np.array(stats.area_90, dtype=np.float64),
    )


def _write_json(path: Path, values: dict[str, object]) -> None:
    path.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_large_domain(
    benchmark_name: str,
    grid_size: tuple[int, int] = (2000, 2000),
    realizations: int = 10,
    sim_duration: int = 3600 * 24,
    step_window: int = 60,
    output_interval: int = 3600,
    output_root: Path | str = "benchmarks/results",
    seed: int = 12345,
) -> LargeDomainResult:
    rows, cols = grid_size
    np.random.seed(seed)
    veg = np.full((rows, cols), 5, dtype=np.int32)
    dem = np.zeros((rows, cols), dtype=np.float32)

    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=realizations,
        fuels=FUEL_SYSTEM_LEGACY,
        do_spotting=False,
        out_of_bounds_mode="raise",
    )

    center_x, center_y = rows // 2, cols // 2
    boundary_condition = BoundaryConditions(
        time=0,
        ignitions=[(center_x, center_y)],
        wind_speed=40.0,
        wind_dir=90.0,
        moisture=0.0,
    )
    simulator.set_boundary_conditions(boundary_condition)

    output_dir = Path(output_root) / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "run_metadata.json",
        {
            "grid_rows": rows,
            "grid_cols": cols,
            "realizations": realizations,
            "sim_duration": sim_duration,
            "step_window": step_window,
            "output_interval": output_interval,
            "seed": seed,
        },
    )

    start_time = perf_counter()
    step_time_init = perf_counter()
    steps = 0
    outputs_saved = 0
    next_output_time = output_interval
    step_times: list[float] = []

    while simulator.time < sim_duration:
        if simulator.next_time() is None:
            break
        try:
            step_start = perf_counter()
            simulator.step(seconds=step_window)
            step_times.append(perf_counter() - step_start)
            steps += 1
        except PropagatorOutOfBoundsError:
            print("Fire reached out of bounds area, stopping simulation.")
            break
        finally:
            if simulator.time >= next_output_time:
                _save_snapshot(output_dir, int(simulator.time), simulator)
                outputs_saved += 1
                elapsed = perf_counter() - step_time_init
                print(
                    f"Time: {timedelta(seconds=int(simulator.time))} | elapsed: {elapsed:.2f} seconds",
                    flush=True,
                )
                next_output_time += output_interval
                step_time_init = perf_counter()

    total_time = perf_counter() - start_time

    _write_json(
        output_dir / "run_summary.json",
        {
            "total_time": total_time,
            "steps_executed": steps,
            "outputs_saved": outputs_saved,
            "final_time": int(simulator.time),
            "step_times": step_times,
        },
    )

    return LargeDomainResult(
        name=benchmark_name,
        grid_size=grid_size,
        realizations=realizations,
        sim_duration=sim_duration,
        step_window=step_window,
        output_interval=output_interval,
        total_time=total_time,
        steps_executed=steps,
        outputs_saved=outputs_saved,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run large-domain benchmark.")
    parser.add_argument("benchmark_name", help="Name for output folder.")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--cols", type=int, default=2000)
    parser.add_argument("--realizations", type=int, default=10)
    parser.add_argument("--sim-duration", type=int, default=3600 * 24)
    parser.add_argument("--step-window", type=int, default=60)
    parser.add_argument("--output-interval", type=int, default=3600)
    parser.add_argument("--output-root", default="benchmarks/results")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    run_large_domain(
        benchmark_name=args.benchmark_name,
        grid_size=(args.rows, args.cols),
        realizations=args.realizations,
        sim_duration=args.sim_duration,
        step_window=args.step_window,
        output_interval=args.output_interval,
        output_root=args.output_root,
        seed=args.seed,
    )
