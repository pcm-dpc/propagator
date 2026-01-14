"""Benchmark suite for propagator performance testing."""

from .benchmark_core import (
    BenchmarkResult,
    benchmark_basic_spread,
    benchmark_grid_sizes,
    benchmark_heterogeneous_fuels,
    benchmark_multiple_realizations,
    benchmark_variable_wind,
    benchmark_with_terrain,
    profile_step_components,
    run_benchmark_suite,
)

__all__ = [
    "BenchmarkResult",
    "benchmark_basic_spread",
    "benchmark_grid_sizes",
    "benchmark_heterogeneous_fuels",
    "benchmark_multiple_realizations",
    "benchmark_variable_wind",
    "benchmark_with_terrain",
    "profile_step_components",
    "run_benchmark_suite",
]
