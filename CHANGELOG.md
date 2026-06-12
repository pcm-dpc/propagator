# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Fire spotting tracking: per-realization ember generation and receiving
  states, exposed as `spotting_generation_probability` and
  `spotting_receiving_probability` output fields and CLI rasters.
- Arrival time metrics: per-cell `min_arrival_time` and `mean_arrival_time`
  output fields and CLI rasters.
- Benchmark suite with large-domain scenarios, comparison tooling, and
  profiling scripts.
- Validation for ignition formats in `BoundaryConditions`, including 3D
  ignition masks and simplified scalar declarations.
- Spotting example with output visualization
  (`example/example_spotting_dynamics.py`).
- Documentation pages for simulation outputs and the fire spotting model.

### Changed

- Propagation scheduling redesigned around a front-tracking kernel:
  propagation events are processed in batched time windows with
  Numba-parallel execution across realizations.
- Scheduler optimized with lazy bounding boxes and bisect insertion.
- `Propagator.step` advances using the configured time resolution with
  streamlined output handling.

### Fixed

- Unknown fuel IDs in the vegetation raster now fall back to the
  non-vegetated fuel instead of failing.
- Numba JIT function caching disabled to avoid
  `ReferenceError: underlying object has vanished` issues.
- Deprecation warnings from outdated imports removed.

## [0.0.2] - 2025-11-05

Enhanced simulation features and improved documentation (#15).

## [0.0.1] - 2025-11-03

First tagged release of the rewritten PROPAGATOR simulation engine
(`propagator.core`, `propagator.io`, CLI), including Shapely-based geometry
handling, 3D ignition masks, time expressed in seconds, statistics in
hectares, a core test suite, and pre-commit tooling with ruff.
