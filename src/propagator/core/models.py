"""Core wildfire propagation engine.

This module defines the main simulation primitives and the `Propagator` class
that evolves a fire state over a grid using wind, slope, vegetation, and
moisture inputs. Public dataclasses capture boundary conditions, actions,
summary statistics, and output snapshots suitable for CLI and IO layers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import numpy.typing as npt

# Integer coords array of shape (n, 3). We can’t encode the shape statically
# with stdlib typing, but we DO lock the dtype to integer families.
FireBehaviourUpdate = tuple[int, int, int, float, float]

UpdateBatchTuple = tuple[
    npt.NDArray[np.integer],
    npt.NDArray[np.integer],
    npt.NDArray[np.integer],
    npt.NDArray[np.integer],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]


@dataclass
class UpdateBatch:
    rows: npt.NDArray[np.integer] = field(
        default_factory=lambda: np.empty((0,), dtype=np.int32)
    )

    cols: npt.NDArray[np.integer] = field(
        default_factory=lambda: np.empty((0,), dtype=np.int32)
    )

    realizations: npt.NDArray[np.integer] = field(
        default_factory=lambda: np.empty((0,), dtype=np.int32)
    )

    rates_of_spread: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )

    fireline_intensities: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )

    # Bounding box (min_row, min_col, max_row, max_col) - computed lazily
    bbox: Optional[tuple[int, int, int, int]] = field(init=False, default=None)
    # Flag to track whether bbox has been computed (avoids recomputation)
    _bbox_computed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        """Validate array lengths and initialize lazy bbox computation.

        OPTIMIZATION: Previously computed bbox (min/max) in __post_init__ for
        every UpdateBatch creation. Profiling showed this was expensive and
        often wasteful since bbox is only needed for bounds checking, which
        is rarely triggered. Now we defer computation until actually needed.
        """
        n = len(self.rows)
        if not (
            len(self.cols) == n
            and len(self.realizations) == n
            and len(self.rates_of_spread) == n
            and len(self.fireline_intensities) == n
        ):
            raise ValueError("All input arrays must have the same length")
        # Initialize bbox as uncomputed
        # Note: Must use object.__setattr__ because dataclass may be frozen
        # or have custom __setattr__ behavior in subclasses
        object.__setattr__(self, "_bbox_computed", False)
        object.__setattr__(self, "bbox", None)

    def _compute_bbox(self) -> None:
        """Compute bounding box from update coordinates.

        This is an expensive operation (4 min/max calls over potentially
        large arrays) so we only do it when bbox is actually accessed.
        Uses object.__setattr__ to bypass any dataclass restrictions.
        """
        if self._bbox_computed:
            return

        n = len(self.rows)
        if n == 0:
            object.__setattr__(self, "bbox", None)
        else:
            # Compute tight bounding box around all updates
            r0 = int(np.min(self.rows))
            c0 = int(np.min(self.cols))
            r1 = int(np.max(self.rows))
            c1 = int(np.max(self.cols))
            object.__setattr__(self, "bbox", (r0, c0, r1, c1))
        object.__setattr__(self, "_bbox_computed", True)

    def get_bbox(self) -> Optional[tuple[int, int, int, int]]:
        """Get bounding box, computing lazily if not yet calculated.

        External code should use this instead of accessing .bbox directly
        to ensure the bbox is computed when needed.
        """
        if not self._bbox_computed:
            self._compute_bbox()
        return self.bbox

    def extend(self, other: "UpdateBatch") -> None:
        """Merge another UpdateBatch into this one.

        Concatenates all arrays and invalidates the cached bounding box.

        OPTIMIZATION: Previously had complex logic to merge bboxes, checking
        if either was None, then computing min/max of merged bounds. This was
        wasteful because:
        1. Often the merged bbox is never used
        2. If it is used, we can compute it directly from the merged arrays

        New approach: Simply invalidate the bbox cache. If someone needs it
        later via get_bbox(), it will be computed fresh from all data.

        Performance: Avoids 8 min/max operations per extend call in the
        common case where bbox isn't needed afterward.
        """
        self.rows = np.concatenate([self.rows, other.rows])
        self.cols = np.concatenate([self.cols, other.cols])
        self.realizations = np.concatenate(
            [self.realizations, other.realizations]
        )
        self.rates_of_spread = np.concatenate(
            [self.rates_of_spread, other.rates_of_spread]
        )
        self.fireline_intensities = np.concatenate(
            [self.fireline_intensities, other.fireline_intensities]
        )
        # Invalidate cached bbox - will be recomputed if needed
        object.__setattr__(self, "_bbox_computed", False)
        object.__setattr__(self, "bbox", None)


@dataclass(frozen=True)
class UpdateBatchWithTime:
    times: npt.NDArray[np.integer]
    rows: npt.NDArray[np.integer]
    cols: npt.NDArray[np.integer]
    realizations: npt.NDArray[np.integer]
    rates_of_spread: npt.NDArray[np.float32]
    fireline_intensities: npt.NDArray[np.float32]

    @staticmethod
    def from_tuple(data: UpdateBatchTuple) -> "UpdateBatchWithTime":
        (
            times,
            rows,
            cols,
            realizations,
            rates_of_spread,
            fireline_intensities,
        ) = data
        return UpdateBatchWithTime(
            times=times,
            rows=rows,
            cols=cols,
            realizations=realizations,
            rates_of_spread=rates_of_spread,
            fireline_intensities=fireline_intensities,
        )

    def split_by_time(self) -> dict[int, UpdateBatch]:
        """Split updates into separate batches grouped by time.

        OPTIMIZATION: This function is called in the hot path (Scheduler.push_updates)
        and was identified as a major bottleneck. Several optimizations:

        1. FAST PATH: Check if all updates are at the same time (common case).
           If so, avoid array indexing and just wrap the existing arrays.
           This happens frequently when processing a batch of cells that all
           spread to neighbors at the same relative time.

        2. REDUCED OBJECT CREATION: Eliminated intermediate variables that
           were created just to pass to UpdateBatch constructor. Now pass
           sliced arrays directly.

        3. TYPE CONVERSION: Move int() conversion outside the inner operations
           to reduce redundant type checking.

        Note: Each UpdateBatch created here will have _bbox_computed=False,
        deferring expensive min/max calculations until/unless needed.
        """
        result: dict[int, UpdateBatch] = {}
        unique_times = np.unique(self.times)

        # Fast path for single time value (common when cells spread uniformly)
        if len(unique_times) == 1:
            time = int(unique_times[0])
            # No need to slice - all updates are at this time
            result[time] = UpdateBatch(
                self.rows,
                self.cols,
                self.realizations,
                self.rates_of_spread,
                self.fireline_intensities,
            )
            return result

        # General case: split by time using boolean indexing
        for time in unique_times:
            # Create boolean mask for this time
            index = self.times == time
            # Create UpdateBatch with sliced arrays (bbox computed lazily)
            result[int(time)] = UpdateBatch(
                self.rows[index],
                self.cols[index],
                self.realizations[index],
                self.rates_of_spread[index],
                self.fireline_intensities[index],
            )

        return result


class PropagatorError(Exception):
    """Domain-specific error raised by PROPAGATOR."""


def validate_ignitions(ignitions):
    if isinstance(ignitions, list):
        for item in ignitions:
            if not (
                isinstance(item, tuple)
                and len(item) in (2, 3)
                and all(isinstance(x, int) for x in item)
            ):
                raise ValueError(
                    "Ignition list items must be (row, col) or (row, col, realization) tuples"
                )
    elif isinstance(ignitions, np.ndarray):
        if ignitions.ndim not in (2, 3):
            raise ValueError("Ignition ndarray must be 2D or 3D boolean array")
    else:
        raise ValueError(
            "Ignitions must be either a list of tuples or a boolean ndarray"
        )


@dataclass(frozen=True)
class BoundaryConditions:
    """
    Boundary conditions applied at or after a given simulation time.


    Attributes
    ----------
    time : int
        Simulation time the conditions refer to (seconds from simulation start).
    moisture : Optional[npt.NDArray[np.floating]]
        Fuel moisture map (%).
    wind_dir : Optional[npt.NDArray[np.floating]]
        Wind direction map (weather convention, degrees clockwise, north is 0).
    wind_speed : Optional[npt.NDArray[np.floating]]
        Wind speed map (km/h).
    ignitions : Optional[
        npt.NDArray[np.bool_]
        | list[tuple[int, int] | tuple[int, int, int]]
    ]
        Ignitions to enqueue. Accepts either a boolean raster (2D applies to
        every realization; 3D maps explicit `realization` planes) or a list of
        `(row, col)` / `(row, col, realization)` tuples.
    additional_moisture : Optional[npt.NDArray[np.floating]]
        Extra moisture to add to fuel (%), can be sparse.
    vegetation_changes : Optional[npt.NDArray[np.floating]]
        Raster of vegetation type overrides (NaN to skip).
    """

    time: int
    moisture: Optional[npt.ArrayLike] = None
    wind_dir: Optional[npt.ArrayLike] = None
    wind_speed: Optional[npt.ArrayLike] = None
    ignitions: Optional[
        npt.NDArray[np.bool_] | list[tuple[int, int] | tuple[int, int, int]]
    ] = None
    additional_moisture: Optional[npt.NDArray[np.floating]] = None
    vegetation_changes: Optional[npt.NDArray[np.floating]] = None

    def __post_init__(self):
        if self.time < 0:
            raise ValueError("BoundaryConditions time must be non-negative")
        if self.ignitions is not None:
            validate_ignitions(self.ignitions)


@dataclass(frozen=True)
class PropagatorStats:
    """Summary statistics for the current simulation state."""

    n_active: int
    area_mean: float
    area_50: float
    area_75: float
    area_90: float

    def to_dict(
        self, c_time: int, ref_date: datetime
    ) -> dict[str, float | int | str]:
        """Serialize stats with the current simulation time expressed in seconds."""
        return dict(
            c_time=c_time,
            ref_date=ref_date.isoformat(),
            n_active=self.n_active,
            area_mean=self.area_mean,
            area_50=self.area_50,
            area_75=self.area_75,
            area_90=self.area_90,
        )


@dataclass(frozen=True)
class PropagatorOutput:
    """Snapshot of simulation outputs at a given time step."""

    time: int  # seconds from simulation start
    fire_probability: npt.NDArray[np.floating]
    ros_mean: npt.NDArray[np.floating]
    ros_max: npt.NDArray[np.floating]
    fli_mean: npt.NDArray[np.floating]
    fli_max: npt.NDArray[np.floating]
    stats: PropagatorStats
