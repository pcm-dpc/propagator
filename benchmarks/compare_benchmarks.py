"""Compare two benchmark runs by diffing saved NPZ snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_snapshots(path: Path) -> dict[int, dict[str, np.ndarray]]:
    snapshots: dict[int, dict[str, np.ndarray]] = {}
    for npz_path in sorted(path.glob("snapshot_*.npz")):
        data = np.load(npz_path)
        time_value = int(np.array(data["time"]).item())
        snapshots[time_value] = {
            key: np.array(data[key]) for key in data.files
        }
    return snapshots


def _compare_arrays(
    a: np.ndarray, b: np.ndarray
) -> tuple[float, float, float] | None:
    if a.shape != b.shape:
        return None
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    nan_a = np.isnan(a64)
    nan_b = np.isnan(b64)
    nan_mismatch = np.any(nan_a != nan_b)
    diff = a64 - b64
    if np.any(nan_a & nan_b):
        diff = diff.copy()
        diff[nan_a & nan_b] = 0.0
    if nan_mismatch:
        return float("inf"), float("inf"), float("inf")
    abs_diff = np.abs(diff)
    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    rms = float(np.sqrt(np.mean(diff * diff)))
    return max_abs, mean_abs, rms


def _resolve_run_path(path: Path) -> Path:
    if path.exists():
        return path
    candidate = Path("benchmarks") / "results" / path
    if candidate.exists():
        return candidate
    return path


def compare_runs(left: Path, right: Path) -> int:
    left = _resolve_run_path(left)
    right = _resolve_run_path(right)
    left_meta = _load_json(left / "run_metadata.json")
    right_meta = _load_json(right / "run_metadata.json")
    left_summary = _load_json(left / "run_summary.json")
    right_summary = _load_json(right / "run_summary.json")

    print(f"Left:  {left}")
    print(f"Right: {right}")
    if left_meta is not None and right_meta is not None:
        if left_meta != right_meta:
            print("Metadata differs.")
    if left_summary is not None and right_summary is not None:
        left_steps = left_summary.get("steps_executed")
        right_steps = right_summary.get("steps_executed")
        left_total = left_summary.get("total_time")
        right_total = right_summary.get("total_time")
        left_times = left_summary.get("step_times", [])
        right_times = right_summary.get("step_times", [])
        left_mean = float(np.mean(left_times)) if left_times else 0.0
        right_mean = float(np.mean(right_times)) if right_times else 0.0
        print(
            "Timing summary: "
            f"left total={left_total} steps={left_steps} mean_step={left_mean:.6g} | "
            f"right total={right_total} steps={right_steps} mean_step={right_mean:.6g}"
        )

    left_snapshots = _load_snapshots(left)
    right_snapshots = _load_snapshots(right)

    common_times = sorted(set(left_snapshots) & set(right_snapshots))
    if not common_times:
        print("No common snapshot times found.")
        return 1

    max_time = common_times[-1]
    print(f"Comparing {len(common_times)} snapshots up to t={max_time}")

    failures = 0
    mismatches = 0
    stats: dict[str, dict[str, float | int]] = {}

    def _ensure_stat(key: str) -> dict[str, float | int]:
        if key not in stats:
            stats[key] = {
                "max_abs": 0.0,
                "mean_abs_sum": 0.0,
                "rms_sum": 0.0,
                "count": 0,
                "mismatch_count": 0,
                "missing_or_shape": 0,
            }
        return stats[key]

    for t in common_times:
        left_data = left_snapshots[t]
        right_data = right_snapshots[t]
        keys = sorted(set(left_data) | set(right_data))
        for key in keys:
            if key == "time":
                continue
            entry = _ensure_stat(key)
            if key not in left_data or key not in right_data:
                failures += 1
                mismatches += 1
                entry["missing_or_shape"] = int(entry["missing_or_shape"]) + 1
                continue
            result = _compare_arrays(left_data[key], right_data[key])
            if result is None:
                failures += 1
                mismatches += 1
                entry["missing_or_shape"] = int(entry["missing_or_shape"]) + 1
                continue
            max_abs, mean_abs, rms = result
            entry["count"] = int(entry["count"]) + 1
            entry["mean_abs_sum"] = float(entry["mean_abs_sum"]) + mean_abs
            entry["rms_sum"] = float(entry["rms_sum"]) + rms
            if max_abs > float(entry["max_abs"]):
                entry["max_abs"] = max_abs
            if max_abs != 0.0 or mean_abs != 0.0 or rms != 0.0:
                entry["mismatch_count"] = int(entry["mismatch_count"]) + 1
                mismatches += 1

    if stats:
        print("\nField comparison (common snapshots):")
        print(
            "field                       max_abs        mean_abs        rms        mismatches"
        )
        print("-" * 78)
        for key in sorted(stats):
            entry = stats[key]
            count = int(entry["count"])
            mean_abs = float(entry["mean_abs_sum"]) / count if count else 0.0
            rms = float(entry["rms_sum"]) / count if count else 0.0
            mismatch_count = int(entry["mismatch_count"])
            missing_or_shape = int(entry["missing_or_shape"])
            mismatch_total = mismatch_count + missing_or_shape
            print(
                f"{key:28s} {entry['max_abs']:12.6g} {mean_abs:12.6g} {rms:10.6g} {mismatch_total:10d}"
            )

    status = "OK" if mismatches == 0 else "MISMATCH"
    print(f"\nComparison result: {status}")
    return 0 if failures == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two benchmark runs using NPZ snapshots."
    )
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    return compare_runs(args.left, args.right)


if __name__ == "__main__":
    raise SystemExit(main())
