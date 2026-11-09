"""Past-only replay using logged time as an explicitly qualified availability proxy."""

import numpy as np


def causal_replay(logged_seconds, positions_m, grid_seconds):
    """Return positions and separate quality arrays; do not add state channels.

    Reject time reversals rather than backdating a late logged row. Exact-time
    duplicates use the last logged row. No Age-field backdating is performed.
    Missing prefix positions remain NaN and must be handled explicitly by the
    caller. The raw ADS-B positions are observations, not independent truth.
    """
    times = np.asarray(logged_seconds, dtype=np.float64)
    positions = np.asarray(positions_m, dtype=np.float64)
    grid = np.asarray(grid_seconds, dtype=np.float64)
    if times.ndim != 1 or grid.ndim != 1 or positions.shape != (len(times), 2):
        raise ValueError("Expected N times, [N,2] positions and T grid times")
    if not np.isfinite(times).all() or not np.isfinite(grid).all() or not np.isfinite(positions).all():
        raise ValueError("Non-finite raw inputs must be explicitly filtered upstream")
    if np.any(np.diff(times) < 0) or np.any(np.diff(grid) < 0):
        raise ValueError("Time reversal: split or exclude segment before replay")
    indices = np.searchsorted(times, grid, side="right") - 1
    available = indices >= 0
    replay = np.full((len(grid), 2), np.nan)
    age = np.full(len(grid), np.nan)
    changed_age = np.full(len(grid), np.nan)
    conflicts = np.zeros(len(grid), dtype=bool)
    if len(times):
        # Changed position is a descriptive freshness proxy, not evidence of a
        # newly acquired independent sensor report.
        changed = np.r_[True, np.any(np.diff(positions, axis=0) != 0, axis=1)]
        latest_changed = np.maximum.accumulate(np.where(changed, np.arange(len(times)), 0))
        duplicate_conflict = np.r_[False, (np.diff(times) == 0) & np.any(np.diff(positions, axis=0) != 0, axis=1)]
        # Propagate any conflict to the final row of an equal-timestamp group.
        group_starts = np.r_[0, np.where(np.diff(times) != 0)[0] + 1]
        group_conflict = np.maximum.reduceat(duplicate_conflict, group_starts)
        duplicate_conflict = np.repeat(group_conflict, np.diff(np.r_[group_starts, len(times)]))
        selected = indices[available]
        replay[available] = positions[selected]
        age[available] = grid[available] - times[selected]
        changed_age[available] = grid[available] - times[latest_changed[selected]]
        conflicts[available] = duplicate_conflict[selected]
    return {"position_m": replay, "available": available,
            "time_since_logged_report_s": age,
            "time_since_changed_position_s": changed_age,
            "same_timestamp_position_conflict": conflicts,
            "source_row_index": indices}
