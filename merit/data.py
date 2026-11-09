"""Leakage-controlled loader for the official TrajAir processed trajectories.

TrajAir processed files contain:
    frame_id, aircraft_id, x(km), y(km), z(km), wind_x(m/s), wind_y(m/s)

The official release interpolates observations at one-second intervals. The
loader preserves the official train/test scene split and creates validation
scenes only from the official training directory.
"""

from __future__ import annotations

import hashlib
import pickle
import warnings
from pathlib import Path

import numpy as np


SAMPLE_INTERVAL_SECONDS = 1.0
POSITION_SCALE_METERS = 1000.0
CACHE_VERSION = "trajair_official_split_v3"
MIN_TRAJECTORY_STEPS = 90


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_contiguous_rows(rows: np.ndarray) -> list[np.ndarray]:
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    _, unique_indices = np.unique(rows[:, 0], return_index=True)
    rows = rows[np.sort(unique_indices)]
    if len(rows) == 0:
        return []
    breaks = np.where(np.diff(rows[:, 0]) > 1.0)[0] + 1
    return [part for part in np.split(rows, breaks) if len(part)]


def _scene_trajectories(path: Path) -> list[np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scene = np.loadtxt(path, delimiter=" ")
    if scene.size == 0:
        return []
    scene = np.atleast_2d(scene)
    if scene.shape[1] < 5:
        return []

    trajectories: list[np.ndarray] = []
    for aircraft_id in np.unique(scene[:, 1].astype(np.int64)):
        aircraft_rows = scene[scene[:, 1].astype(np.int64) == aircraft_id]
        for segment in _split_contiguous_rows(aircraft_rows):
            if len(segment) < MIN_TRAJECTORY_STEPS:
                continue

            x = segment[:, 2] * POSITION_SCALE_METERS
            y = segment[:, 3] * POSITION_SCALE_METERS
            z = segment[:, 4] * POSITION_SCALE_METERS
            vx = np.gradient(x, SAMPLE_INTERVAL_SECONDS)
            vy = np.gradient(y, SAMPLE_INTERVAL_SECONDS)
            ax = np.gradient(vx, SAMPLE_INTERVAL_SECONDS)
            ay = np.gradient(vy, SAMPLE_INTERVAL_SECONDS)

            trajectories.append(
                np.column_stack([x, y, vx, vy, ax, ay]).astype(np.float32)
            )
    return trajectories


def _resolve_processed_dir(data_path: Path) -> Path:
    candidates = [
        data_path / "processed_data",
        data_path / "111_days" / "processed_data",
        data_path.parent / "trajair" / "processed_data",
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise RuntimeError(
        "TrajAir processed data not found. Expected official train/test folders "
        f"under {data_path}."
    )


def _cache_path(data_path: Path) -> Path:
    if data_path.suffix == ".pkl":
        return data_path
    return data_path / "trajair_trajectories_v3.pkl"


def _load_cache(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("cache_version") != CACHE_VERSION:
        raise ValueError(f"Stale TrajAir cache: {path}")
    return payload["trajectories"], payload["splits"], payload["metadata"]


def load_trajair_data(data_path, cache_path=None):
    """Load TrajAir with official test scenes and scene-disjoint validation."""
    data_path = Path(data_path)
    cache_path = Path(cache_path) if cache_path is not None else _cache_path(data_path)
    if cache_path.exists():
        trajectories, splits, metadata = _load_cache(cache_path)
        print(
            "[Data] TrajAir cache: "
            f"{len(trajectories)} trajectories; "
            f"train={len(splits['train'])}, val={len(splits['val'])}, "
            f"test={len(splits['test'])}"
        )
        return trajectories, splits

    processed_dir = _resolve_processed_dir(
        data_path if not data_path.suffix else data_path.parent
    )
    official_train_files = sorted((processed_dir / "train").glob("*.txt"))
    official_test_files = sorted((processed_dir / "test").glob("*.txt"))
    if not official_train_files or not official_test_files:
        raise RuntimeError(f"Incomplete official TrajAir split under {processed_dir}")

    rng = np.random.RandomState(42)
    shuffled_train = np.asarray(official_train_files, dtype=object)[
        rng.permutation(len(official_train_files))
    ]
    n_val_scenes = max(1, int(round(0.10 * len(shuffled_train))))
    val_scene_names = {path.name for path in shuffled_train[:n_val_scenes]}

    trajectories: list[np.ndarray] = []
    splits = {"train": [], "val": [], "test": []}
    split_scene_counts = {"train": 0, "val": 0, "test": len(official_test_files)}

    for path in official_train_files:
        split = "val" if path.name in val_scene_names else "train"
        split_scene_counts[split] += 1
        for trajectory in _scene_trajectories(path):
            splits[split].append(len(trajectories))
            trajectories.append(trajectory)

    for path in official_test_files:
        for trajectory in _scene_trajectories(path):
            splits["test"].append(len(trajectories))
            trajectories.append(trajectory)

    metadata = {
        "cache_version": CACHE_VERSION,
        "source": "TrajAir 111_days processed_data",
        "source_doi": "10.1184/R1/14866251.v1",
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "position_unit": "m",
        "minimum_trajectory_steps": MIN_TRAJECTORY_STEPS,
        "official_scene_files": {
            "train": len(official_train_files),
            "test": len(official_test_files),
        },
        "split_scene_counts": split_scene_counts,
        "split_trajectory_counts": {
            split: len(indices) for split, indices in splits.items()
        },
        "processed_file_manifest_sha256": hashlib.sha256(
            "\n".join(
                f"{path.relative_to(processed_dir).as_posix()}:{_file_sha256(path)}"
                for path in official_train_files + official_test_files
            ).encode("utf-8")
        ).hexdigest(),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "cache_version": CACHE_VERSION,
                "trajectories": trajectories,
                "splits": splits,
                "metadata": metadata,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        "[Data] TrajAir built: "
        f"{len(trajectories)} trajectories; "
        f"train={len(splits['train'])}, val={len(splits['val'])}, "
        f"test={len(splits['test'])}"
    )
    return trajectories, splits


from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset


class AircraftDataset(Dataset):
    """Sliding windows with input [seq_len, 6] and target [label_len+pred_len, 6]."""

    def __init__(self, trajectories, types_or_splits, args, flag="train", size=None):
        self.seq_len, self.label_len, self.pred_len = size or (30, 30, 30)
        if flag not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {flag}")
        self.flag = flag
        self.window_stride = int(getattr(args, "window_stride", max(1, self.seq_len // 2)))
        self.max_windows = int(getattr(args, f"max_{flag}_windows", 0))

        if isinstance(types_or_splits, dict):
            indices = types_or_splits[flag]
        else:
            shuffled = np.random.RandomState(42).permutation(len(trajectories))
            train_end = int(len(trajectories) * 0.75)
            val_end = train_end + int(len(trajectories) * 0.10)
            indices = {
                "train": shuffled[:train_end],
                "val": shuffled[train_end:val_end],
                "test": shuffled[val_end:],
            }[flag]

        if isinstance(trajectories, np.ndarray):
            self.trajectories = trajectories[indices]
        else:
            self.trajectories = [trajectories[int(index)] for index in indices]
        self.num_trajectories = len(self.trajectories)
        if flag == "train":
            self.scaler = StandardScaler().fit(self._valid_points(self.trajectories))
        self._build_windows()

    @staticmethod
    def _valid_length(trajectory):
        valid_rows = np.any(np.abs(trajectory) > 1e-8, axis=1)
        return int(np.where(valid_rows)[0][-1] + 1) if np.any(valid_rows) else 0

    @classmethod
    def _valid_points(cls, trajectories):
        points = [trajectory[: cls._valid_length(trajectory)] for trajectory in trajectories]
        points = [point for point in points if len(point)]
        if points:
            return np.concatenate(points, axis=0)
        array = np.asarray(trajectories)
        return array.reshape(-1, array.shape[-1])

    def set_scaler(self, scaler):
        if scaler is None:
            raise RuntimeError("Training scaler must be created before validation or testing.")
        self.scaler = scaler
        self._build_windows()

    def _build_windows(self):
        window_size = self.seq_len + self.pred_len
        windows = []
        for trajectory in self.trajectories:
            valid = trajectory[: self._valid_length(trajectory)]
            windows.extend(
                valid[start : start + window_size]
                for start in range(0, len(valid) - window_size + 1, self.window_stride)
            )
        self.windows = np.asarray(windows, dtype=np.float32)
        if self.max_windows > 0 and len(self.windows) > self.max_windows:
            split_seed = {"train": 43, "val": 44, "test": 45}[self.flag]
            selected = np.random.RandomState(split_seed).choice(
                len(self.windows), size=self.max_windows, replace=False
            )
            self.windows = self.windows[np.sort(selected)]
        if len(self.windows) and getattr(self, "scaler", None) is not None:
            batch, steps, channels = self.windows.shape
            flattened = self.windows.reshape(-1, channels)
            self.windows = self.scaler.transform(flattened).reshape(batch, steps, channels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        seq_x = window[: self.seq_len]
        seq_y = window[self.seq_len - self.label_len :]
        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
            torch.zeros(self.seq_len, 4),
            torch.zeros(self.label_len + self.pred_len, 4),
        )
