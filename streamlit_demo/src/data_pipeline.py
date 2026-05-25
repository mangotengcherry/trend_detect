"""Data loading, stitching, windowing, feature engineering, rule layer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

def _find_data_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data" / "synthetic_spc_measurements.csv"
        if candidate.exists():
            return candidate.parent
    raise FileNotFoundError("could not locate data/synthetic_spc_measurements.csv")


DATA_DIR = _find_data_dir()
ROOT = DATA_DIR.parent
MEASUREMENTS_CSV = DATA_DIR / "synthetic_spc_measurements.csv"
METADATA_CSV = DATA_DIR / "synthetic_series_metadata.csv"

WINDOW = 30
LONG_SERIES_LEN = 120
STITCH_INTERVAL_HOURS = 6
STITCH_BASE = pd.Timestamp("2026-04-26 00:00:00")

CLASSES = (
    "normal",
    "single_spike_ignore",
    "critical_spike",
    "persistent_high",
    "persistent_low",
    "gradual_up_drift",
    "gradual_down_drift",
    "level_shift",
    "variance_increase",
)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

CHANNEL_NAMES = (
    "value_norm",
    "delta",
    "rolling_slope",
    "ewma",
    "upper_score",
    "lower_score",
    "position",
    "recency",
    "cusum_pos",
    "cusum_neg",
    "we_rule_flag",
    "range_ratio",
)
NUM_CHANNELS = len(CHANNEL_NAMES)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    measurements = pd.read_csv(
        MEASUREMENTS_CSV,
        dtype={
            "series_id": "string",
            "trend_type": "string",
            "sample_index": "int32",
            "lot_id": "string",
            "wafer_id": "string",
            "equipment_id": "string",
            "process_step": "string",
            "sensor_name": "string",
            "point_label": "string",
        },
        parse_dates=["sampled_at"],
    )
    measurements["is_anomalous_point"] = measurements["is_anomalous_point"].astype(str).str.lower() == "true"
    measurements["is_valid_reading"] = measurements["is_valid_reading"].astype(str).str.lower() == "true"
    metadata = pd.read_csv(METADATA_CSV, dtype={"equipment_id": "string", "sensor_name": "string"})
    return measurements, metadata


def list_long_series_keys(metadata: pd.DataFrame | None = None, min_series: int = 1) -> list[tuple[str, str]]:
    """Return sorted (sensor_name, equipment_id) combos that have ≥min_series series."""
    if metadata is None:
        _, metadata = load_raw()
    counts = metadata.groupby(["sensor_name", "equipment_id"]).size()
    return sorted([(s, e) for (s, e), n in counts.items() if n >= min_series])


def _segment_pool(
    metadata: pd.DataFrame, sensor: str, equipment: str, exclude_type: str = "measurement_error"
) -> tuple[list[str], list[str], list[str]]:
    """Return three candidate pools for stitching, in priority order."""
    md = metadata[metadata["trend_type"] != exclude_type]
    primary = md[(md["sensor_name"] == sensor) & (md["equipment_id"] == equipment)]["series_id"].tolist()
    secondary = md[(md["sensor_name"] == sensor) & (md["equipment_id"] != equipment)]["series_id"].tolist()
    tertiary = md[md["sensor_name"] == sensor]["series_id"].tolist()
    return primary, secondary, tertiary


def build_long_series(
    sensor: str,
    equipment: str,
    n_segments: int = 4,
    seed: int = 42,
    measurements: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Stitch n_segments series into a single long sequence with re-stamped timestamps.

    Fallback policy:
      1. (sensor, equipment) match
      2. sensor-only fallback for remaining slots
      3. sampling-with-replacement on sensor pool if still short
    """
    if measurements is None or metadata is None:
        measurements, metadata = load_raw()
    rng = np.random.default_rng(seed + hash((sensor, equipment)) % 100000)

    primary, secondary, tertiary = _segment_pool(metadata, sensor, equipment)
    rng.shuffle(primary)
    rng.shuffle(secondary)

    chosen: list[str] = []
    for sid in primary:
        if len(chosen) >= n_segments:
            break
        chosen.append(sid)
    for sid in secondary:
        if len(chosen) >= n_segments:
            break
        chosen.append(sid)
    while len(chosen) < n_segments:
        if not tertiary:
            raise ValueError(f"no series available for sensor {sensor}")
        chosen.append(tertiary[int(rng.integers(0, len(tertiary)))])

    segments = []
    for seg_idx, sid in enumerate(chosen):
        seg = measurements[measurements["series_id"] == sid].sort_values("sample_index").copy()
        seg["segment_index"] = seg_idx
        seg["segment_series_id"] = sid
        segments.append(seg)
    stitched = pd.concat(segments, ignore_index=True)

    n = len(stitched)
    timestamps = pd.date_range(start=STITCH_BASE, periods=n, freq=f"{STITCH_INTERVAL_HOURS}h")
    stitched["display_time"] = timestamps
    stitched["global_index"] = np.arange(n)
    return stitched


def windows(series: pd.DataFrame, w: int = WINDOW, stride: int = 1) -> Iterator[tuple[int, pd.DataFrame]]:
    n = len(series)
    for start in range(0, n - w + 1, stride):
        yield start, series.iloc[start : start + w].reset_index(drop=True)


def window_label(window_df: pd.DataFrame, mode: str = "last5_majority") -> str:
    """Derive target label for a window. Defaults to majority trend_type of last 5 points."""
    if mode == "last5_majority":
        tail = window_df.tail(5)["trend_type"]
    elif mode == "last_point":
        tail = window_df.tail(1)["trend_type"]
    else:
        tail = window_df["trend_type"]
    counts = tail.value_counts()
    for cand in counts.index.tolist():
        if cand in CLASS_TO_IDX:
            return cand
    return "normal"


def _ewma(x: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _cusum(x: np.ndarray, target: np.ndarray, sigma: np.ndarray, k: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros_like(x)
    neg = np.zeros_like(x)
    for i in range(len(x)):
        z = (x[i] - target[i]) / max(sigma[i], 1e-6)
        prev_p = pos[i - 1] if i > 0 else 0.0
        prev_n = neg[i - 1] if i > 0 else 0.0
        pos[i] = max(0.0, prev_p + z - k)
        neg[i] = min(0.0, prev_n + z + k)
    return pos, neg


def _we_rule(value: np.ndarray, target: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Western Electric Rule 2: 2 of 3 consecutive points beyond 2σ on same side."""
    z = (value - target) / np.maximum(sigma, 1e-6)
    flag = np.zeros_like(z, dtype=float)
    for i in range(2, len(z)):
        window = z[i - 2 : i + 1]
        if np.sum(window > 2) >= 2 or np.sum(window < -2) >= 2:
            flag[i] = 1.0
    return flag


def feature_channels(window_df: pd.DataFrame) -> np.ndarray:
    """Compute the 12-channel feature representation. Returns ndarray of shape [T, C]."""
    value = window_df["value"].to_numpy(dtype=float)
    target = window_df["target"].to_numpy(dtype=float)
    sigma = window_df["sigma"].to_numpy(dtype=float)
    ucl = window_df["ucl"].to_numpy(dtype=float)
    lcl = window_df["lcl"].to_numpy(dtype=float)
    valid = window_df["is_valid_reading"].to_numpy(dtype=bool)
    safe_value = np.where(valid, value, target)

    sigma_safe = np.maximum(sigma, 1e-6)
    value_norm = (safe_value - target) / sigma_safe
    delta = np.concatenate([[0.0], np.diff(safe_value)])

    slope = np.zeros_like(safe_value)
    k = 5
    for i in range(len(safe_value)):
        lo = max(0, i - k + 1)
        xs = np.arange(i - lo + 1, dtype=float)
        ys = safe_value[lo : i + 1]
        if len(xs) >= 2:
            slope[i] = np.polyfit(xs, ys, 1)[0]
    slope_norm = slope / sigma_safe

    ewma_v = _ewma(safe_value)
    ewma_dev = (ewma_v - target) / sigma_safe

    upper_score = np.maximum(0.0, (safe_value - ucl) / sigma_safe)
    lower_score = np.maximum(0.0, (lcl - safe_value) / sigma_safe)

    t = len(safe_value)
    position = np.linspace(-1.0, 1.0, t)
    recency = np.linspace(0.0, 1.0, t)

    cusum_pos, cusum_neg = _cusum(safe_value, target, sigma_safe)
    we_flag = _we_rule(safe_value, target, sigma_safe)

    rolling_std = np.array(
        [np.std(safe_value[max(0, i - 4) : i + 1]) if i >= 1 else 0.0 for i in range(t)]
    )
    range_ratio = rolling_std / sigma_safe

    features = np.stack(
        [
            value_norm,
            delta / sigma_safe,
            slope_norm,
            ewma_dev,
            upper_score,
            lower_score,
            position,
            recency,
            np.tanh(cusum_pos),
            np.tanh(cusum_neg),
            we_flag,
            range_ratio,
        ],
        axis=1,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
    return features.astype(np.float32)


@dataclass
class RuleTrigger:
    five_sigma: bool
    hard_spec: bool
    invalid_reading: bool
    we_rule: bool


def hard_rule_trigger(window_df: pd.DataFrame) -> RuleTrigger:
    value = window_df["value"].to_numpy(dtype=float)
    target = window_df["target"].to_numpy(dtype=float)
    sigma = window_df["sigma"].to_numpy(dtype=float)
    crit_low = window_df["critical_low"].to_numpy(dtype=float)
    crit_high = window_df["critical_high"].to_numpy(dtype=float)
    valid = window_df["is_valid_reading"].to_numpy(dtype=bool)

    five_sigma = bool(np.any(np.abs(value - target) > 5 * np.maximum(sigma, 1e-6)))
    hard_spec = bool(np.any((value > crit_high) | (value < crit_low)))
    invalid = bool(np.any(~valid))
    we_flag = bool(_we_rule(np.where(valid, value, target), target, np.maximum(sigma, 1e-6)).any())
    return RuleTrigger(
        five_sigma=five_sigma, hard_spec=hard_spec, invalid_reading=invalid, we_rule=we_flag
    )


def build_training_set(
    measurements: pd.DataFrame | None = None, metadata: pd.DataFrame | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Use each original 30-point series as one training sample (excluding measurement_error)."""
    if measurements is None or metadata is None:
        measurements, metadata = load_raw()
    md = metadata[metadata["trend_type"] != "measurement_error"]
    X: list[np.ndarray] = []
    y: list[int] = []
    sids: list[str] = []
    for sid in md["series_id"].tolist():
        seg = measurements[measurements["series_id"] == sid].sort_values("sample_index").reset_index(drop=True)
        if len(seg) != WINDOW:
            continue
        X.append(feature_channels(seg))
        ttype = str(seg.iloc[0]["trend_type"])
        y.append(CLASS_TO_IDX.get(ttype, CLASS_TO_IDX["normal"]))
        sids.append(sid)
    return np.stack(X, axis=0), np.asarray(y, dtype=np.int64), sids
