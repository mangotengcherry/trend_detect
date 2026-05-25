#!/usr/bin/env python3
"""Generate synthetic SPC trend data and a faceted scatter chart.

The project intentionally avoids third-party dependencies so the data can be
regenerated in a clean Python environment.
"""

from __future__ import annotations

import csv
import html
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median, pstdev


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"

SEED = 20260525
POINTS_PER_SERIES = 30
SERIES_PER_TYPE = 40

TREND_TYPES = [
    "normal",
    "single_spike_ignore",
    "critical_spike",
    "persistent_high",
    "persistent_low",
    "gradual_up_drift",
    "gradual_down_drift",
    "level_shift",
    "variance_increase",
    "measurement_error",
]

PROCESS_STEPS = [
    "PHOTO",
    "ETCH",
    "CMP",
    "CVD",
    "PVD",
    "DIFFUSION",
    "ION_IMPLANT",
    "METROLOGY",
]

SENSOR_NAMES = [
    "film_thickness_nm",
    "critical_dimension_nm",
    "overlay_error_nm",
    "etch_rate_nm_min",
    "chamber_pressure_mTorr",
    "temperature_c",
    "rf_power_w",
    "particle_count",
]

TYPE_COLORS = {
    "normal": "#2f7d32",
    "single_spike_ignore": "#1f9d8a",
    "critical_spike": "#d73027",
    "persistent_high": "#f46d43",
    "persistent_low": "#4575b4",
    "gradual_up_drift": "#984ea3",
    "gradual_down_drift": "#4daf4a",
    "level_shift": "#ff9f1c",
    "variance_increase": "#7b6888",
    "measurement_error": "#111111",
}


def base_series(rng: random.Random, target: float, sigma: float) -> list[float]:
    return [rng.gauss(target, sigma) for _ in range(POINTS_PER_SERIES)]


def inject_pattern(
    rng: random.Random,
    values: list[float],
    anomaly_type: str,
    target: float,
    sigma: float,
    anomaly_start: int,
) -> tuple[list[bool], list[str], str]:
    labels = ["baseline"] * len(values)
    anomalous = [False] * len(values)
    detail = "none"

    if anomaly_type == "normal":
        return anomalous, labels, detail

    if anomaly_type == "single_spike_ignore":
        idx = rng.randrange(anomaly_start, len(values))
        direction = rng.choice([-1, 1])
        values[idx] += direction * rng.uniform(3.1, 3.8) * sigma
        anomalous[idx] = True
        labels[idx] = "isolated_spike"
        detail = "one isolated control-limit breach; recovery follows"

    elif anomaly_type == "critical_spike":
        count = rng.choice([2, 3])
        for idx in rng.sample(range(anomaly_start, len(values)), count):
            direction = rng.choice([-1, 1])
            values[idx] += direction * rng.uniform(5.2, 7.0) * sigma
            anomalous[idx] = True
            labels[idx] = "critical_spike"
        detail = f"{count} large excursions beyond critical threshold"

    elif anomaly_type == "persistent_high":
        for idx in range(anomaly_start, len(values)):
            values[idx] = target + rng.uniform(3.1, 4.2) * sigma + rng.gauss(0, 0.25 * sigma)
            anomalous[idx] = True
            labels[idx] = "persistent_high"
        detail = "sustained high-side control-limit breach"

    elif anomaly_type == "persistent_low":
        for idx in range(anomaly_start, len(values)):
            values[idx] = target - rng.uniform(3.1, 4.2) * sigma + rng.gauss(0, 0.25 * sigma)
            anomalous[idx] = True
            labels[idx] = "persistent_low"
        detail = "sustained low-side control-limit breach"

    elif anomaly_type == "gradual_up_drift":
        slope = rng.uniform(0.16, 0.27) * sigma
        for idx in range(anomaly_start, len(values)):
            values[idx] += slope * (idx - anomaly_start + 1)
            anomalous[idx] = idx >= anomaly_start + 4
            labels[idx] = "up_drift"
        detail = f"monotonic upward drift; slope={slope:.3f}"

    elif anomaly_type == "gradual_down_drift":
        slope = rng.uniform(0.16, 0.27) * sigma
        for idx in range(anomaly_start, len(values)):
            values[idx] -= slope * (idx - anomaly_start + 1)
            anomalous[idx] = idx >= anomaly_start + 4
            labels[idx] = "down_drift"
        detail = f"monotonic downward drift; slope={slope:.3f}"

    elif anomaly_type == "level_shift":
        direction = rng.choice([-1, 1])
        shift = direction * rng.uniform(2.2, 3.2) * sigma
        for idx in range(anomaly_start, len(values)):
            values[idx] += shift
            anomalous[idx] = True
            labels[idx] = "level_shift_up" if direction > 0 else "level_shift_down"
        detail = f"abrupt baseline shift; shift={shift:.3f}"

    elif anomaly_type == "variance_increase":
        for idx in range(anomaly_start, len(values)):
            values[idx] = target + rng.gauss(0, rng.uniform(2.4, 3.4) * sigma)
            anomalous[idx] = True
            labels[idx] = "high_variance"
        detail = "same mean with wider post-start variance"

    elif anomaly_type == "measurement_error":
        error_count = rng.choice([2, 3, 4])
        error_indices = rng.sample(range(anomaly_start, len(values)), error_count)
        for pos, idx in enumerate(error_indices):
            if pos % 3 == 0:
                values[idx] = 0.0
                labels[idx] = "sensor_dropout"
            elif pos % 3 == 1:
                values[idx] = target + rng.uniform(8.0, 10.5) * sigma
                labels[idx] = "out_of_range_high"
            else:
                values[idx] = target - rng.uniform(8.0, 10.5) * sigma
                labels[idx] = "out_of_range_low"
            anomalous[idx] = True
        detail = f"{error_count} invalid raw readings"

    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    return anomalous, labels, detail


def make_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(SEED)
    rows: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    start_time = datetime(2026, 1, 5, 8, 0, 0)
    global_series_idx = 0

    for trend_type in TREND_TYPES:
        for local_idx in range(1, SERIES_PER_TYPE + 1):
            global_series_idx += 1
            series_id = f"{trend_type}_{local_idx:03d}"
            target = round(100.0 + rng.uniform(-1.5, 1.5), 3)
            sigma = round(rng.uniform(0.75, 1.35), 3)
            anomaly_start = rng.randint(12, 17)
            equipment_id = f"EQP-{rng.randint(1, 12):02d}"
            process_step = rng.choice(PROCESS_STEPS)
            sensor_name = rng.choice(SENSOR_NAMES)
            lot_id = f"LOT-2026-{global_series_idx:04d}"
            wafer_id = f"W{rng.randint(1, 25):02d}"

            values = base_series(rng, target, sigma)
            anomalous, point_labels, detail = inject_pattern(
                rng, values, trend_type, target, sigma, anomaly_start
            )

            ucl = target + 3.0 * sigma
            lcl = target - 3.0 * sigma
            critical_high = target + 5.0 * sigma
            critical_low = target - 5.0 * sigma
            series_start_time = start_time + timedelta(hours=global_series_idx)

            metadata.append(
                {
                    "series_id": series_id,
                    "trend_type": trend_type,
                    "lot_id": lot_id,
                    "wafer_id": wafer_id,
                    "equipment_id": equipment_id,
                    "process_step": process_step,
                    "sensor_name": sensor_name,
                    "target": f"{target:.3f}",
                    "sigma": f"{sigma:.3f}",
                    "anomaly_start": anomaly_start,
                    "detail": detail,
                }
            )

            for sample_index, value in enumerate(values):
                point_label = point_labels[sample_index]
                is_valid = not point_label.startswith(("sensor_dropout", "out_of_range"))
                rows.append(
                    {
                        "series_id": series_id,
                        "trend_type": trend_type,
                        "sample_index": sample_index,
                        "sampled_at": (series_start_time + timedelta(minutes=sample_index * 10)).isoformat(),
                        "lot_id": lot_id,
                        "wafer_id": wafer_id,
                        "equipment_id": equipment_id,
                        "process_step": process_step,
                        "sensor_name": sensor_name,
                        "target": f"{target:.3f}",
                        "sigma": f"{sigma:.3f}",
                        "lcl": f"{lcl:.3f}",
                        "ucl": f"{ucl:.3f}",
                        "critical_low": f"{critical_low:.3f}",
                        "critical_high": f"{critical_high:.3f}",
                        "value": f"{value:.3f}",
                        "point_label": point_label,
                        "is_anomalous_point": str(anomalous[sample_index]).lower(),
                        "is_valid_reading": str(is_valid).lower(),
                    }
                )

    return rows, metadata


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trend_type"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for trend_type in TREND_TYPES:
        group = grouped[trend_type]
        values = [float(row["value"]) for row in group]
        anomalous_count = sum(row["is_anomalous_point"] == "true" for row in group)
        invalid_count = sum(row["is_valid_reading"] == "false" for row in group)
        outside_control = sum(
            float(row["value"]) < float(row["lcl"]) or float(row["value"]) > float(row["ucl"])
            for row in group
        )
        summary_rows.append(
            {
                "trend_type": trend_type,
                "series_count": SERIES_PER_TYPE,
                "point_count": len(group),
                "mean_value": f"{mean(values):.3f}",
                "std_value": f"{pstdev(values):.3f}",
                "min_value": f"{min(values):.3f}",
                "max_value": f"{max(values):.3f}",
                "anomalous_point_count": anomalous_count,
                "invalid_reading_count": invalid_count,
                "outside_control_count": outside_control,
            }
        )

    write_csv(path, summary_rows)
    return summary_rows


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width:.1f}" />'
    )


def svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 12,
    color: str = "#222222",
    anchor: str = "start",
    weight: str = "400",
) -> str:
    escaped = html.escape(text)
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" '
        'font-family="Arial, Helvetica, sans-serif">'
        f"{escaped}</text>"
    )


def render_scatter_svg(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    width = 1060
    panel_w = 460
    panel_h = 190
    left_margin = 70
    top_margin = 105
    gap_x = 70
    gap_y = 68
    plot_pad_l = 44
    plot_pad_r = 16
    plot_pad_t = 22
    plot_pad_b = 34
    cols = 2
    rows_count = math.ceil(len(TREND_TYPES) / cols)
    height = top_margin + rows_count * panel_h + (rows_count - 1) * gap_y + 55

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trend_type"])].append(row)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        svg_text(40, 42, "Synthetic Semiconductor SPC Trend Scatter Chart", 24, "#1f2933", "start", "700"),
        svg_text(
            40,
            70,
            "Each panel contains 40 synthetic sensor series x 30 samples. Dashed lines mark target and 3-sigma control bands.",
            13,
            "#52606d",
        ),
    ]

    for idx, trend_type in enumerate(TREND_TYPES):
        col = idx % cols
        row_idx = idx // cols
        panel_x = left_margin + col * (panel_w + gap_x)
        panel_y = top_margin + row_idx * (panel_h + gap_y)
        plot_x = panel_x + plot_pad_l
        plot_y = panel_y + plot_pad_t
        plot_w = panel_w - plot_pad_l - plot_pad_r
        plot_h = panel_h - plot_pad_t - plot_pad_b
        group = grouped[trend_type]
        values = [float(item["value"]) for item in group]
        targets = [float(item["target"]) for item in group]
        sigmas = [float(item["sigma"]) for item in group]
        y_min = min(values + [mean(targets) - 3 * mean(sigmas)])
        y_max = max(values + [mean(targets) + 3 * mean(sigmas)])
        pad = max((y_max - y_min) * 0.12, 1.0)
        y_min -= pad
        y_max += pad

        def sx(sample_index: float) -> float:
            return plot_x + (sample_index / (POINTS_PER_SERIES - 1)) * plot_w

        def sy(value: float) -> float:
            return plot_y + (1 - ((value - y_min) / (y_max - y_min))) * plot_h

        svg.append(
            f'<rect x="{panel_x:.1f}" y="{panel_y:.1f}" width="{panel_w:.1f}" height="{panel_h:.1f}" '
            'rx="6" fill="#fbfcfd" stroke="#d9e2ec" />'
        )
        svg.append(svg_text(panel_x + 12, panel_y + 18, trend_type, 14, "#1f2933", "start", "700"))

        target_line = median(targets)
        sigma_line = median(sigmas)
        for value, color, dash in [
            (target_line + 3 * sigma_line, "#b91c1c", "5 4"),
            (target_line, "#6b7280", "2 4"),
            (target_line - 3 * sigma_line, "#b91c1c", "5 4"),
        ]:
            y = sy(value)
            svg.append(
                f'<line x1="{plot_x:.1f}" y1="{y:.1f}" x2="{plot_x + plot_w:.1f}" y2="{y:.1f}" '
                f'stroke="{color}" stroke-width="1" stroke-dasharray="{dash}" />'
            )

        svg.append(svg_line(plot_x, plot_y, plot_x, plot_y + plot_h, "#9fb3c8"))
        svg.append(svg_line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, "#9fb3c8"))

        for tick in [0, 10, 20, 29]:
            x = sx(tick)
            svg.append(svg_line(x, plot_y + plot_h, x, plot_y + plot_h + 4, "#9fb3c8"))
            svg.append(svg_text(x, plot_y + plot_h + 19, str(tick), 10, "#627d98", "middle"))

        for value in [y_min, (y_min + y_max) / 2, y_max]:
            y = sy(value)
            svg.append(svg_line(plot_x - 4, y, plot_x, y, "#9fb3c8"))
            svg.append(svg_text(plot_x - 8, y + 3, f"{value:.1f}", 10, "#627d98", "end"))

        svg.append(svg_text(plot_x + plot_w / 2, panel_y + panel_h - 5, "sample index", 10, "#627d98", "middle"))

        color = TYPE_COLORS[trend_type]
        for item in group:
            sample_index = int(item["sample_index"])
            value = float(item["value"])
            series_number = int(str(item["series_id"]).split("_")[-1])
            jitter = ((series_number % 9) - 4) * 0.035
            cx = sx(max(0, min(POINTS_PER_SERIES - 1, sample_index + jitter)))
            cy = sy(value)
            is_anomalous = item["is_anomalous_point"] == "true"
            is_valid = item["is_valid_reading"] == "true"
            radius = 2.6 if is_anomalous else 1.9
            opacity = 0.66 if is_anomalous else 0.35
            stroke = "#ffffff" if is_valid else "#d73027"
            stroke_width = 0.45 if is_valid else 1.3
            svg.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.1f}" '
                f'fill="{color}" fill-opacity="{opacity:.2f}" stroke="{stroke}" '
                f'stroke-width="{stroke_width:.2f}" />'
            )

    svg.append(svg_text(40, height - 24, "Generated from data/synthetic_spc_measurements.csv", 11, "#627d98"))
    svg.append("</svg>")

    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    records, metadata = make_records()
    write_csv(DATA_DIR / "synthetic_spc_measurements.csv", records)
    write_csv(DATA_DIR / "synthetic_series_metadata.csv", metadata)
    write_summary(DATA_DIR / "anomaly_type_summary.csv", records)
    render_scatter_svg(ASSETS_DIR / "spc_trend_scatter.svg", records)

    print(f"Wrote {len(records):,} measurement rows")
    print(f"Wrote {len(metadata):,} series metadata rows")
    print(f"Wrote {ASSETS_DIR / 'spc_trend_scatter.svg'}")


if __name__ == "__main__":
    main()
