#!/usr/bin/env python3
"""Compute PICMUS contrast and resolution metrics for UGNS output images.

The script contains only metric definitions and normalized PICMUS ROI settings;
it intentionally contains no published result values. Results are written to a
user-selected output directory as CSV and JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics_core import cnr_db, contrast, fwhm_6db, gcnr, snr


def _rect(z1, z2, x1, x2):
    return {"type": "rect", "bounds": [z1, z2, x1, x2]}


def _circle(z, x, radius, padding=1.0):
    return {
        "type": "circle",
        "center": [z, x],
        "radius": radius,
        "padding": padding,
    }


def _annular(z, x, inner, outer):
    return {
        "type": "annular",
        "center": [z, x],
        "inner_radius": inner,
        "outer_radius": outer,
    }


SC_CENTERS = [
    (z, x)
    for z in (0.22, 0.56, 0.89)
    for x in (0.17209375, 0.50, 0.82790625)
]

SR_TARGETS = [
    _rect(0.095, 0.195, 0.446, 0.546),
    *[_rect(0.231, 0.331, x, x + 0.1) for x in (0.032, 0.173, 0.309, 0.448, 0.587, 0.723, 0.864)],
    _rect(0.372, 0.472, 0.448, 0.548),
    _rect(0.509, 0.609, 0.448, 0.548),
    _rect(0.649, 0.749, 0.448, 0.548),
    *[_rect(0.786, 0.886, x, x + 0.1) for x in (0.036, 0.173, 0.309, 0.448, 0.587, 0.723, 0.860)],
]

PICMUS_ROIS = {
    "SC": {
        "in": [_circle(z, x, 0.1) for z, x in SC_CENTERS],
        "out": [_annular(z, x, 0.11, 0.14) for z, x in SC_CENTERS],
        "speckle": [
            _rect(0.36046875, 0.44046875, 0.30, 0.38),
            _rect(0.36046875, 0.44046875, 0.62, 0.70),
            _rect(0.69, 0.77, 0.30, 0.38),
            _rect(0.69, 0.77, 0.62, 0.70),
        ],
    },
    "EC": {
        "in": [_circle(0.142, 0.495, 0.045), _circle(0.907, 0.496, 0.05)],
        "out": [
            _annular(0.142, 0.495, 0.055, 0.09),
            _annular(0.907, 0.496, 0.06, 0.1),
        ],
        "speckle": [
            _rect(0.65, 0.80, 0.355, 0.645),
            _rect(0.24, 0.39, 0.15, 0.44),
            _rect(0.24, 0.39, 0.56, 0.85),
        ],
    },
    "ER": {
        "in": [_circle(0.5, 0.21, 0.11)],
        "out": [_annular(0.5, 0.21, 0.125, 0.16)],
        "targets": [
            _rect(0.188, 0.288, 0.436, 0.536),
            _rect(0.448, 0.548, 0.435, 0.535),
            _rect(0.713, 0.813, 0.157, 0.257),
            _rect(0.710, 0.810, 0.441, 0.541),
            _rect(0.712, 0.812, 0.718, 0.818),
        ],
    },
    "SR": {"targets": SR_TARGETS},
}

METRIC_FIELDS = (
    "gCNR",
    "CNR_dB",
    "SNR",
    "Contrast_dB",
    "FWHM_Axial_mm",
    "FWHM_Lateral_mm",
)


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def roi_mask(shape, specification) -> np.ndarray:
    height, width = shape
    kind = specification["type"]
    if kind == "rect":
        z1, z2, x1, x2 = specification["bounds"]
        mask = np.zeros(shape, dtype=bool)
        mask[int(height * z1):int(height * z2), int(width * x1):int(width * x2)] = True
        return mask

    center_z, center_x = specification["center"]
    center_z = int(height * center_z)
    center_x = int(width * center_x)
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((yy - center_z) ** 2 + (xx - center_x) ** 2)
    scale = min(height, width)
    if kind == "circle":
        radius = max(1, int(scale * specification["radius"]) - int(specification.get("padding", 0)))
        return distance <= radius
    if kind == "annular":
        inner = int(scale * specification["inner_radius"])
        outer = int(scale * specification["outer_radius"])
        return (distance >= inner) & (distance <= outer)
    raise ValueError(f"Unknown ROI type: {kind}")


def combined_mask(shape, specifications) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for specification in specifications:
        mask |= roi_mask(shape, specification)
    return mask


def dataset_code(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    match = re.search(r"(?:^|[_-])(EC|ER|SC|SR)(?:[_-]|$)", path.stem.upper())
    if not match:
        raise ValueError("cannot infer EC/ER/SC/SR from the filename; pass --dataset")
    return match.group(1)


def resolution_metrics(image, config, code, pixel_size_mm, window_radius):
    target_mask = combined_mask(image.shape, config["targets"])
    target_only = np.where(target_mask, image, -np.inf)
    peak_z, peak_x = np.unravel_index(np.argmax(target_only), image.shape)
    z1, z2 = max(0, peak_z - window_radius), min(image.shape[0], peak_z + window_radius + 1)
    x1, x2 = max(0, peak_x - window_radius), min(image.shape[1], peak_x + window_radius + 1)
    log_compressed = code == "ER"
    axial = fwhm_6db(image[z1:z2, peak_x], log_compressed=log_compressed)
    lateral = fwhm_6db(image[peak_z, x1:x2], log_compressed=log_compressed)
    return axial * pixel_size_mm, lateral * pixel_size_mm


def calculate_image_metrics(image, code, config, pixel_size_mm, window_radius):
    result = {field: float("nan") for field in METRIC_FIELDS}
    if "in" in config and "out" in config:
        lesion = image[combined_mask(image.shape, config["in"])]
        background = image[combined_mask(image.shape, config["out"])]
        result["gCNR"] = gcnr(lesion, background)
        result["CNR_dB"] = cnr_db(lesion, background)
        result["Contrast_dB"] = contrast(background, lesion)
        if "speckle" in config:
            result["SNR"] = snr(image[combined_mask(image.shape, config["speckle"])])
    if "targets" in config:
        axial, lateral = resolution_metrics(
            image, config, code, pixel_size_mm, window_radius
        )
        result["FWHM_Axial_mm"] = axial
        result["FWHM_Lateral_mm"] = lateral
    return result


def load_roi_settings(path):
    settings = {key: value for key, value in PICMUS_ROIS.items()}
    if not path:
        return settings
    with Path(path).open("r", encoding="utf-8") as handle:
        overrides = yaml.safe_load(handle) or {}
    if not isinstance(overrides, dict):
        raise ValueError("ROI config must contain a YAML mapping")
    settings.update({str(key).upper(): value for key, value in overrides.items()})
    return settings


def image_paths(input_path, pattern, recursive):
    if input_path.is_file():
        return [input_path]
    iterator = input_path.rglob(pattern) if recursive else input_path.glob(pattern)
    return sorted(path for path in iterator if path.is_file())


def json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_results(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ("Image", "Dataset", *METRIC_FIELDS)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    safe_rows = [{key: json_safe(value) for key, value in row.items()} for row in rows]
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(safe_rows, handle, indent=2, allow_nan=False)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Denoised image or directory")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pattern", default="*_denoised.png")
    parser.add_argument("--dataset", choices=("auto", "EC", "ER", "SC", "SR"), default="auto")
    parser.add_argument("--pixel-size-mm", type=float, default=0.1)
    parser.add_argument("--window-radius", type=int, default=30)
    parser.add_argument("--roi-config", type=Path, default=None,
                        help="Optional YAML mapping that overrides built-in PICMUS ROIs")
    parser.add_argument("--no-recursive", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input does not exist: {args.input}")
    files = image_paths(args.input, args.pattern, not args.no_recursive)
    if not files:
        raise SystemExit(f"No images matched {args.pattern!r} under {args.input}")
    settings = load_roi_settings(args.roi_config)
    rows = []
    for path in files:
        try:
            code = dataset_code(path, args.dataset)
            metrics = calculate_image_metrics(
                load_image(path), code, settings[code], args.pixel_size_mm, args.window_radius
            )
        except Exception as error:
            raise RuntimeError(f"Failed to evaluate {path}: {error}") from error
        row = {"Image": str(path), "Dataset": code, **metrics}
        rows.append(row)
        available = [f"{key}={value:.4g}" for key, value in metrics.items() if np.isfinite(value)]
        print(f"{path.name}: " + ", ".join(available))
    output_dir = args.output_dir or ((args.input if args.input.is_dir() else args.input.parent) / "metrics")
    write_results(rows, output_dir)
    print(f"Wrote metrics to {output_dir}")


if __name__ == "__main__":
    main()
