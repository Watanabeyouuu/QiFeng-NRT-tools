#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 QiFeng-NRT authors.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate reconstructed wind fields against a gridded reference.

The input NPZ contains ``gfs``, ``reconstruction`` and ``reference`` arrays in
``(N, L, 2, H, W)`` order.  For one sample, the leading dimension may be
omitted.  Components are eastward and northward wind in m s-1; level 0 is
10-m wind.  Add a ``storm_id`` string array to calculate storm-block intervals.
"""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_LEVELS = ["10m", "850hPa", "700hPa", "500hPa", "200hPa"]


def load_fields(path):
    with np.load(path, allow_pickle=False) as data:
        fields = [np.asarray(data[name]) for name in ("gfs", "reconstruction", "reference")]
        storm_id = np.asarray(data["storm_id"]).astype(str) if "storm_id" in data else None
    fields = [field[None] if field.ndim == 4 else field for field in fields]
    if any(field.ndim != 5 or field.shape[2] != 2 for field in fields):
        raise ValueError("wind arrays must have shape (N,L,2,H,W) or (L,2,H,W)")
    if fields[0].shape != fields[1].shape or fields[0].shape != fields[2].shape:
        raise ValueError("gfs, reconstruction and reference must have the same shape")
    if storm_id is not None:
        storm_id = storm_id.reshape(-1)
        if storm_id.size == 1 and fields[0].shape[0] > 1:
            storm_id = np.repeat(storm_id, fields[0].shape[0])
        if storm_id.size != fields[0].shape[0]:
            raise ValueError("storm_id must contain one value per sample")
    return fields[0], fields[1], fields[2], storm_id


def wind_speed(field):
    return np.hypot(field[:, :, 0], field[:, :, 1])


def error_stats(estimate, reference, mask=None):
    valid = np.isfinite(estimate) & np.isfinite(reference)
    if mask is not None:
        valid &= mask
    error = estimate[valid] - reference[valid]
    if error.size == 0:
        return {"n": 0, "mae": np.nan, "bias": np.nan, "rmse": np.nan}
    return {
        "n": int(error.size),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }


def common_domain_maximum(gfs, reconstruction, reference):
    common = np.isfinite(gfs) & np.isfinite(reconstruction) & np.isfinite(reference)
    maxima = []
    for field in (gfs, reconstruction, reference):
        values = np.where(common, field, np.nan).reshape(field.shape[0], -1)
        maxima.append(
            np.asarray(
                [np.nanmax(row) if np.any(np.isfinite(row)) else np.nan for row in values]
            )
        )
    return maxima


def rmw(field, spacing_km=3.0):
    """Fixed-centre RMW from the peak 3-km azimuthal-mean wind-speed bin."""

    n, height, width = field.shape
    y, x = np.indices((height, width), dtype=float)
    radius = np.hypot(y - (height - 1) / 2, x - (width - 1) / 2) * spacing_km
    n_bins = min(height, width) // 2
    bin_index = np.clip(
        np.floor(radius / spacing_km).astype(int), 0, n_bins - 1
    )
    result = np.full(n, np.nan)
    for sample in range(n):
        valid = np.isfinite(field[sample])
        count = np.bincount(bin_index[valid], minlength=n_bins)
        total = np.bincount(
            bin_index[valid], weights=field[sample][valid], minlength=n_bins
        )
        profile = np.divide(
            total,
            count,
            out=np.full(n_bins, np.nan),
            where=count > 0,
        )
        if np.any(np.isfinite(profile)):
            result[sample] = (np.nanargmax(profile) + 0.5) * spacing_km
    return result


def storm_bootstrap(gfs_error, reconstruction_error, storm_id, statistic, n_resamples, seed):
    """Bootstrap GFS-minus-reconstruction improvement by resampling whole storms."""

    common = np.isfinite(gfs_error) & np.isfinite(reconstruction_error)
    if statistic == "rmse":
        gfs_value = np.where(common, gfs_error ** 2, 0.0)
        reconstruction_value = np.where(common, reconstruction_error ** 2, 0.0)
    else:
        gfs_value = np.where(common, np.abs(gfs_error), 0.0)
        reconstruction_value = np.where(common, np.abs(reconstruction_error), 0.0)

    axes = tuple(range(1, gfs_value.ndim))
    gfs_sum = gfs_value.sum(axis=axes) if axes else gfs_value
    reconstruction_sum = reconstruction_value.sum(axis=axes) if axes else reconstruction_value
    count = common.sum(axis=axes) if axes else common.astype(int)

    storms = np.unique(storm_id)
    block_gfs = np.asarray([gfs_sum[storm_id == storm].sum() for storm in storms])
    block_reconstruction = np.asarray(
        [reconstruction_sum[storm_id == storm].sum() for storm in storms]
    )
    block_count = np.asarray([count[storm_id == storm].sum() for storm in storms])
    keep = block_count > 0
    storms = storms[keep]
    block_gfs = block_gfs[keep]
    block_reconstruction = block_reconstruction[keep]
    block_count = block_count[keep]
    if len(storms) == 0:
        raise ValueError("no common valid values for storm-block bootstrap")
    rng = np.random.default_rng(seed)
    reductions = np.empty(n_resamples)
    relative = np.empty(n_resamples)
    for i in range(n_resamples):
        draw = rng.integers(0, len(storms), len(storms))
        denominator = block_count[draw].sum()
        gfs_metric = block_gfs[draw].sum() / denominator
        reconstruction_metric = block_reconstruction[draw].sum() / denominator
        if statistic == "rmse":
            gfs_metric = np.sqrt(gfs_metric)
            reconstruction_metric = np.sqrt(reconstruction_metric)
        reductions[i] = gfs_metric - reconstruction_metric
        relative[i] = reductions[i] / gfs_metric if gfs_metric != 0 else np.nan
    relative_ci = (
        np.nanpercentile(relative, [2.5, 97.5])
        if np.any(np.isfinite(relative))
        else np.array([np.nan, np.nan])
    )
    return {
        "n_storms": int(len(storms)),
        "n_resamples": int(n_resamples),
        "gfs_minus_reconstruction_ci": np.percentile(reductions, [2.5, 97.5]),
        "relative_reduction_ci": relative_ci,
    }


def evaluate(gfs, reconstruction, reference, levels, spacing_km, storm_id, bootstrap, seed):
    gfs_speed, reconstruction_speed, reference_speed = map(
        wind_speed, (gfs, reconstruction, reference)
    )
    result = {"levels": {}}
    for level, name in enumerate(levels):
        common = (
            np.isfinite(gfs_speed[:, level])
            & np.isfinite(reconstruction_speed[:, level])
            & np.isfinite(reference_speed[:, level])
        )
        gfs_max, reconstruction_max, reference_max = common_domain_maximum(
            gfs_speed[:, level], reconstruction_speed[:, level], reference_speed[:, level]
        )
        result["levels"][name] = {
            "wind_speed": {
                "gfs": error_stats(gfs_speed[:, level], reference_speed[:, level], common),
                "reconstruction": error_stats(
                    reconstruction_speed[:, level], reference_speed[:, level], common
                ),
            },
            "domain_maximum_wind_speed": {
                "gfs": error_stats(gfs_max, reference_max),
                "reconstruction": error_stats(reconstruction_max, reference_max),
            },
        }

    common_10m = (
        np.isfinite(gfs_speed[:, 0])
        & np.isfinite(reconstruction_speed[:, 0])
        & np.isfinite(reference_speed[:, 0])
    )
    gfs_10m = np.where(common_10m, gfs_speed[:, 0], np.nan)
    reconstruction_10m = np.where(common_10m, reconstruction_speed[:, 0], np.nan)
    reference_10m = np.where(common_10m, reference_speed[:, 0], np.nan)
    gfs_rmw = rmw(gfs_10m, spacing_km)
    reconstruction_rmw = rmw(reconstruction_10m, spacing_km)
    reference_rmw = rmw(reference_10m, spacing_km)
    result["surface_rmw_km"] = {
        "gfs": error_stats(gfs_rmw, reference_rmw),
        "reconstruction": error_stats(reconstruction_rmw, reference_rmw),
    }

    if bootstrap:
        if storm_id is None:
            raise ValueError("storm_id is required when --bootstrap is used")
        gfs_max, reconstruction_max, reference_max = common_domain_maximum(
            gfs_speed[:, 0], reconstruction_speed[:, 0], reference_speed[:, 0]
        )
        result["storm_block_bootstrap"] = {
            "field_rmse_reduction_ms": storm_bootstrap(
                gfs_speed[:, 0] - reference_speed[:, 0],
                reconstruction_speed[:, 0] - reference_speed[:, 0],
                storm_id, "rmse", bootstrap, seed,
            ),
            "vmax_mae_reduction_ms": storm_bootstrap(
                gfs_max - reference_max,
                reconstruction_max - reference_max,
                storm_id, "mae", bootstrap, seed,
            ),
            "rmw_mae_reduction_km": storm_bootstrap(
                gfs_rmw - reference_rmw,
                reconstruction_rmw - reference_rmw,
                storm_id, "mae", bootstrap, seed,
            ),
        }
    return result


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--levels", default=",".join(DEFAULT_LEVELS))
    parser.add_argument("--grid-spacing-km", type=float, default=3.0)
    parser.add_argument("--bootstrap", type=int, default=0, metavar="N")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    gfs, reconstruction, reference, storm_id = load_fields(args.input)
    levels = [value.strip() for value in args.levels.split(",")]
    if len(levels) != gfs.shape[1]:
        raise ValueError("--levels must contain one name per vertical level")
    result = evaluate(
        gfs, reconstruction, reference, levels, args.grid_spacing_km,
        storm_id, args.bootstrap, args.seed,
    )
    text = json.dumps(json_ready(result), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
