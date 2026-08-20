#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright (c) 2026 QiFeng-NRT authors.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a storm-centred GFS--TCVitals input for QiFeng-NRT.

The script reads the 0.25-degree GFS f000 analysis product, locates the GFS
storm centre from mean sea-level pressure and 10-m winds, and interpolates the
five-level u/v fields to the 256 x 256 model grid.  TCVitals Vmax and RMW are
added as two constant condition channels.  One compressed NPZ file is written.
"""

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from read_tcvitals import read_tcvitals


GRID_SIZE = 256
HALF_WIDTH_KM = 384.0
KM_PER_DEGREE = 111.0

# Model channel order: u/v at 10 m, 850, 700, 500 and 200 hPa.
WIND_FIELDS = [
    ("u10m", "10u", "heightAboveGround", 10),
    ("v10m", "10v", "heightAboveGround", 10),
    ("u850", "u", "isobaricInhPa", 850),
    ("v850", "v", "isobaricInhPa", 850),
    ("u700", "u", "isobaricInhPa", 700),
    ("v700", "v", "isobaricInhPa", 700),
    ("u500", "u", "isobaricInhPa", 500),
    ("v500", "v", "isobaricInhPa", 500),
    ("u200", "u", "isobaricInhPa", 200),
    ("v200", "v", "isobaricInhPa", 200),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gfs-grib", type=Path, required=True,
        help="global 0.25-degree GFS f000 GRIB2 file"
    )
    parser.add_argument(
        "--tcvitals", type=Path, nargs="+", required=True,
        help="one or more operational TCVitals files"
    )
    parser.add_argument(
        "--storm", required=True,
        help="storm identifier, for example 13L or AL132023"
    )
    parser.add_argument(
        "--cycle", required=True, help="analysis cycle in YYYYMMDDHH format"
    )
    parser.add_argument(
        "--input-statistics",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "input_statistics.json",
        help="training-input mean and standard deviation file",
    )
    parser.add_argument("--output", type=Path, required=True, help="output NPZ file")
    return parser.parse_args()


def check_cycle(cycle):
    if len(cycle) != 10 or not cycle.isdigit():
        raise ValueError("--cycle must have format YYYYMMDDHH")
    try:
        datetime.strptime(cycle, "%Y%m%d%H")
    except ValueError as exc:
        raise ValueError("--cycle must have format YYYYMMDDHH") from exc
    if cycle[-2:] not in {"00", "06", "12", "18"}:
        raise ValueError("--cycle must be a 00, 06, 12 or 18 UTC cycle")


def select_tcvitals(records, storm, cycle):
    """Select one exact-cycle record by ATCF ID or TCVitals storm ID."""

    storm = storm.upper()
    record = records.get((storm, cycle))
    if record is None:
        matches = [
            item
            for item in records.values()
            if item["cycle"] == cycle
            and storm in {item["storm_id"].upper(), item["atcf_id"].upper()}
        ]
        if len(matches) != 1:
            raise ValueError(f"found {len(matches)} TCVitals records for {storm} at {cycle}")
        record = matches[0]

    required = ["center_lat", "center_lon", "vmax_ms", "rmw_km"]
    values = np.array([float(record[name]) for name in required])
    if not np.isfinite(values).all():
        raise ValueError("selected TCVitals record contains non-finite values")
    if not -90 <= values[0] <= 90 or not -180 <= values[1] <= 180:
        raise ValueError("selected TCVitals centre is outside the latitude/longitude range")
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError("TCVitals Vmax and RMW must be positive")
    return record


def grib_message(grib, short_name, level_type=None, level=None):
    selector = {"shortName": short_name}
    if level_type is not None:
        selector["typeOfLevel"] = level_type
    if level is not None:
        selector["level"] = level
    grib.seek(0)
    matches = grib.select(**selector)
    if not matches:
        raise RuntimeError(f"missing GFS field: {selector}")
    return matches[0]


def message_cycle(message):
    """Return the GRIB valid time and require an f000 field."""

    try:
        date = int(message.dataDate)
        time = int(message.dataTime)
        forecast_hour = int(message.forecastTime)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("GRIB field is missing cycle metadata") from exc
    if forecast_hour != 0:
        raise ValueError(f"expected GFS f000, found forecastTime={forecast_hour}")
    reference = datetime.strptime(f"{date:08d}{time:04d}", "%Y%m%d%H%M")
    return (reference + timedelta(hours=forecast_hour)).strftime("%Y%m%d%H")


def as_array(values):
    return np.asarray(np.ma.asarray(values, dtype=np.float64).filled(np.nan))


def read_gfs(path, expected_cycle):
    try:
        import pygrib
    except ImportError as exc:
        raise RuntimeError("reading GRIB2 requires pygrib and ecCodes") from exc

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    grib = pygrib.open(str(path))
    try:
        try:
            pressure_message = grib_message(grib, "prmsl")
        except (RuntimeError, ValueError):
            pressure_message = grib_message(grib, "msl")

        field_cycle = message_cycle(pressure_message)
        if field_cycle != expected_cycle:
            raise ValueError(
                f"GFS valid time {field_cycle} does not match --cycle {expected_cycle}"
            )
        pressure = as_array(pressure_message.values)

        winds = []
        coordinate_message = None
        for _, short_name, level_type, level in WIND_FIELDS:
            message = grib_message(grib, short_name, level_type, level)
            if message_cycle(message) != field_cycle:
                raise ValueError("selected GFS fields do not share one valid time")
            winds.append(as_array(message.values))
            if coordinate_message is None:
                coordinate_message = message
        latitude_grid, longitude_grid = coordinate_message.latlons()
    finally:
        grib.close()

    latitude = np.asarray(latitude_grid[:, 0], dtype=np.float64)
    longitude = np.mod(np.asarray(longitude_grid[0, :], dtype=np.float64), 360.0)
    winds = np.stack(winds)

    if latitude[0] > latitude[-1]:
        latitude = latitude[::-1]
        pressure = pressure[::-1]
        winds = winds[:, ::-1]

    order = np.argsort(longitude)
    longitude = longitude[order]
    pressure = pressure[:, order]
    winds = winds[:, :, order]

    if latitude.shape != (721,) or longitude.shape != (1440,):
        raise ValueError("expected the global 721 x 1440 GFS 0.25-degree grid")
    if not np.allclose(np.diff(latitude), 0.25, atol=1e-6):
        raise ValueError("GFS latitude spacing is not 0.25 degrees")
    if not np.allclose(np.diff(longitude), 0.25, atol=1e-6):
        raise ValueError("GFS longitude spacing is not 0.25 degrees")
    return pressure, winds, latitude, longitude


def storm_grid(center_lat, center_lon):
    offset = np.linspace(-HALF_WIDTH_KM, HALF_WIDTH_KM, GRID_SIZE)
    x_km, y_km = np.meshgrid(offset, offset)
    cos_lat = max(abs(np.cos(np.deg2rad(center_lat))), 0.15)
    latitude = center_lat + y_km / KM_PER_DEGREE
    longitude = center_lon + x_km / (KM_PER_DEGREE * cos_lat)
    longitude = (longitude + 180.0) % 360.0 - 180.0
    return latitude, longitude, x_km, y_km


def interpolate(field, source_lat, source_lon, target_lat, target_lon):
    """Bilinear interpolation with periodic longitude."""

    if target_lat.min() < source_lat[0] or target_lat.max() > source_lat[-1]:
        raise ValueError("storm-centred grid extends beyond the GFS latitude range")
    base = source_lon[0]
    periodic_lon = np.concatenate([source_lon, [base + 360.0]])
    periodic_field = np.concatenate([field, field[:, :1]], axis=1)
    sample_lon = (target_lon - base) % 360.0 + base
    points = np.column_stack([target_lat.ravel(), sample_lon.ravel()])
    result = RegularGridInterpolator(
        (source_lat, periodic_lon), periodic_field, method="linear"
    )(points)
    return result.reshape(target_lat.shape)


def minimum_index(field, mask):
    valid = mask & np.isfinite(field)
    if not valid.any():
        raise ValueError("no finite values in the centre-search region")
    return np.unravel_index(np.argmin(np.where(valid, field, np.inf)), field.shape)


def locate_center(pressure, winds, source_lat, source_lon, seed_lat, seed_lon):
    """Refine the TCVitals first guess using GFS pressure and 10-m winds."""

    lat, lon, x_km, y_km = storm_grid(seed_lat, seed_lon)
    p_crop = interpolate(pressure, source_lat, source_lon, lat, lon)
    u10 = interpolate(winds[0], source_lat, source_lon, lat, lon)
    v10 = interpolate(winds[1], source_lat, source_lon, lat, lon)
    speed = np.hypot(u10, v10)

    p_row, p_col = minimum_index(p_crop, np.ones_like(p_crop, dtype=bool))
    p_x, p_y = x_km[p_row, p_col], y_km[p_row, p_col]

    if np.hypot(p_x, p_y) <= 100.0:
        local = np.hypot(x_km - p_x, y_km - p_y) <= 50.0
        row, col = minimum_index(speed, local)
        method = "mslp_min+local_wind_min"
    else:
        local = np.hypot(x_km, y_km) <= 100.0
        row, col = minimum_index(speed, local)
        method = "local_wind_min"
    return float(lat[row, col]), float(lon[row, col]), method


def load_statistics(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    names = [item[0] for item in WIND_FIELDS]
    if config["channel_names"] != names:
        raise ValueError("input-statistics channel order does not match the model")
    mean = np.array([config["input"][name]["mean"] for name in names], dtype=float)
    std = np.array([config["input"][name]["std"] for name in names], dtype=float)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("invalid mean or standard deviation in input statistics")

    scales = config["scales"]
    wind_scale = float(scales["wind_ms"])
    vmax_scale = float(scales["vmax_ms"])
    rmw_scale = float(scales["rmw_km"])
    if not np.allclose([wind_scale, vmax_scale, rmw_scale], [150.0, 150.0, 200.0]):
        raise ValueError("expected wind/Vmax/RMW scales of 150, 150 and 200")
    return mean[:, None, None], std[:, None, None]


def main():
    args = parse_args()
    check_cycle(args.cycle)
    tcvitals = select_tcvitals(
        read_tcvitals(args.tcvitals), args.storm, args.cycle
    )
    pressure, winds, source_lat, source_lon = read_gfs(
        args.gfs_grib, args.cycle
    )

    center_lat, center_lon, center_method = locate_center(
        pressure,
        winds,
        source_lat,
        source_lon,
        float(tcvitals["center_lat"]),
        float(tcvitals["center_lon"]),
    )
    latitude, longitude, x_km, y_km = storm_grid(center_lat, center_lon)
    wind_mps = np.stack(
        [
            interpolate(field, source_lat, source_lon, latitude, longitude)
            for field in winds
        ]
    ).astype(np.float32)
    mslp_pa = interpolate(
        pressure, source_lat, source_lon, latitude, longitude
    ).astype(np.float32)

    mean, std = load_statistics(args.input_statistics)
    wind_standardized = ((wind_mps / 150.0 - mean) / std).astype(np.float32)
    conditions = np.array(
        [float(tcvitals["vmax_ms"]) / 150.0, float(tcvitals["rmw_km"]) / 200.0],
        dtype=np.float32,
    )
    condition_maps = np.broadcast_to(
        conditions[:, None, None], (2, GRID_SIZE, GRID_SIZE)
    ).copy()
    model_input = np.concatenate([wind_standardized, condition_maps], axis=0)

    if not np.isfinite(model_input).all() or not np.isfinite(mslp_pa).all():
        raise ValueError("prepared input contains non-finite values")

    output = (
        args.output
        if args.output.suffix == ".npz"
        else args.output.with_suffix(".npz")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        model_input=model_input,
        wind_mps=wind_mps,
        mslp_pa=mslp_pa,
        latitude=latitude.astype(np.float32),
        longitude=longitude.astype(np.float32),
        x_km=x_km.astype(np.float32),
        y_km=y_km.astype(np.float32),
        channel_names=np.array([item[0] for item in WIND_FIELDS] + ["vmax", "rmw"]),
        cycle=np.array(args.cycle),
        atcf_id=np.array(tcvitals["atcf_id"]),
        storm_id=np.array(tcvitals["storm_id"]),
        tcvitals_center=np.array(
            [tcvitals["center_lat"], tcvitals["center_lon"]], dtype=np.float32
        ),
        gfs_center=np.array([center_lat, center_lon], dtype=np.float32),
        center_method=np.array(center_method),
        vmax_ms=np.float32(tcvitals["vmax_ms"]),
        rmw_km=np.float32(tcvitals["rmw_km"]),
    )
    print(f"saved {model_input.shape} input for {tcvitals['storm_id']} {args.cycle}: {output}")


if __name__ == "__main__":
    main()
