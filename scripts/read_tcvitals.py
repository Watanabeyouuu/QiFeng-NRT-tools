#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 QiFeng-NRT authors.
# SPDX-License-Identifier: Apache-2.0

"""Read operational TCVitals files and write the matched records as CSV."""

import argparse
import csv
import math
import sys
from pathlib import Path


SYNOPTIC_TIMES = {"0000", "0600", "1200", "1800"}
BASIN_CODES = {
    "L": "AL",
    "E": "EP",
    "C": "CP",
    "W": "WP",
    "O": "WP",
    "T": "WP",
    "S": "SH",
    "P": "SH",
    "U": "SH",
    "A": "IO",
    "B": "IO",
}
CSV_FIELDS = [
    "atcf_id",
    "cycle",
    "storm_id",
    "agency",
    "center_lat",
    "center_lon",
    "vmax_ms",
    "rmw_km",
]


def coordinate(token, positive, negative):
    token = token.upper()
    if len(token) < 2 or token[-1] not in (positive, negative):
        raise ValueError(f"invalid coordinate: {token}")
    value = float(token[:-1]) / 10.0
    return -value if token[-1] == negative else value


def parse_line(line):
    """Return one usable record, or None for an incomplete/non-synoptic line."""

    fields = line.split()
    date_index = next(
        (i for i, value in enumerate(fields) if len(value) == 8 and value.isdigit()),
        None,
    )
    if date_index is None or date_index < 2 or date_index + 10 >= len(fields):
        return None

    storm_id = fields[1].upper()
    if len(storm_id) < 3 or not storm_id[:-1].isdigit():
        return None
    storm_number = int(storm_id[:-1])
    basin = BASIN_CODES.get(storm_id[-1])
    if basin is None or not 1 <= storm_number <= 49:
        return None

    date = fields[date_index]
    hhmm = fields[date_index + 1][:4]
    if hhmm not in SYNOPTIC_TIMES:
        return None

    try:
        lat = coordinate(fields[date_index + 2], "N", "S")
        lon = coordinate(fields[date_index + 3], "E", "W")
        vmax = float(fields[date_index + 9])
        rmw = float(fields[date_index + 10])
    except (ValueError, IndexError):
        return None
    if not all(math.isfinite(x) for x in (lat, lon, vmax, rmw)):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180 and vmax > 0 and rmw > 0):
        return None

    return {
        "atcf_id": f"{basin}{storm_number:02d}{date[:4]}",
        "cycle": date + hhmm[:2],
        "storm_id": storm_id,
        "agency": fields[0].upper(),
        "center_lat": lat,
        "center_lon": lon,
        "vmax_ms": vmax,
        "rmw_km": rmw,
    }


def read_tcvitals(paths):
    """Read files into {(ATCF id, YYYYMMDDHH): record}; prefer NHC duplicates."""

    records = {}
    for path in paths:
        with Path(path).open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record = parse_line(line)
                if record is None:
                    continue
                key = (record["atcf_id"], record["cycle"])
                previous = records.get(key)
                if previous is None or (
                    record["agency"] == "NHC" and previous["agency"] != "NHC"
                ):
                    records[key] = record
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="CSV output (stdout by default)")
    args = parser.parse_args()

    records = sorted(
        read_tcvitals(args.inputs).values(),
        key=lambda row: (row["cycle"], row["atcf_id"]),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        handle = args.output.open("w", encoding="utf-8", newline="")
    else:
        handle = sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    finally:
        if args.output:
            handle.close()


if __name__ == "__main__":
    main()
