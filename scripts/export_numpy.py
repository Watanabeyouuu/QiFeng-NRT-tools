#!/usr/bin/env python3
"""Export an example NetCDF file to a NumPy archive."""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path, help='Input NetCDF file')
    parser.add_argument('output', type=Path, help='Output .npz file')
    args = parser.parse_args()
    if args.output.suffix != '.npz':
        parser.error('The output filename must end in .npz.')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')

    with xr.open_dataset(args.file, engine='netcdf4') as ds:
        with args.output.open('xb') as output:
            np.savez_compressed(
                output,
                wind_speed=ds['wind_speed'].values,
                level=ds['level'].values.astype(str),
                latitude=ds['latitude'].values,
                longitude=ds['longitude'].values,
                latitude_bounds=ds['latitude_bounds'].values,
                longitude_bounds=ds['longitude_bounds'].values,
                analysis_time_utc=np.asarray(ds.attrs['analysis_time_utc']),
                wind_speed_units=np.asarray(ds['wind_speed'].attrs['units']),
            )
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
