#!/usr/bin/env python3
"""Read a QiFeng-NRT example wind field and optionally save a map."""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path, help='Example NetCDF file')
    parser.add_argument('--level', default='10m', help='Level label, e.g. 10m or 850hPa')
    parser.add_argument('--plot', type=Path, help='Save a wind-speed map to this path')
    args = parser.parse_args()

    with xr.open_dataset(args.file, engine='netcdf4') as ds:
        levels = ds['level'].values.tolist()
        if args.level not in levels:
            parser.error(f'Level {args.level!r} is unavailable. Choose from {levels}.')
        field = ds['wind_speed'].sel(level=args.level)
        print(f"Case: {ds.attrs['case_name']} ({ds.attrs['storm_id']})")
        print(f"Analysis time: {ds.attrs['analysis_time_utc']}")
        print(f"Paper panels: {ds.attrs['figure_panels']}")
        print(f"Available levels: {', '.join(levels)}")
        print(f'Wind-speed array: {field.shape}, {field.dtype}, {field.attrs["units"]}')
        print(f'Wind-speed range: {float(field.min()):.3f} to {float(field.max()):.3f} m/s')

        if args.plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Use the original plotting-cell bounds stored with the sample.
            lon_bounds = ds['longitude_bounds'].values
            lat_bounds = ds['latitude_bounds'].values
            lon_edges = np.r_[lon_bounds[:, 0], lon_bounds[-1, 1]]
            lat_edges = np.r_[lat_bounds[:, 0], lat_bounds[-1, 1]]
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            image = ax.pcolormesh(lon_edges, lat_edges, field.values, shading='flat', cmap='viridis', vmin=0)
            fig.colorbar(image, ax=ax, label='Wind speed (m/s)')
            ax.set(xlabel='Longitude (degrees east)', ylabel='Latitude (degrees north)',
                   title=f"{ds.attrs['case_name']} | {args.level}\n{ds.attrs['analysis_time_utc']}")
            ax.set_aspect(1 / np.cos(np.deg2rad(float(ds['latitude'].mean()))))
            fig.savefig(args.plot, dpi=180)
            plt.close(fig)
            print(f'Saved {args.plot}')


if __name__ == '__main__':
    main()
