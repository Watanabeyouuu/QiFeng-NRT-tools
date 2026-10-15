#!/usr/bin/env python3
"""Plot the available wind-speed levels in an example NetCDF file."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path, help='Input NetCDF file')
    parser.add_argument('output', type=Path, help='Output image file')
    args = parser.parse_args()

    with xr.open_dataset(args.file, engine='netcdf4') as ds:
        levels = ds['level'].values.tolist()
        columns = min(3, len(levels))
        rows = (len(levels) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows),
                                 squeeze=False, constrained_layout=True)
        lon = ds['longitude_bounds'].values
        lat = ds['latitude_bounds'].values
        lon_edges = np.r_[lon[:, 0], lon[-1, 1]]
        lat_edges = np.r_[lat[:, 0], lat[-1, 1]]
        vmax = float(ds['wind_speed'].max())
        aspect = 1 / np.cos(np.deg2rad(float(ds['latitude'].mean())))

        for ax, level in zip(axes.flat, levels):
            field = ds['wind_speed'].sel(level=level).values
            mesh = ax.pcolormesh(lon_edges, lat_edges, field, shading='flat',
                                 cmap='viridis', vmin=0, vmax=vmax)
            ax.set(title=level, xlabel='Longitude (degrees east)',
                   ylabel='Latitude (degrees north)')
            ax.set_aspect(aspect)
        for ax in list(axes.flat)[len(levels):]:
            ax.set_visible(False)
        fig.colorbar(mesh, ax=list(axes.flat)[:len(levels)], label='Wind speed (m/s)')
        fig.suptitle(f"{ds.attrs['case_name']} | {ds.attrs['analysis_time_utc']}")
        fig.savefig(args.output, dpi=180)
        plt.close(fig)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
