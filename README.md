# Near real time reconstruction of multilevel tropical cyclone wind fields at hurricane resolving scales

Example QiFeng-NRT wind fields and scripts for reading, exporting and plotting them.

## Examples

| File in `examples/` | Analysis time (UTC) | Levels | Paper panels |
| --- | --- | --- | --- |
| `bret_20230620T0000Z.nc` | 20 June 2023, 00:00 | 10 m | Figure 3b |
| `dora_20230802T1200Z.nc` | 2 August 2023, 12:00 | 10 m, 850, 700, 500 and 200 hPa | Figure 3e and Supplementary Figure S3b,e,h,k,n |
| `dora_20230806T0600Z.nc` | 6 August 2023, 06:00 | 10 m | Figure 3h |

Each file contains the plotted QiFeng-NRT wind-speed values on a 256 × 256 grid with nominal 3 km spacing. `wind_speed(level, y, x)` is in m/s. `latitude` and `longitude` give cell midpoints in degrees north/east; their bounds preserve the original plotting mesh. `time` is the analysis time in UTC. File attributes identify the storm, paper panels and source archive.

The examples are available in the paper's [figure source data](https://doi.org/10.5281/zenodo.23360500).

## Read and plot

```bash
pip install -r requirements.txt
python scripts/read_example.py examples/bret_20230620T0000Z.nc
python scripts/read_example.py examples/dora_20230802T1200Z.nc --level 850hPa --plot dora_850hPa.png
```

The reader prints the case, available levels, array shape, units and value range. `--plot` saves a map using the stored cell bounds.

To plot all available levels with a common colour scale:

```bash
python scripts/plot_levels.py examples/dora_20230802T1200Z.nc dora_levels.png
```

To export the wind-speed array, level labels, coordinates, cell bounds, time and units:

```bash
python scripts/export_numpy.py examples/dora_20230802T1200Z.nc dora.npz
```

Read the exported arrays with `numpy.load("dora.npz", allow_pickle=False)`.

To read an array directly:

```python
import xarray as xr

with xr.open_dataset("examples/dora_20230802T1200Z.nc", engine="netcdf4") as ds:
    wind_speed = ds["wind_speed"].sel(level="850hPa").values
    latitude = ds["latitude"].values
    longitude = ds["longitude"].values
```

## License

Code is distributed under Apache License 2.0; see `LICENSE` and `NOTICE`. Example data are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), matching the source-data archive. Please cite the associated paper and the source archive when using these samples.
