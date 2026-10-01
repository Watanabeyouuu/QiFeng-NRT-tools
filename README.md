# QiFeng-NRT preprocessing and evaluation scripts

This repository contains data-reading, input-preparation and gridded-field
evaluation scripts accompanying the QiFeng-NRT study.

## Files

- `scripts/read_tcvitals.py` reads operational TCVitals records at 00, 06, 12
  and 18 UTC and writes the usable records to CSV.
- `scripts/prepare_gfs_tcvitals.py` reads a 0.25° GFS f000 file, refines the
  storm centre, constructs the storm-centred multilevel wind crop and appends
  the TCVitals maximum-wind and radius-of-maximum-wind condition channels.
- `scripts/evaluate_fields.py` calculates wind-speed, domain-maximum wind and
  radius-of-maximum-wind errors. Optional confidence intervals resample whole
  storms so that all cycles and HAFS configurations remain in the same block.
- `config/input_statistics.json` contains the ten-channel GFS input statistics
  calculated from the 4,438 training samples.

The scripts require Python 3.9 or later. Install NumPy and SciPy, and install
`pygrib` with ecCodes when reading GFS GRIB2 files:

```bash
python -m pip install -r requirements.txt
```

## Examples

Read TCVitals records:

```bash
python scripts/read_tcvitals.py examples/sample_tcvitals.txt -o tcvitals.csv
```

Prepare one 12-channel GFS–TCVitals input:

```bash
python scripts/prepare_gfs_tcvitals.py \
  --gfs-grib /path/to/gfs.t00z.pgrb2.0p25.f000 \
  --tcvitals /path/to/gfs.t00z.syndata.tcvitals.tm00 \
  --storm AL132023 --cycle 2023090700 \
  --output prepared/AL132023_2023090700.npz
```

The output contains the ten physical wind channels in m s⁻¹ and the final
12-channel model input. The latter uses the published channel statistics after
dividing winds by 150 m s⁻¹, followed by the constant `Vmax/150` and `RMW/200`
maps.

Evaluate gridded fields:

```bash
python scripts/evaluate_fields.py fields.npz \
  --bootstrap 20000 --seed 0 -o evaluation.json
```

Here `fields.npz` contains `gfs`, `reconstruction` and `reference` arrays in
`(N, L, 2, H, W)` order, with `u` and `v` in m s⁻¹. The first level is 10 m.
For bootstrap intervals, add one `storm_id` string per sample.

## Data sources

- [GFS archive](https://rda.ucar.edu/datasets/d084001/)
- [Operational GFS and TCVitals](https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/)
- [HAFS](https://nomads.ncep.noaa.gov/pub/data/nccf/com/hafs/prod/)
- [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)
- [NOAA Hurricane Research Division dropsondes](https://www.aoml.noaa.gov/hrd/data_sub/hurr.html)
- [NOAA/STAR tropical-cyclone SAR winds](https://www.star.nesdis.noaa.gov/socd/mecb/sar/sarwinds_tropical.php)

The scripts are released under the Apache License 2.0.
