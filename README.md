# Near real time reconstruction of multilevel tropical cyclone wind fields at hurricane resolving scales

Code and source data for *Near real time reconstruction of multilevel tropical cyclone wind fields at hurricane resolving scales*.

## Code

- `scripts/read_tcvitals.py`: read operational TCVitals records.
- `scripts/prepare_gfs_tcvitals.py`: prepare storm-centred GFS wind fields and TCVitals conditions.
- `scripts/evaluate_fields.py`: evaluate reconstructed wind fields.
- `config/input_statistics.json`: input normalisation statistics.

Install the Python dependencies with `pip install -r requirements.txt`. Reading GFS GRIB2 files also requires pygrib and ecCodes. See each script's `--help` for usage.

## Source data

Download the figure and table source data from [Zenodo](https://doi.org/10.5281/zenodo.23153076).

The source-data archive contains the numerical values underlying the article's figures and tables. Its README describes the files, variables and units.

## License

Code is distributed under the Apache License 2.0. See `LICENSE` and `NOTICE`.
