# Raga-Makam Similarity

Code and exported results for symbolic makam-raga similarity analysis.

## What Is In This Repo

- `src/`: 11-step research pipeline scripts plus `run_pipeline.py`
- `scripts/`: utility/analysis scripts for manifest building, statistics, and table/figure generation
- `results/`: exported artifacts currently tracked in this repo
	- `features.csv`
	- `label_map.csv`
	- `manifest.csv`
	- `matched_pairs.csv`
	- `predictive_checks.json`
	- `stats.json`
- `.gitignore`, `LICENSE`, `README.md`

## Sourcing datasets

This repository currently does not include raw data folders such as `data_turkish/`, `data_indian/`, or `symbtr/`. Some scripts in `src/` and `scripts/` expect those external folders. These datasets can be sourced from [Karaosmanoğlu et al. SymbTr v3](https://zenodo.org/records/15470412) and [Korade et al. Indian Raga Notation Dataset](https://github.com/sohamkorade/indian-raga-dataset) respectively.

The tracked `results/` files are the precomputed outputs available in this repo.

## Running Code

If you have the required external data directories in place, you can run:

```bash
python src/run_pipeline.py
```

Or run utilities individually, for example:

```bash
python scripts/symbtr_pipeline.py
```

## Author

Arjun Bahuguna  
Universitat Pompeu Fabra  
arjun.bahuguna01@estudiant.upf.edu
