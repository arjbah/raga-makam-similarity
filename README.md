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

## Important Scope Note

This repository currently does not include raw data folders such as `data_turkish/`, `data_indian/`, or `symbtr/`, and does not include manuscript folders such as `paper/`.

Some scripts in `src/` and `scripts/` expect those external folders. The tracked `results/` files are the precomputed outputs available in this repo.

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
