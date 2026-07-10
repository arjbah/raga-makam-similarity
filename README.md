# Computing Modal Similarity in Indian Art Music and Ottoman-Turkish Makam Music

Code, data processing pipeline, and manuscript sources for a symbolic cross-cultural study of modal similarity between Turkish makam and Indian raga corpora.

## Repository Contents

- `paper/`: ISMIR-style manuscript sources (`main.tex`) and bibliography.
- `src/`: end-to-end analysis scripts (ingestion, normalization, features, similarity, tests, figures, paper generation).
- `data/`: processed tables and intermediate outputs used by the paper.
- `data_turkish/`: Turkish symbolic files (raw corpus subset/full set).
- `data_indian/`: Indian notation corpus and auxiliary analysis files.
- `figures/`: manuscript figures.
- `results/`: exported analysis artifacts.

## Reproducibility

Run the full pipeline from the project root:

```bash
python src/run_pipeline.py
```

Compile the paper:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper paper/main.tex
```

## Notes

- The comparison is tonic-relative and operates in 12 pitch classes.
- Turkish microtonal detail is intentionally collapsed for cross-corpus comparability.
- Top-pair exports are deduplicated by manuscript display labels to avoid repeated rows.

## Citation

If you use this repository, cite the accompanying manuscript in `paper/main.tex`.

## Author

Arjun Bahuguna  
Universitat Pompeu Fabra  
arjun.bahuguna01@estudiant.upf.edu
