#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

from symbtr_pipeline import RESULTS_DIR, TABLES_DIR


if __name__ == "__main__":
    df = pd.read_csv(RESULTS_DIR / "features.csv")
    with open(RESULTS_DIR / "stats.json", "r", encoding="utf-8") as handle:
        stats = json.load(handle)

    sample_rows = [
        ("Retained pieces", stats["n_retained"]),
        ("Saz semaisi", stats["n_sazsemaisi"]),
        ("Türkü", stats["n_turku"]),
        ("Median duration (quarters)", round(df["duration_quarters"].median(), 1)),
        ("Median note count", int(df["n_notes"].median())),
    ]
    with open(TABLES_DIR / "sample_summary.tex", "w", encoding="utf-8") as handle:
        handle.write("\\centering\\begin{tabular}{lr}\\toprule\n")
        handle.write("  Item & Value \\\\ \n")
        handle.write("  \\midrule\n")
        for item, value in sample_rows:
            handle.write(f"  {item} & {value} \\\\ \n")
        handle.write("  \\bottomrule\\end{tabular}\n")

    feature_names = ["rhythmic_complexity", "note_density", "pitch_class_entropy", "phase_entropy"]
    with open(TABLES_DIR / "effect_sizes.tex", "w", encoding="utf-8") as handle:
        handle.write("\\centering\\begin{tabular}{lrrr}\\toprule\n")
        handle.write("  Feature & Difference & p-value & Effect size \\\\ \n")
        handle.write("  \\midrule\n")
        for feature in feature_names:
            entry = stats["features"][feature]
            label = feature.replace("_", "\\_")
            handle.write(f"  {label} & {entry['difference']:.3f} & {entry['p_value']:.3g} & {entry['effect_size']:.3f} \\\\ \n")
        handle.write("  \\bottomrule\\end{tabular}\n")

    with open(RESULTS_DIR / "predictive_checks.json", "r", encoding="utf-8") as handle:
        predictive = json.load(handle)

    with open(TABLES_DIR / "classifier_results.tex", "w", encoding="utf-8") as handle:
        handle.write("\\centering\\begin{tabular}{lrr}\\toprule\n")
        handle.write("  Feature set & Accuracy & Balanced accuracy \\\\ \n")
        handle.write("  \\midrule\n")
        for name in ["rhythm_only", "melody_only", "combined", "with_controls"]:
            entry = predictive[name]
            label = name.replace("_", "\\_")
            handle.write(f"  {label} & {entry['accuracy']:.3f} & {entry['balanced_accuracy']:.3f} \\\\ \n")
        handle.write("  \\bottomrule\\end{tabular}\n")

    print("Wrote LaTeX tables to tables/")
