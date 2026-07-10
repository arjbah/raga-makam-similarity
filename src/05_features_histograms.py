#!/usr/bin/env python3
"""
Task 5: Compute Pitch-Class Histograms
First-order feature: frequency distribution of pitch classes per melody.
"""

import csv
from pathlib import Path
import numpy as np
from collections import defaultdict

MAKAM_INPUT = Path(__file__).parent.parent / "data" / "processed" / "makam_seqs.csv"
RAGA_INPUT = Path(__file__).parent.parent / "data" / "processed" / "raga_seqs.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

print("Computing pitch-class histograms (first-order features)...\n")

def get_pitch_histogram(pitch_seq_str):
    """Return normalized 12-bin pitch-class histogram."""
    if not pitch_seq_str:
        return np.zeros(12)
    
    pitch_seq = [int(ch) for ch in pitch_seq_str if ch.isdigit()]
    if not pitch_seq:
        return np.zeros(12)
    
    hist = np.bincount(pitch_seq, minlength=12)
    return hist / max(1, hist.sum())  # Normalize to [0, 1]

# Process Turkish makams
print("Processing Turkish makams...")
makam_data = []
with open(MAKAM_INPUT, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    makam_data = list(reader)

# Store per-composition histograms
makam_hists = []
for row in makam_data:
    hist = get_pitch_histogram(row['pitch_seq_normalized_12edo'])
    makam_hists.append(hist)
    # Add histogram as columns to row
    for i, val in enumerate(hist):
        row[f'hist_pc_{i}'] = f"{val:.6f}"

# Save with histogram columns
with open(MAKAM_INPUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=makam_data[0].keys())
    writer.writeheader()
    writer.writerows(makam_data)

print(f"  ✓ Computed histograms for {len(makam_data)} Turkish compositions")

# Process Indian ragas
print("Processing Indian ragas...")
raga_data = []
with open(RAGA_INPUT, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    raga_data = list(reader)

raga_hists = []
for row in raga_data:
    hist = get_pitch_histogram(row['pitch_seq_normalized'])
    raga_hists.append(hist)
    for i, val in enumerate(hist):
        row[f'hist_pc_{i}'] = f"{val:.6f}"

# Save with histogram columns
with open(RAGA_INPUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=raga_data[0].keys())
    writer.writeheader()
    writer.writerows(raga_data)

print(f"  ✓ Computed histograms for {len(raga_data)} Indian compositions")

# Aggregate per makam/raga
print("\nAggregating per makam/raga...")

# Turkish makam profiles
makam_profiles = defaultdict(list)
for row, hist in zip(makam_data, makam_hists):
    makam_profiles[row['makam']].append(hist)

makam_profile_agg = {}
for makam, hists in makam_profiles.items():
    makam_profile_agg[makam] = np.mean(hists, axis=0)

# Indian raga profiles
raga_profiles = defaultdict(list)
for row, hist in zip(raga_data, raga_hists):
    raga_profiles[row['raga']].append(hist)

raga_profile_agg = {}
for raga, hists in raga_profiles.items():
    raga_profile_agg[raga] = np.mean(hists, axis=0)

# Save aggregated profiles
makam_profile_file = OUTPUT_DIR / "makam_profiles.csv"
with open(makam_profile_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['makam', 'n_compositions'] + [f'pc_{i}' for i in range(12)]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for makam, hist in makam_profile_agg.items():
        row = {'makam': makam, 'n_compositions': len(makam_profiles[makam])}
        for i, val in enumerate(hist):
            row[f'pc_{i}'] = f"{val:.6f}"
        writer.writerow(row)

raga_profile_file = OUTPUT_DIR / "raga_profiles.csv"
with open(raga_profile_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['raga', 'n_compositions'] + [f'pc_{i}' for i in range(12)]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for raga, hist in raga_profile_agg.items():
        row = {'raga': raga, 'n_compositions': len(raga_profiles[raga])}
        for i, val in enumerate(hist):
            row[f'pc_{i}'] = f"{val:.6f}"
        writer.writerow(row)

print(f"  ✓ Makam profiles: {len(makam_profile_agg)} unique makams → {makam_profile_file.name}")
print(f"  ✓ Raga profiles: {len(raga_profile_agg)} unique ragas → {raga_profile_file.name}")

# Statistics
print(f"\nHistogram statistics:")
print(f"  Turkish makams: {len(makam_profile_agg)}")
print(f"  Indian ragas: {len(raga_profile_agg)}")
print(f"\n  Top 5 makams by composition count:")
for makam, count in sorted([(m, len(makam_profiles[m])) for m in makam_profiles], key=lambda x: -x[1])[:5]:
    print(f"    - {makam}: {count}")
print(f"\n  Top 5 ragas by composition count:")
for raga, count in sorted([(r, len(raga_profiles[r])) for r in raga_profiles], key=lambda x: -x[1])[:5]:
    print(f"    - {raga}: {count}")

print(f"\n✓ Pitch-class histogram extraction complete")
