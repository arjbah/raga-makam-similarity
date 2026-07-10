#!/usr/bin/env python3
"""
Task 8: Generate Similarity Matrix & Top Pairs
Aggregates distance metrics and identifies best makam-raga matches.
"""

import csv
from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

print("Generating similarity matrix and top pairs...\n")

# Load distance matrices
kl_div_matrix = np.load(OUTPUT_DIR / "kl_div_matrix.npy")
wasserstein_matrix = np.load(OUTPUT_DIR / "wasserstein_matrix.npy")
cosine_matrix = np.load(OUTPUT_DIR / "cosine_matrix.npy")

# Load indices
makam_names = []
with open(OUTPUT_DIR / "makam_index.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    makam_names = [row['makam'] for row in reader]

raga_names = []
with open(OUTPUT_DIR / "raga_index.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    raga_names = [row['raga'] for row in reader]

print(f"Matrix shape: {len(makam_names)} makams × {len(raga_names)} ragas")


def canonical_display_raga(raga_name):
    """Normalize close spelling variants for display without changing source IDs."""
    mapping = {
        'C_shankarabharana': 'C_shankarabharanam',
        'C_dharmavathi': 'C_dharmavati',
    }
    return mapping.get(raga_name, raga_name)


def canonical_display_makam(makam_name):
    """Lightweight display cleanup for manuscript-facing exports."""
    return makam_name.replace('_', ' ')

# Create composite similarity metric (lower is better for KL/Wasserstein, higher for cosine)
# Normalize each metric to [0, 1]
kl_norm = (kl_div_matrix - kl_div_matrix.min()) / (kl_div_matrix.max() - kl_div_matrix.min() + 1e-10)
wasserstein_norm = (wasserstein_matrix - wasserstein_matrix.min()) / (wasserstein_matrix.max() - wasserstein_matrix.min() + 1e-10)
cosine_norm = 1.0 - cosine_matrix  # Convert cosine similarity to distance

# Composite distance: average of normalized metrics (lower = more similar)
composite_distance = (kl_norm + wasserstein_norm + cosine_norm) / 3.0
np.save(OUTPUT_DIR / "composite_distance.npy", composite_distance)

print(f"Composite distance: min={composite_distance.min():.4f}, max={composite_distance.max():.4f}")

# Find top pairs (lowest composite distance)
print("\nFinding top matching pairs...")
pairs = []
for i, makam in enumerate(makam_names):
    for j, raga in enumerate(raga_names):
        pairs.append({
            'makam': makam,
            'raga': raga,
            'display_makam': canonical_display_makam(makam),
            'display_raga': canonical_display_raga(raga),
            'kl_divergence': float(kl_div_matrix[i, j]),
            'wasserstein_distance': float(wasserstein_matrix[i, j]),
            'cosine_similarity': float(cosine_matrix[i, j]),
            'composite_distance': float(composite_distance[i, j]),
        })

# Sort by composite distance (ascending)
pairs_sorted = sorted(pairs, key=lambda x: x['composite_distance'])

# Keep only unique display pairs for manuscript-facing exports.
# This avoids repeated rows when different source labels are canonically
# mapped to the same display label.
pairs_unique_display = []
seen_display_pairs = set()
for pair in pairs_sorted:
    key = (pair['display_makam'], pair['display_raga'])
    if key in seen_display_pairs:
        continue
    seen_display_pairs.add(key)
    pairs_unique_display.append(pair)
    if len(pairs_unique_display) >= 100:
        break

# Save top-100 pairs
top_pairs_file = OUTPUT_DIR / "top_pairs.csv"
with open(top_pairs_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['rank', 'makam', 'raga', 'display_makam', 'display_raga', 'kl_divergence', 'wasserstein_distance', 'cosine_similarity', 'composite_distance']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for rank, pair in enumerate(pairs_unique_display, 1):
        pair['rank'] = rank
        writer.writerow(pair)

print(f"  ✓ Top 100 pairs → {top_pairs_file.name}")

# Print top-10
print(f"\nTop 10 makam-raga matches (by composite distance):")
for rank, pair in enumerate(pairs_unique_display[:10], 1):
    print(f"  {rank:2d}. {pair['makam']:15s} ↔ {pair['raga']:25s}  KL={pair['kl_divergence']:.4f}  Cosine={pair['cosine_similarity']:.4f}  Composite={pair['composite_distance']:.4f}")

# Save full similarity matrix as CSV (readable form)
similarity_csv = OUTPUT_DIR / "similarity_matrix_composite.csv"
df_sim = pd.DataFrame(composite_distance, index=makam_names, columns=raga_names)
df_sim.index.name = 'makam'
df_sim.to_csv(similarity_csv)
print(f"\n  ✓ Full similarity matrix → {similarity_csv.name}")

# Also save per-metric matrices as CSV for reference
pd.DataFrame(kl_div_matrix, index=makam_names, columns=raga_names).to_csv(OUTPUT_DIR / "similarity_kl_divergence.csv")
pd.DataFrame(wasserstein_matrix, index=makam_names, columns=raga_names).to_csv(OUTPUT_DIR / "similarity_wasserstein.csv")
pd.DataFrame(cosine_matrix, index=makam_names, columns=raga_names).to_csv(OUTPUT_DIR / "similarity_cosine.csv")

print(f"\n✓ Similarity matrix generation complete")
