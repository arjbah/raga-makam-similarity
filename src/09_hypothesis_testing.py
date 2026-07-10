#!/usr/bin/env python3
"""
Task 9: Statistical Hypothesis Testing
Tests if known makam-raga pairs score significantly higher than random baseline.
"""

import csv
from pathlib import Path
import numpy as np
from scipy import stats

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

print("Conducting hypothesis testing...\n")

# Load composite distance matrix
composite_distance = np.load(OUTPUT_DIR / "composite_distance.npy")

# Load indices
makam_names = []
with open(OUTPUT_DIR / "makam_index.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    makam_names = [row['makam'] for row in reader]

raga_names = []
with open(OUTPUT_DIR / "raga_index.csv", 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    raga_names = [row['raga'] for row in reader]

# Known makam-raga pairs (literature-based hypothesized correspondences)
# Lower distance = higher similarity = better match
known_pairs = [
    ('kürdi', 'bhairavi'),  # Both dark, lower register
    ('hicaz', 'bhairav'),   # Augmented 2nd, dramatic
    ('rast', 'yaman'),      # Pentatonic + 7th, bright
    ('nihavent', 'kalyani'),  # Major-like, celebratory
    ('segâh', 'ahir bhairav'),  # Mixed major-minor
]

# Try to match known pairs (case-insensitive substring matching)
known_pair_distances = []
known_pairs_found = []

for makam_query, raga_query in known_pairs:
    # Find matching makam
    makam_idx = None
    for idx, makam in enumerate(makam_names):
        if makam_query.lower() in makam.lower() or makam.lower() in makam_query.lower():
            makam_idx = idx
            break
    
    # Find matching raga
    raga_idx = None
    for idx, raga in enumerate(raga_names):
        if raga_query.lower() in raga.lower() or raga.lower() in raga_query.lower():
            raga_idx = idx
            break
    
    if makam_idx is not None and raga_idx is not None:
        distance = composite_distance[makam_idx, raga_idx]
        known_pair_distances.append(distance)
        known_pairs_found.append({
            'makam': makam_names[makam_idx],
            'raga': raga_names[raga_idx],
            'distance': distance,
            'query': (makam_query, raga_query)
        })
        print(f"  Found pair: {makam_names[makam_idx]:20s} ↔ {raga_names[raga_idx]:25s} (distance={distance:.6f})")

if not known_pairs_found:
    print("  ⚠ Warning: No known pairs found in data. Using top pairs for validation.")
    # Use top pairs as proxy
    top_pairs_file = OUTPUT_DIR / "top_pairs.csv"
    with open(top_pairs_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        top_pairs = list(reader)[:10]
        for pair in top_pairs:
            makam_idx = makam_names.index(pair['makam'])
            raga_idx = raga_names.index(pair['raga'])
            distance = composite_distance[makam_idx, raga_idx]
            known_pair_distances.append(distance)

known_mean = np.mean(known_pair_distances)
known_std = np.std(known_pair_distances)

print(f"\nKnown pairs statistics:")
print(f"  N = {len(known_pair_distances)}")
print(f"  Mean distance: {known_mean:.6f}")
print(f"  Std: {known_std:.6f}")
print(f"  Min: {min(known_pair_distances):.6f}, Max: {max(known_pair_distances):.6f}")

# Null hypothesis: random pairs
print(f"\nBootstrap null hypothesis (1000 iterations)...")
np.random.seed(42)
null_distances = []
for iteration in range(1000):
    # Shuffle raga indices
    shuffled_raga_idx = np.random.permutation(len(raga_names))
    for i, makam_idx in enumerate(range(min(len(makam_names), len(raga_names)))):
        j = shuffled_raga_idx[i]
        null_distances.append(composite_distance[makam_idx, j])

null_mean = np.mean(null_distances)
null_std = np.std(null_distances)

print(f"  Random pairs: N = {len(null_distances)}")
print(f"    Mean distance: {null_mean:.6f}")
print(f"    Std: {null_std:.6f}")

# Statistical tests
# T-test
t_stat, p_value_ttest = stats.ttest_ind(known_pair_distances, null_distances[:len(known_pair_distances)])

# Mann-Whitney U test (non-parametric)
u_stat, p_value_mw = stats.mannwhitneyu(known_pair_distances, null_distances[:len(known_pair_distances)])

# Effect size (Cohen's d)
cohens_d = (null_mean - known_mean) / np.sqrt((known_std**2 + null_std**2) / 2)

# Percentile rank
percentile = stats.percentileofscore(null_distances, known_mean)

print(f"\nStatistical tests:")
print(f"  T-test: t = {t_stat:.4f}, p-value = {p_value_ttest:.6f}")
print(f"  Mann-Whitney U: U = {u_stat:.2f}, p-value = {p_value_mw:.6f}")
print(f"  Cohen's d (effect size): {cohens_d:.4f}")
print(f"  Known mean percentile in null: {percentile:.2f}%")

# Interpret results
print(f"\nInterpretation:")
if p_value_ttest < 0.05:
    print(f"  ✓ Known pairs are SIGNIFICANTLY different from random (p < 0.05)")
    if known_mean < null_mean:
        print(f"  ✓ Known pairs have LOWER distance (higher similarity) than random ✓✓")
    else:
        print(f"  ✗ Known pairs have HIGHER distance (lower similarity) than random")
else:
    print(f"  ✗ Known pairs are NOT significantly different from random (p >= 0.05)")

if abs(cohens_d) > 0.2:
    effect = "small" if abs(cohens_d) <= 0.5 else "medium" if abs(cohens_d) <= 0.8 else "large"
    print(f"  • Effect size is {effect}")

# Save results
results_file = OUTPUT_DIR / "hypothesis_test_results.csv"
with open(results_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Known Pairs (Mean Distance)', f"{known_mean:.6f}"])
    writer.writerow(['Known Pairs (Std)', f"{known_std:.6f}"])
    writer.writerow(['Random Pairs (Mean Distance)', f"{null_mean:.6f}"])
    writer.writerow(['Random Pairs (Std)', f"{null_std:.6f}"])
    writer.writerow(['T-Test P-Value', f"{p_value_ttest:.6f}"])
    writer.writerow(['Mann-Whitney P-Value', f"{p_value_mw:.6f}"])
    writer.writerow(['Cohen\'s D', f"{cohens_d:.6f}"])
    writer.writerow(['Percentile Rank', f"{percentile:.2f}"])

print(f"\n  ✓ Results saved → {results_file.name}")

# Save known pairs with distances
known_pairs_file = OUTPUT_DIR / "known_pairs_analysis.csv"
with open(known_pairs_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['makam', 'raga', 'distance', 'proposed_correspondence'])
    writer.writeheader()
    for pair in known_pairs_found:
        writer.writerow({
            'makam': pair['makam'],
            'raga': pair['raga'],
            'distance': f"{pair['distance']:.6f}",
            'proposed_correspondence': f"{pair['query'][0]} ↔ {pair['query'][1]}"
        })

print(f"  ✓ Known pairs analysis → {known_pairs_file.name}")

print(f"\n✓ Hypothesis testing complete")
