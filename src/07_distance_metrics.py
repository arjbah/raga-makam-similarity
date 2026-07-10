#!/usr/bin/env python3
"""
Task 7: Compute Distance Metrics
Calculates KL-divergence, Wasserstein, and cosine similarity for all makam-raga pairs.
"""

import csv
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy as scipy_entropy

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

MAKAM_PROFILE = OUTPUT_DIR / "makam_profiles.csv"
RAGA_PROFILE = OUTPUT_DIR / "raga_profiles.csv"
MAKAM_BIGRAM = OUTPUT_DIR / "makam_bigrams.pkl"
RAGA_BIGRAM = OUTPUT_DIR / "raga_bigrams.pkl"

print("Computing distance metrics for all makam-raga pairs...\n")

# Load profiles
print("Loading profiles...")
makam_profiles = {}
with open(MAKAM_PROFILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        makam = row['makam']
        hist = np.array([float(row[f'pc_{i}']) for i in range(12)])
        makam_profiles[makam] = hist

raga_profiles = {}
with open(RAGA_PROFILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        raga = row['raga']
        hist = np.array([float(row[f'pc_{i}']) for i in range(12)])
        raga_profiles[raga] = hist

print(f"  ✓ Loaded {len(makam_profiles)} makam profiles")
print(f"  ✓ Loaded {len(raga_profiles)} raga profiles")

# Load bigrams
print("Loading bigram matrices...")
with open(MAKAM_BIGRAM, 'rb') as f:
    makam_bigrams = pickle.load(f)

with open(RAGA_BIGRAM, 'rb') as f:
    raga_bigrams = pickle.load(f)

print(f"  ✓ Loaded {len(makam_bigrams)} makam bigrams")
print(f"  ✓ Loaded {len(raga_bigrams)} raga bigrams")

def symmetric_kl_div(P, Q):
    """Symmetric KL-divergence between two distributions."""
    eps = 1e-10
    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)
    kl_pq = np.sum(P * np.log(P / Q))
    kl_qp = np.sum(Q * np.log(Q / P))
    return (kl_pq + kl_qp) / 2.0

def wasserstein_distance(P, Q):
    """Wasserstein distance (EMD) treating pitch classes as positions on a ring."""
    from scipy.stats import wasserstein_distance as wd
    return wd(P, Q)

def cosine_similarity_matrices(M1, M2):
    """Cosine similarity between two flattened transition matrices."""
    v1 = M1.flatten()
    v2 = M2.flatten()
    # Handle zero vectors
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return 1.0 - cosine(v1, v2)  # 1 - cosine_distance = cosine_similarity

# Compute pairwise distances
print("\nComputing pairwise distances...")
makam_names = sorted(makam_profiles.keys())
raga_names = sorted(raga_profiles.keys())

kl_div_matrix = np.zeros((len(makam_names), len(raga_names)))
wasserstein_matrix = np.zeros((len(makam_names), len(raga_names)))
cosine_matrix = np.zeros((len(makam_names), len(raga_names)))

for i, makam in enumerate(makam_names):
    for j, raga in enumerate(raga_names):
        P_makam = makam_profiles[makam]
        P_raga = raga_profiles[raga]
        
        kl_div_matrix[i, j] = symmetric_kl_div(P_makam, P_raga)
        wasserstein_matrix[i, j] = wasserstein_distance(P_makam, P_raga)

        # Bigram similarity (only if both have bigrams)
        if makam in makam_bigrams and raga in raga_bigrams:
            M_makam = makam_bigrams[makam]
            M_raga = raga_bigrams[raga]
            cosine_matrix[i, j] = cosine_similarity_matrices(M_makam, M_raga)
    
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i + 1} / {len(makam_names)} makams")

print(f"  ✓ Computed {len(makam_names) * len(raga_names)} pairwise distances")

# Save distance matrices
np.save(OUTPUT_DIR / "kl_div_matrix.npy", kl_div_matrix)
np.save(OUTPUT_DIR / "wasserstein_matrix.npy", wasserstein_matrix)
np.save(OUTPUT_DIR / "cosine_matrix.npy", cosine_matrix)

# Save as CSV for inspection
with open(OUTPUT_DIR / "kl_div_matrix.csv", 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['makam'] + raga_names)
    writer.writeheader()
    for i, makam in enumerate(makam_names):
        row = {'makam': makam}
        for j, raga in enumerate(raga_names):
            row[raga] = f"{kl_div_matrix[i, j]:.6f}"
        writer.writerow(row)

print(f"\n✓ Distance metrics computed and saved:")
print(f"  - kl_div_matrix.npy / .csv")
print(f"  - wasserstein_matrix.npy")
print(f"  - cosine_matrix.npy")

# Statistics
print(f"\nDistance statistics:")
print(f"  KL-divergence: min={kl_div_matrix.min():.4f}, max={kl_div_matrix.max():.4f}, mean={kl_div_matrix.mean():.4f}")
print(f"  Wasserstein: min={wasserstein_matrix.min():.4f}, max={wasserstein_matrix.max():.4f}, mean={wasserstein_matrix.mean():.4f}")
print(f"  Cosine similarity: min={cosine_matrix.min():.4f}, max={cosine_matrix.max():.4f}, mean={cosine_matrix.mean():.4f}")

# Save index mappings
with open(OUTPUT_DIR / "makam_index.csv", 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['idx', 'makam'])
    for idx, makam in enumerate(makam_names):
        writer.writerow([idx, makam])

with open(OUTPUT_DIR / "raga_index.csv", 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['idx', 'raga'])
    for idx, raga in enumerate(raga_names):
        writer.writerow([idx, raga])

print(f"\n✓ Distance metric computation complete")
