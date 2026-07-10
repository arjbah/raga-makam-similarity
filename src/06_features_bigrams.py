#!/usr/bin/env python3
"""
Task 6: Compute Bigram Transition Matrices
Second-order feature: note-to-note transition probabilities.
"""

import csv
import pickle
from pathlib import Path
import numpy as np
from collections import defaultdict

MAKAM_INPUT = Path(__file__).parent.parent / "data" / "processed" / "makam_seqs.csv"
RAGA_INPUT = Path(__file__).parent.parent / "data" / "processed" / "raga_seqs.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

print("Computing bigram transition matrices (second-order features)...\n")

def get_bigram_matrix(pitch_seq_str):
    """Return 12x12 transition probability matrix from pitch sequence."""
    if not pitch_seq_str or len(pitch_seq_str) < 2:
        return np.zeros((12, 12))
    
    pitch_seq = [int(ch) for ch in pitch_seq_str if ch.isdigit()]
    if len(pitch_seq) < 2:
        return np.zeros((12, 12))
    
    # Build transition matrix
    mat = np.zeros((12, 12))
    for i in range(len(pitch_seq) - 1):
        from_pc = pitch_seq[i]
        to_pc = pitch_seq[i + 1]
        mat[from_pc, to_pc] += 1
    
    # Normalize rows to probabilities (avoid division by zero)
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = np.divide(mat, row_sums, where=(row_sums > 0), out=np.zeros_like(mat))
    
    return mat

# Process Turkish makams
print("Processing Turkish makams...")
makam_data = []
with open(MAKAM_INPUT, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    makam_data = list(reader)

makam_bigrams = []
for row in makam_data:
    mat = get_bigram_matrix(row['pitch_seq_normalized_12edo'])
    makam_bigrams.append(mat)

print(f"  ✓ Computed bigrams for {len(makam_data)} Turkish compositions")

# Process Indian ragas
print("Processing Indian ragas...")
raga_data = []
with open(RAGA_INPUT, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    raga_data = list(reader)

raga_bigrams = []
for row in raga_data:
    mat = get_bigram_matrix(row['pitch_seq_normalized'])
    raga_bigrams.append(mat)

print(f"  ✓ Computed bigrams for {len(raga_data)} Indian compositions")

# Aggregate per makam/raga
print("\nAggregating per makam/raga...")

makam_profiles_bigram = defaultdict(list)
for row, bigram in zip(makam_data, makam_bigrams):
    makam_profiles_bigram[row['makam']].append(bigram)

makam_bigram_agg = {}
for makam, bigrams in makam_profiles_bigram.items():
    makam_bigram_agg[makam] = np.mean(bigrams, axis=0)

raga_profiles_bigram = defaultdict(list)
for row, bigram in zip(raga_data, raga_bigrams):
    raga_profiles_bigram[row['raga']].append(bigram)

raga_bigram_agg = {}
for raga, bigrams in raga_profiles_bigram.items():
    raga_bigram_agg[raga] = np.mean(bigrams, axis=0)

# Save as pickle (more efficient than CSV for matrices)
makam_bigram_file = OUTPUT_DIR / "makam_bigrams.pkl"
with open(makam_bigram_file, 'wb') as f:
    pickle.dump(makam_bigram_agg, f)

raga_bigram_file = OUTPUT_DIR / "raga_bigrams.pkl"
with open(raga_bigram_file, 'wb') as f:
    pickle.dump(raga_bigram_agg, f)

print(f"  ✓ Makam bigrams: {len(makam_bigram_agg)} makams → {makam_bigram_file.name}")
print(f"  ✓ Raga bigrams: {len(raga_bigram_agg)} ragas → {raga_bigram_file.name}")

# Statistics
print(f"\nBigram statistics:")
print(f"  Turkish makams: {len(makam_bigram_agg)}")
print(f"  Indian ragas: {len(raga_bigram_agg)}")

# Sample bigram for inspection
print(f"\n  Sample bigram matrix (Makam {list(makam_bigram_agg.keys())[0]}):")
sample_bigram = list(makam_bigram_agg.values())[0]
print(f"    Shape: {sample_bigram.shape}")
print(f"    Non-zero entries: {np.count_nonzero(sample_bigram)}")
print(f"    Row sums (should be ~1.0 or 0):")
row_sums = sample_bigram.sum(axis=1)
print(f"      Min: {row_sums.min():.4f}, Max: {row_sums.max():.4f}, Mean: {row_sums.mean():.4f}")

print(f"\n✓ Bigram transition matrix extraction complete")
