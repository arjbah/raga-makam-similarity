#!/usr/bin/env python3
"""
Task 4: Tonic Alignment & Pitch-Class Normalization
Normalizes all sequences to tonic=0 for cross-cultural comparison.
"""

import csv
from pathlib import Path
import numpy as np

MAKAM_INPUT = Path(__file__).parent.parent / "data" / "processed" / "makam_seqs.csv"
RAGA_INPUT = Path(__file__).parent.parent / "data" / "processed" / "raga_seqs.csv"

print("Normalizing pitch sequences to tonic-relative space (tonic=0)...")

def normalize_sequence(pitch_seq_str):
    """
    Normalize a pitch sequence to tonic-relative space.
    First note or most frequent low note becomes 0.
    """
    if not pitch_seq_str:
        return ""
    
    # Parse pitch sequence
    pitch_seq = [int(ch) for ch in pitch_seq_str if ch.isdigit()]
    
    if not pitch_seq:
        return ""
    
    # Use first note as tonic (most common approach)
    tonic = pitch_seq[0]
    
    # Normalize: (pc - tonic) % 12
    normalized = [(pc - tonic) % 12 for pc in pitch_seq]
    
    return ''.join(map(str, normalized))

# Process Turkish MIDI sequences
print(f"\nProcessing: {MAKAM_INPUT.name}")
makam_data = []
with open(MAKAM_INPUT, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    makam_data = list(reader)

for row in makam_data:
    row['pitch_seq_normalized_12edo'] = normalize_sequence(row['pitch_sequence_12edo'])
    row['pitch_seq_normalized_53edo'] = normalize_sequence(row['pitch_sequence_53edo'])

# Save
with open(MAKAM_INPUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=makam_data[0].keys())
    writer.writeheader()
    writer.writerows(makam_data)

print(f"  ✓ Normalized {len(makam_data)} Turkish sequences")

# Process Indian raga sequences
print(f"\nProcessing: {RAGA_INPUT.name}")
raga_data = []
with open(RAGA_INPUT, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    raga_data = list(reader)

for row in raga_data:
    row['pitch_seq_normalized'] = normalize_sequence(row['pitch_sequence'])

# Save
with open(RAGA_INPUT, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=raga_data[0].keys())
    writer.writeheader()
    writer.writerows(raga_data)

print(f"  ✓ Normalized {len(raga_data)} Indian sequences")

# Sanity checks
print(f"\nSanity checks:")
print(f"  Turkish sequences with normalized form: {sum(1 for d in makam_data if d.get('pitch_seq_normalized_12edo'))}")
print(f"  Indian sequences with normalized form: {sum(1 for d in raga_data if d.get('pitch_seq_normalized'))}")

# Sample
print(f"\n  Sample normalizations:")
print(f"    Turkish (Makam {makam_data[0]['makam']}):")
print(f"      Original: {makam_data[0]['pitch_sequence_12edo'][:50]}...")
print(f"      Normalized: {makam_data[0]['pitch_seq_normalized_12edo'][:50]}...")
print(f"    Indian (Raga {raga_data[0]['raga']}):")
print(f"      Original: {raga_data[0]['pitch_sequence'][:50]}...")
print(f"      Normalized: {raga_data[0]['pitch_seq_normalized'][:50]}...")

print(f"\n✓ Tonic normalization complete")
