#!/usr/bin/env python3
"""
Task 3: Normalize Raga Names
Standardizes raga names across sources (phonetic variants, prefixes, etc.)
"""

import csv
from pathlib import Path
import re

INPUT_FILE = Path(__file__).parent.parent / "data" / "processed" / "raga_seqs.csv"
OUTPUT_FILE = INPUT_FILE  # Overwrite

# Normalization rules: map variants to canonical names
RAGA_NORMALIZATION = {
    # Hindustani ragas (canonical names)
    'bhoop': 'bhoopali',
    'bhupali': 'bhoopali',
    'bhoopali': 'bhoopali',
    'bhup': 'bhoopali',
    
    'yaman': 'yaman',
    'yaman_kalyan': 'yaman',
    
    'bhairavi': 'bhairavi',
    'bhairvi': 'bhairavi',
    'bhairav': 'bhairav',
    'bhairavbahar': 'bhairavbahar',
    
    'kafi': 'kafi',
    'kafi_kanada': 'kafi',
    
    'malkauns': 'malkauns',
    'malkosh': 'malkauns',
    
    'rast': 'rast',
    'rast_bahar': 'rast',
    
    'khamaj': 'khamaj',
    
    'durga': 'durga',
    
    'bhimpalasi': 'bhimpalasi',
    'bhimpalasy': 'bhimpalasi',
    
    'desh': 'desh',
    
    # Carnatic ragas (will be prefixed with "C_")
    'arabhi': 'arabhi',
    'devamanohari': 'devamanohari',
    'sankarabharanam': 'sankarabharanam',
    'kalyani': 'kalyani',
    'dheerashankarabharanam': 'dheerashankarabharanam',
}

def normalize_raga_name(raga_name, source):
    """Normalize a raga name based on source and phonetic variants."""
    if not raga_name:
        return 'unknown'
    
    raga_name = raga_name.strip().lower()
    
    # Carnatic ragas: prefix with "C_"
    if source in ['carnaticnotations', 'swarasindhu', 'patantara', 'shivkumar']:
        # Check if already prefixed
        if raga_name.startswith('c_'):
            return raga_name
        else:
            return f"C_{raga_name}"
    
    # Hindustani: check normalization map
    if raga_name in RAGA_NORMALIZATION:
        return RAGA_NORMALIZATION[raga_name]
    
    # Try substring matching for Carnatic with C_ prefix
    for source in ['carnaticnotations', 'swarasindhu', 'patantara', 'shivkumar']:
        if source in raga_name.lower():
            return f"C_{raga_name.replace(source, '').strip()}"
    
    # Default: return as-is (lowercase)
    return raga_name

print(f"Reading from: {INPUT_FILE}")

raga_data = []
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    raga_data = list(reader)

print(f"Loaded {len(raga_data)} sequences")

# Normalize raga names
for row in raga_data:
    original_raga = row['raga']
    normalized_raga = normalize_raga_name(row['raga'], row['source'])
    row['raga'] = normalized_raga
    row['raga_original'] = original_raga

# Save back
keys = raga_data[0].keys()
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(raga_data)

# Statistics
unique_ragas = set(d['raga'] for d in raga_data)
print(f"\n✓ Normalized raga names")
print(f"  Total sequences: {len(raga_data)}")
print(f"  Unique ragas: {len(unique_ragas)}")
print(f"\n  Top 15 ragas by frequency:")
from collections import Counter
raga_counts = Counter(d['raga'] for d in raga_data)
for raga, count in raga_counts.most_common(15):
    print(f"    - {raga}: {count}")
