#!/usr/bin/env python3
"""
Task 2: Parse Indian Notation TSVs
Extracts raga notation sequences from TSV files and converts to pitch classes.
"""

import csv
import glob
from pathlib import Path
import numpy as np

DATA_INDIAN = Path(__file__).parent.parent / "data_indian"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "processed" / "raga_seqs.csv"

# SwarLipi mapping (from InRaMu paper §VII)
# Base swaras: s=0, r=2, g=4, m=5, p=7, d=9, n=11
# Uppercase variants add +1 semitone
SWARA_MAP = {
    's': 0, 'S': 0,     # Sa
    'r': 2, 'R': 3,     # Re / Re#
    'g': 4, 'G': 5,     # Ga / Ga#
    'm': 5, 'M': 6,     # Ma / Ma# (Teevra)
    'p': 7, 'P': 7,     # Pa (unchanged)
    'd': 9, 'D': 10,    # Dha / Dha#
    'n': 11, 'N': 12,   # Ni / Ni#
}

print(f"Reading Indian notation files from: {DATA_INDIAN}")
print(f"Output: {OUTPUT_FILE}")

raga_data = []
seq_id = 0
errors = []

for tsv_file in sorted(glob.glob(str(DATA_INDIAN / "*.tsv"))):
    source_name = Path(tsv_file).stem
    
    # Skip non-data files
    if source_name.startswith('analysis') or source_name == 'README':
        continue
    
    print(f"  Processing {source_name}.tsv...")
    
    try:
        with open(tsv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row_num, row in enumerate(reader):
                try:
                    # Extract raga name
                    raga = None
                    for raga_field in ['raga', 'orig_raga', 'Raga']:
                        if raga_field in row and row[raga_field]:
                            raga = row[raga_field].strip()
                            break
                    
                    if not raga:
                        continue
                    
                    # Extract notation sequence
                    notation = None
                    for notation_field in ['seq', 'swaras', 'orig_seq', 'notation']:
                        if notation_field in row and row[notation_field]:
                            notation = row[notation_field].strip()
                            break
                    
                    if not notation or len(notation) < 3:
                        continue
                    
                    # Parse notation to pitch classes
                    # Remove octave markers (. and ')
                    notation_clean = notation.replace('.', '').replace("'", '')
                    
                    # Map swaras to pitch classes
                    pitch_classes = []
                    for ch in notation_clean:
                        if ch in SWARA_MAP:
                            pc = SWARA_MAP[ch] % 12  # Normalize to 12 semitones
                            pitch_classes.append(pc)
                        elif ch == '-' or ch == ' ':
                            # Duration/separator; skip
                            continue
                    
                    if not pitch_classes:
                        continue
                    
                    # Extract composition type
                    comp_type = row.get('composition_type', row.get('Composition Type', 'unknown'))
                    
                    # Extract tala
                    tala = row.get('tala', row.get('Tala', 'unknown'))
                    
                    raga_data.append({
                        'raga': raga,
                        'source': source_name,
                        'seq_id': seq_id,
                        'pitch_sequence': ''.join(map(str, pitch_classes)),
                        'n_notes': len(pitch_classes),
                        'composition_type': comp_type if comp_type else 'unknown',
                        'tala': tala if tala else 'unknown',
                        'raw_notation': notation[:100]  # Store first 100 chars for reference
                    })
                    
                    seq_id += 1
                
                except Exception as e:
                    errors.append((source_name, row_num, str(e)))
    
    except Exception as e:
        print(f"    ERROR reading file: {e}")
        errors.append((tsv_file, "file", str(e)))

# Save to CSV
if raga_data:
    keys = raga_data[0].keys()
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(raga_data)
    
    print(f"\n✓ Extracted {len(raga_data)} raga sequences")
    print(f"  Unique ragas: {len(set(d['raga'] for d in raga_data))}")
    print(f"  Unique sources: {len(set(d['source'] for d in raga_data))}")
    print(f"  Average sequence length: {np.mean([d['n_notes'] for d in raga_data]):.1f} notes")
    
    # Show breakdown by source
    print(f"\n  Breakdown by source:")
    for source in sorted(set(d['source'] for d in raga_data)):
        count = sum(1 for d in raga_data if d['source'] == source)
        print(f"    - {source}: {count} compositions")
    
    if errors:
        print(f"\n⚠ Encountered {len(errors)} parse errors (skipped)")
else:
    print("✗ No sequences extracted!")
    exit(1)
