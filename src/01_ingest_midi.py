#!/usr/bin/env python3
"""
Task 1: Extract Turkish MIDI Pitch Sequences
Extracts note pitch classes from SymbTr MIDI files and saves to CSV.
"""

import csv
import glob
import re
from pathlib import Path
import numpy as np

try:
    import pretty_midi
except ImportError:
    print("ERROR: pretty_midi not installed. Run: pip install pretty_midi")
    exit(1)

DATA_TURKISH = Path(__file__).parent.parent / "data_turkish"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "processed" / "makam_seqs.csv"

print(f"Reading MIDI files from: {DATA_TURKISH}")
print(f"Output: {OUTPUT_FILE}")

makam_data = []
errors = []
seq_id = 0

for midi_file in sorted(glob.glob(str(DATA_TURKISH / "*.mid"))):
    try:
        # Parse filename: [makam]--[form]--[tala]--[title]--[composer].mid
        fname = Path(midi_file).stem
        match = re.match(r"([^-]+)--([^-]*)--([^-]*)--(.*)--(.+)", fname)
        
        if not match:
            match = re.match(r"([^-]+)--", fname)
            makam_name = match.group(1) if match else "unknown"
            form = "unknown"
        else:
            makam_name = match.group(1)
            form = match.group(2)
        
        # Load MIDI
        pm = pretty_midi.PrettyMIDI(midi_file)
        
        if not pm.instruments:
            continue
        
        # Extract first instrument (assume melody)
        notes = pm.instruments[0].notes
        if not notes:
            continue
        
        # Convert to pitch classes (12-EDO)
        pitch_classes_12 = [int(n.pitch) % 12 for n in notes]
        
        # Also store 53-EDO for future analysis (approximated)
        # 53-EDO: each semitone = 53/12 ≈ 4.42 steps
        pitch_classes_53 = [round((int(n.pitch) * 53) / 12) % 53 for n in notes]
        
        if not pitch_classes_12:
            continue
        
        makam_data.append({
            'makam': makam_name,
            'seq_id': seq_id,
            'pitch_sequence_12edo': ''.join(map(str, pitch_classes_12)),
            'pitch_sequence_53edo': ''.join(map(str, pitch_classes_53)),
            'n_notes': len(pitch_classes_12),
            'form': form,
            'midi_file': Path(midi_file).name
        })
        
        seq_id += 1
        
        if seq_id % 500 == 0:
            print(f"  Processed {seq_id} compositions...")
    
    except Exception as e:
        errors.append((midi_file, str(e)))

# Save to CSV
if makam_data:
    keys = makam_data[0].keys()
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(makam_data)
    
    print(f"\n✓ Extracted {len(makam_data)} makam sequences")
    print(f"  Unique makams: {len(set(d['makam'] for d in makam_data))}")
    print(f"  Average sequence length: {np.mean([d['n_notes'] for d in makam_data]):.1f} notes")
    
    if errors:
        print(f"\n⚠ Skipped {len(errors)} files due to errors")
else:
    print("✗ No sequences extracted!")
    exit(1)
