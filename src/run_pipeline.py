#!/usr/bin/env python3
"""
Master pipeline script for Makam-Raga cross-cultural modal comparison.
Executes all 11 tasks sequentially to produce paper-ready output.
"""

import os
import sys
import time
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SRC_DIR = PROJECT_ROOT / "src"

# Create output dirs
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "metadata").mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "paper" / "figures").mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("MAKAM-RAGA CROSS-CULTURAL MODAL COMPARISON PIPELINE")
print("=" * 80)
print(f"Project Root: {PROJECT_ROOT}")
print(f"Data Dir: {PROCESSED_DIR}")
print()

def run_task(task_num, task_name, script_path):
    """Run a task script and report status."""
    print(f"\n{'='*80}")
    print(f"TASK {task_num}: {task_name}")
    print(f"{'='*80}")
    start = time.time()
    
    try:
        # Import and run the module
        spec = __import__('importlib.util').util.spec_from_file_location(f"task_{task_num}", script_path)
        module = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        elapsed = time.time() - start
        print(f"✓ Task {task_num} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"✗ Task {task_num} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# Define task sequence
tasks = [
    (1, "Extract Turkish MIDI sequences", SRC_DIR / "01_ingest_midi.py"),
    (2, "Parse Indian notation TSVs", SRC_DIR / "02_ingest_notation.py"),
    (3, "Normalize raga names", SRC_DIR / "03_normalize_ragas.py"),
    (4, "Tonic alignment & normalization", SRC_DIR / "04_normalize_tonic.py"),
    (5, "Compute pitch-class histograms", SRC_DIR / "05_features_histograms.py"),
    (6, "Compute bigram transition matrices", SRC_DIR / "06_features_bigrams.py"),
    (7, "Compute distance metrics", SRC_DIR / "07_distance_metrics.py"),
    (8, "Generate similarity matrix", SRC_DIR / "08_similarity_matrix.py"),
    (9, "Statistical hypothesis testing", SRC_DIR / "09_hypothesis_testing.py"),
    (10, "Create visualizations", SRC_DIR / "10_visualizations.py"),
    (11, "Generate LaTeX paper", SRC_DIR / "11_generate_paper.py"),
]

# Execute pipeline
failed_tasks = []
for task_num, task_name, script_path in tasks:
    if script_path.exists():
        if not run_task(task_num, task_name, script_path):
            failed_tasks.append((task_num, task_name))
    else:
        print(f"\n✗ Task {task_num}: Script not found at {script_path}")
        failed_tasks.append((task_num, task_name))

# Summary
print("\n" + "=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)
if failed_tasks:
    print(f"✗ {len(failed_tasks)} task(s) failed:")
    for task_num, task_name in failed_tasks:
        print(f"  - Task {task_num}: {task_name}")
    sys.exit(1)
else:
    print("✓ All tasks completed successfully!")
    print(f"\nOutput files:")
    for f in sorted(PROCESSED_DIR.glob("*.csv")):
        print(f"  - {f.name}")
    print("\nFigures generated:")
    for f in sorted((PROJECT_ROOT / "paper" / "figures").glob("*.pdf")):
        print(f"  - {f.name}")
    print("\nPaper generated:")
    paper_file = PROJECT_ROOT / "paper" / "main.tex"
    if paper_file.exists():
        print(f"  - {paper_file.name}")
    print("\n✓ Pipeline complete. Ready for paper compilation.")
