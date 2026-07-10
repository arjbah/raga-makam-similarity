#!/usr/bin/env python3
"""
Task 10: Create Visualizations
Generates publication-ready figures for the paper.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
FIGURES_DIR = Path(__file__).parent.parent / "paper" / "figures"

print("Generating visualizations...\n")

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style("whitegrid")


def style_axes(ax):
    """Apply consistent spine, grid, and tick styling."""
    ax.grid(axis='y', alpha=0.25, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#bdbdbd')
    ax.tick_params(width=0.8, color='#6e6e6e')


def load_index(path, key):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row[key] for row in reader]


def load_profiles(path, id_key):
    profiles = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row[id_key]
            profiles[item_id] = np.array([float(row[f'pc_{i}']) for i in range(12)])
    return profiles


composite_distance = np.load(OUTPUT_DIR / "composite_distance.npy")
makam_names = load_index(OUTPUT_DIR / "makam_index.csv", 'makam')
raga_names = load_index(OUTPUT_DIR / "raga_index.csv", 'raga')

# FIGURE 1: Similarity Heatmap
print("Figure 1: Similarity heatmap...")
top_k_makams = 20
top_k_ragas = 25

makam_variance = composite_distance.var(axis=1)
raga_variance = composite_distance.var(axis=0)

top_makam_idx = np.argsort(-makam_variance)[:top_k_makams]
top_raga_idx = np.argsort(-raga_variance)[:top_k_ragas]

sub_distance = composite_distance[np.ix_(top_makam_idx, top_raga_idx)]
sub_makam_names = [makam_names[i] for i in top_makam_idx]
sub_raga_names = [raga_names[i] for i in top_raga_idx]

fig, ax = plt.subplots(figsize=(11.6, 7.2), constrained_layout=True)
im = ax.imshow(sub_distance, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(sub_raga_names)))
ax.set_yticks(range(len(sub_makam_names)))
ax.set_xticklabels(sub_raga_names, rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(sub_makam_names, fontsize=7)
ax.set_xlabel('Raga', fontsize=10)
ax.set_ylabel('Makam', fontsize=10)
ax.set_title('Makam-Raga Similarity Matrix\n(Lower values = Higher similarity)', fontsize=11, fontweight='bold')
for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color('#bdbdbd')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Composite Distance', fontsize=9)
fig.savefig(FIGURES_DIR / "fig1_heatmap.pdf", dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / "fig1_heatmap.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ {FIGURES_DIR / 'fig1_heatmap.pdf'}")

# FIGURE 2: Pitch-Class Distribution Comparisons
print("Figure 2: Pitch-class distributions...")
makam_profiles = load_profiles(OUTPUT_DIR / "makam_profiles.csv", 'makam')
raga_profiles = load_profiles(OUTPUT_DIR / "raga_profiles.csv", 'raga')

known_pairs = []
known_pairs_file = OUTPUT_DIR / "known_pairs_analysis.csv"
if known_pairs_file.exists():
    with open(known_pairs_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        known_pairs = list(reader)[:4]

plot_count = min(4, len(known_pairs))
fig, axes = plt.subplots(1, plot_count, figsize=(11.6, 3.1), sharey=True, constrained_layout=True)
if plot_count == 1:
    axes = [axes]

global_max = 0.0
for pair in known_pairs:
    makam = pair['makam']
    raga = pair['raga']
    if makam in makam_profiles and raga in raga_profiles:
        global_max = max(global_max, makam_profiles[makam].max(), raga_profiles[raga].max())
global_max = global_max * 1.08 if global_max else 1.0

for idx, (ax, pair) in enumerate(zip(axes, known_pairs)):
    makam = pair['makam']
    raga = pair['raga']
    if makam not in makam_profiles or raga not in raga_profiles:
        continue

    x = np.arange(12)
    width = 0.35
    ax.bar(x - width / 2, makam_profiles[makam], width, label='Makam', alpha=0.7, color='steelblue')
    ax.bar(x + width / 2, raga_profiles[raga], width, label='Raga', alpha=0.7, color='coral')
    ax.set_xlabel('Pitch Class', fontsize=9)
    if idx == 0:
        ax.set_ylabel('Frequency', fontsize=9)
    ax.set_title(f'{makam}\nvs. {raga}', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'], fontsize=8)
    ax.set_ylim(0, global_max)
    style_axes(ax)

if plot_count:
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=8, frameon=True, bbox_to_anchor=(0.5, 1.04))

fig.savefig(FIGURES_DIR / "fig2_pitch_overlay.pdf", dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / "fig2_pitch_overlay.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ {FIGURES_DIR / 'fig2_pitch_overlay.pdf'}")

# FIGURE 3: Distribution Statistics
print("Figure 3: Distribution statistics...")
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4), constrained_layout=True)

# 3a: Histogram of all distances
ax = axes[0]
ax.hist(composite_distance.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax.set_xlabel('Composite Distance', fontsize=9)
ax.set_ylabel('Frequency', fontsize=9)
ax.set_title('Distribution of All Pairwise Distances', fontsize=10, fontweight='bold')
style_axes(ax)

# 3b: Top 10 pairs
ax = axes[1]
top_pairs_data = []
top_pairs_file = OUTPUT_DIR / "top_pairs.csv"
if top_pairs_file.exists():
    with open(top_pairs_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        top_pairs_data = list(reader)[:10]

if top_pairs_data:
    pairs_labels = [f"{p.get('display_makam', p['makam'])[:8]}—{p.get('display_raga', p['raga'])[:12]}" for p in top_pairs_data]
    pairs_distances = [float(p['composite_distance']) for p in top_pairs_data]
    ax.barh(pairs_labels, pairs_distances, color='darkgreen', alpha=0.7)
    ax.set_xlabel('Composite Distance', fontsize=9)
    ax.set_title('Top 10 Matching Pairs', fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    style_axes(ax)

# 3c: Known vs Random bar chart
ax = axes[2]
results = {}
results_file = OUTPUT_DIR / "hypothesis_test_results.csv"
if results_file.exists():
    with open(results_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            results[row[0]] = float(row[1])

if results:
    known_mean = results.get('Known Pairs (Mean Distance)', 0)
    random_mean = results.get('Random Pairs (Mean Distance)', 0)
    bars = ax.bar(['Known Pairs', 'Random Pairs'], [known_mean, random_mean], color=['darkgreen', 'lightcoral'], alpha=0.7)
    ax.set_ylabel('Mean Distance', fontsize=9)
    ax.set_title('Known vs. Random Pairs', fontsize=10, fontweight='bold')
    style_axes(ax)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.4f}', ha='center', va='bottom', fontsize=8)

fig.savefig(FIGURES_DIR / "fig3_statistics.pdf", dpi=300, bbox_inches='tight')
fig.savefig(FIGURES_DIR / "fig3_statistics.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ {FIGURES_DIR / 'fig3_statistics.pdf'}")

print(f"\n✓ All visualizations created successfully")
print(f"  Figures saved in: {FIGURES_DIR}")
