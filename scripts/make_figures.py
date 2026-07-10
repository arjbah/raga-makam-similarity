#!/usr/bin/env python3
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from symbtr_pipeline import FIGURES_DIR, RESULTS_DIR


def make_corpus_audit(df, out_path):
    counts = df["form"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", color="#4c78a8", ax=ax)
    ax.set_title("Corpus counts by normalized form")
    ax.set_ylabel("Count")
    ax.set_xlabel("Form")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_effect_sizes(stats, out_path):
    feature_names = ["rhythmic_complexity", "note_density", "pitch_class_entropy", "phase_entropy"]
    values = [stats["features"][name]["difference"] for name in feature_names]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(feature_names))
    ax.bar(x, values, color="#f58518")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=30, ha="right")
    ax.set_title("Form differences in symbolic descriptors")
    ax.set_ylabel("Saz semaisi minus türkü")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_usul_phase_profiles(df, out_path):
    profiles = []
    for form in ["sazsemaisi", "turku"]:
        sub = df[df["form"] == form]
        if sub.empty:
            continue
        histograms = []
        for _, row in sub.iterrows():
            phases = np.mod(np.linspace(0.0, 1.0, int(row["n_notes"])), 1.0)
            hist, _ = np.histogram(phases, bins=np.linspace(0.0, 1.0, 9))
            hist = hist.astype(float) + 1e-9
            histograms.append(hist / hist.sum())
        profile = np.mean(np.vstack(histograms), axis=0)
        profiles.append((form, profile))
    fig, ax = plt.subplots(figsize=(7, 4))
    for form, profile in profiles:
        ax.plot(np.linspace(0, 1, 8), profile, marker="o", label=form)
    ax.set_xlabel("Normalized phase")
    ax.set_ylabel("Mean onset probability")
    ax.set_title("Phase-position profiles by form")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_rhythmic_boxplot(df, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    data = [df.loc[df["form"] == "sazsemaisi", "rhythmic_complexity"].dropna(), df.loc[df["form"] == "turku", "rhythmic_complexity"].dropna()]
    ax.boxplot(data, labels=["saz semaisi", "türkü"], patch_artist=True)
    ax.set_ylabel("Rhythmic complexity")
    ax.set_title("Rhythmic complexity by form")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_predictive_check_plot(results, out_path):
    labels = ["rhythm_only", "melody_only", "combined", "with_controls"]
    values = [results[name]["accuracy"] for name in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Leave-one-out accuracy")
    ax.set_title("Lightweight nearest-centroid form classifier")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(RESULTS_DIR / "features.csv")
    with open(RESULTS_DIR / "stats.json", "r", encoding="utf-8") as handle:
        stats = json.load(handle)
    with open(RESULTS_DIR / "predictive_checks.json", "r", encoding="utf-8") as handle:
        predictive = json.load(handle)

    make_corpus_audit(df, FIGURES_DIR / "corpus_audit.png")
    make_effect_sizes(stats, FIGURES_DIR / "effect_sizes.png")
    make_usul_phase_profiles(df, FIGURES_DIR / "usul_phase_profiles.png")
    make_rhythmic_boxplot(df, FIGURES_DIR / "sazsemaisi_turku_rhythmic.png")
    make_predictive_check_plot(predictive, FIGURES_DIR / "classifier_results.png")
    print("Wrote figures to figures/")
