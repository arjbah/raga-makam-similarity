#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def balanced_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        recall = tp / max(tp + fn, 1)
        scores.append(recall)
    return float(np.mean(scores)) if scores else 0.0


def evaluate_feature_set(X, y, feature_names):
    n = len(y)
    preds = []
    truths = []
    for i in range(n):
        train_idx = np.arange(n) != i
        test_x = X[i]
        train_x = X[train_idx]
        train_y = y[train_idx]

        train_mean = train_x.mean(axis=0)
        train_std = train_x.std(axis=0)
        train_std = np.where(train_std < 1e-8, 1.0, train_std)
        train_z = (train_x - train_mean) / train_std
        test_z = (test_x - train_mean) / train_std

        centroids = []
        for label in np.unique(train_y):
            class_rows = train_z[train_y == label]
            centroids.append(class_rows.mean(axis=0))
        centroids = np.vstack(centroids)
        labels = np.unique(train_y)
        distances = np.sum((centroids - test_z[None, :]) ** 2, axis=1)
        pred = labels[np.argmin(distances)]
        preds.append(pred)
        truths.append(y[i])
    preds = np.asarray(preds)
    truths = np.asarray(truths)
    acc = float(np.mean(preds == truths))
    bacc = balanced_accuracy(truths, preds)
    return {"accuracy": acc, "balanced_accuracy": bacc, "n": int(n)}


def main():
    df = pd.read_csv(RESULTS_DIR / "features.csv")
    df = df[df["form"].isin(["sazsemaisi", "turku"])].copy()
    labels = (df["form"] == "sazsemaisi").astype(int).to_numpy()

    feature_sets = {
        "rhythm_only": ["rhythmic_complexity", "phase_entropy"],
        "melody_only": ["pitch_class_entropy", "note_density"],
        "combined": ["rhythmic_complexity", "note_density", "pitch_class_entropy", "phase_entropy"],
        "with_controls": ["rhythmic_complexity", "note_density", "pitch_class_entropy", "phase_entropy", "duration_quarters", "n_notes"],
    }

    results = {}
    for name, cols in feature_sets.items():
        X = df[cols].astype(float).to_numpy()
        results[name] = evaluate_feature_set(X, labels, cols)

    output_path = RESULTS_DIR / "predictive_checks.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
