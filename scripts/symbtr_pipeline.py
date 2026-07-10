#!/usr/bin/env python3
import json
import math
import re
from pathlib import Path

import mido
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SYMBTR_DIR = ROOT / "symbtr"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"
DOCS_DIR = ROOT / "docs"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

FORM_ALIASES = {
    "saz semaisi": "sazsemaisi",
    "sazsemaisi": "sazsemaisi",
    "saz sema isi": "sazsemaisi",
    "turku": "turku",
    "türkü": "turku",
    "turku": "turku",
    "sarki": "sarki",
    "pesrev": "pesrev",
    "ilahi": "ilahi",
    "beste": "beste",
    "kupe": "kupe",
    "nakis": "nakis",
    "mars": "mars",
}

USUL_ALIASES = {
    "duyek": "duyek",
    "devrikebir": "devrikebir",
    "aksak": "aksak",
    "sofyan": "sofyan",
    "yuruksemai": "yuruksemai",
    "agiraksak": "agiraksak",
    "nimsofyan": "nimsofyan",
    "muhammes": "muhammes",
    "curcuna": "curcuna",
    "nihavend": "nihavend",
}


def normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("ı", "i").replace("ü", "u").replace("ö", "o").replace("ş", "s").replace("ç", "c").replace("ğ", "g")
    return text.strip()


def canonical_form(value):
    label = normalize_text(value)
    if not label:
        return "unknown"
    return FORM_ALIASES.get(label, label)


def canonical_usul(value):
    label = normalize_text(value)
    if not label:
        return "unknown"
    return USUL_ALIASES.get(label, label)


def parse_filename(path):
    stem = path.stem
    parts = [part for part in stem.split("--") if part != ""]
    if len(parts) < 3:
        return {
            "piece_id": stem,
            "makam": "unknown",
            "form": "unknown",
            "usul": "unknown",
            "title": "",
            "composer": "",
        }
    makam = parts[0] if len(parts) > 0 else "unknown"
    form = parts[1] if len(parts) > 1 else "unknown"
    usul = parts[2] if len(parts) > 2 else "unknown"
    title = parts[3] if len(parts) > 3 else ""
    composer = parts[4] if len(parts) > 4 else ""
    return {
        "piece_id": stem,
        "makam": canonical_form(makam) if makam == parts[0] and form == parts[1] else makam,
        "form": canonical_form(form),
        "usul": canonical_usul(usul),
        "title": title,
        "composer": composer,
    }


def parse_filename(path):
    stem = path.stem
    parts = [part for part in stem.split("--") if part != ""]
    if len(parts) < 3:
        return {
            "piece_id": stem,
            "makam": "unknown",
            "form": "unknown",
            "usul": "unknown",
            "title": "",
            "composer": "",
        }
    makam = parts[0]
    form = parts[1]
    usul = parts[2]
    title = parts[3] if len(parts) > 3 else ""
    composer = parts[4] if len(parts) > 4 else ""
    return {
        "piece_id": stem,
        "makam": normalize_text(makam),
        "form": canonical_form(form),
        "usul": canonical_usul(usul),
        "title": title,
        "composer": composer,
    }


def entropy_from_counts(values):
    if len(values) == 0:
        return 0.0
    counts = np.bincount(values)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def extract_midi_features(path):
    mid = mido.MidiFile(str(path))
    ticks_per_beat = max(mid.ticks_per_beat, 1)
    current_tick = 0
    pitches = []
    onset_ticks = []
    has_pitch_bend = False

    for msg in mido.merge_tracks(mid.tracks):
        current_tick += msg.time
        if msg.type == "pitchwheel":
            has_pitch_bend = True
        elif msg.type == "note_on" and msg.velocity > 0:
            onset_ticks.append(current_tick / ticks_per_beat)
            pitches.append(msg.note % 12)

    if not pitches:
        return None

    durations = np.array(onset_ticks)
    duration_quarters = max(float(current_tick / ticks_per_beat), 1e-6)
    if len(durations) >= 2:
        iois = np.diff(durations)
        iois = iois[iois > 1e-6]
        if len(iois) >= 2:
            bins = np.linspace(0.0, max(iois.max(), 1.0), 8)
            hist, _ = np.histogram(iois, bins=bins)
            hist = hist.astype(float) + 1e-9
            probs = hist / hist.sum()
            rhythmic_complexity = float(-np.sum(probs * np.log2(probs)))
        else:
            rhythmic_complexity = 0.0
    else:
        rhythmic_complexity = 0.0

    pitch_classes = np.array(pitches, dtype=int)
    pitch_class_entropy = entropy_from_counts(pitch_classes)
    note_density = float(len(pitch_classes) / max(duration_quarters, 1.0))
    pitch_range = float(max(pitch_classes) - min(pitch_classes)) if len(pitch_classes) > 0 else 0.0
    unique_pitch_count = float(len(np.unique(pitch_classes)))

    if duration_quarters <= 0:
        phase_hist = np.zeros(8)
    else:
        phases = np.mod(np.array(onset_ticks) / duration_quarters, 1.0)
        bins = np.linspace(0.0, 1.0, 9)
        hist, _ = np.histogram(phases, bins=bins)
        hist = hist.astype(float) + 1e-9
        phase_hist = hist / hist.sum()
    phase_entropy = float(-np.sum(phase_hist * np.log2(phase_hist)))

    return {
        "source_format": "midi",
        "duration_quarters": float(duration_quarters),
        "n_notes": int(len(pitch_classes)),
        "pitch_class_entropy": pitch_class_entropy,
        "rhythmic_complexity": rhythmic_complexity,
        "note_density": note_density,
        "pitch_range": pitch_range,
        "unique_pitch_count": unique_pitch_count,
        "phase_entropy": phase_entropy,
        "has_pitch_bend_or_perde": bool(has_pitch_bend),
    }


def build_manifest(path_out=None):
    if path_out is None:
        path_out = RESULTS_DIR / "manifest.csv"
    files = sorted(SYMBTR_DIR.rglob("*.mid"))
    rows = []
    for path in files:
        meta = parse_filename(path)
        feats = extract_midi_features(path)
        if feats is None:
            continue
        row = {"filename": path.name, "path": str(path.relative_to(ROOT)), **meta, **feats}
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No MIDI files could be parsed from the SymbTr directory.")
    df.to_csv(path_out, index=False)
    return df


def build_label_map(path_out=None):
    if path_out is None:
        path_out = RESULTS_DIR / "label_map.csv"
    rows = []
    for key, value in sorted(FORM_ALIASES.items()):
        rows.append({"kind": "form", "raw": key, "canonical": value})
    for key, value in sorted(USUL_ALIASES.items()):
        rows.append({"kind": "usul", "raw": key, "canonical": value})
    pd.DataFrame(rows).to_csv(path_out, index=False)
    return pd.DataFrame(rows)


def read_manifest(path=None):
    if path is None:
        path = RESULTS_DIR / "manifest.csv"
    return pd.read_csv(path)


def compute_feature_table(manifest_path=None, features_path=None):
    if manifest_path is None:
        manifest_path = RESULTS_DIR / "manifest.csv"
    if features_path is None:
        features_path = RESULTS_DIR / "features.csv"
    manifest = read_manifest(manifest_path)
    manifest.to_csv(features_path, index=False)
    return manifest


def permutation_test(values_a, values_b, n_permutations=20000, seed=7):
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    observed = float(a.mean() - b.mean())
    combined = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    permuted = np.empty(int(n_permutations), dtype=float)
    for idx in range(int(n_permutations)):
        shuffled = rng.permutation(combined)
        perm_a = shuffled[:len(a)]
        perm_b = shuffled[len(a):]
        permuted[idx] = float(perm_a.mean() - perm_b.mean())
    count_extreme = int(np.sum(np.abs(permuted) >= abs(observed)))
    p_value = float((count_extreme + 1) / (int(n_permutations) + 1))
    pooled_sd = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    effect_size = float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else 0.0
    return observed, p_value, effect_size


def matched_pairs(df, feature_name="rhythmic_complexity"):
    pairs = []
    a_rows = df[df["form"] == "sazsemaisi"].copy()
    b_rows = df[df["form"] == "turku"].copy()
    used_b = set()
    for _, row_a in a_rows.iterrows():
        candidates = b_rows[~b_rows.index.isin(used_b)]
        if candidates.empty:
            break
        same_makam = candidates[candidates["makam"] == row_a["makam"]]
        if not same_makam.empty:
            row_b = same_makam.iloc[0]
        else:
            same_usul = candidates[candidates["usul"] == row_a["usul"]]
            if not same_usul.empty:
                row_b = same_usul.iloc[0]
            else:
                length_bin = int(float(row_a["duration_quarters"]) // 50)
                candidate_bin = candidates[candidates["duration_quarters"].apply(lambda x: int(float(x)) // 50) == length_bin]
                row_b = candidate_bin.iloc[0] if not candidate_bin.empty else candidates.iloc[0]
        pairs.append({
            "piece_a": row_a["piece_id"],
            "piece_b": row_b["piece_id"],
            "makam": row_a["makam"],
            "usul": row_a["usul"],
            "feature_diff": float(row_a[feature_name] - row_b[feature_name]),
        })
        used_b.add(row_b.name)
    return pd.DataFrame(pairs)


def run_statistics(features_path=None, stats_path=None, matched_path=None):
    if features_path is None:
        features_path = RESULTS_DIR / "features.csv"
    if stats_path is None:
        stats_path = RESULTS_DIR / "stats.json"
    if matched_path is None:
        matched_path = RESULTS_DIR / "matched_pairs.csv"

    df = pd.read_csv(features_path)
    df = df[df["form"].isin(["sazsemaisi", "turku"])].copy()
    df = df.dropna(subset=["rhythmic_complexity", "note_density", "pitch_class_entropy", "phase_entropy"])

    feature_names = ["rhythmic_complexity", "note_density", "pitch_class_entropy", "phase_entropy"]
    stats = {}
    for feature in feature_names:
        a = df.loc[df["form"] == "sazsemaisi", feature].astype(float).dropna().to_numpy()
        b = df.loc[df["form"] == "turku", feature].astype(float).dropna().to_numpy()
        raw_diff, p_value, effect_size = permutation_test(a, b)
        stats[feature] = {
            "n_sazsemaisi": int(len(a)),
            "n_turku": int(len(b)),
            "difference": float(raw_diff),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
        }

    matched = matched_pairs(df)
    if not matched.empty:
        stats["matched_rhythmic_complexity"] = {
            "n_pairs": int(len(matched)),
            "mean_difference": float(matched["feature_diff"].mean()),
            "median_difference": float(matched["feature_diff"].median()),
        }
    matched.to_csv(matched_path, index=False)

    summary = {
        "n_retained": int(len(df)),
        "n_sazsemaisi": int((df["form"] == "sazsemaisi").sum()),
        "n_turku": int((df["form"] == "turku").sum()),
        "features": stats,
    }
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
