#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import math
import mido
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
SYMBTR_DIR = ROOT / "symbtr"
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def entropy_from_counts(values):
    if len(values) == 0:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def parse_filename(path: Path):
    stem = path.stem
    parts = stem.split("--")
    makam = parts[0] if len(parts) > 0 else "unknown"
    form = parts[1] if len(parts) > 1 else "unknown"
    usul = parts[2] if len(parts) > 2 else "unknown"
    title = parts[3] if len(parts) > 3 else ""
    composer = parts[4] if len(parts) > 4 else ""
    return {
        "makam": makam,
        "form": form,
        "usul": usul,
        "title": title,
        "composer": composer,
    }


def extract_features(path: Path):
    mid = mido.MidiFile(str(path))
    current_time = 0
    pitches = []
    onset_times = []
    active_notes = {}

    for msg in mido.merge_tracks(mid.tracks):
        current_time += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            onset_times.append(current_time)
            pitches.append(msg.note % 12)
            active_notes[msg.note] = current_time
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active_notes:
                active_notes.pop(msg.note)

    if not pitches:
        return None

    melodic_entropy = entropy_from_counts(pitches)
    note_density = len(pitches) / max(current_time, 1)
    pitch_range = max(pitches) - min(pitches)

    if len(onset_times) >= 2:
        iois = np.diff(onset_times)
        iois = iois[iois > 0]
        if len(iois) > 1:
            bins = np.linspace(0, max(iois), 8)
            hist, _ = np.histogram(iois, bins=bins)
            hist = hist.astype(float) + 1e-9
            probs = hist / hist.sum()
            rhythmic_complexity = -np.sum(probs * np.log2(probs))
        else:
            rhythmic_complexity = 0.0
    else:
        rhythmic_complexity = 0.0

    return {
        "melodic_entropy": melodic_entropy,
        "rhythmic_complexity": rhythmic_complexity,
        "note_density": note_density,
        "pitch_range": pitch_range,
        "num_notes": len(pitches),
        "total_ticks": current_time,
    }


def select_sample_files():
    files = sorted(SYMBTR_DIR.rglob("*.mid"))
    pesrev_files = [p for p in files if "--pesrev--" in p.name]
    sarki_files = [p for p in files if "--sarki--" in p.name]
    return pesrev_files + sarki_files


def build_dataset():
    sample_files = select_sample_files()
    records = []
    for path in sample_files:
        meta = parse_filename(path)
        features = extract_features(path)
        if features is None:
            continue
        row = {"path": str(path), "filename": path.name, **meta, **features}
        records.append(row)

    df = pd.DataFrame(records)
    df = df[df["form"].isin(["pesrev", "sarki"])].copy()
    df["form_binary"] = (df["form"] == "pesrev").astype(int)
    return df


def test_rhythmic_complexity_by_form(df: pd.DataFrame):
    pesrev = df[df["form"] == "pesrev"]["rhythmic_complexity"].to_numpy(dtype=float)
    sarki = df[df["form"] == "sarki"]["rhythmic_complexity"].to_numpy(dtype=float)
    observed_diff = float(pesrev.mean() - sarki.mean())

    combined = np.concatenate([pesrev, sarki])
    rng = np.random.default_rng(42)
    permuted_diffs = []
    for _ in range(2000):
        shuffled = rng.permutation(combined)
        perm_pesrev = shuffled[:len(pesrev)]
        perm_sarki = shuffled[len(pesrev):]
        permuted_diffs.append(float(perm_pesrev.mean() - perm_sarki.mean()))

    p_value = float(np.mean(np.abs(permuted_diffs) >= abs(observed_diff)))
    effect_size = float((pesrev.mean() - sarki.mean()) / np.sqrt((np.var(pesrev, ddof=1) + np.var(sarki, ddof=1)) / 2))
    return observed_diff, p_value, effect_size


def write_outputs(df: pd.DataFrame, result: dict):
    df.to_csv(DATA_DIR / "symbtr_full_features.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="form", y="rhythmic_complexity", palette=["#4c78a8", "#f58518"], ax=ax)
    sns.stripplot(data=df, x="form", y="rhythmic_complexity", color="black", size=4, alpha=0.7, ax=ax)
    ax.set_title("Rhythmic Complexity by Ottoman Form")
    ax.set_xlabel("Form")
    ax.set_ylabel("Rhythmic Complexity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "form_rhythmic_complexity.png", dpi=200)
    plt.close(fig)

    summary = {
        "n_files": len(df),
        "pesrev_mean": float(df.loc[df["form"] == "pesrev", "rhythmic_complexity"].mean()),
        "sarki_mean": float(df.loc[df["form"] == "sarki", "rhythmic_complexity"].mean()),
        "difference": result["difference"],
        "p_value": result["p_value"],
        "effect_size": result["effect_size"],
    }
    with open(DATA_DIR / "summary.json", "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, indent=2)


def main():
    df = build_dataset()
    if df.empty:
        raise RuntimeError("No MIDI files were processed.")

    result = {
        "difference": None,
        "p_value": None,
        "effect_size": None,
    }
    result["difference"], result["p_value"], result["effect_size"] = test_rhythmic_complexity_by_form(df)
    write_outputs(df, result)

    print(f"Processed {len(df)} files")
    print(f"Pesrev mean rhythmic complexity: {df.loc[df['form'] == 'pesrev', 'rhythmic_complexity'].mean():.3f}")
    print(f"Sarki mean rhythmic complexity: {df.loc[df['form'] == 'sarki', 'rhythmic_complexity'].mean():.3f}")
    print(f"Difference (pesrev - sarki): {result['difference']:.3f}")
    print(f"Permutation p-value: {result['p_value']:.3f}")
    print(f"Effect size: {result['effect_size']:.3f}")
    print("Results written to data/ and figures/")


if __name__ == "__main__":
    main()
