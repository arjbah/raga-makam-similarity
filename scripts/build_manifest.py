#!/usr/bin/env python3
from symbtr_pipeline import build_label_map, build_manifest

if __name__ == "__main__":
    build_label_map()
    build_manifest()
    print("Wrote results/manifest.csv and results/label_map.csv")
