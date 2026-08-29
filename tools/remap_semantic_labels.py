#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm


def load_mapping(path: Path, key: str):
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if key not in data:
        raise KeyError(f"mapping key not found in {path}: {key}")
    return {int(k): int(v) for k, v in data[key].items()}


def read_labels(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        labels = np.load(path)
    else:
        labels = np.loadtxt(path, dtype=int)
    return np.asarray(labels, dtype=int).reshape(-1)


def main():
    parser = argparse.ArgumentParser(description="Remap keypoint semantic labels.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--map-yaml", default=Path("configs/semantic_label_map.yaml"), type=Path)
    parser.add_argument("--map-key", default="reset_map_kitti360")
    parser.add_argument("--input-ext", default=".txt", choices=[".txt", ".npy"])
    parser.add_argument("--discard-value", type=int, default=14)
    parser.add_argument("--discard-to", type=int, default=-1)
    parser.add_argument("--expected-rows", type=int, default=0)
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(args.map_yaml, args.map_key)
    files = sorted(args.input_dir.glob(f"*{args.input_ext}"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"no {args.input_ext} files found in {args.input_dir}")

    moving_count = 0
    total_count = 0
    for path in tqdm(files, desc="remap labels"):
        labels = read_labels(path)
        if args.expected_rows and labels.size != args.expected_rows:
            raise ValueError(f"{path} has {labels.size} labels, expected {args.expected_rows}")

        remapped = []
        for label in labels:
            if int(label) not in mapping:
                raise KeyError(f"label {int(label)} in {path} is not present in mapping {args.map_key}")
            new_label = mapping[int(label)]
            if new_label == args.discard_value:
                new_label = args.discard_to
                moving_count += 1
            remapped.append(new_label)
            total_count += 1

        output_path = args.output_dir / f"{path.stem}.txt"
        np.savetxt(output_path, np.asarray(remapped, dtype=int), fmt="%d")

    print(f"discarded_or_reassigned: {moving_count}/{total_count}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
