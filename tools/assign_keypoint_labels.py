#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign raw per-point semantic labels to extracted keypoints."
    )
    parser.add_argument("--raw-scan-dir", required=True, type=Path)
    parser.add_argument("--semantic-label-dir", required=True, type=Path)
    parser.add_argument("--keypoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--index-json", type=Path, default=None,
                        help="Optional JSON whose keys are raw frame ids to process.")
    parser.add_argument("--raw-padding", type=int, default=6,
                        help="Zero padding for raw .bin frame files, e.g. 6 for KITTI and 10 for KITTI-360.")
    parser.add_argument("--voxel-size", type=float, default=0.30)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--keypoint-index-mode", choices=["frame", "ordinal"], default="frame")
    parser.add_argument("--output-index-mode", choices=["frame", "ordinal"], default="frame")
    return parser.parse_args()


def discover_frames(label_dir: Path, index_json: Path):
    if index_json:
        with index_json.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return [int(k) for k in data.keys()]
    return sorted(int(path.stem) for path in label_dir.glob("*.npy"))


def majority_label(labels: np.ndarray, indices: np.ndarray) -> int:
    valid = [int(labels[idx]) for idx in indices if idx >= 0]
    if not valid:
        return -1
    return Counter(valid).most_common(1)[0][0]


def assign_one(args, frame_id: int, ordinal: int):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "assign_keypoint_labels.py requires the optional Python package open3d. "
            "Install tool dependencies with: pip install -r tools/requirements.txt"
        ) from exc

    raw_path = args.raw_scan_dir / f"{frame_id:0{args.raw_padding}d}.bin"
    label_path = args.semantic_label_dir / f"{frame_id}.npy"
    keypoint_index = ordinal if args.keypoint_index_mode == "ordinal" else frame_id
    keypoint_path = args.keypoint_dir / f"{keypoint_index}.ply"
    output_index = ordinal if args.output_index_mode == "ordinal" else frame_id
    output_path = args.output_dir / f"{output_index}.txt"

    if not raw_path.is_file():
        raise FileNotFoundError(f"raw scan not found: {raw_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"semantic label not found: {label_path}")
    if not keypoint_path.is_file():
        raise FileNotFoundError(f"keypoint cloud not found: {keypoint_path}")

    points = np.fromfile(raw_path, dtype=np.float32).reshape((-1, 4))[:, :3]
    labels = np.asarray(np.load(label_path), dtype=int).reshape(-1)
    if labels.shape[0] != points.shape[0]:
        raise ValueError(
            f"{label_path} has {labels.shape[0]} labels but {raw_path} has {points.shape[0]} points"
        )

    raw_pcd = o3d.geometry.PointCloud()
    raw_pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd, _, trace = raw_pcd.voxel_down_sample_and_trace(
        voxel_size=args.voxel_size,
        min_bound=raw_pcd.get_min_bound(),
        max_bound=raw_pcd.get_max_bound(),
        approximate_class=False,
    )
    down_points = np.asarray(down_pcd.points)
    down_labels = np.asarray([majority_label(labels, np.asarray(group)) for group in trace], dtype=int)

    keypoint_pcd = o3d.io.read_point_cloud(str(keypoint_path))
    keypoints = np.asarray(keypoint_pcd.points)
    if keypoints.size == 0:
        raise ValueError(f"keypoint cloud is empty: {keypoint_path}")
    if down_points.size == 0:
        raise ValueError(f"downsampled raw cloud is empty: {raw_path}")

    nearest = cKDTree(down_points).query(keypoints, k=1)[1]
    keypoint_labels = down_labels[nearest]
    np.savetxt(output_path, keypoint_labels, fmt="%d")


def main():
    args = parse_args()
    for directory in [args.raw_scan_dir, args.semantic_label_dir, args.keypoint_dir]:
        if not directory.is_dir():
            raise FileNotFoundError(f"directory does not exist: {directory}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = discover_frames(args.semantic_label_dir, args.index_json)
    if args.max_frames > 0:
        frames = frames[:args.max_frames]
    if not frames:
        raise RuntimeError("no frames to process")

    for ordinal, frame_id in enumerate(tqdm(frames, desc="assign keypoint labels")):
        assign_one(args, frame_id, ordinal)

    print(f"processed_frames: {len(frames)}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
