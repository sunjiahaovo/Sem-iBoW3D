# Data Preparation

Sem-iBoW3D separates the online place-recognition code from feature extraction and semantic segmentation. Before running the C++ executable, prepare point clouds, local descriptors, ground-truth poses, loop ids, and keypoint semantic labels.

## Expected Files

For a dataset `KITTI` and sequence `05`, the default layout is:

```text
data/KITTI/05/GTposes.csv
data/KITTI/05/loop_lst.csv
res_data/KITTI/05/D3F_allpoints/0.ply
descriptor_txt/D3F/KITTI/05/descriptors_0.txt
feature_txt/D3F/KITTI/05/descriptors_0.txt
res_data/KITTI/05/D3F_keypoints_label_reset_SF/0.txt
```

`GTposes.csv` has one pose per line. The executable reads the first three comma-separated values as `x,y,z`.

`loop_lst.csv` stores one query frame id per line. A query frame in this file is counted as a false negative if the system returns no loop.

`D3F_allpoints/<frame>.ply` stores the downsampled point cloud used by Open3D registration.

`descriptor_txt/D3F/<dataset>/<seq>/descriptors_<frame>.txt` stores keypoint descriptors. The default setting expects 20 rows and 32 space-separated float values per row.

`feature_txt/D3F/<dataset>/<seq>/descriptors_<frame>.txt` stores all-point descriptors aligned with the points in `D3F_allpoints/<frame>.ply`.

`D3F_keypoints_label_reset_SF/<frame>.txt` stores one integer semantic class per keypoint. The default setting expects 20 rows. Static classes should be `0..13`; ignored or dynamic classes should be `-1`.

Descriptor parsing is strict. Each descriptor row must contain exactly `--feature-dim` finite floating-point values; missing values, extra values, non-numeric tokens, `NaN`, and `Inf` are rejected with a file and line number. Semantic-label parsing is also strict: each row must contain exactly one integer, and the only valid values are `-1` or `[0, --semantic-num - 1]`.

## KITTI-360 Defaults

For KITTI-360, `--project-root` defaults to the following pose and loop files:

```text
data/KITTI360/2013_05_28_drive_00<seq>_sync/GTposes.csv
data/KITTI360/2013_05_28_drive_00<seq>_sync/loop_lst.csv
```

The processed point clouds, descriptors, and keypoint labels still use:

```text
res_data/KITTI360/<seq>/D3F_allpoints/
descriptor_txt/D3F/KITTI360/<seq>/
feature_txt/D3F/KITTI360/<seq>/
res_data/KITTI360/<seq>/D3F_keypoints_label_reset_SF/
```

## Semantic Label Tools

Use `tools/assign_keypoint_labels.py` to assign raw per-point semantic labels to extracted keypoints:

```bash
python3 tools/assign_keypoint_labels.py \
  --raw-scan-dir /path/to/sequences/05/velodyne \
  --semantic-label-dir /path/to/semantic_predictions/05 \
  --keypoint-dir /path/to/res_data/KITTI/05/D3F_keypoints \
  --output-dir /path/to/res_data/KITTI/05/D3F_keypoints_label_SF \
  --raw-padding 6
```

Then use `tools/remap_semantic_labels.py` to merge labels into the static semantic categories used by Sem-iBoW3D:

```bash
python3 tools/remap_semantic_labels.py \
  --input-dir /path/to/res_data/KITTI/05/D3F_keypoints_label_SF \
  --output-dir /path/to/res_data/KITTI/05/D3F_keypoints_label_reset_SF \
  --map-yaml configs/semantic_label_map.yaml \
  --map-key reset_map_kitti360 \
  --discard-value 14 \
  --discard-to -1 \
  --expected-rows 20
```

The default `reset_map_kitti360` maps labels to 15 merged classes and the example above discards the merged dynamic class `14`, producing the 14 static classes used by the C++ code.

## Overriding Paths

Every data path can be overridden:

```bash
./build/sem_ibow3d \
  --dataset KITTI \
  --sequence 05 \
  --scan-dir /path/to/ply \
  --pose-file /path/to/GTposes.csv \
  --loop-file /path/to/loop_lst.csv \
  --key-feature-dir /path/to/key_descriptors \
  --all-feature-dir /path/to/all_descriptors \
  --label-dir /path/to/keypoint_labels \
  --result-dir results/custom
```

This is useful when keeping datasets outside the source checkout.

## Reproducibility Scope

The repository intentionally does not ship datasets, predicted semantic labels, or precomputed descriptors. The included CTest fixtures verify the public code paths and input contracts with synthetic data, including a registration-path smoke test. Reproducing paper P/R/F1 tables requires the original prepared datasets, descriptor extraction settings, semantic predictions or annotations, and loop ground truth.
