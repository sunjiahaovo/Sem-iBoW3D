# Sem-iBoW3D

Sem-iBoW3D is the reference implementation for:

```bibtex
@article{sun2026semantic,
  title={Semantic-aided bag-of-words for LiDAR-based place recognition},
  author={Sun, Jiahao and Lin, Yuxiaotong and Zhang, Jian and Yi, Peiqi and Li, Liang},
  journal={Pattern Recognition},
  pages={114335},
  year={2026},
  publisher={Elsevier},
  url={https://www.sciencedirect.com/science/article/pii/S0031320326013002}
}
```

Paper link: https://www.sciencedirect.com/science/article/pii/S0031320326013002

The code implements a semantic-aided bag-of-words pipeline for LiDAR place recognition. It builds a semantic-aware visual word dictionary from local 3D descriptors, retrieves candidate loops with BoW histograms, and verifies loop candidates with feature-based point cloud registration.

## Repository Layout

```text
apps/                    Command line entry point
include/, src/            Sem-iBoW3D C++ core
tools/                    Data preparation utilities
configs/                  Semantic class remapping configuration
docs/                     Data and release notes
examples/                 Example run scripts
```

This release intentionally does not include datasets, extracted descriptors, experiment logs, paper drafts, or third-party source trees.

## Dependencies

Required:

- CMake 3.16 or newer
- A C++17 compiler. GCC 9 or newer is recommended because the code uses C++17 parallel algorithms.
- Eigen 3.4 or newer
- OpenCV
- Open3D C++ SDK, tested with Open3D 0.17.0
- TBB or oneTBB

Optional for Python data utilities:

- Python 3.8+
- numpy
- scipy
- open3d
- PyYAML
- tqdm

## Build

If all dependencies are installed in standard system locations:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If you use a local Open3D/Eigen/TBB installation, pass the prefixes explicitly:

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=g++-9 \
  -DCMAKE_PREFIX_PATH="/path/to/open3d;/path/to/eigen" \
  -DTBB_ROOT=/path/to/tbb \
  -DLIBCXX_RUNTIME_DIR=/path/to/libcxx-runtime
cmake --build build -j
```

`LIBCXX_RUNTIME_DIR` is only needed when your Open3D binary package depends on a non-system `libc++.so.1`.

## Data Layout

By default the executable expects a project root with:

```text
<project-root>/
  data/<dataset>/<seq>/GTposes.csv
  data/<dataset>/<seq>/loop_lst.csv
  res_data/<dataset>/<seq>/D3F_allpoints/<frame>.ply
  descriptor_txt/D3F/<dataset>/<seq>/descriptors_<frame>.txt
  feature_txt/D3F/<dataset>/<seq>/descriptors_<frame>.txt
  res_data/<dataset>/<seq>/D3F_keypoints_label_reset_SF/<frame>.txt
```

For KITTI-360, the default pose and loop paths are:

```text
data/KITTI360/2013_05_28_drive_00<seq>_sync/GTposes.csv
data/KITTI360/2013_05_28_drive_00<seq>_sync/loop_lst.csv
```

All paths can be overridden from the command line. See [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) for details.

## Run

Show options:

```bash
./build/sem_ibow3d --help
```

Run SemanticKITTI sequence 05 using the default layout:

```bash
./build/sem_ibow3d \
  --dataset KITTI \
  --sequence 05 \
  --project-root /path/to/prepared/data \
  --result-dir results/KITTI_05
```

A short smoke test:

```bash
./build/sem_ibow3d \
  --dataset KITTI \
  --sequence 05 \
  --project-root /path/to/prepared/data \
  --max-frames 5 \
  --init-pcd-num 3 \
  --update-num 2 \
  --result-dir results/smoke_KITTI_05 \
  --time-dir results/smoke_KITTI_05/time_info
```

Disable semantic labels:

```bash
./build/sem_ibow3d \
  --dataset KITTI \
  --sequence 05 \
  --project-root /path/to/prepared/data \
  --no-semantic
```

Enable asynchronous BoW dictionary updates:

```bash
./build/sem_ibow3d \
  --dataset KITTI \
  --sequence 05 \
  --project-root /path/to/prepared/data \
  --async-update
```

Default thresholds follow the paper experiments:

| Dataset | fit / fit2 | score / score2 | search_num | max_iter | check_th |
|---|---:|---:|---:|---:|---:|
| SemanticKITTI / KITTI | 0.94 | 1.4 | 5 | 1000 | 1000000 |
| KITTI-360 | 0.90 | 1.6 | 5 | 1000 | 1000000 |

Use `--fit-th`, `--score-th`, `--fit-th2`, `--score-th2`, `--search-num`, `--max-iter`, and `--check-th` to override them.

## Outputs

The result directory contains:

- `results.txt`: run configuration and TP/FP/FN/P/R/F1 values.
- `looplist.txt`: accepted loop pairs as `query_frame, candidate_frame`.
- `false_negative_frames.txt`: GT loop frame ids missed by the method.

If `--time-dir` is set, timing text files are also written for retrieval, candidate selection, geometric verification, registration, and BoW update stages.

## Notes

- The code expects precomputed local descriptors. The paper experiments used D3Feat-style 32D descriptors stored as text files.
- Semantic labels are expected at keypoint level and should be remapped to the 14 static classes used by the method, with dynamic classes marked as `-1`.
- The FGR backend is exposed for experimentation with `--registration-backend fgr`; the default and paper setting is RANSAC.
- The Python utilities are provided for data preparation convenience and are not required to build the C++ executable.

## License

This code is released under the MIT License. Dependencies and datasets are governed by their own licenses.
