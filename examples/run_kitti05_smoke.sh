#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/path/to/prepared/data}"

./build/sem_ibow3d \
  --dataset KITTI \
  --sequence 05 \
  --project-root "${PROJECT_ROOT}" \
  --max-frames 5 \
  --init-pcd-num 3 \
  --update-num 2 \
  --result-dir results/smoke_KITTI_05 \
  --time-dir results/smoke_KITTI_05/time_info
