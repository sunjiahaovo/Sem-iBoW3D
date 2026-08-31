#!/usr/bin/env python3
import shutil
import subprocess
import sys
from pathlib import Path


POINTS = [
    (1.0, 1.0, 1.0),
    (2.0, 1.0, 1.0),
    (1.0, 2.0, 1.0),
    (1.0, 1.0, 2.0),
    (2.0, 2.0, 1.0),
    (2.0, 1.0, 2.0),
]

ALL_DESCRIPTORS = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
]

KEY_DESCRIPTORS = ALL_DESCRIPTORS[:4]


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_ply(path: Path) -> None:
    body = "\n".join(f"{x} {y} {z}" for x, y, z in POINTS)
    write_text(
        path,
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(POINTS)}",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                body,
                "",
            ]
        ),
    )


def write_descriptors(path: Path, rows) -> None:
    write_text(path, "\n".join(" ".join(str(value) for value in row) for row in rows) + "\n")


def make_fixture(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)

    write_text(root / "data/KITTI/00/GTposes.csv", "0,0,0\n1,0,0\n0,0,0\n1,0,0\n")
    write_text(root / "data/KITTI/00/loop_lst.csv", "2\n3\n")

    for frame_id in range(4):
        write_ply(root / f"res_data/KITTI/00/D3F_allpoints/{frame_id}.ply")
        write_descriptors(root / f"descriptor_txt/D3F/KITTI/00/descriptors_{frame_id}.txt", KEY_DESCRIPTORS)
        write_descriptors(root / f"feature_txt/D3F/KITTI/00/descriptors_{frame_id}.txt", ALL_DESCRIPTORS)
        write_text(root / f"res_data/KITTI/00/D3F_keypoints_label_reset_SF/{frame_id}.txt", "0\n0\n0\n0\n")


def run_smoke(exe: Path, root: Path, backend: str) -> None:
    result_dir = root / f"results/{backend}"
    command = [
        str(exe),
        "--dataset",
        "KITTI",
        "--sequence",
        "00",
        "--project-root",
        str(root),
        "--max-frames",
        "4",
        "--init-pcd-num",
        "2",
        "--update-num",
        "2",
        "--keypoint-num",
        "4",
        "--feature-dim",
        "3",
        "--init-words-num",
        "2",
        "--words-num-add",
        "1",
        "--semantic-num",
        "2",
        "--lambda-word",
        "0",
        "--near-num",
        "2",
        "--gap-num",
        "0",
        "--search-num",
        "2",
        "--ransac-n",
        "3",
        "--max-iter",
        "50",
        "--check-th",
        "1000000",
        "--result-dir",
        str(result_dir),
        "--time-dir",
        str(result_dir / "time_info"),
        "--registration-backend",
        backend,
    ]

    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{backend} smoke failed with code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if "Registration time:" not in completed.stdout:
        raise RuntimeError(f"{backend} smoke did not enter registration path\nstdout:\n{completed.stdout}")

    registration_time = result_dir / "time_info/RegistrationTime.txt"
    if not registration_time.exists() or not registration_time.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"{backend} smoke did not write RegistrationTime.txt")


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: run_tiny_registration_smoke.py EXE FIXTURE_ROOT BACKEND", file=sys.stderr)
        return 2

    exe = Path(sys.argv[1]).resolve()
    root = Path(sys.argv[2]).resolve()
    backend = sys.argv[3]
    if backend not in {"ransac", "fgr"}:
        print("backend must be ransac or fgr", file=sys.stderr)
        return 2

    make_fixture(root)
    run_smoke(exe, root, backend)
    print(f"tiny registration smoke passed for {backend}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
