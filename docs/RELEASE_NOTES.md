# Release Notes

## 1.0.0

- Initial public release for the Pattern Recognition paper "Semantic-aided bag-of-words for LiDAR-based place recognition".
- Provides the C++ Sem-iBoW3D core and a non-interactive command line executable.
- Keeps the paper default RANSAC geometric verification path.
- Exposes asynchronous BoW dictionary updates through `--async-update`.
- Exposes Open3D FGR as an experimental registration backend through `--registration-backend fgr`.
- Adds parameterized data-preparation utilities for assigning and remapping keypoint semantic labels.
- Excludes datasets, extracted descriptors, experiment logs, third-party dependency source trees, and paper draft files from the release package.

Release cleanup changes relative to the internal experiment tree:

- Removed machine-specific absolute paths from the public source tree.
- Replaced interactive stdin prompts with explicit command line options.
- Added input validation for missing pose, loop, descriptor, point cloud, and label files.
- Fixed the feature-container update buffer reset for all-point descriptors.
- Avoided making small_gicp a default build dependency; the public default backend remains Open3D RANSAC.

## Post-1.0.0 Hardening

- Added strict descriptor parsing: exact column count, numeric conversion, and finite-value checks.
- Added strict semantic-label parsing and runtime label range checks.
- Rejected invalid CLI boundaries such as `--init-pcd-num 1` and `--near-num 0`.
- Fixed candidate-island mean distance calculation to use floating-point accumulation and division.
- Added CTest regression coverage for input contracts, CLI help, parameter rejection, and registration-path smoke testing.
- Added a sanitizer build option, `-DSEM_IBOW3D_ENABLE_SANITIZERS=ON`, for local ASan/UBSan checks.
