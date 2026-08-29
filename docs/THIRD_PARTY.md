# Third-Party Components

Sem-iBoW3D links against external libraries but does not vendor them in this source release.

Required dependencies:

- Eigen
- OpenCV
- Open3D
- TBB or oneTBB

Optional Python utility dependencies:

- numpy
- scipy
- open3d
- PyYAML
- tqdm

The semantic remapping configuration in `configs/semantic_label_map.yaml` follows the SemanticKITTI label taxonomy and the class merging used in the Sem-iBoW3D experiments. Dataset files, semantic predictions, extracted descriptors, and model weights are not distributed in this repository and remain subject to their own licenses and terms.
