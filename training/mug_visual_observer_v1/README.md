# Compact synthetic mug-pose inputs

This package contains every one of the 4,096 training and 512 development examples used by the [stopped observer fit](../../docs/robotics/evidence/mug-visual-observer-v1/README.md). It does not contain a new manipulation dataset: the images are posed renders without physics stepping or a valid-grasp claim.

Each split contains `data.npz`, its manifest, six predetermined illustrative mosaics, and the complete reproduction audit. The NPZ preserves RGB features, geometric labels, presence, view availability, all joint/object positions, episode index, exposure, every RGB-byte hash and the sampled perturbation recipe. Absent examples remain included. Full RGB arrays are reconstructed from these compact recipes rather than duplicated in the repository.

The original scene/initial-state inputs are already packaged under `training/mug_release_v1/episodes`; the original trajectory input is `training/dinner_suite/mug/retrieval.npz`. Neither is modified. `reproduction-inputs.json` and each split manifest identify all source/asset hashes; `source/` preserves exact Python files used in this experiment.

In a clean checkout with the documented training environment:

```powershell
.venv-training/Scripts/python.exe scripts/collect_mug_visual_observer.py --split training
.venv-training/Scripts/python.exe scripts/collect_mug_visual_observer.py --split development
.venv-training/Scripts/python.exe scripts/audit_mug_visual_observer.py --split training
.venv-training/Scripts/python.exe scripts/audit_mug_visual_observer.py --split development
.venv-training/Scripts/python.exe scripts/train_mug_visual_observer.py
```

All outputs must be unused; scripts preserve prior data and enforce a 10 GiB reserve plus expected writes. The cumulative raw/model/input/evidence budget is 256 MiB. Model optimization uses the RTX GPU, while development scores use CPU FP32. Deterministic seeds and exact inputs are retained; elapsed time and allocator measurements are specific to the recorded PC.

Both checkpoints fail accuracy, so stop before export or generation of the reserved fresh split. `scripts/package_mug_visual_observer.py --verify-only` checks an existing raw run and its packages, including independent CPU reproduction of both candidate scores. All 13,824 image hashes and every label/feature already reproduced exactly in the preserved data audits.
