# Observer camera adaptation: stopped at fresh perception

This separate [protocol](protocol.json) trains the existing observer on 1,024 new paired original/opposite-camera states, with 256 different paired development states. The generator/protocol were committed in `b06be89`; the strict trainer was committed in `a3a27fe` before fitting. The four datasets finish in 87.17 seconds and occupy about 270 MiB locally. Each camera pair uses identical sampled qpos and light states.

All **6,000 declared training updates** complete in **207.78 seconds**, using 2.703 GiB peak Torch-allocated / 2.912 GiB reserved GPU memory on the RTX 4070. The 3,000-update checkpoint is selected before fresh perception because it accepts more opposite-view development states than the 6,000-update checkpoint. Both pass their development gates; the predeclared ordering favors coverage before accepted p95 error.

| Checkpoint | Original accepted present | Opposite accepted present | Opposite error p95 | Opposite maximum | Absent false accepts, each layout |
| --- | --- | --- | --- | --- | --- |
| 3,000 updates, selected | 214/233 | 221/233 | 2.662 mm | 4.700 mm | 0/23 |
| 6,000 updates | 215/233 | 219/233 | 2.156 mm | 4.334 mm | 0/23 |

[Both checkpoints, all development observations and exact training sources](training-manifest.json) are frozen here. The selected SHA is `2d82fc20ee8bc7d209be57ca1436361c4865b3ee3fb1d49a176d2ca2df5f1109`. Data hashes are in [training.json](training.json); compact reproduction recipes and subsequent checks are separate additions. The original observer and V5 motor remain unchanged.

The selection was committed in `ea7139c` before the export/fresh-test sequence. FP32 OpenVINO export passes all 12 numerical probes for each network. CPU development rescoring changes **zero acceptance decisions**; its largest position-score difference from CUDA is 0.0064 mm. The actual IR hashes and successful CPU development gate are frozen before generating fresh perception seed 2026098203.

| Fresh CPU perception, 256 paired states | Accepted present | Absent false accepts | Accepted p95 | Accepted maximum | Gate |
| --- | --- | --- | --- | --- | --- |
| Original cameras, comparison | 218/232 | 0/24 | 2.299 mm | 6.516 mm | Fail maximum |
| Opposite cameras, selected | 226/232 | 0/24 | 2.293 mm | 7.124 mm | Fail maximum |

**The selected candidate fails the unchanged 6 mm maximum-error gate and stops before physical trials.** Coverage and p95 pass; the maximum does not. The comparison configuration also fails. No checkpoint is reselected, no threshold is relaxed, and all reserved physical training/development/final scenes remain unexposed. The selected dinner controller and hosted model stay unchanged. Earlier experiments use different fresh scenes, so their headline counts are not controlled improvement estimates.

[The validation package](validation/manifest.json) retains all CPU observations, exact scene recipes and training labels for 1,536 unique paired states, per-image hashes, calibration, runtime sources, model exports and representative images. Its [independent audit](validation/audit.json) recomputes **all 2,560 baseline/candidate/CPU observation scores** and verifies 3,072 camera-specific state records against MuJoCo geometry, with zero score discrepancy. Raw RGB and failed packaging attempts remain local; no original data is overwritten. The original training manifest and pre-test selection remain unchanged snapshots, including their historical zero-exposure fields.

The six collections report 99.23 seconds in total. Training takes 207.78 seconds and 2.912 GiB peak Torch-reserved memory. Raw data plus packages occupy about 0.357 GiB, below the declared 1 GiB cap; over 522 GiB remains free. The validation package itself contains 126 manifest entries and 12,814,133 bytes. Original private protocol paths in four dataset manifests are normalized explicitly; their original hashes and all shard hashes are preserved. Compact labels are exact concatenated arrays, while RGB pixels are represented by exact hashes and rendering recipes.

## Reproduce and inspect

From the repository root in the documented training environment:

```powershell
.venv-training/Scripts/python scripts/package_rgb_servo_camera_training.py --verify-only
.venv-training/Scripts/python scripts/replay_rgb_servo_camera_recipe.py --dataset evaluation-opposite_obliques --indices 0 32 128 228 255 --output .run/observer-camera-replay-review
```

The replay command needs a fresh output directory, checks disk space and hashes, regenerates RGB/masks from the portable scene and sampled states, and executes the selected OpenVINO observer. Omit `--indices` to replay the entire already-exposed split. The other available splits are `train`, `development` and `evaluation`, each suffixed with `-original` or `-opposite_obliques`. Rendering may differ across drivers; per-pixel differences are reported explicitly instead of silently treated as identical input. Full retraining follows [the declared collection/training protocol](protocol.json), using the saved collector/trainer sources and packaged warm-start model; raw datasets must be generated into unused directories and their hashes checked before fitting.

The [opposite-view failure image](validation/worst-opposite_obliques.png) shows exposed frame 228; [the original-view failure](validation/worst-original.png) is frame 210. Green markers are predictions, red markers are scoring-only references. These are static perception failures, not physical trials or Intel evidence.

[Reproduction checks](replay/manifest.json) re-render five preserved states per camera configuration, including each worst-error frame. All 30 view images and their training labels match exactly on this PC; CPU acceptance and position scores also match exactly. This verifies the compact recipe on selected exposed samples, not another independent perception test.
