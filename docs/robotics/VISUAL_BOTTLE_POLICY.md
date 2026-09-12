# Upright visual bottle experiment

The revised policy separates bottle localization from the table's visual appearance. Its classical RGB detector supplies an initial overhead centroid to a trained trajectory network, while motor feedback regulates progress. All inference behavior and limits are described in the [model card](../../models/bottle_visual/README.md).

The first broader PCA candidate did not survive physical evaluation even after its training loss fell. A direct saved/live image comparison found zero pixel differences in all three cameras, ruling out a rendering mismatch for that case. The next experiment changed both the visual encoding and offline trajectory alignment and restricted its scope to upright bottles. Its improved results are not a controlled ablation attributing the gain to one change.

## Measured results

| Evaluation | Outcome | Scope |
| --- | --- | --- |
| Default task scene, seed 42 | Pass | Upright bottle, familiar setup |
| Six development seeds, 2026091101–1106 | 6/6 | Small bottle jitter and full-scene randomization |
| Three additional wider-position training starts | 1/3 | Exposed development cases, not held out |
| Frozen OpenVINO test, 2026091301–1310 | 10/10 | New task-preset scene seeds at declaration |
| Blank image test | Refused before movement | Missing bottle detection |
| Cancellation at 1.5 simulated seconds | Paused and closed renderer | Physical state preserved |

All ten frozen tests lifted the bottle by more than 6.8 cm, maintained physical finger support, and released it upright at its goal. Final errors were 0.36–1.66 mm. The policy completed in 28.4–29.3 simulated seconds. A 10/10 result does not establish universal reliability or full workspace coverage. Failed wider-position cases remain recorded.

## Reproduce an evaluation

```powershell
.\.venv-training\Scripts\python scripts/verify_live_policy.py --checkpoint models/bottle_visual --seed 42 --output .run/visual-bottle-check.json
.\.venv-training\Scripts\python scripts/run_command_demo.py --text "place the bottle" --mode learned_bottle --checkpoint models/bottle_visual --output .run/visual-bottle-recording
```

For training, collect a fresh bounded compact batch with `collect_compact_bottle.py`, then use `prepare_visual_bottle_primitive.py`, `train_primitive_policy.py`, and `export_primitive_openvino.py`. The saved original batch used 32 attempts, 28 physically eligible outcomes, then selected 22 upright examples using the original detector threshold. The later detector removes small disconnected color fragments, so regenerating the current input may select additional episodes. Reproducing training exactly therefore requires the frozen compact input, not an assumption that every script version selects identical rows. Fresh outputs and a 10 GiB free-space reserve are required.

The benchmark excludes rendering, RGB localization, feedback guards and physics. It reports network-only timings on an Intel i7-10850H and Intel UHD iGPU. Current camera rendering uses NVIDIA. No Core Ultra or NPU execution claim is made.
