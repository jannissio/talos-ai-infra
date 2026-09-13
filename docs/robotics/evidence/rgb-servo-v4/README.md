# V4 full-batch motor refinement: stopped before physics

The [declared protocol](protocol.json) used the existing 4,056 training points with a full-batch L-BFGS optimizer, starting from the preserved V3 checkpoint. It completed all **500 update calls / 1,012 gradient evaluations** in **32.13 seconds**, with about **0.073 GiB peak Torch-allocated memory** on the RTX 4070. These are offline training measurements, not physical execution times or total device memory use.

| Candidate | Accepted exposed development points | Position error p95 | Median | Maximum |
| --- | --- | --- | --- | --- |
| V3 warm start | 361/362 | 0.564 mm | 0.208 mm | 1.746 mm |
| 150 updates | 361/362 | 0.562 mm | 0.195 mm | 1.714 mm |
| 500 updates, selected | 361/362 | 0.512 mm | 0.183 mm | 1.711 mm |

The selected candidate **misses the unchanged 0.500 mm p95 gate**. The experiment stops without physical trials, OpenVINO export or deployment. Both checkpoints, every per-point score and the structured curve are retained in [training.json](training.json), with [summary](summary.json), [manifest](manifest.json), and exact source copies. The reserved physical development/final scenes remain unexposed; no further updates belong to this protocol.

The [independent CPU audit](audit.json) reloads all three checkpoints and recomputes robot geometry through MuJoCo. Every point retains its original acceptance decision; maximum score discrepancy is below 0.000129 mm. The raw console log remains local because a PyTorch warning contains a personal installation path. Its structured training curve is preserved without modification.

The training source was committed in `7abbd0d` before the run. Its differentiable geometry, original data and refusal thresholds are unchanged from V3. No controller, observer, demonstration, pretrained weight or deployed dinner model changed. Lower offline error alone does not establish broader physical placement coverage.

From the repository root, verify saved weights without retraining:

```powershell
.\.venv-training\Scripts\python scripts/package_rgb_servo_full_batch.py --verify-only --output docs/robotics/evidence/rgb-servo-v4
```

To reproduce the bounded training on CUDA, choose a fresh output folder:

```powershell
.\.venv-training\Scripts\python scripts/refine_rgb_servo_full_batch.py --protocol docs/robotics/experiments/rgb-servo-bottle-v4.json --output .run/reproduce-rgb-servo-v4
```

The trainer and packager check free space before output/checkpoint writes, preserving 10 GiB plus expected writes. The two candidates and the original model stay separate; source snapshots are for auditing, while the reproduction command uses the matching scripts at the repository root.
