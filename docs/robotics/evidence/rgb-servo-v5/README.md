# V5 motor refinement: offline gate passed; physical testing pending

The [separately declared protocol](protocol.json) starts from the preserved V4 checkpoint and allows 3,000 full-batch L-BFGS update calls. The fit completed in **192.55 seconds / 6,018 gradient evaluations**, with about **0.073 GiB peak Torch-allocated memory** on the RTX 4070. Both candidates are retained; selection used the original exposed kinematic development set only.

| Candidate | Accepted points | Position error p95 | Median | Maximum |
| --- | --- | --- | --- | --- |
| V4 warm start | 361/362 | 0.512 mm | 0.183 mm | 1.711 mm |
| 1,000 updates | 361/362 | 0.470 mm | 0.143 mm | 1.283 mm |
| 3,000 updates, selected | 361/362 | 0.367 mm | 0.098 mm | 0.989 mm |

This passes the unchanged offline gate: at least 360/362 accepted points and at most 0.500 mm p95. The [independent CPU audit](audit.json) reloads all three checkpoints and reproduces every acceptance decision; maximum per-point discrepancy is below 0.000138 mm. The [full original report](training.json) preserves every score, the optimization curve and the source/data/protocol hashes. The [manifest](manifest.json) covers exact copies of both checkpoints and training sources. No new demonstration, kinematic label, observer, controller or pretrained weight was added.

**Offline precision is not physical placement success.** The selected motor remains experimental. Fresh physical development seeds 2026097401–7406 will use all four declared conditions, with every refusal and failure retained. The final seeds 2026097501–7512 stay unexposed unless the development gate passes. V3/V4 remain stopped; their physical seed sets are not reused. The offline summary records the packaging state before these physical tests.

Verify the saved checkpoints without training:

```powershell
.\.venv-training\Scripts\python scripts/package_rgb_servo_full_batch.py --verify-only --output docs/robotics/evidence/rgb-servo-v5
```

Repeat the declared CUDA fit only into an unused output folder:

```powershell
.\.venv-training\Scripts\python scripts/refine_rgb_servo_full_batch.py --protocol docs/robotics/experiments/rgb-servo-bottle-v5.json --output .run/rgb-servo-v5-reproduce-training
```

The original training labels remain under `training/bottle_servo_v1/motor`. The source declaration is committed before training in `67fef17`. Dataset/checkpoint/export operations retain 10 GiB plus expected writes. Raw console logs remain local because a library warning contains a personal installation path; their structured curves are preserved in the report. The dinner submission models remain unchanged.
