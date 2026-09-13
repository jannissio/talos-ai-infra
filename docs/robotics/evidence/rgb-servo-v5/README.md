# V5 motor refinement: stopped at physical development gate

The [separately declared protocol](protocol.json) starts from the preserved V4 checkpoint and allows 3,000 full-batch L-BFGS update calls. The fit completed in **192.55 seconds / 6,018 gradient evaluations**, with about **0.073 GiB peak Torch-allocated memory** on the RTX 4070. Both candidates are retained; selection used the original exposed kinematic development set only.

| Candidate | Accepted points | Position error p95 | Median | Maximum |
| --- | --- | --- | --- | --- |
| V4 warm start | 361/362 | 0.512 mm | 0.183 mm | 1.711 mm |
| 1,000 updates | 361/362 | 0.470 mm | 0.143 mm | 1.283 mm |
| 3,000 updates, selected | 361/362 | 0.367 mm | 0.098 mm | 0.989 mm |

This passes the unchanged offline gate: at least 360/362 accepted points and at most 0.500 mm p95. The [independent CPU audit](audit.json) reloads all three checkpoints and reproduces every acceptance decision; maximum per-point discrepancy is below 0.000138 mm. The [full original report](training.json) preserves every score, the optimization curve and the source/data/protocol hashes. The [manifest](manifest.json) covers exact copies of both checkpoints and training sources. No new demonstration, kinematic label, observer, controller or pretrained weight was added.

**Offline precision is not physical placement success.** All 24 physical development trials now finish: **3/6 nominal live, 1/6 nominal frozen, 3/6 pushed live and 0/6 pushed frozen**. The unchanged gate requires at least 4/6 nominal live and 4/6 pushed live. V5 therefore stops without final evaluation or promotion. Final seeds 2026097501–7512 remain unexposed. V3/V4 remain stopped; their physical seed sets are not reused. The original offline summary retains its packaging state before these physical tests.

| Development seed | Nominal live | Nominal frozen | Pushed live | Pushed frozen |
| --- | --- | --- | --- | --- |
| 2026097401 | RGB unavailable | Placement failed | RGB unavailable | Placement failed |
| 2026097402 | Pass, 5.293 mm | Placement failed | Pass, 5.353 mm | Placement failed |
| 2026097403 | Pass, 1.250 mm | Pass, 7.902 mm | Pass, 1.253 mm | Placement failed |
| 2026097404 | RGB unavailable | Placement failed | RGB unavailable | Placement failed |
| 2026097405 | Pass, 5.225 mm | Placement failed | Pass, 5.668 mm | Placement failed |
| 2026097406 | Invalid start | Invalid start | Invalid start | Invalid start |

Seed 7402 completes both table-supported relay legs in each live condition. Seed 7403 is the only case passing both undisturbed controls; its pushed comparison is 1.253 mm/live-pass versus 11.373 mm/frozen-fail. This is development evidence on one paired scene, not a final robustness result. Two live scenes stop when current RGB is unavailable for more than 0.6 seconds; one overlaps or tips at reset. No refusal is removed from a denominator, and the 8 mm placement threshold is unchanged.

The [physical audit](physical/audit.json) checks every report against its declared inputs, exact weights, evaluated source, paired starting scene and unchanged release/parking/contact limits. It verifies **13,711 synchronized state frames and 10 completed route legs**. The [139-file physical manifest](physical/manifest.json) covers approximately 21.9 MB of reports, traces, observations, image-action comparisons, portable scenes, source blobs and the actual evaluated model. All raw files remain in the local archive. The FP32 motor export passes 12 numerical probes on AMD OpenVINO CPU, with maximum output difference 7.153e-7; the deployed experimental observer IR is an exact copy of V1. This is not Intel execution evidence.

Verify the physical package without exposing any new scene:

```powershell
.\.venv-training\Scripts\python scripts/package_rgb_servo_full_batch_physics.py --verify-only --offline docs/robotics/evidence/rgb-servo-v5 --output docs/robotics/evidence/rgb-servo-v5/physical
```

The physical driver was committed in `2b83401` before these trials. No final selection is created because the development gate failed. A future experiment should isolate missing visual observations during approach/descent, with its own budget and fresh splits; additional motor fitting alone is not supported by these failure categories. The selected dinner workflow remains unchanged.

Verify the saved checkpoints without training:

```powershell
.\.venv-training\Scripts\python scripts/package_rgb_servo_full_batch.py --verify-only --output docs/robotics/evidence/rgb-servo-v5
```

Repeat the declared CUDA fit only into an unused output folder:

```powershell
.\.venv-training\Scripts\python scripts/refine_rgb_servo_full_batch.py --protocol docs/robotics/experiments/rgb-servo-bottle-v5.json --output .run/rgb-servo-v5-reproduce-training
```

The original training labels remain under `training/bottle_servo_v1/motor`. The source declaration is committed before training in `67fef17`. Dataset/checkpoint/export operations retain 10 GiB plus expected writes. Raw console logs remain local because a library warning contains a personal installation path; their structured curves are preserved in the report. The dinner submission models remain unchanged.
