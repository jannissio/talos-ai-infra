# Fresh camera perception test: stopped before physical trials

The [protocol](protocol.json) and [evaluator](evaluate_rgb_servo_camera_perception.py) were committed in `67166b0` before generating fresh RNG seed **2026098101**. The opposite oblique camera angles had already been selected on the separate exposed diagnostic. Neither the angles, observer weights nor acceptance limits changed after this test began.

All **256 sampled states / 512 paired observations** finish. Both camera configurations observe the same states, with varied bottle position, height, tilt, lighting and known exposed arm-pose families. These are static perception samples, not physical manipulation or unseen robot-pose families.

| Configuration | Accepted present | False accepted absent | Accepted error p95 | Maximum | Gate |
| --- | --- | --- | --- | --- | --- |
| Original cameras, comparison | 225/237 (94.9%) | 0/19 | 2.824 mm | 11.548 mm | Fail: maximum error |
| Opposite obliques, preselected | 202/237 (85.2%) | 0/19 | 2.383 mm | 3.737 mm | Fail: acceptance |

The unchanged gate requires at least 90% acceptance of present bottles, no accepted absent bottles, accepted p95 error at most 3 mm and maximum error at most 6 mm. **The selected camera configuration fails and stops before physical trials.** Its accurate accepted estimates do not compensate for inadequate coverage. The original-camera comparison also fails this new set's maximum-error limit; it is not selected as an alternative. These perception samples are now exposed and must not be reused as a fresh selection test.

The [independent audit](audit.json) loads every sampled state into MuJoCo and recomputes all accepted geometric errors with zero discrepancy. The [13-file manifest](manifest.json) preserves all outcomes, exact sampled qpos/light states, calibration, source/protocol/model-input hashes and eight RGB previews. The raw report also records software versions and the evaluated parent revision. No bulk image dataset, new model or motor action is produced. The run takes **33.31 seconds** on AMD/OpenVINO CPU, with 3.41 MB raw output. Raw plus packaged evidence stays below the declared 64 MiB limit; every export checks the 10 GiB reserve.

Verify without generating more scenes:

```powershell
.\.venv-training\Scripts\python scripts/package_rgb_servo_camera_perception.py --verify-only --output docs/robotics/evidence/rgb-servo-camera-perception-v1
```

The next justified experiment would add a compact training set for the changed camera views, mixing original and opposite-angle examples to check retention. It needs its own training budget, new development/perception seeds and physical development gate. No further fitting, threshold relaxation or camera revision belongs to this completed test. The V5 physical final seeds remain unexposed. The selected dinner/relay demo remains unchanged and private.
