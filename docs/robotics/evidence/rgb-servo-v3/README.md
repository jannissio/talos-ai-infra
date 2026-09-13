# V3 motor refinement: stopped at the kinematic gate

The separate [declared protocol](protocol.json) completed its 6,000 additional Adam steps on the RTX 4070. It improved offline gripper geometry but missed the unchanged **0.500 mm p95** gate. The experiment stops here: **no physical trials, no OpenVINO export, no promotion**. The deployed dinner models and all V1/V2 evidence remain unchanged.

| Candidate | Accepted development points | Position error p95 | Median | Maximum |
| --- | --- | --- | --- | --- |
| Preserved motor, rescored before refinement | 355/362 | 1.084 mm | 0.451 mm | 2.729 mm |
| 3,000 additional steps | 360/362 | 0.634 mm | 0.254 mm | 1.939 mm |
| 6,000 additional steps, selected | 361/362 | 0.564 mm | 0.208 mm | 1.746 mm |

These are the same exposed kinematic development points used by the earlier motor experiment, not fresh physical test scenes. Selection used acceptance count, then p95 error. The gate required both at least 360/362 accepted points and at most 0.500 mm p95. Acceptance retained the original 1.5 mm position, 0.025 axis and joint-limit guards. No threshold was relaxed after seeing the result. All 6,000 allowed steps are consumed; the reserved physical development and final seeds remain unexposed.

## What changed and what was verified

The same motor architecture and 4,056 original training labels were refined with gripper-position and vertical-axis losses, plus a small joint-reference loss. There were no new demonstrations, kinematic labels, pretrained weights, controller changes or observer changes. Differentiable forward geometry exists only in the offline training scripts; deployed controllers do not import it or solve inverse kinematics.

Tests compare 512 random joint configurations across both arms to independent MuJoCo geometry, and compare autodiff to MuJoCo finite-difference gradients. The actual CUDA float32 geometry check had a maximum absolute discrepancy of `1.720e-7` across point/rotation components. The last training-step timer recorded **151.70 seconds**, with **0.0221 GiB peak Torch-allocated memory**; this is not total graphics-device occupancy or physical execution time.

The [CPU audit](audit.json) reloads the original and both saved candidate checkpoints and recomputes all 362 points with MuJoCo. Every per-point acceptance decision matches; maximum position-score difference from the CUDA training report is below 0.000153 mm. It independently reproduces the failed gate.

## Preserved package

- [Full original training report](training.json): all per-point scores for all three checkpoints, training curve, protocol/input/source hashes and CUDA device.
- [Compact summary](summary.json) and [byte-level manifest](manifest.json).
- Both candidates in `candidates/`, plus exact copies of the two training sources in `evaluated-source/`.
- Original shared inputs remain at `training/bottle_servo_v1/motor` and `models/bottle_servo_v1/motor.safetensors`; their hashes are checked without duplicating the arrays.

The raw console log remains local because a library warning includes a personal absolute path. Its structured training curve is already preserved in `training.json`. No scores or checkpoint bytes were sanitized or rewritten. Shared dependency hashes were checked during packaging against the unchanged parent revision; that check is distinguished from the two source hashes recorded by training itself.

## Reproduce without retraining

From the repository root, after installing the documented training environment:

```powershell
.\.venv-training\Scripts\python scripts/package_rgb_servo_motor_refinement.py --verify-only --device cpu
.\.venv-training\Scripts\python -m unittest training_tests.test_torch_robot_geometry -v
```

To repeat the bounded training experiment on CUDA, use a new output directory:

```powershell
.\.venv-training\Scripts\python scripts/refine_rgb_servo_motor.py --output .run/reproduce-rgb-servo-v3
```

The trainer checks disk before output and checkpoints, retaining 10 GiB plus expected writes. A reproduction does not make this checkpoint eligible for deployment or authorize tuning on reserved final scenes. Any further experiment needs a separate declared budget and selection protocol.
