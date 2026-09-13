# Reproducing the RGB bottle experiment

The selected observer uses 10,112 rendered states: 4,096 procedural states, 1,920 frames from 15 physically replayed motions, and 4,096 additional procedural states with exposed right-arm poses. The packaged compact inputs preserve exactly the arrays consumed by the image generator. Original rendered-shard hashes remain under `original-dataset-manifests/`; different OpenGL drivers may produce different pixels. This is a reproducible generation/training recipe, not a promise of identical GPU training bits on every machine.

`robot-poses/` contains only the 12 robot joint positions sampled by the static renderer, in their original order and dtype. `motion/` contains the exact 128 full-state rows used for each of the 15 accepted motions, all 24 original attempt reports, and a target-label correction. **The original collector's requested destination field was not applied to its teacher. Those motions establish fallback-goal replay success only.** Use these stored states to reproduce the RGB labels; rerunning the corrected motion collector produces different motions. `motor/` contains the exact 4,056 training labels from 6,144 attempted kinematic points and all refusals. These are offline IK supervision, not inference-time object-state access.

Use the RTX training environment described in the root README. Every command requires an unused output path and checks actual disk space, preserving 10 GiB plus expected writes. Generating the image inputs needs roughly 1 GiB; budget 2 GiB for this reproduction. Do not delete existing datasets automatically.

Generate the selected chain's inputs:

```powershell
.\.venv-training\Scripts\python scripts/collect_rgb_servo_observer.py --protocol docs/robotics/experiments/rgb-servo-observer-v2.json --split train --recording training/bottle_servo_v1/robot-poses/dinner-42 --output .run/reproduce-rgb-train-v2
.\.venv-training\Scripts\python scripts/collect_rgb_servo_observer.py --protocol docs/robotics/experiments/rgb-servo-observer-v1.json --split development --recording training/bottle_servo_v1/robot-poses/dinner-42 --output .run/reproduce-rgb-dev
.\.venv-training\Scripts\python scripts/render_rgb_servo_motion_labels.py --source training/bottle_servo_v1/motion --output .run/reproduce-rgb-motion
.\.venv-training\Scripts\python scripts/collect_rgb_servo_observer.py --protocol docs/robotics/experiments/rgb-servo-observer-v4.json --split train --recording training/bottle_servo_v1/robot-poses/relay-42 --output .run/reproduce-rgb-train-v4
.\.venv-training\Scripts\python scripts/combine_rgb_observer_data.py --inputs .run/reproduce-rgb-train-v2 .run/reproduce-rgb-motion .run/reproduce-rgb-train-v4 --output .run/reproduce-rgb-combined
```

Train from scratch, then reproduce the two warm-start stages. V2 did not warm-start from V1:

```powershell
.\.venv-training\Scripts\python scripts/train_rgb_servo_observer.py --train .run/reproduce-rgb-train-v2 --development .run/reproduce-rgb-dev --steps 6000 --save-every 6000 --output .run/reproduce-rgb-fit-v2
.\.venv-training\Scripts\python scripts/train_rgb_servo_observer.py --train .run/reproduce-rgb-train-v2 --extra-train .run/reproduce-rgb-motion --development .run/reproduce-rgb-dev --warm-start .run/reproduce-rgb-fit-v2/step-006000/model.safetensors --steps 4000 --save-every 4000 --output .run/reproduce-rgb-fit-v3
.\.venv-training\Scripts\python scripts/train_rgb_servo_observer.py --train .run/reproduce-rgb-combined --development .run/reproduce-rgb-dev --warm-start .run/reproduce-rgb-fit-v3/step-004000/model.safetensors --learning-rate 0.0003 --steps 4000 --save-every 4000 --output .run/reproduce-rgb-fit-v4
.\.venv-training\Scripts\python scripts/train_rgb_servo_motor.py --dataset training/bottle_servo_v1/motor --output .run/reproduce-rgb-motor
```

The selected observer stages took 210.84, 141.86 and 151.16 seconds on the RTX 4070, peaking at 0.994 GiB allocated VRAM. The motor fit took 28.24 seconds. Including rejected V1/V5 observer fits and the motor, the original experiment consumed its declared 30,000 Adam-step budget. All five fit logs are retained in `fits/`. Recovery supervision used by rejected V5 is recorded in the history but is not part of the selected V4 chain.

Evaluate a reproduced model using the [frozen physical commands](../../models/bottle_servo_v1/README.md), substituting its new paths. Report it as a new reproduction, keep every failure, and never tune on the already exposed final seeds. The archived selected weights remain authoritative for the reported 48-trial result.
