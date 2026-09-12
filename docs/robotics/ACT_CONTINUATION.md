# Matched ACT continuation — September 11, 2026

## Objective and unchanged contract

Following the [livestream review](../hackathon/LIVESTREAM_REVIEW_2026-09-11.md), continue the camera-driven bottle learning diagnostic. The talk supports our approach; this experiment remains a prerequisite rather than the complete language/bimanual submission.

Resume the [matched step-1,000 checkpoint](ACT_MATCHED_EXPERIMENT.md) with the same five episodes, seed, batch size four, optimizer settings, normalization, image capture contract and fixed 0.25 Nm gripper adapter. Preserve all previous checkpoints and results. The current run starts at update 1,001, targeting 2,000 total updates. Resume restores weights, optimizer, scaler and Torch RNG, but restarts shuffled dataset iteration; it is not a bitwise continuation of an uninterrupted run.

`successful_updates` in each training log counts updates in that invocation, not the global step number. Thus the step-2,000 run should end with 1,000 successful new updates. All current training uses the local NVIDIA GPU. This is not Intel deployment evidence.

## Predeclared continuation gate

After step 2,000, evaluate all five familiar physical starts and 128 fixed offline observations. Extend toward 5,000 total updates only if there is a new task success, **or** at least a 25% reduction in first-five-endpoint left-arm prediction MAE without fewer task successes, relative to step 1,000. This is a bounded compute-budget decision, not a convergence criterion or a guarantee of eventual learning. Variational training loss alone does not trigger continuation. Record full-chunk and gripper errors, supported hold and video evidence alongside the gate.

Physical success checks remain unchanged: bilateral supported hold above 5 cm, upright release and stable placement, and limits on disturbances. Keep the reserved physical poses untouched. A passing continuation gate is not the same as passing the physical task.

## Reproduce

Use fresh output folders. Run training and GPU evaluation sequentially so inference timings are not contaminated by another training process.

```powershell
.\.venv-training\Scripts\python.exe scripts/train_act.py --dataset .run/act-matched-nvidia-fit5 --output .run/act-matched-2000 --resume .run/act-matched-1000/step-001000 --steps 2000 --batch-size 4 --save-every 500 --log-every 50
.\.venv-training\Scripts\python.exe scripts/evaluate_act_checkpoint.py --checkpoint .run/act-matched-2000/step-002000 --output .run/act-matched-evaluation-2000
```

The evaluation driver records five videos, normal-image and black-image scores, and `comparison.json`. The comparison checks matching dataset lineage and physical start names. Runtime records now include the actual model device and OpenGL vendor/renderer; a CPU flag alone is not enough to establish Intel graphics execution.

If the step-2,000 gate passes:

```powershell
.\.venv-training\Scripts\python.exe scripts/train_act.py --dataset .run/act-matched-nvidia-fit5 --output .run/act-matched-5000 --resume .run/act-matched-2000/step-002000 --steps 5000 --batch-size 4 --save-every 500 --log-every 100
.\.venv-training\Scripts\python.exe scripts/evaluate_act_checkpoint.py --checkpoint .run/act-matched-5000/step-005000 --output .run/act-matched-evaluation-5000 --baseline .run/act-matched-evaluation-2000
```

Stop after reviewing step 5,000 in this experiment; any further run needs a new evidence-based budget decision. Do not expand to more utensils or claim generalization solely from prediction error improvements.

## Completed outcome — September 12

The step-2,000 checkpoint completed all 1,000 additional finite optimizer updates and saved successfully. The laptop/session paused overnight between logged updates 1,900 and 1,950. The recorded 45,925-second elapsed time includes that suspension and must not be reported as active compute time.

**The continuation gate failed; no step-5,000 training was started.** Physical success remains 0/5, supported hold remains zero in all five trials, and first-five-endpoint action error increased from 0.06715 to 0.06784 rad (about 1.03% worse). The hold-current-position reference remains 0.00880 rad. Full-chunk left-arm error fell slightly, from 0.06676 to 0.06358 rad, but that is not improvement in the short horizon actually executed. Gripper error is 0.10806 rad. Black-image full-chunk error is 0.67024 rad; the model is image-sensitive under this intervention, without demonstrating robust manipulation.

| Start | Maximum bottle height increase | Best supported hold | Final placement error |
| --- | ---: | ---: | ---: |
| upright-01 | 3.47 cm | 0 s | 63.2 mm |
| upright-04 | 3.56 cm | 0 s | 57.9 mm |
| sideways-01 | approximately 0 cm | 0 s | 202.7 mm |
| sideways-03 | approximately 0 cm | 0 s | 153.8 mm |
| sideways-11 | 0.53 cm | 0 s | 152.0 mm |

Maximum height is not verified grasp/lift success. The inspected upright-01 video shows the bottle ends on its side; sideways-01 stalls without transfer. All five trials completed normally as 60-second task timeouts. Each video decoded into 300 valid 320×240 frames at 5 fps. The two-example contact sheet is `.run/act-matched-evaluation-2000/contact-sheet.png`; all videos and detailed logs are beside it.

The [combined result](act-continuation-results.json) preserves checkpoint hash, log endpoints, device metadata, physical trials, both offline scores and video validation. The three reserved physical poses were not consumed. Two training-boundary tests passed after the runtime-reporting change; the CPU MuJoCo smoke and [OpenVINO export parity check](INTEL_INFERENCE.md) also executed successfully. Existing simulation control code was not changed in this continuation.

## Next diagnostic, not more blind scaling

Update: the [300-update single-episode inference-objective diagnostic](ACT_INFERENCE_DIAGNOSTIC.md) has completed. It improved some prediction metrics but still failed its one physical trial. Storage and faster submission movements are now explicitly budgeted/planned there.

Keep the verified physical controller and matched images fixed. First localize the remaining action error by joint, execution horizon and trajectory phase using labels only for offline analysis. Then test a small deterministic imitation baseline on one upright demonstration, with the inference and training objectives aligned, before broadening the data again. Treat removal of the VAE objective and relative joint-action targets as separate controlled experiments, not simultaneous fixes. Current observations do not establish whether representation, variational train/inference differences, dataset ambiguity or insufficient optimization is the main cause.

Require successful physical reproduction before promoting a model to more layouts, other utensils or coordinated skills. This failed gate does not prove ACT is unsuitable or converged; it means this bounded extra-compute experiment did not earn the next allocation. The useful parallel result is a verified FP32 OpenVINO CPU export of checkpoint 1,000, while Intel rendering integration and a clean benchmark remain unfinished.
