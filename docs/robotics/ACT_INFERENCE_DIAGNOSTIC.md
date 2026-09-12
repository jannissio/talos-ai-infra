# Small inference-objective diagnostic — September 12, 2026

## Goal and budget

Diagnose the stalled ACT controller by matching the training forward pass to inference, using the existing upright-01 episode. No new demonstrations or image copies. Start from checkpoint 2,000, reset the optimizer, keep its normalization and architecture, and perform 300 new updates with batch size four and the existing 1e-5 learning rate. Save one final resumable checkpoint and one rollout video. New experiment artifacts should fit within 1 GiB.

The [storage audit](storage-audit.json) found about 109 GiB free, 13.25 GiB of local `.run` artifacts, and 7.97 GiB in the training environment. Counts are logical file sizes. No existing artifact was deleted. Training now checks estimated checkpoint writes plus a 10 GiB reserve, including before each checkpoint. The matched-image collector and converter also check headroom. These estimates are conservative preflight checks, not disk quotas; other processes can consume space concurrently.

## Controlled changes and limits

`--objective inference-l1` puts ACT in evaluation mode while keeping autograd enabled. Its forward pass uses the same zero latent and disabled dropout as inference, and directly optimizes masked action L1 without the VAE KL objective. Tests confirm that this loss matches the inference prediction error, active model weights receive gradients, and the unused VAE encoder receives none. `eval()` is not the same as `no_grad()`.

`--warm-start` preserves the old checkpoint and starts a fresh experiment counter and optimizer. Dataset lineage must match. `--episode upright-01` selects existing rows without copying images; future-target padding remains bounded by the original episode. Resume requires the same objective and episode selection.

This simultaneously narrows the fitting distribution and changes objective/dropout behavior. It is a learning diagnostic, **not a causal ablation proving the VAE caused failure**. If useful, later compare matched-budget objectives on the same subset. Relative actions and faster trajectories are deliberately not added to this run.

## Before-training measurements

On 128 fixed observations from upright-01, checkpoint 2,000 has first-five-endpoint left-arm MAE of 0.04646 rad. Current-observation phase labels are used only for scoring; they never enter model inputs. The largest phase-average arm errors are park (0.1130 rad) and approach (0.0706 rad), with grasp-relevant errors during lift (0.0397 rad). Gripper errors include lift (0.1258 rad) and lower (0.1658 rad). These describe the sampled action predictions, not root causes of the closed-loop failure.

The diagnostic will score exactly those observations again and run the physical upright-01 trial for up to 60 simulated seconds, recording video. Existing supported-grasp and placement criteria remain unchanged. A lower prediction error alone is insufficient for promoting the model to more layouts or utensils.

```powershell
.\.venv-training\Scripts\python.exe scripts/train_act.py --dataset .run/act-matched-nvidia-fit5 --output .run/act-inference-upright-300 --warm-start .run/act-matched-2000/step-002000 --episode upright-01 --objective inference-l1 --steps 300 --batch-size 4 --save-every 300 --log-every 50
.\.venv-training\Scripts\python.exe scripts/score_act.py --checkpoint .run/act-inference-upright-300/step-000300 --dataset .run/act-matched-nvidia-fit5 --episode upright-01 --phase-source .run/bottle-matched-nvidia --samples 128 --output .run/act-diagnostic-after.json
.\.venv-training\Scripts\python.exe scripts/evaluate_act.py --checkpoint .run/act-inference-upright-300/step-000300 --episode .run/bottle-matched-nvidia/upright-01 --output .run/act-diagnostic-rollout.json --video .run/act-diagnostic-rollout.mp4
```

## Completed result

All 300 finite updates completed in 363 seconds. New artifacts total **0.448 GiB**, below the 1 GiB budget; the final check found 108.25 GiB free. No new demonstration images or duplicated dataset were created, and no older artifacts were deleted. One resumable checkpoint is retained.

| Metric on the same 128 upright-01 observations | Before | After |
| --- | ---: | ---: |
| Left arm, full 20-target MAE | 0.04393 rad | 0.03312 rad |
| Left arm, first five endpoints | 0.04646 rad | 0.04151 rad |
| Gripper, full chunk | 0.07490 rad | 0.03554 rad |
| Current-position reference, first five | 0.00770 rad | 0.00770 rad |

The short-horizon improvement is 10.65%, and gripper error falls 52.55%. However, approach-phase arm error worsens from 0.07062 to 0.08781 rad. Grasp closure predictions improve, but the approach still fails in closed-loop use. This mixed result does not isolate the VAE as the cause of the earlier failure.

**Physical result: 0/1 success.** The 60-second familiar-start trial times out, with zero supported hold, negligible bottle lift and 84.45 mm final placement error. The other arm moves at most 0.055° and other objects about 0.011 mm. The video shows approach without a completed grasp. All 300 frames decode successfully at 5 fps / 320×240, and representative initial, five-second and final frames were inspected.

The [complete report](act-inference-diagnostic-results.json) preserves phase/joint scores, checkpoint hash, physical metrics, disk measurements and video validation. The video is `.run/act-diagnostic-rollout.mp4`, with a contact sheet beside it. Four training tests passed, including inference-loss equivalence, gradients, subset bounds and a low-disk rejection. Existing simulator control code and training trajectory timing are unchanged. Reserved physical poses remain untouched.

## Next decision

Do not promote this checkpoint or add more layouts yet. The next bounded experiment should test relative joint targets (desired target minus measured current joint position) on this same episode, with explicit normalization and reconstruction back to absolute actuator targets. This may better represent the small per-step corrections, but it is a hypothesis requiring physical verification. Compare against this direct-inference baseline and preserve phase-specific scoring so an improved average cannot hide a worse approach. Avoid adding speed changes or new data at the same time.

The separate [movement-speed plan](MOVEMENT_SPEED_PLAN.md) records the user's submission improvement, the measured 38-second teacher baseline and proposed validated speed increments. It has not yet accelerated the physical controller.
