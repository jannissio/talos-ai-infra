# Matched-controller ACT experiment

This follows the unsuccessful 200-update experiment documented in `ACT_TRAINING.md`. The next decision is based on a new physical experiment, not on assuming that lower training loss means the arm can grasp.

## Corrections

1. Re-record the same five familiar starts by replaying the successful nominal trajectories through the exact policy actuator adapter. Both capture and evaluation use `simulation_lab.policy_control.apply_targets`: position bounds, 200 Hz feedback and a fixed 0.25 Nm left-gripper torque cap. No bottle pose or task stage selects the cap.
2. Capture fresh joint measurements and lossless PNG images from the resulting physics. Do not copy the original teacher's observations, which were generated with different gripper limits.
3. Use direct 320×240 views, no shadows and no multisampling in both capture and evaluation. An initial comparison exposed an additional hardware difference: the simulation environment rendered on Intel UHD, while CUDA policy inference rendered on the NVIDIA Quadro T2000. Capture now runs in `.venv-training` with CUDA initialized before creating the renderer. The initial images then match exactly on all three cameras.

The first Intel-rendered attempt and its partial conversion remain under `.run/bottle-matched` and `.run/act-matched-fit5` for diagnosis; they are superseded and must not be used for this experiment. The authoritative new recording root is `.run/bottle-matched-nvidia`, converted to `.run/act-matched-nvidia-fit5`.

## Verification gates

- Each nominal trajectory must physically pass both 200 Hz and interpolated 20 Hz replay checks before capture. These check supported hold, stable upright placement, contact loss and displacement of other objects/arm.
- Reconstruct every saved joint state and applied command independently from the initial state. Require agreement within 1e-10.
- Compare four points along each episode in all three cameras against the saved PNGs. This checks 60 images, not only a stationary initial image.
- Preserve whole-episode boundaries, source manifest hashes and the gripper cap in the converted dataset lineage. A checkpoint with a single declared cap is rejected if evaluation requests a different one.
- No teacher updates, state resets after initialization, welds or external forces during capture or policy evaluation. Recorded teacher stage names remain evaluation metadata, never policy inputs.

## Learning run

The recording gates passed: all ten physical replay checks, exact reconstruction of every recorded position/velocity and applied motor command, and zero pixel differences in all 60 sampled images. See `act-matched-capture.json`, `act-matched-reconstruction.json` and `act-matched-image-check.json`. The legacy `outcome` snapshot in each recording is inherited teacher provenance; matched-run measurements are the `replays` and reconstruction reports. The declared runtime gripper cap is the top-level 0.25 Nm value.

A fresh ACT model with the same ImageNet ResNet-18 initialization, transformer, 20-step action chunks and four-interval replanning as the first experiment. Seed 42, batch size 4, 1,000 updates, learning rate 1e-5, on the local Quadro T2000. Save intermediate and final checkpoints, then evaluate all five familiar starts physically for up to 60 simulated seconds each and record videos.

This changes recording consistency and training exposure together. It is not an ablation isolating their individual effects. The old 200-update score and the new score are both familiar-start diagnostics. Physical evaluation of the reserved poses remains gated on learning the familiar starts successfully.

Commands:

```powershell
.\.venv-training\Scripts\python.exe scripts/collect_matched_act.py --output .run/bottle-matched-nvidia --cuda-renderer
.\.venv-training\Scripts\python.exe scripts/convert_act_dataset.py --source .run/bottle-matched-nvidia --output .run/act-matched-nvidia-fit5 --split fit5
.\.venv-training\Scripts\python.exe scripts/audit_matched_act.py
.\.venv-training\Scripts\python.exe scripts/train_act.py --dataset .run/act-matched-nvidia-fit5 --output .run/act-matched-1000 --steps 1000 --batch-size 4 --save-every 500 --log-every 50
.\.venv-training\Scripts\python.exe scripts/evaluate_act_batch.py --checkpoint .run/act-matched-1000/step-001000 --episode-root .run/bottle-matched-nvidia --output .run/act-matched-evaluation-1000 --record-video
```

Use fresh output directories. Bulk data, videos and checkpoints remain local under `.run`; evidence reports belong in this documentation folder.

## Interpretation and controls

One thousand updates at batch size four draws 4,000 examples, slightly fewer than the 4,255 recorded observations. This is still a short fitting experiment, not a converged-training claim. Training loss also uses ACT's variational encoder, whereas rollout predicts with the inference-time zero latent; offline scores use that same inference path.

The final diagnostic compares joint prediction error with repeating the current joint position, both over the full 20-target chunk and the first five endpoints used for execution. It also scores the same observations with black camera images. These comparisons help identify weak prediction or dependence on images; they are not substitutes for physical trials, and black images are outside the training distribution.

The saved stage profile in `act-matched-data-profile.json` is for analysis only. No single stage exceeds 13% of the samples. Stage labels and object poses are never policy inputs. All three reserved physical evaluation poses remain unused.

The gripper-contract rejection was exercised against the step-500 checkpoint: requesting 0.35 Nm instead of its recorded 0.25 Nm raised the expected error before loading the policy. Two training-boundary tests and 29 simulator/controller tests passed. The optional evaluation video path was checked by decoding all ten frames of a two-second smoke rollout.

## Completed results

Training completed all 1,000 finite updates in 1,551 seconds (25.9 minutes), with 1.16 GiB peak allocated GPU memory. The final checkpoint is `.run/act-matched-1000/step-001000`. Serialized model weights were checked against the in-memory weights at both saved checkpoints.

**Learned physical task success: 0/5 familiar starts.** All trials executed normally for 60 simulated seconds and ended in task timeout. None achieved a supported hold above 5 cm. The successful nominal replays establish that these five demonstrations are physically executable with this controller; they do not establish that ACT has learned them.

| Start | Maximum bottle lift | Best supported hold | Final placement error |
| --- | ---: | ---: | ---: |
| upright-01 | approximately 0 cm | 0 s | 84.5 mm |
| upright-04 | approximately 0 cm | 0 s | 75.8 mm |
| sideways-01 | 4.49 cm | 0 s | 177.9 mm |
| sideways-03 | approximately 0 cm | 0 s | 153.8 mm |
| sideways-11 | approximately 0 cm | 0 s | 161.4 mm |

All five MP4 videos decoded successfully: 300 frames each at 5 fps, 320×240. They are in `.run/act-matched-evaluation-1000/`. Contact sheets from upright-01 and sideways-01 were visually inspected. The upright example approaches but settles without lifting; the sideways example moves the bottle but does not complete the task. Maximum height alone is not proof of a secure grasp. Physics waits during inference in these offline trials, so their timings are not a real-time deployment benchmark.

Offline scoring uses 128 evenly spaced familiar observations and masks padded future targets:

| Left-arm action prediction MAE | Full 20-target chunk | First five executed endpoints |
| --- | ---: | ---: |
| ACT, normal images | 0.06676 rad | 0.06715 rad |
| Repeat current joint positions | 0.04051 rad | 0.00880 rad |
| ACT, black images | 0.70226 rad | 0.70143 rad |

ACT's short-horizon error is about 3.85°, versus 0.50° for the current-position reference. The reference cannot complete the task by staying still; this comparison shows that low-looking absolute errors must be judged against the small motion required between observations. It does not prove that ACT learned an identity shortcut. Black images strongly worsen prediction, showing image sensitivity under this intervention, not robust visual understanding.

The combined evidence is saved in [act-matched-results.json](act-matched-results.json), reproducible with `scripts/summarize_matched_act.py`. The stage counts remain in [act-matched-data-profile.json](act-matched-data-profile.json). Older 200-update results used different recordings; this is not a controlled causal comparison with them.

## Next decision

Update: the [step-2,000 continuation](ACT_CONTINUATION.md) completed on September 12 and failed its predeclared continuation gate. The 5,000-update run was therefore not started. The proposal below records the decision before that experiment.

Keep the corrected controller and recording contract fixed. The next learning experiment should resume this checkpoint on the same five demonstrations, evaluate at 2,000 and 5,000 total updates, and proceed toward 10,000 only if zero-latent prediction or physical progress improves. Use free Colab or the RTX 4070 for that longer run; a rental is unnecessary for the current model size. Use a packaged snapshot of the matched dataset and preserve its lineage and normalization. Cross-machine camera rendering still needs checking before comparing closed-loop results; local evaluation remains the reference.

At each checkpoint, run the same familiar physical trials and prediction/reference comparison. If progress stalls, inspect action timing, chunk execution and gripper errors before spending on broader data. Test a change such as relative joint actions or execution horizon as a separate experiment, rather than changing several factors together. Do not claim that additional updates are guaranteed to solve the task.

Only after reliable familiar-start manipulation should we train on the wider demonstration split, test the development layouts, and eventually evaluate the untouched reserved poses. More utensils and broad placement coverage remain subsequent milestones. This experiment is complete; reliable learned manipulation remains unfinished.
