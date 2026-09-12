# ACT learning milestone — 11 September 2026

## What this milestone establishes

The physical teacher and its verified demonstrations are now connected to an actual LeRobot ACT training and MuJoCo evaluation pipeline. A learned policy is not considered successful merely because its training loss decreases. The first run is a bounded, 200-update familiar-start diagnostic. Generalization to other bottle placements, other objects or both arms remains unproven.

The current Colab notebook is [Talos ACT Bottle Pilot](https://colab.research.google.com/drive/1LmRi42yWNRJ_-nTTEiTiGFC2RpIx-tHz). Its reviewable source is `notebooks/Talos_ACT_Bottle_Pilot.ipynb`. No dataset or checkpoint is published to the Hub, and no GPU was rented. Colab allocated a Tesla T4 with 15,360 MiB memory. Its default Python was 3.13; the experiment uses an isolated Python 3.12 environment instead.

## Reproduction and action contract

- LeRobot 0.6.1, PyTorch 2.8.0, torchvision 0.23.0, NumPy 2.2.6 and MuJoCo 3.12.0. See `requirements-training.txt`. The simulator's original `.venv` is preserved; local training uses `.venv-training` with official PyTorch CUDA 12.8 wheels.
- ACT uses its ResNet-18 ImageNet backbone, transformer and conditional VAE. It predicts 20 joint-target vectors, replanning after four 50 ms intervals. Five endpoints are interpolated at the 200 Hz physics rate. This is a short-horizon receding-horizon controller, not a language model directly driving motors.
- Inputs: overhead, left-wrist and right-wrist RGB at 320×240, 12 joint positions and 12 joint velocities. No object coordinates, teacher stage, contacts or clock are supplied to the policy. Contact information is reserved for the outcome evaluator.
- Targets: the 12 recorded nominal actuator position targets. Normalization uses only the training split. Standard deviations have a 0.02 floor; RGB uses the ResNet ImageNet mean/std. No validation normalization enters the model.
- Initial physical states are restored once per trial. During execution there are no state resets, welds, external forces or teacher corrections. Actions are clipped to actuator control ranges; the left gripper uses an explicit fixed torque limiter.
- Physics waits during image inference in this offline evaluation. Nominal control frequency is not a claim of real-time performance. Median and p95 inference latency are reported separately.

## Dataset boundaries

`scripts/convert_act_dataset.py` creates local LeRobot v3 datasets with images embedded in Parquet. The five-episode fit diagnostic has 4,255 nonterminal samples: sideways-01, sideways-03, sideways-11, upright-01 and upright-04. The source policy loader rejects excluded/incomplete episodes and emits only allowed fields. `talos_lineage.json` records episode IDs and SHA256 hashes of source manifests.

The larger development split reserves sideways-05, sideways-08, upright-08 and upright-10 for validation, leaving 16 training episodes. Splits are whole episodes, never shuffled frames across train/validation. These are nearby poses in one surrounding layout, not a convincing out-of-distribution benchmark. The three previously reserved poses remain separate. The five-episode fit set is a subset of training and must never be described as held-out evaluation.

All three conversions completed and were reopened successfully: `.run/act-fit5` (4,255 frames), `.run/act-train16` (13,428 frames) and `.run/act-validation4` (3,393 frames). Their source-ID separation is recorded in `act-dataset-conversion.json`.

## Gripper mismatch found during preparation

The original teacher uses 0.15 Nm for upright bottles and 0.65 Nm for sideways ones. Inferring that cap from simulator bottle orientation at policy runtime would leak privileged information. Fixed-cap physical replay was therefore checked independently.

| Fixed cap | Passing pilot episodes | Limitation |
| --- | --- | --- |
| 0.15 Nm | 18/20 | sideways-01 and sideways-05 fail |
| 0.18 Nm | 18/20 | sideways-13 and upright-10 fail |
| 0.25 Nm | 18/20 | upright-08 and upright-10 fail |
| 0.35 Nm | 18/20 | upright-08 and upright-10 fail |

The selected five-episode fit set passes physical 20 Hz teacher-target replay at 0.25 Nm. That cap is explicitly used for the first diagnostic. Original images were recorded under the teacher's original caps, so even these starts have some observation distribution mismatch. **Do not interpret a 20-episode policy score until this contract is resolved.** The next data iteration should either recollect demonstrations under a verified common low-level controller, or explicitly learn the commanded gripper force limit as an additional action. It must not select a cap from ground-truth bottle pose at runtime.

## Running locally

```powershell
.\.venv-training\Scripts\python.exe scripts/convert_act_dataset.py --output .run/act-fit5 --split fit5
.\.venv-training\Scripts\python.exe scripts/train_act.py --dataset .run/act-fit5 --output .run/my-act-run --steps 200 --batch-size 4 --device cuda
.\.venv-training\Scripts\python.exe scripts/evaluate_act.py --checkpoint .run/my-act-run/step-000200 --episode .run/bottle-pilot/upright-01 --output .run/my-act-run/eval-upright-01.json --seconds 60 --gripper-cap .25
```

Use a new output directory. Training saves weights, configuration, normalization, source lineage, optimizer/scaler state and metrics. `--resume PATH --steps NEW_TOTAL` resumes optimization, but restarts shuffled data iteration; it is not bitwise-identical continuation. Optimizer state is loaded with `weights_only=True`. The saver verifies that every serialized parameter equals the in-memory parameter. Each log records actual finite optimizer updates. An initial mixed-precision smoke attempt had nonfinite gradients; reducing the scaler's initial scale to 128 fixed this, and the failed attempt is not counted as trained.

## Physical evaluation and checks

The first Colab run completed 200/200 finite updates at batch size 8 in 336.48 seconds (about 1.68 seconds/update), with peak allocated GPU memory 1,751,501,312 bytes (1.63 GiB). Logged minibatch L1 was 0.615 initially and 0.180 at step 200; these are different training minibatches, not a validation score. This processes 1,600 sampled observations, less than one pass over the 4,255-frame fit set. Do not treat it as converged training.

The first completed Colab physical trial (`upright-01`, 60 simulated seconds) **failed**: no supported hold, no stable release, and final placement error 61.4 mm. It used zero teacher updates and one initial reset. Median inference was 23.87 ms (p95 30.51 ms), but the full simulation took 230.56 seconds wall time. The subsequent cloud trial was interrupted deliberately so evaluation could move to the faster local simulator. This is not a five-trial score.

Colab exported `talos-act-result.zip` (570,161,387 bytes, SHA256 `8020f35cb49b0248eed464476a09a48b786b0d439f5c311cb908cd6ecadff014`). It includes `step-000200`, optimizer/scaler state, logs, environment freeze and the first evaluation. The original uploaded source/data bundle is preserved locally as `.run/talos-act-colab-first-run.zip`; its SHA256 is `98c1f276eb73e46e6b9eed1b46a4347f5be4bd813f6fc51219850f52cf87e5ed`. The rebuilt upload bundle contains subsequent evaluator diagnostics; its hash is generated beside it and inserted into the local notebook.

**Transfer verified:** the user saved `.run/talos-act-result.zip`. Its SHA256 matches the Colab export and every ZIP entry passed CRC verification. The checkpoint, optimizer/scaler state and results are extracted in `.run/act-colab-200`. The trained checkpoint loads on the local Quadro T2000; `scripts/evaluate_act_batch.py` runs the five familiar starts sequentially and writes `.run/act-local-evaluation-200/summary.json`. The Colab runtime is no longer the only copy of the trained checkpoint.

### Completed local physical evaluation

The unchanged step-200 checkpoint completed all five 60-second trials, with **0/5 successes** and zero verified supported hold in every trial. Every trial ended as a task timeout, not a program error. These are familiar training starts, not held-out physical trials. Results are preserved in `act-local-evaluation-200.json`.

| Familiar start | Final placement error | Wall time | Median prediction latency |
| --- | ---: | ---: | ---: |
| upright-01 | 57.7 mm | 24.8 s | 41.3 ms |
| upright-04 | 46.3 mm | 33.5 s | 56.5 ms |
| sideways-01 | 202.7 mm | 33.4 s | 53.2 ms |
| sideways-03 | 153.8 mm | 29.6 s | 50.3 ms |
| sideways-11 | 161.4 mm | 28.6 s | 49.0 ms |

All trials used the same 0.25 Nm cap, one initial reset, zero teacher updates and no clipped predictions. The local and Colab `upright-01` trajectories differ; these measurements do not establish identical rendering or trajectories across platforms. Both platforms agree that this checkpoint fails the task. The simulator's existing HTTP state endpoint still reports ready.

`scripts/evaluate_act.py` observes physical contacts independently of the policy. Success requires bilateral support above 5 cm for at least 1.49 seconds, upright placement within 12 mm, low translational speed, table support and released jaws for one stable second, limited contact loss, less than 4 mm movement of other objects and at most 1° movement of the other arm. A timeout is a failure. This evaluator does not call the teacher's planning or update methods.

The initial three-update local checkpoint was intentionally tested as an inadequately trained policy: it loaded, moved through the normal actuator interface, and timed out without a successful lift. Checkpoint resume performed a further finite update. Dedicated tests check normalization round-tripping, finite values with constant joints, exclusion of privileged fields, camera shapes and blank-image input. The browser/server path is unchanged.

`scripts/score_act.py` additionally measures offline zero-latent action prediction error with the same inference path as rollout. It removes target actions before prediction, applies padding masks only when scoring, and uses checkpoint normalization on the development split. That metric is distinct from both VAE-conditioned training loss and closed-loop physical success. The existing 28 simulator tests and two new training-boundary tests passed.

The downloaded step-200 checkpoint was scored on 128 evenly spaced observations per dataset. Mean absolute left-arm joint error was **0.0885 rad on fit5** and **0.1142 rad on the four development-validation episodes**; gripper error was 0.1172 and 0.1069 rad respectively. These are offline action-prediction errors, not physical success rates. The three separately reserved poses were not evaluated. Both score reports are in `act-offline-scores-200.json`.

Cross-machine resume also passed: the downloaded Colab optimizer/scaler state produced one finite local update at step 201 and a verified checkpoint under `.run/act-colab-resume-local/step-000201`. This is a resume check only; all reported physical and prediction results use the original step-200 checkpoint.

## Next decision

Update: the matched-controller follow-up has now completed. See [ACT_MATCHED_EXPERIMENT.md](ACT_MATCHED_EXPERIMENT.md) for verified recording alignment, the fresh 1,000-update run, five physical failures, video evidence and the next learning gate. The figures above remain the historical 200-update experiment.

Inspect physical failures from this first run before spending on larger training. Resolve the gripper action contract, fit a small demonstration set successfully, then train on the 16-episode split and evaluate the four development episodes. Expand collision-free position/orientation coverage and camera/appearance variation only with successful physical baselines. Use the RTX 4070 or free Colab before renting a GPU. Scaling a model that cannot reliably reproduce its familiar starts is unlikely to solve the underlying data/control mismatch.

The immediate next experiment should recollect the five pilot demonstrations under the exact fixed gripper controller used for rollout, then perform a longer small-set fit with periodic physical evaluation. The 200-update diagnostic is less than one dataset pass, and the control/observation mismatch is known; this experiment has not isolated which factor contributes most to failure. Passing familiar-start physical trials is the gate before claiming generalization or broadening the task to more utensils. The training-pipeline milestone is complete; reliable learned manipulation is still outstanding.

This implementation follows the ACT/LeRobot evidence and staged evaluation plan in `TRAINING_RESEARCH_PLAN.md`; it does not claim a novel algorithm or state-of-the-art benchmark result. The camera policy remains separate from the final Intel hardware inference and submission requirements.
