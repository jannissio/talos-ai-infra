# Bottle robustness milestone

This follow-up tests whether a small, targeted addition to the demonstration dataset improves physical behavior beyond the five familiar starts. It continues the nonparametric camera/joint retrieval baseline; it does not claim successful ACT neural training.

## Fixed experiment

[Protocol](bottle-robustness-protocol.json) was written before new data collection and evaluation. It specifies five new upright training placements, a recovery episode from the old controller's actual stalled state, four development cases, and four separate test positions. The old reserved validation file remains untouched. Offsets are relative to the original upright-01 state and remain a small local neighborhood, not the full reachable workspace.

The old controller scores **2/4 on development**: the +4 mm and -4 mm y shifts succeed, while a (+2 mm, +2 mm) diagonal shift and a +2 mm y shift under 90% diffuse lighting time out. These development outcomes may guide selection; test outcomes must not.

The recovery snapshot comes from replaying the old controller's known +2 mm y failure through the unchanged physical evaluator for 60 simulated seconds. A new teacher episode starts from that saved state once. The teacher may use simulator positions to generate labels; the fitted controller only sees RGB images and 12 joint positions plus 12 joint velocities. No teacher corrections run during policy evaluation.

## Physical and storage contract

- Preserve the 200 Hz simulator, 20 Hz recorded observations and interpolated nominal joint targets, and 0.25 Nm feedback gripper cap.
- Require a successful physical teacher trajectory and both 200 Hz / 20 Hz replays before image capture or training eligibility.
- Capture all three 320×240 RGB cameras directly from the resulting physics on the existing NVIDIA rendering path. No old image copies and no image reconstruction from partial saved state.
- Keep new milestone artifacts below 2 GiB, with at least 10 GiB free. Earlier datasets/checkpoints remain intact.
- Preserve failed attempts explicitly. A teacher failure is not evidence of an impossible physical task, and must not be included as a successful demonstration.

New data lives under `.run/bottle-robustness-data`. `scripts/bottle_robustness.py` creates starts, collects verified training demonstrations and writes a lineage referencing the old five episodes in place. The fitter checks episode manifest hashes and frame counts. `scripts/evaluate_bottle_protocol.py` refuses held-out evaluation without a recorded checkpoint/source freeze.

## Results

All six new demonstrations passed the teacher and both replay checks. They contain 4,785 new observations and 14,355 camera images; the fit combines them with the old 4,255 observations by reference (9,040 total). The first recovery attempt failed the teacher's parked-arm precondition and remains saved as `teacher-result-attempt0.json`. The successful retry physically opens the gripper, follows a checked upward retreat and park path, settles, then performs the task. There is one initial state restoration per capture/replay, no teleport into the parked pose.

| Development candidate | Successes |
|---|---:|
| Preserved five-demo baseline, progress weight 0.01 | 2/4 |
| Eleven-demo retrieval, progress weight 0.01 | **3/4** |
| Eleven-demo retrieval, progress weight 0.05 | 3/4 |
| Two-neighbor interpolation, progress weight 0.01 | 2/4 |
| Two-neighbor interpolation, progress weight 0.05 | 3/4 |

The selected candidate is `.run/bottle-robustness-model`: the simplest tied candidate, with the original 0.01 progress weight. It fixes the diagonal and lighting cases but fails the -4 mm y case that the old baseline handled. Its six new training starts all succeed. An aggregate improvement must not hide this individual regression.

Interpolation was a disclosed development experiment. It blends only demonstrations with identical offline phase timelines, with weights obtained from initial camera/joint similarity. Runtime inputs remain camera/joint observations; teacher phase labels are used only for offline alignment. It achieved a lift on the difficult negative-y case but still stalled later and did not improve the overall development score, so it was not selected. Old models default to single-demonstration retrieval.

## Frozen test result: do not promote

The checkpoint, controller/success-check code, runtime versions, protocol and test states were frozen in [the freeze record](bottle-robustness-freeze.json) before either test run. Exact source copies are archived under `.run/bottle-robustness-frozen-sources`; these are archival copies, not a standalone runnable package. There was no tuning after viewing these outcomes.

| Unseen initial offset | Old baseline | Selected candidate |
|---|---|---|
| x +3 mm, y +3 mm | Pass | Pass |
| x -3 mm, y +3 mm | Pass | **Fail: timeout, maximum lift 2.97 cm** |
| x +3 mm, y -3 mm | Pass | Pass |
| x -3 mm, y -3 mm | Pass | Pass |
| **Total** | **4/4** | **3/4** |

**The candidate failed the predeclared promotion gate. The browser on port 8766 continues to use `.run/retrieval-fit5-v4`.** Development improvements did not predict improved performance on the separate test cases. Four nearby tests are a small diagnostic, not a statistically strong estimate of general workspace reliability.

The selected candidate still passes all five original familiar starts, all six new training starts, and a two-second actuator pause during approach. Blank/nonfinite inputs and invalid camera dimensions are rejected; existing control/dataset tests and training/normalization tests passed. The [contract checks](bottle-robustness-contract-checks.json) and [complete compact results](bottle-robustness-results.json) retain the failures as well as successes.

The candidate's recovery run is recorded in `.run/bottle-robustness-recovery-demo.mp4`: it begins from the actual stalled state, physically opens/retreats/parks, then lifts 6.85 cm and places within 0.79 mm. This is a **familiar recovery demonstration**, not evidence that it can recover from arbitrary failures. During this run the fitted controller supplies all actions; the teacher only supplied the offline training labels.

New milestone artifacts use approximately **433 MiB**, below the 2 GiB cap. About **102 GiB remained free** at completion. No old training images were copied or deleted, no paid/cloud resources were used, and normal motion timing and the live simulator controls were retained. The AC-only keep-awake helper is removed when the goal finishes.

## Reproduce and inspect

Read the six new manifests and `teacher-result.json` files under `.run/bottle-robustness-data/`. Each eligible episode has both physical replay results. `fit-lineage.json` references all eleven episodes in place and records manifest hashes. The original recovery precondition failure remains separately preserved.

Refit without duplicating image data, using a fresh output directory:

```powershell
.\.venv-training\Scripts\python.exe scripts/fit_retrieval_policy.py --lineage .run/bottle-robustness-data/fit-lineage.json --output .run/bottle-robustness-refit --progress-weight 0.01
```

Repeat the familiar recovery demonstration with fresh output names:

```powershell
.\.venv-training\Scripts\python.exe scripts/evaluate_act.py --checkpoint .run/bottle-robustness-model --episode .run/bottle-robustness-data/recovery-yplus2 --device cpu --output .run/recovery-repeat.json --video .run/recovery-repeat.mp4
```

The existing test reports remain in `.run/bottle-robustness-baseline-test` and `.run/bottle-robustness-selected-test`. Do not claim those four positions are still an untouched evaluation set in a future tuning cycle. Keep the older reserved validation poses separate.

## Next step

Retain retrieval as a physical reference, and stop tuning nearest-demonstration matching for now. The next learning experiment should use short observation/action history or a recurrent policy to represent progress through the movement, train on the verified recovery sequence as well as nominal motions, and include closed-loop physical validation from the beginning. The new examples are available through `load_policy_episode`; they have not yet been converted into a new LeRobot/Colab dataset or used to train a new ACT network.

If a sequence-aware policy still leaves its training trajectories, collect teacher corrections from those specific states within another fixed budget. Define a new development/test boundary before using current test failures to guide learning. A larger GPU alone does not address the demonstrated action-selection and feedback errors.
