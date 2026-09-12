# Neural bottle controller — larger milestone results

## Outcome

A working neural motion primitive is available at **http://127.0.0.1:8768/**. It was trained locally on all eleven verified demonstrations, using initial three-camera PCA features to condition generated joint trajectories. A programmed motor-feedback guard controls progression and handles the tested short actuator pause. No demonstrated action arrays, exact object poses, simulator clock, teacher corrections, attachments or teleports are used during execution. The neural model has its own internal progress variable; it does **not** continuously correct its trajectory from vision.

| Evaluation | Neural primitive | Existing retrieval |
|---|---:|---:|
| Original familiar starts, included in training | 5/5 | Previously 5/5 |
| All eleven training starts, including the five above | 9/11 | Not rerun in this goal |
| Eight previously exposed development positions/conditions | 4/8 | Not rerun in this goal |
| Six newly frozen held-out starts | **3/6** | **6/6** |

The neural model passed the recorded recovery start. The two failed training starts were `train-yplus6` and `train-yminus6`. New-test failures were southeast and both shifted sideways starts. The new test evidence was not used for tuning. These small fixed suites are diagnostics, not broad success-rate estimates.

The upright repeat lifted approximately 6.84 cm and placed within 1.730 mm in 27.205 simulated seconds. A two-second actuator pause beginning at 3 s passed with 1.729 mm error in 29.205 seconds. A 20-second injected stall stopped at 13.2 seconds. Rendering/inference are synchronous offline; these are not real-time or full-Intel-rendering claims. Physics and actuator timing remain 200 Hz, with 20 Hz trajectory endpoints and 5 Hz policy calls.

**Decision:** retain the retrieval demo at port 8766 as the established baseline. The neural model is a separate learning milestone, not a robustness improvement. Port 8767 retains the earlier single-start guarded GRU example. Automatic approval review rejected restarting that server without a specific reason, so the new model received its own port instead.

The next development target is continuous visual correction and corrective demonstrations for failed placements, using a fresh evaluation split. Renting a larger GPU is not the immediate remedy for missing feedback/data coverage. The present model also does not support arbitrary reachable arrangements, changed camera calibration, moving objects, utensils beyond the bottle, or autonomous use of both arms. Do not expand submission claims from these results.

Final interface checks: the new port 8768 page displayed the correct model description and all five original starts. A server-triggered sideways-01 trial passed (9.339 cm lift, 0.240 mm placement error); a normal upright repeat passed; blank cameras rejected before any simulation step; cancellation returned the runner to idle. The interface plays completed trial recordings. Nine learning tests, syntax checks and post-comparison frozen-file hash verification passed. The main simulator still reports ready, and the retrieval demo remains available. Sequence-goal artifacts occupy approximately **70.5 MiB**, with **101.5 GiB free**; no additional training images were copied or old artifacts deleted. The temporary AC keep-awake sentinel was removed after training and evaluations finished.

The larger goal was physical success with a neural camera/joint controller, repeatability and recovery, followed by a newly frozen comparison on unseen starts. Recurrent action learning was investigated first; measured failures motivated the simpler motion-primitive architecture. No paid services were used. Additional artifacts are capped at 2 GiB with a 10 GiB disk reserve.

## Representation and execution

`SequenceNet` has a 56→128 input encoder, a two-layer 128-unit GRU and an action head producing 20×12 joint targets. Inputs are 32 fixed PCA RGB features and 24 measured joint values (positions/velocities). PCA was fitted only on the eleven training demonstrations. This is neural action learning over fixed visual features, not end-to-end image training, ACT, or a VLA.

Training uses complete episodes sampled at the evaluator's 5 Hz replanning rate. The existing 200 Hz actuator adapter executes four 20 Hz intervals per prediction. A causal/incremental inference test verifies that recurrent step-by-step inference matches full-sequence training. No simulator clock, object poses or teacher stages enter the network. The checkpoint contains neural weights, state/output normalization and the PCA transform; it does not contain demonstrated actions or nearest-neighbor tables.

There are relative-target and absolute-target diagnostics. `sequence.json`'s `action_mode` is the authoritative decoding convention. Small proprioceptive/image-feature perturbations are synthetic augmentation, not additional physically verified demonstrations. For relative targets, perturbed joint observations are subtracted from the targets too, preserving the intended absolute action. Absolute-target labels stay unchanged.

The existing ACT OpenVINO exporter rejects recurrent checkpoints: a correct export must expose hidden state explicitly.

## Experiments so far

- `.run/bottle-sequence-first`: eleven-episode relative model, 3,000 updates. Upright-01 lifted 3.40 cm but timed out without success.
- `.run/bottle-sequence-refine`: 15,000 additional updates, lower training loss. The 10,000-update diagnostic lifted 3.28 cm without success; the final checkpoint also failed. Better training fit alone did not solve closed-loop behavior.
- `.run/bottle-sequence-noise`: eleven-episode relative model with 0.003 rad joint perturbations and 0.01 normalized visual-feature noise. The 10,000-update physical test failed.
- `.run/bottle-sequence-one`: single upright-01 relative model with the same perturbations, 15,000 updates. Physical test failed without lifting.
- `.run/bottle-sequence-absolute`: single upright-01 absolute-target model, 10,000 updates. Physical evaluation failed. A recorded-image audit nevertheless confirmed training/runtime feature and action equivalence to below 0.000001 rad for executed endpoints.
- `.run/bottle-sequence-absolute-feedback`: first-five-endpoint loss and stronger position/feature perturbations. Failed physical evaluations.
- **`.run/bottle-sequence-velocity/step-010000`: first neural physical success**, trained only on upright-01 with position, velocity and visual-feature perturbations. Lift 6.8559 cm, supported hold 3.595 s, final placement error 0.994 mm, 27.405 simulated seconds. Repeated successfully through the separate demo server. This is the identical familiar initial state, not randomized robustness. Two-second actuator pause at 3 s failed. Blank-camera trial rejected at zero simulated seconds.
- `.run/bottle-sequence-all-feedback/step-025000`: eleven-episode expansion failed all 11 training-start physical trials. The 10,000-update checkpoint also failed the original five. No promotion.
- `.run/bottle-sequence-stutter`: synthetic repeated pre-grasp observations to mimic actuator pauses. Both 10,000- and 20,000-update normal trials failed; no promotion. These are observation augmentations, not new verified demonstrations.
- `.run/bottle-sequence-context`: retaining the first camera/joint observation as explicit recurrent context did not solve physical control: the final checkpoint failed all eleven starts. Incremental execution has a matching contract test.
- `.run/bottle-sequence-guard`: endpoint-hold diagnostic. Threshold was chosen above the normal trial's maximum 0.00270 rad endpoint lag. The short-pause trial passed lift/hold but failed placement.
- **`.run/bottle-sequence-guard-replay`: normal and short-pause physical success.** This separately saved variant repeats the last neural-predicted segment when arm-joint lag exceeds 0.005 rad, without advancing recurrent memory. It contains no demonstrated action table. Normal: 6.8559 cm lift, 0.994 mm error, 27.405 simulated seconds. Two-second actuator pause at 3 s: 6.8531 cm lift, 1.385 mm error, 29.205 seconds, 10 guarded waiting calls. A 20-second injected stall rejected at 13.2 seconds with a tracking-timeout explanation. This is **programmed tracking recovery combined with neural action prediction**, not evidence of learned recovery. The short-pause trial repeated successfully through the demo server. Three contract tests pass, including unchanged recurrent memory during waiting and bounded rejection.
- **`.run/bottle-sequence-primitive-tight/step-000600`: selected eleven-episode neural motion primitive.** A three-layer 256-unit network maps initial PCA camera features and internal progress Fourier features to joint targets. Measured-joint tracking gates progress. Cameras subsequently provide rejection checks, **not continuous visual corrections**. This is a learned motion primitive, not a GRU, ACT or VLA. Initial Adam fits failed, as did early full-dataset L-BFGS refinement. Tightening numerical stopping tolerances eventually produced a physically successful fit: all five original familiar starts passed, including three sideways scenes. Upright-01 placed within 1.730 mm in 27.205 simulated seconds. A two-second approach pause also passed, within 1.729 mm in 29.205 seconds. A persistent stall rejected at 13.2 seconds. Nine learning contract tests passed. Final results are 9/11 training starts and 3/6 new held-out starts; retrieval passed 6/6 on those same new starts.

All runs reuse compact features from `.run/bottle-robustness-model/retrieval.npz`; no training images were copied. Prior weights/results remain saved. Low training loss is not success evidence. The final selection is supported by physical familiar-start and recovery successes, with generalization limitations independently measured.

The separate neural experiment is at `http://127.0.0.1:8767/`, using the guarded successful upright-01 checkpoint. The retrieval demo at port 8766 is unchanged. The neural page explicitly identifies the algorithm, familiar-start scope, programmed tracking guard and diagnostic fault tests. Cancellation and blank-camera rejection were verified on the unguarded neural server before promotion. It displays a completed recording, not a live teleoperation stream.

Restart the current limited neural demo:

```powershell
.\.venv-training\Scripts\python.exe scripts/bottle_policy_demo.py --checkpoint .run/bottle-sequence-primitive-tight/step-000600 --evidence .run/bottle-sequence-primitive-demo-normal.json --port 8768
```

Rebuild compact physical results and the disk audit with `scripts/summarize_sequence_goal.py`; see [machine-readable results](bottle-sequence-results.json). The GRU demo's weights came from a single upright episode. The later selected motion primitive was trained on all eleven episodes and now has broader familiar-start evidence, but no continuous visual correction.

## Evaluation boundary

[The new protocol](bottle-sequence-protocol.json) defines six new test starts. The eight exposed robustness development/test positions are now development data and will not be relabeled unseen. New held-out trials must follow a checkpoint/code/state freeze; no tuning from those results in this goal. The older reserved validation poses remain untouched.

The selected primitive, runtime source snapshot and all six test states are recorded in [the immutable selection freeze](bottle-sequence-freeze.json). Selection used original familiar-start and pause success; the new test results must not influence weights, control or criteria in this goal. All training is now stopped.

Reproduce the selected model (CUDA training environment, fixed seed 31):

```powershell
.\.venv-training\Scripts\python.exe scripts/train_primitive_policy.py --output .run/bottle-sequence-reproduce-adam --steps 40000 --save-every 10000
.\.venv-training\Scripts\python.exe scripts/train_primitive_policy.py --output .run/bottle-sequence-reproduce-lbfgs --steps 200 --save-every 50 --lr 1 --optimizer lbfgs --lbfgs-tolerance-change 1e-9 --lbfgs-tolerance-grad 1e-7 --warm-start .run/bottle-sequence-reproduce-adam/step-040000
.\.venv-training\Scripts\python.exe scripts/train_primitive_policy.py --output .run/bottle-sequence-reproduce-precise --steps 1000 --save-every 250 --lr 1 --optimizer lbfgs --lbfgs-tolerance-change 1e-9 --lbfgs-tolerance-grad 1e-7 --warm-start .run/bottle-sequence-reproduce-lbfgs/step-000200
.\.venv-training\Scripts\python.exe scripts/train_primitive_policy.py --output .run/bottle-sequence-reproduce-tight --steps 600 --save-every 200 --lr 1 --optimizer lbfgs --warm-start .run/bottle-sequence-reproduce-precise/step-001000
```

These commands reproduce the selected optimizer schedule: Adam 40,000 → initial L-BFGS 200 (original default tolerances) → 1,000 more outer calls (mostly stopped early) → tighter L-BFGS 600 outer calls. GPU numerics can still prevent bit-identical retraining. Every parent checkpoint and its arguments remain saved. The final weights are pinned by SHA-256 for exact evaluation reproducibility. Do not expect a fresh fit to pass without physical evaluation.

```powershell
.\.venv-training\Scripts\python.exe scripts/evaluate_act.py --checkpoint .run/bottle-sequence-primitive-tight/step-000600 --episode .run/bottle-matched-nvidia/upright-01 --device cpu --output .run/bottle-sequence-repeat.json
```
