# Functioning bottle model goal — September 12, 2026

The user authorized unattended local work toward successful learned bottle manipulation. The goal is physical grasp, supported lift, upright placement and release from RGB/joint observations, without privileged controller inputs, welds, teleports or runtime teacher corrections. A one-start success is followed by repeatability and additional familiar-start checks; broader generalization must be measured separately.

## Constraints

- Reuse the matched recordings and keep at least 10 GiB free. Preserve previous checkpoints; save bounded runs rather than a checkpoint every few updates.
- No paid services or required user actions. A process-scoped helper prevents idle sleep only while on AC power; its sentinel is removed when this goal ends. Permanent power settings are unchanged.
- Keep the verified physical adapter, motion timing and outcome evaluator fixed. Faster submission trajectories remain a separate work item.

## Relative-action ACT experiment

Warm-start the visual/state encoder and transformer from the previous 300-update inference-objective diagnostic. Reset the action head to zero and fit nominal target minus current measured joint position. Use zero residual mean and per-joint residual RMS (minimum 0.02 rad), computed from valid numeric action windows on upright-01 without decoding/copying more images. Keep the direct inference L1 objective, batch four and learning rate 1e-5.

One thousand updates are saved in `.run/act-relative-upright-1000`. Its representation is explicitly stored in normalization metadata. Evaluation adds the current observed joint positions back to predicted residuals before the unchanged actuator adapter. Old absolute-action checkpoints remain compatible. Tests verify round-tripping and reject relative decoding without a measured state. OpenVINO export has the same representation branch, but a new relative checkpoint still requires fresh export validation.

## Separate observation-based retrieval baseline

The baseline fits a PCA representation of the three RGB cameras and stores demonstrated state/action pairs. At runtime it selects a demonstration from current observations, then searches a bounded local window using camera features, joint positions and joint velocities. A weak four-frame temporal tie-breaker resolves nearly indistinguishable stationary observations. It retrieves future action chunks and executes them through the same physical adapter.

This is a **nonparametric, demonstration-conditioned baseline, not ACT, a VLA, or proof of a neural policy learning the task**. It contains demonstrated action sequences. It does not receive the evaluation episode name, simulator clock, object pose, contact state or teacher phase. Successful exact starts alone could resemble replay; actuator-delay and camera-mismatch tests are necessary to assess whether feedback changes progression.

Nearest-neighbor visual imitation has a research precedent in [Pari et al., RSS 2022](https://www.roboticsproceedings.org/rss18/p010.pdf). That work uses learned visual representations and action regression; our PCA/joint features and sequence tie-breaker are engineering choices, not a reproduction of VINN or a state-of-the-art claim.

The first baseline successfully completed upright-01 in 27.245 simulated seconds, with 6.80 cm maximum lift, supported hold and 2.17 mm placement error. A two-second initial actuator hold shifted completion to 29.245 seconds while retaining success. This is evidence against unconditional clock-based playback, not broad robustness. The initial black-camera test timed out rather than explicitly rejecting; version two adds a reconstruction-error threshold derived from the fitted training image residuals. Preserve both versions and their separate results.

Artifacts: `.run/retrieval-fit5`, `.run/retrieval-fit5-v2`, `.run/retrieval-evaluation-fit5`, and the `retrieval-stall` / `retrieval-blank` reports. Tests during concurrent neural training are physical checks, not clean performance benchmarks.

## Selected working baseline and measured limits

The selected artifact is `.run/retrieval-fit5-v4`, a 1.80 MiB nonparametric model fitted from the five existing demonstrations. It uses progress weight `0.01`, selected on the development probes. The original `1e-8` tie-breaker stalled on small observation changes; `0.001` and `0.01` both passed four of five nearby probes. The four-frame prior is an explicit algorithmic contribution: this is observation-conditioned sequence retrieval with memory, not a memoryless visual policy. No runtime teacher is involved.

| Physical check | Outcome |
|---|---|
| Five familiar starts: two upright, three sideways | **5/5 success** |
| Initial bottle x shift +2 mm / -2 mm | Both succeed |
| Initial bottle y shift -2 mm | Succeeds |
| Initial bottle y shift +2 mm | **Fails: timeout without lift** |
| Light diffuse intensity at 90% | Succeeds |
| Two-second actuator hold beginning three seconds into the approach | Succeeds; completion shifts from 27.245 to 29.095 simulated seconds |
| All three cameras blank | Rejects at 0.0 simulated seconds, before movement |

The small shifts are applied once to the initial state. No object state is modified after that reset. The lighting and position probes were used during tuning, so they are **development results, not held-out generalization**. Reserved evaluation poses remain untouched. The failed +2 mm y shift demonstrates that even very nearby arrangements can fail. Do not advertise arbitrary reachable bottle placements.

Upright-01 reaches 6.80 cm lift and finishes within 2.18 mm of the destination. The unchanged observer requires bilateral finger support during lift/hold, no supporting rack/table contact while carrying, upright placement and stable release, limited motion of other objects, and a parked other arm. All five successes have no unsupported carry gap. The policy never receives these observer measurements.

Full evidence and source/checkpoint hashes: [compact results](bottle-model-goal-results.json). Per-cycle traces and videos: `.run/retrieval-v4-familiar/`. Fault reports: `.run/retrieval-v4-midstall.json` and `.run/retrieval-v4-blank.json`. Browser re-runs are preserved separately in `.run/bottle-policy-demo/`.

Verification also passed five training/normalization tests and two control/dataset tests. Browser runs repeated upright-01 and sideways-03 successfully, and cancellation restored the idle controls. The main simulator remained ready. [Input-contract checks](bottle-model-contract-checks.json) confirm rejection of blank images, nonfinite joint/camera values and wrong camera dimensions, plus rejection of unsupported demo starts and cross-origin control requests. Source observations that happen to pass these checks can still be outside the model's useful operating range; these are not comprehensive out-of-distribution guarantees.

## ACT result: improved prediction, still no physical success

The relative-action ACT run completed 1,000 finite updates in 1,317 seconds, using about 0.95 GiB peak allocated GPU memory. On 128 sampled observations from the same upright training start, first-five-target arm MAE fell to 0.003391 rad, versus 0.007701 rad for simply retaining current joint positions. Full-chunk arm MAE was 0.020683 rad and gripper MAE 0.027240 rad.

Despite this, the physical trial timed out at 60 simulated seconds with **no lift**. Offline accuracy is insufficient evidence of a working feedback policy. The relative checkpoint is preserved in `.run/act-relative-upright-1000/step-001000`; its score and failed rollout are `.run/act-relative-upright-score.json` and `.run/act-relative-upright-rollout.json`. Further blind training on these same five nominal trajectories is not the recommended next step.

The delivered functioning controller is the retrieval baseline, **not a successfully trained ACT network**. Its limited success does not complete the hackathon submission or establish a general learned manipulation solution.

## Run and view locally

Open **http://127.0.0.1:8766/** for the separate bottle-policy demo. Choose one of the five supported starts and click **Run policy**. It runs an actual isolated MuJoCo trial and shows the resulting video and physical verdict. The initial video is explicitly a saved verified trial. **Cancel** terminates only that demo trial. Fault modes let you check actuator-pause recovery and blank-camera rejection. The main simulator on port 8765 keeps its existing controls and is independent of this page.

If the demo server is stopped, run from the project root:

```powershell
.\.venv-training\Scripts\python.exe scripts/bottle_policy_demo.py --checkpoint .run/retrieval-fit5-v4 --evidence .run/retrieval-v4-familiar
```

To run a trial directly:

```powershell
.\.venv-training\Scripts\python.exe scripts/evaluate_act.py --checkpoint .run/retrieval-fit5-v4 --episode .run/bottle-matched-nvidia/sideways-03 --device cpu --output .run/my-bottle-trial.json --video .run/my-bottle-trial.mp4
```

Use a new output name to preserve previous results. The evaluator's historical filename supports both ACT and retrieval; its result explicitly records the policy kind. The browser runner uses unique output folders, checks for 10 GiB free plus a small trial budget, permits one trial at a time, and enforces a five-minute wall-time limit. Physics advances at the unchanged 200 Hz and prediction chunks cover four 20 Hz intervals. Evaluation waits for inference; this is **not a real-time benchmark**. Retrieval inference runs on the Intel CPU, while these development renders use NVIDIA. No claim of a fully Intel graphics stack is made.

Refit the same baseline without duplicating images:

```powershell
.\.venv-training\Scripts\python.exe scripts/fit_retrieval_policy.py --output .run/retrieval-refit --progress-weight 0.01
```

ACT OpenVINO export now rejects retrieval checkpoints to prevent tracing NumPy operations into a misleading constant-output graph. Relative ACT export handles its action representation but still needs fresh numerical export validation before use.

## Next development decision

1. Preserve this successful physical baseline as a regression reference. Its feedback and fault tests are more useful than a lower offline loss alone.
2. Expand training around the known failure with a small, storage-budgeted set of varied starts and recovery demonstrations. Record how to correct off-trajectory states, rather than only replaying nominal motions. Keep reserved poses separate and fix the validation protocol before tuning.
3. Train a policy with short observation/action history or an explicit learned progress representation to handle nearly identical stationary observations. Compare its physical success to retrieval, including lighting, actuator-delay and placement-shift checks. Reuse the existing laptop first; a bigger GPU alone does not solve the demonstrated failure.
4. Only after physical success generalizes: integrate language selection and additional dinner objects, validate the required Intel execution path, then evaluate a faster motion profile independently. These submission requirements remain open.

The unattended goal establishes a functioning, limited demonstration-fitted controller and a repeatable local demonstration. It does not establish neural ACT success or arbitrary-placement capability.
