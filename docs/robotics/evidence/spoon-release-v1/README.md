# Spoon release reliability experiment

The revised spoon model **does not pass its full-workflow development gate and is not promoted**. All 24 paired trials finish: the original and candidate each complete **4/6 standard dinner workflows and 4/6 left-reach workflows**. Every run that reaches the spoon succeeds with either model. Two scenes fail earlier at the unchanged mug skill. The reserved final seeds, 2026099301–2026099312, remain unexposed.

| Development seed | Standard: baseline / candidate | Left reach: baseline / candidate | Failure in both models |
| --- | --- | --- | --- |
| 2026099201 | Fail / Fail | Fail / Fail | Mug XY error: 8.515 / 8.522 mm by preset |
| 2026099202 | Pass / Pass | Pass / Pass | None |
| 2026099203 | Pass / Pass | Pass / Pass | None |
| 2026099204 | Fail / Fail | Fail / Fail | Mug XY error: 8.018 / 8.010 mm by preset |
| 2026099205 | Pass / Pass | Pass / Pass | None |
| 2026099206 | Pass / Pass | Pass / Pass | None |

The acceptance limit remains 8 mm, including the near-boundary failures. Each model reaches and passes eight spoon attempts out of twelve full workflows. These are conditional spoon outcomes, not twelve successful sequences. Baseline spoon error ranges from 0.201 to 3.384 mm (median 0.836); the candidate ranges from 0.607 to 2.229 mm (median 2.055). The smaller observed maximum does not establish improved reliability. No scene is removed, tolerance relaxed or checkpoint reselected.

## Intervention and selection

The [declared protocol](protocol.json) follows an [exposed release diagnostic](../dinner-spoon-diagnostic-v1/README.md). It changes only the existing spoon demonstration's lower/release/retract action labels: stop lowering 3 mm higher, open the fingers over one second while holding the arm still for 1.25 seconds, then retreat. The original path, durations, preceding/following labels, visual features, inactive-arm actions and runtime remain fixed. This allows a small settling drop under the unchanged physical monitor.

All [39 exact modified training trajectories](../../../../training/spoon_release_v1/README.md) pass independent physical replay. One 12,000-update warm-start fit finishes on the RTX 4070 in **43.674 seconds**, using **0.0488 GiB peak Torch-reserved memory**. FP32 OpenVINO export passes 32 synthetic parity batches on this PC's AMD CPU and NVIDIA GPU. These measurements exclude rendering and physics and are not Intel results.

The evaluator and both complete model/configuration sets were frozen in `e20ed34239ef89f1a01854e0b2ab4abe9fac8e6b` before physical selection. The original protocol was already committed in `b68d4ea`. The candidate SHA-256 is `8a6bec4cf6f0fc4e2743ac9336b2714f86d684be3750a6d2ba05ed0b7836a03c`. The exact [model and export](../../../../models/spoon_release_v1/README.md) are preserved separately from the selected dinner suite.

The development gate requires at least 5/6 complete candidate workflows per preset and no loss against the paired baseline. It fails at 4/6 in each preset. The protocol therefore stops before its 48 final paired trials. It provides no evidence for arbitrary reachable positions, new destinations, continuous visual correction, Intel execution or human voice.

## Physical evidence

The evaluator invokes the actual browser engine's `set the table` language/RGB planning path. Standard starts select six skills; left-reach starts select both reverse bottle-relay legs followed by plate, mug, drawer, fork and spoon. No supplied skill list replaces the planner. Only the spoon model changes, and the privileged monitor only stops/scores execution.

- [All 24 outcomes](development/summary.json), [exact pre-run freeze](development/frozen-inputs.json), [package manifest](package-manifest.json) and [independent audit](audit.json).
- 112,320 synchronized state frames, all 24 initial states and portable scene XMLs, every report and per-trial console log. Only mesh-directory references change in scene XMLs.
- All 12 paired starts match. An independent array comparison confirms identical states, times, motor targets and progress before the spoon stage in every pair.
- Every passed skill retains release, parking, contact, collision, disturbance and placement checks. All controller updates are checked for physical-state writes and hidden forces. Reconstructing the final spoon geometry agrees with the reports to within 1.18e-9 mm.
- The complete batch takes 420.73 wall seconds with at most three isolated workers. This is an offline evaluation timing, not hosted-demo latency.

The package retains 188 files totaling 95,544,718 bytes before these README files and the subsequent portability check. The cumulative protocol budget is 1 GiB including raw and packaged evidence; every batch/episode/export checks the 10 GiB reserve. All original published files remain unchanged.

An [isolated portability check](portability.json) copies only the required packaged/repository files into a new minimal checkout, restores the frozen lookup paths and executes exposed candidate seed 2026099202. All six skills pass in 247.455 simulated / 52.78 wall seconds. The initial state, all physical metrics and **every synchronized state array match exactly**. The [trace audit](portability-trace-audit.json), [complete physical report](portability-report.json), source procedure and [separate manifest](portability-manifest.json) preserve this reproduction without changing the original package manifest. It is not another development or final selection case.

## Reproduction

From a fresh clone with the documented training/runtime environment:

```powershell
.venv-training/Scripts/python scripts/package_spoon_release.py --verify-only
.venv-training/Scripts/python scripts/package_spoon_release.py --restore-evaluation-inputs
.venv-training/Scripts/python scripts/evaluate_spoon_release.py trial --freeze docs/robotics/evidence/spoon-release-v1/development/frozen-inputs.json --seed 2026099202 --preset upright --controller candidate --output .run/spoon-release-reproduction-9202
```

Restoration recreates only the exact frozen model/report lookup paths under `.run/spoon-release-v1`, using packaged files. It refuses an existing destination. On the original development PC these inputs already exist, so omit restoration. Always use unused trial and console-log paths. These development seeds are now exposed reproductions; do not relabel them as new holdouts or tune this stopped protocol further.

The selected browser, hosted Space and submission media continue using `models/dinner_suite/suite.json`. A future reliability experiment must diagnose the preceding mug failures and declare fresh selection scenes separately; more spoon fitting within this protocol is not justified by these results.
