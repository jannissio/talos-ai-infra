# Mug release fit: failed development gate

The mug candidate is **not promoted**. All 24 paired development workflows finish: the preserved baseline passes **6/6 standard and 6/6 left-reach starts**; the candidate passes **4/6 in each preset**. It regresses on two mug-placement scenes that the baseline passes. The reserved final seeds, 2026099701–2026099712, remain unexposed.

| Development seed | Standard: baseline / candidate | Left reach: baseline / candidate | Candidate failure |
| --- | --- | --- | --- |
| 2026099601 | Pass / Pass | Pass / Pass | None |
| 2026099602 | Pass / Fail | Pass / Fail | Mug XY error: 8.623 / 8.634 mm by preset |
| 2026099603 | Pass / Pass | Pass / Pass | None |
| 2026099604 | Pass / Pass | Pass / Pass | None |
| 2026099605 | Pass / Fail | Pass / Fail | Mug XY error: 10.655 / 10.662 mm by preset |
| 2026099606 | Pass / Pass | Pass / Pass | None |

The unchanged placement limit is 8 mm. The development gate requires at least 5/6 complete candidate workflows per preset and no fewer completed workflows or passed mug attempts than the baseline. The candidate fails both requirements. All four failures remain in the denominator; no tolerance, scene, checkpoint or runtime is changed after observing these results.

## Declared intervention

An [exposed-state diagnosis](../dinner-mug-diagnostic-v1/README.md) found error both before lowering and during release. The [separate protocol](protocol.json) therefore combines two changes: revised stationary finger opening and a weighted warm fit. It does not isolate which component causes a result.

All [41 modified float32 demonstrations](../../../../training/mug_release_v1/README.md) pass physical action replay, with a maximum placement error of 5.951 mm. Lowering stops 1 mm higher on the existing path. The fingers open over one second while the arm holds still for 1.25 seconds, then withdraw sideways. Original durations, visual inputs, other stages, inactive-arm commands and all physical thresholds remain unchanged.

One **18,000-update** fit weights alignment/lowering/release endpoints fourfold and retains every training endpoint. It uses the original v6 normalization and architecture and finishes on the RTX 4070 in **70.569 seconds**, with **0.0488 GiB peak Torch-reserved memory**. Weighted normalized MSE drops from 0.012604 to 0.0000751, but that improvement in label agreement does not translate into better physical completion.

The sole candidate has SHA-256 `e3866a1782fb7525823106d0c44421e412eed67a3682c37e29022de7eb1d391d`. Its FP32 OpenVINO export passes 32 synthetic parity batches on this PC's AMD CPU and NVIDIA GPU. These are network-only measurements, not Intel execution or physical success evidence. [Exact fit, export and suite](../../../../models/mug_release_v1/README.md) are retained separately from the selected baseline.

## Frozen physical comparison

The evaluator and both suites were frozen in `8af8223a32fe9305befd21f54b2e42c45ef957e1` before any of these workflows ran. The protocol was already committed in `1d014c7a`. The evaluator checks that no gripper, camera, normalization or controller metadata changed; all other models remain byte-identical.

Each isolated trial calls the actual language/RGB engine with `set the table`. Standard starts select six learned skills; left-reach starts select both reverse bottle-relay legs followed by plate, mug, drawer, fork and spoon. The planner receives no supplied skill list. Later spoon or other failures would still count as failed full workflows.

- [Every outcome](development/summary.json), [exact freeze](development/frozen-inputs.json), [package manifest](package-manifest.json) and [independent audit](audit.json).
- All 24 initial states and portable scenes, **119,878 synchronized state frames**, every report, per-trial log and frozen source. Only mesh-directory references change in scene XMLs.
- All 12 paired starts match exactly; every pre-mug state, time, target and progress array is identical within its pair. Passed skills retain the unchanged contact, release, parked-arm, collision and disturbance checks.
- The audit reconstructs final mug geometry separately from the error reported at mug completion, which occurs before later skills. The largest difference is only **0.000577 mm**. The existing sequence's 4 mm displacement allowance is unchanged; the audit does not treat different timestamps as identical states.
- The batch takes **443.57 wall seconds** with at most three workers. It is an offline evaluation timing, not browser or cloud latency.

The core package contains 189 files totaling 101,677,508 bytes before these README files and the separate portability record. Disk checks preserve 10 GiB plus expected writes; the protocol allows at most 1 GiB across raw and packaged artifacts. All earlier evidence and original models remain intact.

An [isolated portability run](portability.json) restores the required packaged files into a fresh minimal checkout and repeats exposed candidate seed 2026099601. All six skills pass in 247.475 simulated / 53.52 wall seconds. Every physical metric, initial state and synchronized state array matches exactly, including the NPZ file hash. The [physical report](portability-report.json), captured reproduction script and [separate manifest](portability-manifest.json) retain this additional deployment check. It is not a new selection case; its identical state arrays are already preserved in the original trace.

## Reproduction and next work

From a fresh clone with the documented environment:

```powershell
.venv-training/Scripts/python scripts/package_release_workflows.py --protocol docs/robotics/experiments/mug-release-v1.json --verify-only
.venv-training/Scripts/python scripts/package_release_workflows.py --protocol docs/robotics/experiments/mug-release-v1.json --restore-evaluation-inputs
.venv-training/Scripts/python scripts/evaluate_release_workflows.py --protocol docs/robotics/experiments/mug-release-v1.json trial --freeze docs/robotics/evidence/mug-release-v1/development/frozen-inputs.json --seed 2026099601 --preset upright --controller candidate --output .run/mug-release-reproduction-9601
```

Restoration refuses an existing `.run/mug-release-v1`; omit it on the original development PC, where the exact inputs already exist. Use unused output and console-log paths. These development seeds are exposed reproductions and cannot become new holdouts.

This stopped fit does not justify another similar warm fit. The next architectural direction is to measure and correct placement visually while the object is held, beginning with a separately declared RGB observability check. That capability remains unfinished. No selected browser, hosted model, submission media or physical acceptance rule changes as a result of this experiment. No broader workspace, Intel, microphone or anonymous-access claim is made.
