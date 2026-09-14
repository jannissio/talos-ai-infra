# Mug release experiment: exact training inputs

All **41/41 revised float32 action trajectories pass independent physical replay** with the unchanged dinner monitor. The maximum placement error is **5.951 mm**, below the existing 8 mm limit, with physical release and parked arms. Summed replay wall time on this PC is 231.51 seconds. These are demonstration results, not learned-policy or Intel evidence.

The [separate protocol](protocol.json) changes only the existing mug lower/release/retract labels. Lowering stops 1 mm higher on its existing path; fingers open over one second while the right arm holds still for 1.25 seconds, followed by the original lateral withdrawal and a tapered return to the original retract path. The largest active joint-label change is 0.027315 radians. All initial visual features, other arrays, inactive-arm targets, stage durations and undeclared actions remain unchanged.

[Every original collection manifest and initial integration state](manifest.json) is packaged with portable scene XMLs, exact modified arrays, all replay outcomes, source snapshots and asset hashes. The independent [audit](audit.json) reconstructs all 41 initial scenes and checks lineage, action-change limits and physical acceptance. Only mesh-directory references change in scene XMLs.

```powershell
.venv-training/Scripts/python scripts/package_release_inputs.py --protocol docs/robotics/experiments/mug-release-v1.json --verify-only
.venv-training/Scripts/python scripts/verify_dinner_learning_inputs.py --source training/mug_release_v1 --dataset training/mug_release_v1/episodes --output .run/mug-release-input-reproduction.json
```

Use unused output and console-log paths and keep the 10 GiB reserve. The next gate is a separately frozen, single 18,000-update weighted warm fit, followed by export parity and new paired full-workflow scenes. Training replays cannot replace candidate physical evaluation. The selected dinner suite, prior spoon candidate, hosted demo and submitted materials remain unchanged.
