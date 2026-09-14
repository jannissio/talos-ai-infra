# Spoon release candidate: exact training inputs

All **39/39 modified float32 action trajectories pass independent physical replay** using the unchanged dinner monitor. The maximum placement error is 5.201 mm against the 8 mm limit; all successful replays release the spoon and park the arms. The batch takes 286.70 seconds of summed replay wall time on this PC. This is demonstration validation, not learned-policy or Intel evidence.

The [protocol](protocol.json), declared in `b68d4ea` before transformation, modifies only lowering, release and retract in the existing 39 training episodes. Lowering stops 3 mm higher on the original path. The fingers open over one second while the arm remains stationary for 1.25 seconds, followed by the sideways retreat. The small joint offset returns to zero during retract. The largest active-arm label change is 0.025731 radians; all other input arrays, inactive-arm actions and undeclared stages remain unchanged.

[Exact inputs and every outcome](manifest.json) include all initial integration states, original collection manifests and portable scene XMLs. The independent [package audit](audit.json) loads every scene/state and verifies lineage, action-change limits and the physical gate. Mesh-directory references are the only scene XML transformation. Original Talos files remain untouched.

From the repository root:

```powershell
.venv-training/Scripts/python scripts/package_spoon_release_inputs.py --verify-only
.venv-training/Scripts/python scripts/verify_dinner_learning_inputs.py --source training/spoon_release_v1 --dataset training/spoon_release_v1/episodes --output .run/spoon-release-input-reproduction.json
```

Use a fresh output path. These commands check available space and preserve the 10 GiB reserve. The complete input gate must pass before the separately declared 12,000-update spoon fit. Development/final selection uses new scenes and full workflows; the training replays cannot substitute for that evidence. The current dinner suite remains the selected baseline until the physical promotion gate is satisfied.
