# Bottle data preparation — first pilot

This milestone establishes a physically checked demonstration and recording pipeline. It does **not** establish a robot that handles every reachable arrangement. The teacher still reads simulator object positions; no learned model is running.

The final selected dataset contains **50,523 synchronized images**, **16,841 observations** (including terminal observations), **168,119 motor-command steps**, and **840.595 seconds** of simulated demonstrations. All selected files passed timestamp, shape, image-reference and SHA-256 integrity checks. The integrity report records episode-level hashes.

## Physical teacher

The existing upright neck grasp remains available. A sideways bottle is grasped around its body, approximately 70 mm from the bottom. The arm lifts it, verifies an unsupported hold, rotates the physically held bottle upright, places it in the serving area, releases it and parks. The other arm remains parked.

The sideways body grasp needs a wider opening (0.85 rad) and a 0.65 Nm gripper torque limit, compared with the upright neck grasp's 0.15 Nm limit. These are simulation settings, not calibrated hardware recommendations. Robot links, bottle geometry, inertias and table dimensions are unchanged. The sideways reset puts the bottle at a different initial position/orientation; reset is explicitly separate from manipulation.

IK constrains the held bottle's axis in the wrist frame during reorientation. The controller checks candidate paths, contacts, unsupported lift/hold, placement and nearby-object disturbance. It does not write authoritative object poses or velocities, use external applied forces, teleport, attach or weld the bottle. A fixed intermediate reorientation region is a practical limitation of this teacher.

## Coverage and pilot results

- **20 successful demonstrations:** 10 upright, 10 sideways. All use the left arm. The ten sideways trials span three horizontal heading families (approximately 45°, 135°, 225°) and several starting positions; upright trials cover small position/heading variations near the original start.
- **40 successful physical replays:** each episode was replayed at original 200 Hz and with nominal targets sampled at 20 Hz and interpolated for 200 Hz motor control. Replay uses one initial integration-state reset, then only motor commands and physics. It never restores recorded object poses during movement and never calls the teacher's update method. The 20 Hz replay knows both endpoints of each interpolation interval from the saved trajectory; a future action-chunk policy must predict those endpoints. This is not a test of zero-order hold, inference latency or closed-loop learned control.
- The independent replay checks require bilateral finger support during the recorded hold, at least 1.49 s verified hold, table support after release, final XY error below 12 mm, tilt below 5°, and low final linear speed. They also check retention throughout carrying, no external support during rotation/alignment, nearby-object movement and the parked arm. The final 40-run audit measured **zero missing-contact intervals while carrying**, maximum placement error **2.13 mm**, and effectively zero parked-arm movement. Replay is a reproducibility check on the same initial state, not a generalization test.
- A negative control holds the gripper open throughout replay: it correctly fails, with zero verified hold and about 20 cm placement error.
- Two additional failed full rollouts are retained. One reaches a later planning failure after pickup; the other fails initial planning. Three earlier successful movements were excluded because their initial bottle poses overlapped the plate before settling; clean replacements passed. Exclusions are enforced by the loader. A planning failure is **not proof that an object is physically unreachable**.

The coarse approach scan covers a 30 × 20 cm region at 5 cm spacing, both arms, upright and four horizontal neck directions. Nine local 1 cm refinement samples bring the total to **359**. After rejecting invalid initial overlaps, there are **85 approach candidates**, **5 tested-path collision rejections**, **177 unresolved IK/planning results**, and **92 invalid reset combinations**. Approach candidates have not all passed full pickup-to-placement trials. Untested regions and orientations remain unknown; this is not a complete workspace boundary.

Three separate bottle poses are reserved in `.run/bottle-pilot/reserved-validation.json`; they were not used for demonstrations or tuning and have not yet been evaluated. Surrounding dishes remain at seed 42. Training later needs substantially more layout, heading, object and visual diversity, especially right-arm and coordinated tasks.

Evidence: [pilot motions/replays](bottle-pilot-results.json), [coverage map](bottle-coverage.svg), [approach coverage](bottle-workspace.json), [clean reset check](bottle-reset-check.json), [failed rollouts](bottle-failed-rollouts.json), [negative replay control](bottle-replay-negative-control.json), [complete dataset integrity check](bottle-dataset-validation.json).

## Cameras and observations

The selected pilot configuration is **overhead + left wrist + right wrist**, at **640 × 480 RGB, 20 Hz**. The browser can independently select the new **Opposite side** camera. Changing the browser view does not change recorded cameras.

The comparison uses identical states at approach, grasp, hold, rotation and lowering. Overhead makes table positions and placement guides easy to locate. The opposite view exposes more of the bottle during grasp/hold: in the sampled hold frame the bottle has 307 visible pixels versus 137 overhead, measured at 480 × 360. Overhead gives a larger initial bottle view (1045 versus 472 pixels). These counts measure visibility, not information content or learned-policy quality. They support a provisional camera choice, not a claim of optimality.

Wrist images supply a close view where the arm occludes the overhead image. The bottle occupies much of the active wrist view; the parked wrist contributes little to this single-arm pilot. Camera placement and input ablations should be tested before scaling to the full challenge.

See [comparison](bottle-camera-comparison.jpg), [actual training-view samples](bottle-observation-preview.jpg), and [frozen calibration](bottle-camera-calibration.json). Calibration includes resolution, intrinsics, camera body, local pose and field of view. Wrist world extrinsics depend on the robot joints.

Each image is rendered offline from exactly the corresponding recorded observation. Rendering may be slow but cannot change the controller's simulated timing. Physics and low-level control run at 200 Hz; observations and policy targets are on a 20 Hz grid. A final terminal observation may be off-grid and is excluded from policy samples.

The policy-facing loader exposes only RGB paths, 12 joint positions, 12 joint velocities and nominal action targets. Full object state, contact/evaluator data and simulator actuator forces remain privileged diagnostics. Actuator force is a simulator quantity, not a claim that the physical motors provide a calibrated force sensor. The initial policy need not consume it.

## Use and reproduce

Use the existing `.venv`:

```powershell
# Browser demo: start server, click Load sideways bottle practice, then Start dinner goal.
.\start-lab.ps1 -NoBrowser

# Full pilot collection; refuses to overwrite existing successful episode folders.
.\.venv\Scripts\python.exe scripts/prepare_bottle_data.py collect
.\.venv\Scripts\python.exe scripts/prepare_bottle_data.py render
.\.venv\Scripts\python.exe scripts/summarize_bottle_pilot.py

# Reproduce one movement using saved motor targets; no teacher commands.
.\.venv\Scripts\python.exe scripts/replay_bottle_pilot.py .run/bottle-pilot/sideways-01 --hz 20

# Deliberately broken grasp must fail evaluation.
.\.venv\Scripts\python.exe scripts/replay_bottle_pilot.py .run/bottle-pilot/sideways-01 --open-gripper

# Recompute the approach scan independently.
.\.venv\Scripts\python.exe scripts/bottle_workspace.py
```

Bulk data lives under `.run/bottle-pilot/`, excluded from Git. Each successful episode has scene XML, full initial integration state, 200 Hz nominal/applied commands, 20 Hz observations, three camera streams, image hashes, outcome and replay checks. Preserve the repository's referenced SO-101 assets when copying data. This is a local schema; LeRobot conversion and model training are next steps.

The existing 27 simulator tests pass, including cancellation, stalled motion, gripper failure, disturbance rejection, chemistry transfer and slow-render timing. A new loader test checks that privileged simulator fields cannot enter policy input (**28 tests total**). Recorded trajectories additionally assert that each teacher update leaves authoritative qpos/qvel untouched and applies no external forces. Failed full-task command/state traces remain under `.run/bottle-failures/`; excluded overlapping starts remain under `.run/bottle-pilot/` with explicit exclusion metadata.

To package the selected dataset after validation, run `.\.venv\Scripts\python.exe scripts/package_bottle_pilot.py`. The ZIP can be extracted at the root of a matching repository checkout on another machine. It contains only the selected episodes, calibration and documentation; the checkout provides the robot assets and code.

## Next decision

Use this pilot first to verify an ACT training/data-loading run and image/joint normalization. Then expand demonstrations and evaluate on the reserved layouts. Do not interpret overfitting these 20 examples as a successful manipulation policy. Broader reachable-pose coverage, right-arm skills and recovery are necessary before the dinner challenge's complete task. The [research-backed training plan](TRAINING_RESEARCH_PLAN.md) remains the algorithm roadmap.
