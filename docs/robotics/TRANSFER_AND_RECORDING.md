# Rack transfer and synchronized demonstration recording

Implemented September 8, 2026 for the local BenchLab learning experiment. This extends the [lift-and-return baseline](LIFT_AND_RETURN.md); it remains separate from the final hackathon submission.

## Outcome

A selected arm now picks up a tube, verifies its physical grasp, carries it into a different rack, places it upright in an empty slot, withdraws and parks. The other arm remains parked. The implementation supports mirrored practice layouts for either arm and keeps the original lift-and-return goal.

**22/22 practice transfers passed**: seeds 0–9 and 42 in each arm's layout. A browser run also completed and produced a validated recording with **8,734 actions, 875 observations and 657 camera images**. No object weld, attachment, teleportation or applied lifting force is used by the controller.

## Run it

```powershell
.\start-lab.ps1
```

Open [BenchLab](http://127.0.0.1:8765/), then:

1. Choose **Load transfer practice**. This selects the left arm, source tube **A2**, and empty destination **B2**. Select Right arm before loading the preset to mirror the scene.
2. Leave **Record demonstration** checked to save the episode, or uncheck it for a quicker repeat without image export.
3. Press **Start transfer**. The physical task takes approximately **44 simulated seconds**.
4. Watch progress beside the camera or in Robot goals. After the arm finishes, wait for **Demonstration and synchronized images saved** in the recording panel.
5. Use **Open recording manifest** to inspect the saved episode. **Reset scene** restores the starting arrangement for a clean new run.

**Pause** freezes simulation and recording time. **Step 50 ms** advances both. **Cancel** ends the goal and pauses physics, retaining the current physical state; a cancelled recording is labeled accordingly and is not accepted as a successful training episode. Manual controls are available again after cancellation. Camera controls remain available throughout a goal.

Image export is separate from physical task completion and may temporarily lower camera FPS. Only one recorded episode is saved/exported at a time. While it exports, uncheck recording to run another physical goal, or wait before starting another recorded goal.

## Scene and reachability

The original lift practice puts one rack near each arm. A single arm cannot generally reach both racks with the required finger orientation. The transfer preset instead puts **both racks within the selected arm's workspace**. The inactive arm stays in its original position.

The left layout places rack centers near X = −28 cm and −5.5 cm, Y = 9 cm; the right layout mirrors them. Centers remain more than 20 cm apart. Seeds vary centers by up to 3 mm and orientations by up to 3° around approximately −15° / +15°. This is deliberately a narrow practice distribution.

Rack A has three tubes: front middle and rear corners. Rack B has two rear tubes, leaving its front row empty. The flat-bottom training tubes and 22 mm guides introduced for lift practice are retained. The first lift is about 5.6–5.7 cm; the arm then raises another 2.5 cm before carrying, giving roughly 8 cm maximum elevation and additional rack clearance.

The **tube ID** names the physical object and does not change when it moves. The **slot ID** names a fixed location. After transfer, tube **A2 is in slot B2**. The API and browser now compute occupancy from current physical poses instead of reusing initial rack contents. A subsequent goal can move that same tube back to its original rack.

## Planning and verification

The controller still uses exact simulator positions, MuJoCo Jacobians, five-joint inverse kinematics and physical gripper contacts. Position plus an upright finger-axis constraint leaves the arm's available orientation freedom intact. Quintic trajectories use sampled Cartesian waypoints and bounded nominal joint speed. Gripper torque is limited to 0.15 Nm through the existing actuator.

The transfer adds these checks and stages:

- Select an existing tube and a destination in a **different rack**. Reject occupied, blocked or unreachable destinations.
- Check the planned arm path **and the carried tube's geometry** against the scene. Hypothetical object poses are changed only in scratch planning data.
- After the verified lift/hold, raise for clearance and carry to the destination.
- Recheck destination occupancy after pickup and before placement. Compensate for the observed tube position within the fingers when centering over the target.
- Lower until the **destination rack base** supplies measured support; open while moving away from the fixed finger, withdraw and park.
- Report success only after the tube is upright, centered, stable and supported by the destination with no finger support. Verify that the other arm and other tubes stayed essentially undisturbed.

The earlier bounded grasp retry, stage timeouts, 75-second simulation-time task limit, lost-grasp checks, collision monitoring and cancellation remain active. Path checks are sampled and local; this is not a general motion planner or a formal continuous collision guarantee.

## Demonstration format

Episodes are saved in `.run/demonstrations/{episode_id}/`. No external service or new Python dependency is required.

| File | Contents |
| --- | --- |
| `manifest.json` | Schema version, layout, units, joint order, timing contracts, task outcome, completeness and image-export status |
| `scene.xml` | The actual scene description, with a relative reference to the workspace's robot assets |
| `actions.jsonl.gz` | 200 Hz nominal joint targets and actual actuator control values, timestamps and task stage |
| `observations.jsonl.gz` | 20 Hz MuJoCo `qpos` / `qvel`, corresponding action index, simulation time, and a terminal observation |
| `images.jsonl` | Image-to-observation/action mapping, timestamps, camera names, paths and SHA-256 hashes |
| `images/*.jpg` | 640 × 360 central, left-wrist and right-wrist images, at a nominal 5 Hz per camera |

Each ordinary observation at time **t** precedes the corresponding action applied for **[t, t + 5 ms)**. The terminal observation has no following action. All 12 actuator targets use radians, ordered as left shoulder pan/lift, elbow, wrist flex/roll, gripper, then the same right-arm order. `target` is the nominal position target; `ctrl` includes the gripper torque cap used during execution.

Images are rendered **afterward from exactly the referenced saved observation**. They are not paired with delayed live-stream frames. The saved model and recorded coordinates supply wrist-camera poses as well as object poses. A terminal image can make the final interval shorter than the nominal 0.2-second image spacing; explicit timestamps remain authoritative.

The physics thread only enqueues copies into a bounded in-memory queue. A writer thread owns compression/file I/O; a separate process exports images. Queue overflow or write failure marks the recording incomplete instead of silently accepting missing samples. Temporary Windows file locks are retried. Failure to update the optional image-progress file does not destroy the image export. An interrupted image export can be repeated from the intact trajectory without rerunning the robot.

`training_eligible` means this local schema passed its completion conditions and the physical task succeeded; it is **not a claim that a model has been trained or that the episode generalizes**. Cancelled/failed tasks are retained as such. Exact object state is privileged teacher/evaluator information. A later camera-based learner must explicitly select its allowed observation fields, such as images and robot joint state.

This is a documented local recording schema, **not yet LeRobot format**. Full bitwise physics replay is not promised: the recording captures actions and observation states, not every internal solver/warm-start field. The camera reconstruction uses the saved states directly.

## Validation and results

| Practice layout | Runs | Maximum slot error | Maximum final tilt | Task duration |
| --- | --- | --- | --- | --- |
| Left arm | **11 / 11** successful | **0.148 mm** | **0.006°** | 43.515–43.750 simulated seconds |
| Right arm | **11 / 11** successful | **0.164 mm** | **0.007°** | 43.425–43.650 simulated seconds |

Sources: [left-arm evaluation](transfer-left-evaluation.json), [right-arm evaluation](transfer-right-evaluation.json). These are simulator results with idealized rigid props and small scene variations, not measurements on a physical robot or arbitrary unseen rack layouts. Seeds were used for development/regression, not an untouched benchmark.

**18 unit/physics/timing tests passed**, covering the original scene and lift baseline, both transfer layouts, occupied/same-rack destinations, an obstacle intersecting only the carried tube's path, a destination blocked after pickup, two consecutive transfers of the same tube, synchronized recording, queue overflow, and rendering delays. Six live API check groups also passed, preserving cameras, streaming, manual controls, pause/step, seed/layout behavior and validation.

The recorded browser episode `20260908T124919_e76058e978` contains **43.67 seconds**, **8,734 actions**, **875 observations**, and **657 images**. Validation checks all action/observation timestamps and indices, state dimensions and finite values, control shapes/ranges, image dimensions, image references, and hashes. See [validation result](transfer-recording-validation.json), [live result](transfer-live-demo.json), and the [recording manifest](../../.run/demonstrations/20260908T124919_e76058e978/manifest.json).

The earlier recording's raw trajectory was preserved after the Windows progress-file lock and successfully re-exported using the recovery command below. The final browser episode completed its automatic image export successfully.

Recorded camera examples: [carrying between racks](transfer-carry.jpg) and [placed in the destination](transfer-placed.jpg).

## Reproduce the checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\evaluate_autonomy.py --transfer left
.\.venv\Scripts\python.exe scripts\evaluate_autonomy.py --transfer right
.\.venv\Scripts\python.exe scripts\check_live_lab.py
.\.venv\Scripts\python.exe scripts\validate_demonstration.py .run\demonstrations\EPISODE_ID
```

The live API check resets physical state. The headless evaluator does not touch the live simulator. To regenerate an episode's images/index while preserving its action and observation files:

```powershell
.\.venv\Scripts\python.exe scripts\export_demonstration.py .run\demonstrations\EPISODE_ID
```

Keep the workspace assets with the episode, since `scene.xml` references them. Recordings remain local under the ignored `.run` directory and are not automatically uploaded or versioned.

## Next boundary

This milestone uses **one arm per transfer**, with the other parked. Both arms can perform the skill in their corresponding layouts, but coordinated hand-off is still future work. A next step is a shared transfer region and explicit giver/receiver support checks. Broader layout evaluation, camera perception, dataset conversion and learned policies also remain separate tasks.
