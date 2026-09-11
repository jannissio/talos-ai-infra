# Talos simulator · Dinner challenge and BenchLab practice

**September 11:** the default browser scene is now the dinner challenge. Select **Dinner challenge**, a seed, **Task start** or **Target example**, and **Load seed**. Select **Open for inspection** before loading to see the cutlery in the passive drawer. Every preset rebuilds the physical scene; the target example is not autonomous execution. Dinner-specific robot goals are the next milestone.

The scene contains seven free rigid tableware objects, a passive drawer, two unchanged SO-101 arms and five cameras. See [dinner setup, model dimensions and verification](../docs/robotics/DINNER_SCENE.md), [standalone dinner models](assets/dinner/README.md), and [submission build plan](../docs/hackathon/BUILD_PLAN.md).

**The chemistry instructions below remain available:** first choose **Chemistry practice → Load seed**. Existing lift/transfer goals, recording, rack editing, and manual controls are preserved.

A working **MuJoCo 3.12.0** learning sandbox, built on September 8, 2026 at the user's request. The browser displays actual camera frames from this PC. It now includes an autonomous **Lift & return** skill using simulator positions, inverse kinematics, and physical finger contacts.

It also supports **Rack-to-rack transfer** and demonstration recording. These remain exact-state physical baselines; no learned policy is loaded.

## Open and run

The default address is [http://127.0.0.1:8765](http://127.0.0.1:8765/).

From the project folder in PowerShell:

```powershell
.\start-lab.ps1
```

This starts a hidden local server and opens your default browser. It reuses an already running instance. Use `-NoBrowser` to start without opening another tab. Use `-Port 8766` if the default port is occupied.

Stop the server:

```powershell
.\stop-lab.ps1
```

Pass the same `-Port` if changed. The stop script verifies the workspace Python launcher and its server child before stopping them. Logs and process information are under `.run/`.

For a foreground server that you can stop with Ctrl+C:

```powershell
.\.venv\Scripts\python.exe -m simulation_lab.server
```

The server binds to **127.0.0.1**. It is accessible from this PC, not published to the Internet. Viewers share one scene and camera selection. After 15 seconds without viewer/control requests, simulation advancement stops and the renderer idles; reconnecting resumes it. An active autonomous task continues without a viewer. The separate camera process exits when the server stops.

## Run the autonomous demonstration

For the newest milestone, select **Load transfer practice → Start transfer**. This loads two racks within the selected arm's workspace, with tube **A2** and empty destination **B2** selected. Choose Right arm before loading the preset to mirror the layout. A transfer takes approximately **44 simulated seconds**. The other arm stays parked.

**Record demonstration** is checked by default in the browser. Actions are saved at 200 Hz and observations at 20 Hz. After the physical task finishes, a separate process renders synchronized central/left-wrist/right-wrist images at 5 Hz. The recording panel shows progress and a manifest link when finished. Image export takes additional time and can lower viewer FPS, while physics and controls remain independent. Uncheck recording for a quick repeat while an earlier recording exports.

The original lift-and-return goal is still available:

1. Open [BenchLab](http://127.0.0.1:8765/) after starting the server.
2. Keep **Seed 42**, then select **Load lift practice** in the Robot goals panel.
3. Leave Arm and Tube on **Choose automatically**, or choose the left arm with A2 / right arm with B2.
4. Press **Start lift & return**. A complete run takes about **35 seconds of simulated time**. The panel and camera-side status strip show each stage and its result.
5. Use **Pause**, **Step 50 ms** and the five cameras to inspect it. **Cancel** ends the goal and pauses physics, retaining the current state even if a tube is held. Reset the scene before another clean attempt.

The practice scene has two racks with three flat-bottom training tubes each, exposed front slots, and 22 mm guide openings. The original randomized scene retains its rounded tube bottoms and looser guides; **Load seed** returns to that scene. **Reset scene** preserves the current mode and rack poses. The practice adjustments make the first manipulation exercise tractable without changing the robot meshes, inertias or joint limits.

The task first checks reach and clearance. It closes the fingers slowly with a 0.15 Nm software torque cap, verifies contact on both sides, lifts about 5.7 cm, holds clear for 1.5 seconds, centers over the original slot, lowers until rack support is detected, releases, withdraws, parks, then checks stable placement. Tubes are never attached, welded or teleported. An unverified grasp gets at most one retry; loss of support, collision, unreachable goals or timeouts produce explicit failure and pause physics.

Joint controls and motion presets are unavailable while the goal owns the arms. Camera and playback controls remain available. See the [full implementation notes and evidence](../docs/robotics/LIFT_AND_RETURN.md).

Tube IDs remain stable after movement: **tube A2 in slot B2** is still tube A2. The destination selector shows current occupancy, rather than the initial rack contents. Occupied, blocked, same-rack and unreachable transfer destinations are rejected. The controller rechecks the destination after pickup and checks collisions along the carried tube's path as well as the arm's. See [transfer and recording details](../docs/robotics/TRANSFER_AND_RECORDING.md).

## What is in the scene

- Two copies of the official Menagerie **SO-101** model, with the original joints, limits, inertias, actuators, and collision geometry.
- A **96 × 78 cm** tabletop at **76 cm** height, with a 10 cm reference grid.
- **Two to four racks**, each with six possible tube positions and four to six occupied slots. Seed 42 starts with three racks and 18 tubes.
- Racks at different positions and rotations. Their poses and occupied slots are reproducible from the seed.
- Dynamic rigid tubes approximately **18 mm wide and 88–108 mm tall**, with caps and colored visual markers.
- A central camera between/behind the arm bases, plus an overview, overhead, and both wrist cameras.

The tube shells/caps have mass, gravity, and collisions. Their colored contents are attached visual geometry. The rack bodies are fixed during simulation; changing a rack layout rebuilds/resets the scene. Liquid dynamics, chemical behavior, vision, learned policies and general-purpose motion planning are not implemented. The new grasp controller uses sampled trajectory collision checks and runtime contact monitoring.

## First experiments

1. Switch between **Between arms** and **Overhead** to inspect how the same arrangement looks from different viewpoints. The overhead view is useful for identifying occluded tubes and checking spacing.
2. Select an arm and move a joint slider a few degrees. The values are position targets, in degrees; the physics uses radians internally.
3. Use **Gentle motion test** for a six-second small joint movement. Moving a slider cancels it. Use **Home both arms** to return their targets to the initial pose.
4. Pause, then **Step 50 ms** to inspect motion incrementally.
5. Load another seed or use **New arrangement**. The rack count is selectable from two to four.
6. In **Rack layout**, choose a colored rack, change X/Y/rotation, and apply it. Rack centers must stay at least 20 cm apart. **Reset scene** preserves the current rack poses; **Load seed** regenerates the seeded layout.

Manual joint commands respect joint ranges but do not perform collision avoidance. The autonomous skill checks each candidate's reachability, but full workspace reachability is not certified. The rear racks and opposite arm's rack may be unreachable with the required grasp orientation. Arbitrary randomized layouts can be rejected or fail during manipulation; use the practice scene for the first milestone.

## Rendering and performance

The tested graphics device is **Intel UHD Graphics** on this PC. The stream uses **960 × 540 JPEG frames**, with a target of 20 frames/s. Actual rate is shown in the viewer and varies with the camera and graphics load. **Shadows** are disabled by default to improve responsiveness; enable them for more depth cues at a lower frame rate.

The visual meshes were simplified from the original source meshes. The unique mesh triangle count was reduced from 322,564 to 94,518, excluding unchanged geometry such as the camera mount. Full original files remain included. Tests confirm the visual optimization preserves body mass and inertia, and the original collision meshes remain in use.

Physics and control advance in **5 ms fixed steps (200 Hz in simulation time)** in the engine thread. A **separate process** owns rendering and its own MuJoCo data/model, receiving snapshots through bounded queues. Slow images are dropped rather than delaying or queuing controller actions. The viewer shows simulation / real time separately from camera FPS. There is a bounded catch-up after CPU/OS stalls, so this is not a hard real-time system.

A timing regression test adds **400 ms delay to every rendered frame** and verifies that physics still advances near real time, with pause and 50 ms stepping preserved. The browser is a viewer/controller, not the physics engine. These are local simulator measurements, not policy-inference or qualifying Intel-hardware benchmarks.

## Setup on another machine

Use a clean **CPython 3.12** environment. The existing `.venv` was based on the available bundled CPython after this PC's Anaconda interpreter produced a MuJoCo DLL initialization error. No global Python installation or graphics-driver setting was changed.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m simulation_lab.server
```

The robot assets and lightweight visual meshes are already included. To download the pinned originals again:

```powershell
.\.venv\Scripts\python.exe scripts\fetch_robot_assets.py
```

To regenerate the lightweight visual meshes, install the optional tooling and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-assets.txt
.\.venv\Scripts\python.exe scripts\prepare_visual_meshes.py
```

Linux/Core Ultra cloud execution has not yet been tested. It will need an appropriate graphics context and drivers for camera rendering, in addition to checking the actual processor model. The engine/scene separation makes this a practical codebase to port without changing the browser interface.

## Files and interfaces

| File | Purpose |
| --- | --- |
| [scene.py](scene.py) | Seeded scene generation, camera placement, two-arm composition and rack/tube geometry |
| [engine.py](engine.py) | Fixed-step physics, control ownership, commands and telemetry |
| [autonomy.py](autonomy.py) | Five-joint IK, sampled trajectories, task state machine and physical success checks |
| [rendering.py](rendering.py) | Independent camera process and bounded snapshot/frame transport |
| [slots.py](slots.py) | Current tube locations, slot occupancy and entry-column blockers |
| [recording.py](recording.py) | Asynchronous action/state recording and synchronized camera export |
| [server.py](server.py) | Local HTTP service and live camera stream |
| [web/index.html](web/index.html) | Browser interface |
| [web/app.js](web/app.js) | Camera, joint, seed and rack interactions |
| [web/style.css](web/style.css) | Responsive layout and appearance |
| [assets/so101/provenance.json](assets/so101/provenance.json) | Original asset commit, source URLs and file hashes |
| [assets/so101/assets/lod/provenance.json](assets/so101/assets/lod/provenance.json) | Derived visual mesh records |

The small API is available for subsequent experiments:

- `POST /api/reset` with `{"scenario":"dinner","seed":42,"dinner_preset":"task","drawer_open":false}`: load the challenge scene. Use `reference` for the target example, or `drawer_open:true` for inspection.
- `GET /api/state` in dinner mode adds `scenario`, `objects`, `targets`, `drawer` and `challenge`. Object poses and grasp sites are simulator ground truth for inspection. `challenge.autonomy_available` is false. Tube-goal and rack-layout calls in dinner mode return HTTP 400 with an explanation.
- `POST /api/reset` without `scenario` keeps the old chemistry API behavior. The application sends the selected scene explicitly; server startup defaults to dinner.

- `GET /api/state`: joint targets/positions, rack poses, tube state and timing.
- `GET /frame.jpg`: current rendered camera image.
- `GET /stream`: multipart JPEG stream.
- `POST /api/control`: camera, running state, shadows, selected arm's six joint targets, home/motion presets, and stepping.
- `POST /api/reset`: seed and rack count.
- `POST /api/reset` with `{"seed":42,"rack_count":2,"practice":true}`: load the practice scene.
- `POST /api/reset` with `{"seed":42,"transfer_side":"left"}`: load transfer practice. Use `right` for the mirrored setup.
- `POST /api/racks`: a custom rack layout, followed by a physics reset.
- `POST /api/task` with `{"action":"start","arm":"auto","tube_id":null}`: start Lift & return. Arm can also be `left` / `right`; tube can be an existing ID such as `A2`.
- `POST /api/task` with `{"action":"cancel"}`: end the active goal and pause physics.
- `POST /api/task` with `{"action":"start","kind":"transfer","arm":"left","tube_id":"A2","destination_slot":"B2","record":true}`: transfer and record an episode. Omit `destination_slot` for automatic empty-slot selection. `record_images:false` saves only the action/state trajectory.
- `GET /api/recordings/{episode_id}/manifest`: view a saved demonstration manifest.

`GET /api/state` includes `task` (stage, outcome, metrics, history), `practice`, `scene_version`, `real_time_factor` and frame metadata. The newest image may trail the physics state: check `frame_camera`, `frame_scene_version` and `frame_simulation_time_s` when capturing evidence. Completed, failed and cancelled tasks are saved under `.run/autonomy/`, with the latest in `.run/autonomy-last.json`.

State also includes `transfer_side`, current `slots`, and `recording` status. Demonstrations are saved under `.run/demonstrations/{episode_id}/`. Recording is opt-in in the API and on by default in the browser. An incomplete recording is explicitly marked invalid; it is never silently accepted as a training episode. Recorded cameras are reconstructed from the exact saved observation, not paired with the possibly delayed live stream.

The simulator has **12 controlled joints** total: five arm joints and one gripper per side. The API includes simulator ground truth for debugging; any future vision-policy experiment must explicitly document which fields the policy can see.

## Verification

Dinner-scene verification:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_dinner_scene.py
.\.venv\Scripts\python.exe scripts\check_live_dinner.py
```

On September 11, **22 unit/physics/timing tests**, **33/33 dinner scene-settling configurations**, and the live browser/API checks passed. These are scene and infrastructure results, not dinner manipulation success. The older milestone-specific counts below remain historical.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\check_live_lab.py
.\.venv\Scripts\python.exe scripts\evaluate_autonomy.py
.\.venv\Scripts\python.exe scripts\evaluate_autonomy.py --transfer left
.\.venv\Scripts\python.exe scripts\evaluate_autonomy.py --transfer right
```

The first command checks the original scene/physics invariants, successful manipulation by either arm, no controller changes to physical coordinates, torque limits, missing/lost grasp, bounded retry, timeout, cancellation, unreachable targets and independent timing under slow rendering.

The second requires the server at port 8765. It exercises pause/step, all five camera images, the JPEG stream, seeded resets, custom rotation, joint clamping, motion presets and invalid inputs. **It resets physical state**, then restores the original layout/control settings. Use it when a current experiment can be reset.

The third command runs independent headless practice trials for seeds 0–9 and 42 and writes `.run/autonomy-evaluation.json`. The saved [milestone evaluation](../docs/robotics/autonomy-evaluation.json) records **11/11 completed practice trials**. These are small variations of the accessible practice layout, not a claim of general manipulation success. This is a task evaluator, not yet a training-dataset recorder.

Eleven unit/physics/timing tests and six live API check groups passed. Browser checks verified goal start/progress, physical holding, pause/step, cancellation, failure feedback and completion. The earlier loading-indicator fix (`hidden = state.ready === true`) is retained. See [results and limitations](../docs/robotics/LIFT_AND_RETURN.md).

The transfer milestone adds physical tests for either arm, occupied destinations, carried-object collision, destination revalidation and two consecutive transfers with the same tube ID. Recording tests verify timestamp/image alignment and explicit failure on queue overflow. The [transfer evaluation](../docs/robotics/TRANSFER_AND_RECORDING.md) covers 11 seeds per arm layout (22 runs).

Validate or re-export a saved recording without moving the robot:

```powershell
.\.venv\Scripts\python.exe scripts\validate_demonstration.py .run\demonstrations\EPISODE_ID
.\.venv\Scripts\python.exe scripts\export_demonstration.py .run\demonstrations\EPISODE_ID
```

The re-export command replaces that episode's generated images/index; it preserves the original action and observation files. Keep the workspace robot assets with the recording: the saved scene references them by a relative path. The format is a documented local schema, not yet a LeRobot dataset or a trained policy.

## Provenance and scope

The SO-101 assets come from [Google DeepMind's MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101), pinned to `ac6b2b09983786f3036cab1000221017fa2193b4`, under [Apache-2.0](assets/so101/LICENSE). Their original notices are retained. Our scene changes include arm placement/colors, visual-only simplification, cameras, lab props and controls. See [NOTICE.md](NOTICE.md).

This learning sandbox was developed with Codex assistance on September 8. Its chemistry setting is a user-requested experiment, not a change to the published dinner-table hackathon brief. Keep the dated preparation history if any code is later considered for the submission. The competition architecture and eligibility of reused preparation code remain separate decisions.

See [Intel cloud assessment](../docs/robotics/INTEL_CLOUD.md) and the earlier [robotics research](../docs/robotics/RESEARCH.md).
