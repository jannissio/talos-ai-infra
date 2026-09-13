# Camera visibility diagnostic: a candidate for fresh validation

The [protocol](protocol.json) was committed in `dcd84ca` before running this diagnostic. It compares three fixed camera configurations with the **unchanged V1 observer weights and OpenVINO IR**. It replays 32 saved approach frames from each of four exposed V5 development traces: both nominal visibility failures, the successful relay and the only both-nominal-pass seed. No controller runs and no new physical trial or neural training occurs.

| Camera configuration | Accepted frames | Recovered original refusals | Accepted error p95 | Maximum | Diagnostic gate |
| --- | --- | --- | --- | --- | --- |
| Original | 117/128 | 0/11 | 1.325 mm | 2.900 mm | Fail |
| Both oblique cameras opposite | 122/128 | 11/11 | 2.524 mm | 4.160 mm | Pass |
| One oblique camera opposite | 117/128 | 0/11 | 1.956 mm | 3.907 mm | Fail |

The selected configuration retains the overhead camera and places the two angled cameras at azimuths 225° and 315°, elevation −55°, distance 0.72 m, with the same fixed look-at point. The observer receives the actual calibration for each configuration. Its confidence, known-shape and reprojection limits are unchanged. It recovers all 11 old refusals but loses six other accepted frames. There is no claim of perfect visibility or lower error than the original cameras.

Selection requires at least 90% overall acceptance, recovery of at least 75% of original refusals, accepted p95 error at most 3 mm and maximum error at most 6 mm. Both opposite oblique cameras pass this **exposed diagnostic gate only**. Any use requires a new, separately declared perception test and physical development/final protocol. V5 stays stopped, all reserved final scenes remain unexposed and the submission controller is unchanged.

All **384 observations**, per-view predictions/confidence, calibration matrices and actual input hashes are in [report.json](report.json). The [audit](audit.json) independently reloads all 128 saved states through MuJoCo and recomputes every accepted geometric error, with zero discrepancy. The [manifest](manifest.json) preserves the full report, declared protocol, exact probe source and 12 final-frame camera strips. Two strips were visually checked: the old views lose the bottle behind the approaching arm or outside an oblique view, while the selected views show it from the opposite side. This does not establish detection accuracy on unseen scenes.

The probe takes **26.36 seconds** using OpenVINO CPU on the AMD PC, with 4.74 MB of raw output. Raw plus packaged output remains within the declared 32 MiB allowance, retaining the 10 GiB disk reserve. Camera images are generated from saved simulator states; they are not hardware camera captures or Intel evidence.

Verify the package without new observations:

```powershell
.\.venv-training\Scripts\python scripts/package_rgb_servo_visibility.py --verify-only --output docs/robotics/evidence/rgb-servo-camera-visibility-v1
```

Reproduce only into a new folder and check that any redirected log path is also unused:

```powershell
.\.venv-training\Scripts\python scripts/probe_rgb_servo_visibility.py --output .run/rgb-servo-camera-visibility-reproduction
```
