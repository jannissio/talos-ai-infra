# RGB feedback: active experiment record

September 13. The selected six-skill/relay baseline stays unchanged. This experiment addresses broader bottle positions and corrective actions driven by current images. It is not yet a deployed robot capability.

| Stage | Measured result | Consequence |
| --- | --- | --- |
| 48 grid/arm approach pairs | 27 candidate approaches; at least one arm for 21/24 positions | Approach feasibility only; full manipulation still needs tests |
| Existing camera/color detector, 96 physical-trace frames | Zero frames with two unique views | Amber arm causes ambiguity; drawer view misses the bottle |
| Three new fixed cameras and custom keypoint CNN, 1,280 training states | 101 seconds, 0.994 GiB VRAM; accepted 3D error p95 3.598 mm | Misses 3 mm development gate; preserved |
| Expanded 4,096-state observer | 211 seconds, 0.994 GiB VRAM; p95 3.240 mm on 222/225 accepted present development states | Two-view cases are weaker; define a three-view consistency rule |
| Frozen fresh synthetic observer test, seed 2026095404 | 210/229 present states accepted; 0/27 absent states accepted; p95 2.767 mm, maximum 5.960 mm | Passes synthetic accepted-error gate; refusals retained |
| Exposed seed-42 physical grasping footage, 256 frames | 216 accepted, 40 refused; p95 5.065 mm, maximum 6.489 mm | Fails motion accuracy gate; not integrated into robot actions |
| Wider nominal physical training collection | 15/24 complete saved-action replays pass; all nine failures retained | Adds real grasp/carry/release appearances across varied starts and three goals |
| Motion-conditioned observer v3 | Training on 1,920 rendered states from those 15 replays plus the 4,096 procedural states | Active; requires physical-trace and new final evaluation before promotion |

The new network outputs two bottle keypoints, visibility and a foreground mask. Calibrated triangulation estimates position and vertical axis. It does not infer absolute yaw of the nearly rotationally symmetric bottle. Simulator segmentation, body points and saved physical states are training/scoring sources only; the observer's `observe` method accepts RGB images and fixed camera calibration.

The consistency rule requires all three confident views, at most 1.5-pixel reprojection residual, and the known 103 mm keypoint spacing within 6 mm. Rejecting uncertain views improves accepted accuracy but reduces coverage. Both quantities are reported. No rejected state is silently omitted from the denominator.

Protocols: [overall experiment](experiments/rgb-servo-bottle-v1.json), [observer v1](experiments/rgb-servo-observer-v1.json), [expanded v2](experiments/rgb-servo-observer-v2.json), [confidence rule](experiments/rgb-servo-confidence-v1.json), [frozen observer evaluation](experiments/rgb-servo-observer-evaluation-v1.json), [motion-conditioned v3](experiments/rgb-servo-observer-v3.json).

The final eight training seeds, 2026095125–2026095132, remain reserved for recovery demonstrations after a disclosed physical-push calibration. Physical development seeds 2026095201–5206 and final evaluation seeds 2026095301–5312 have not been consumed by these perception tests. A neural corrective action model and paired live/frozen-image disturbance trials still need implementation and execution. Merely refreshing images is not accepted as proof of corrective control.

Raw datasets and failed candidates are retained in `.run/`; they are not automatically published or substituted for the selected models. Dataset/export scripts check actual free space before each batch, episode/shard and periodic writes, retaining 10 GiB plus expected output.
