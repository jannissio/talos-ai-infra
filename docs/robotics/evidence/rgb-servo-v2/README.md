# RGB routing experiment V2 — not promoted

All 48 reserved trials completed on September 13, 2026. Source and model hashes were frozen before evaluation. The selected R1 controller reuses the unchanged [V1 neural weights](../../../../models/bottle_servo_v1/README.md); no extra training was performed. The six-skill dinner submission baseline remains unchanged.

| Condition | Physical passes | All attempted scenes |
| --- | ---: | ---: |
| Undisturbed, live RGB | 6 | 12 |
| Undisturbed, initial RGB frozen per leg | 2 | 12 |
| Pushed, live RGB | 5 | 12 |
| Pushed, initial RGB frozen per leg | 1 | 12 |

The declared gate required at least 8/12 in both live conditions and a live advantage among at least four seeds passing both undisturbed controls. Only seed 2026095803 passes both controls; its pushed live result is 4.02 mm/pass versus 17.73 mm/fail with frozen images. That is one paired example, not broad robustness. These are different final scenes from V1, so comparing their raw success counts does not establish a controlled improvement.

## All final outcomes

Seeds below have prefix `202609`. A pass requires physical grasp, supported lift, release, placement within 8 mm, both arms parked and the unchanged collision/displacement checks. A placement near the goal alone is insufficient.

| Seed suffix | Nominal live | Nominal frozen | Pushed live | Pushed frozen |
| --- | --- | --- | --- | --- |
| 5801 | Fail | Fail | Fail | Fail |
| 5802 | Fail | Fail | Fail | Fail |
| 5803 | Pass | Pass | Pass | Fail |
| 5804 | Pass | Fail | Pass | Fail |
| 5805 | Fail | Fail | Fail | Fail |
| 5806 | Fail | Pass | Pass | Fail |
| 5807 | Pass | Fail | Pass | Pass |
| 5808 | Fail | Fail | Fail | Fail |
| 5809 | Pass | Fail | Pass | Fail |
| 5810 | Fail | Fail | Fail | Fail |
| 5811 | Pass | Fail | Fail | Fail |
| 5812 | Pass | Fail | Fail | Fail |

Three seeds are refused before motion in every condition. Other failures include post-grasp transport refusals, placement error and a neural geometry guard reached before release/parking completed. The guard remains 1.5 mm; rounded messages near that boundary do not justify changing the outcome. No failed seed is omitted.

Live table-supported relays complete both legs on nominal seeds 5809/5812 and pushed seeds 5806/5809. These release onto the table between arms. They do not establish direct airborne exchange, other-object handoffs or general reachable-position coverage.

## Selection and scope

R1 selects a supported lift and a separate transport height, and tries a table-supported relay when neither arm can serve both endpoints directly. Only RGB bottle estimates, robot joint measurements, neural motor predictions and robot forward geometry generate targets. Simulator state belongs to the independent scorer and disclosed disturbance, not the action policy. Carry after closure is proprioceptive.

R2 adds two lower transport heights, three shared-point choices and RGB checks at the align/lower transitions. It failed on occluded images during carry. On all 24 development cases, R1 passed 3/6 nominal live and 3/6 pushed live; R2 passed 2/6 in each. Both passed 0/6 nominal frozen and 1/6 pushed frozen. R1 was selected before final seeds were exposed. Both revisions and all eight exposed training trials are retained.

The finite XY region is x ∈ [-0.14, 0.16] m, y ∈ [-0.18, -0.06] m, with yaw ±0.6 rad and three declared destinations. Each pushed case uses the published 0.75 N force for 0.1 seconds at simulation time 1.5 seconds. The push occurs once, before the first grasp. Frozen-image trials acquire a fresh initial image for each relay leg after the preceding physical release and parking. This is a per-leg ablation, not a globally frozen camera.

The 156-point kinematic probe found only one case where the solver passed but the neural guard refused. A separate exposed-offset diagnostic found no solver solution at its 19 refused transport probes. Neither finding proves physical unreachability. They leave grasp orientation, observed grasp offsets and route geometry as open issues; extra GPU training alone has not been shown to solve them.

## Reproduction and retained evidence

- [Final audit](audit.json), [selection freeze](frozen-selection.json), [48 trial summaries](final/summary.json) and [package verification](release-checks.json).
- Every final and development physical report, original synchronized state archive, compact RGB observation and image-action counterfactual. No numeric state array was reduced or rewritten.
- Every exposed trial and both diagnostic reports, including refusals. Camera PNGs remain in the local raw archive.
- [Package manifest](package-manifest.json) records source/destination hashes. Scenes change only `compiler.meshdir` to resolve the same repository meshes. Unique evaluated Python files are deduplicated under `source-blobs/<SHA256>.py`; each trial's `source_sha256` maps its original filename to that content.

Run from the repository root with the documented training/OpenVINO environment:

```powershell
.venv-training/Scripts/python scripts/summarize_rgb_servo_routing.py docs/robotics/evidence/rgb-servo-v2/final --freeze docs/robotics/evidence/rgb-servo-v2/frozen-selection.json
```

For a fresh physical reproduction of an already exposed relay case, choose an unused output directory:

```powershell
.venv-training/Scripts/python scripts/evaluate_rgb_servo_routing.py --routing docs/robotics/experiments/rgb-servo-routing-v2.json --split evaluation --seed 2026095809 --output .run/rgb-servo-v2-reproduce-5809
```

A repeat is a reproduction, not another independent holdout. The full suite command requires the final freeze and refuses an existing output folder or a changed frozen source. No further selection may use these final seeds. New development needs a separate bounded protocol and new final scenes. Disk checks preserve the 10 GiB reserve; raw and packaged V2 evidence together remain within the 256 MiB experiment budget.
