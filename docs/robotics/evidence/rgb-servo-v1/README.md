# Frozen RGB feedback result: 48 physical trials

September 13, 2026. Candidate frozen after 18 development trials; every final seed 2026095301–2026095312 was then tested under all four conditions. Inference used OpenVINO CPU FP32 on the AMD Ryzen 7 5800X PC. This is not Intel execution evidence.

| Condition | Passed / all 12 seeds | Failures |
| --- | --- | --- |
| Live RGB, no push | 2/12 | 7 pre-motion refusals, 1 placement failure, 1 missing RGB estimate, 1 descent timeout |
| Initial image frozen, no push | 1/12 | 7 pre-motion refusals, 3 placement failures, 1 descent timeout |
| Live RGB, disclosed push | 3/12 | 7 pre-motion refusals, 1 missing RGB estimate, 1 descent timeout |
| Initial image frozen, same force schedule | 0/12 | 7 pre-motion refusals, 4 placement failures, 1 descent timeout |

**Decision: keep experimental.** Reliable broad bottle placement is not established. Every refusal remains in the denominator. A refused neural prediction does not prove the position physically unreachable. The selected six-skill dinner suite is unchanged.

The force schedule is +0.75 N world X for 0.1 seconds, starting at simulation time 1.5 seconds at 12 mm above the bottle origin. The simulation applies this disclosed test force; the controller is not told its value or timing. Five seeds reached the disturbance, with measured translations about 6.4–11.6 mm. Paired force schedules are identical; their largest measured translation difference is about 0.000009 mm.

Only seed **2026095309** passed both undisturbed controls. On that seed, the roughly 8.77 mm push resulted in **1.53 mm / pass with live images** versus **11.34 mm / fail with frozen images**, against the unchanged 8 mm placement threshold. Same-joint-state counterfactual motor commands differ by up to 0.081 rad. This supports a narrow corrective result in one controlled case. It does not support a general robustness claim. The full test also contains a case where the push helps a nominal failure.

The [50.4-second supplementary comparison](../../../../submission/final-v6/rgb-feedback-experiment.mp4) shows this one pair at 1× speed, labels its selection and displays the complete test counts. It is an offline rendering of recorded physical states, not a replacement for the main dinner/relay demo. All 1,008 video frames decoded successfully and the captions and physical sequence were visually checked.

`summary.json` preserves the original batch summary; `audit.json` verifies all 48 reports against the frozen model, IR and source hashes and records every failure category. Each trial folder includes its original report, compact authoritative states and same-joint-state action counterfactuals. Scene XML only remaps mesh paths to the same assets. `evaluated-source/` deduplicates the exact source snapshots. `package-manifest.json` records original and packaged hashes plus every transformation. Original RGB shards, per-frame observer reports and preview images remain in the local experiment archive.

`history/` preserves earlier failed physical candidates and fresh perception results, including V3/V4 accuracy failures. The final rigid observer passed its separately frozen perception test: 233/236 present states accepted, 0/20 absent states accepted, 1.938 mm p95 and 5.182 mm maximum accepted keypoint error. Repeating those images through OpenVINO checks runtime parity; it is not another independent test.

Audit the packaged evidence without running simulation:

```powershell
.\.venv-training\Scripts\python scripts/summarize_rgb_servo_evaluation.py --input docs/robotics/evidence/rgb-servo-v1 --freeze docs/robotics/evidence/rgb-servo-v1/frozen-selection.json --output .run/reproduce-rgb-audit.json
```

The [model card](../../../../models/bottle_servo_v1/README.md) gives live/frozen reproduction commands. The [training inputs](../../../../training/bottle_servo_v1/README.md) explain data generation and the corrected historical target-label bug. No final outcome was used to change this candidate.
