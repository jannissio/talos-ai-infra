# Talos completion checklist

Updated September 13, 2026. Finish and hand over by **September 14**. The user works September 15. The live public dashboard still describes September 10–16 and shows submissions open; that does not extend our internal target.

This is the active development and release plan. It supersedes the narrower scope decision in `RELEASE_PLAN_2026-09-14.md`. Preparing files or stopping an experiment does not complete the full submission goal.

## Sources checked now

- [Intel online challenge PDF](https://drive.google.com/file/d/1xSisqTQUAFQiLOpjLZrCVTCsQi4bMCpO/view), all five pages read in the browser September 13.
- [Live event dashboard](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon/live), September 13: submissions open, online September 10–16.
- [Submission form guide](https://lablab.ai/ai-articles/hackathon-guidelines), September 13. The authenticated entry form still needs checking.
- [Intel cloud request guide](https://cloud.intel.com/docs/how_to_request.html) and [registration guide](https://cloud.intel.com/docs/how_to_register.html), September 13. See [cloud findings](../robotics/INTEL_CLOUD.md).
- The user's laptop status, the [PC baseline](../robotics/PC_BASELINE.md), and [preserved bottle coverage result](../robotics/RTX4070_COVERAGE_EXPERIMENT.md).
- [Licensing recheck](LICENSING_2026-09-13.md): rendered event page and terms section 16 require original, open-source, MIT-compliant submissions. Original Talos materials remain MIT; upstream notices remain separate.

## Required submission artifacts

`Partial` means an older artifact exists but does not yet demonstrate the expanded final system. No unchecked item below is claimed complete.

| ID | Requirement | Current evidence/status | Completion evidence |
| --- | --- | --- | --- |
| R1 | Reproducible public repository: setup, dependencies, simulation assets, training, evaluation, inference | Partial: published repository, models, compact bottle training input; fresh PC installs verified | Final release revision; clean setup and reproduction commands for every promoted policy |
| R2 | Reproducible randomized dual SO-101 MuJoCo dinner simulation | Partial: complete scene and seeded variation; broad task robustness unproven | Frozen evaluation specifications, per-seed physical outcomes and disclosed ranges |
| R3 | Intel benchmark script with latency, throughput, device and precision | Partial: scripts and historical laptop results; no final qualifying machine run | Actual CPU SKU, inference device, OpenGL renderer, versioned benchmark JSON and final physical run |
| R4 | Video demonstrating successful task execution across ten randomized seeds | Partial: original narrow bottle montage exists; whole workflow is not demonstrated across ten seeds | Final ten-seed outcome table and video showing instructions, variation and outcomes, retaining failures |
| R5 | Technical architecture README covering model, coordination, training, robustness and OpenVINO mapping | Partial: original limited system documented | Architecture and claims rewritten around the final verified implementation |
| R6 | Final MuJoCo and inference on Intel hardware | Open external dependency | Core Ultra Series 2/3 execution preferred to avoid written/livestream ambiguity; actual evidence required |
| R7 | Platform enrollment/team and primary Intel online track | User confirms enrolled with a team September 13; track/form still to inspect | Primary Intel online track confirmed in final form |
| R8 | Title ≤50 characters; summary ≤255; long description ≥100 words; technologies | Partial: `submission/PROJECT.md` | Updated copy validated against the actual form |
| R9 | Cover, ≤5-minute video <300 MB, presentation PDF | Partial: preserved original files exist | Versioned final assets visually checked, playback/size/duration checked |
| R10 | Public GitHub and judge-access/demo URL | Authenticated check found existing GitHub repository is private; Pages preview prepared; interactive hosting open | Tested anonymous repository/site access; resolve actual platform field without representing localhost as public |
| R11 | Final upload and submission confirmation | Open; handoff assigns upload to the user | Actual submitted-entry confirmation, not merely a ready folder |
| R12 | Licenses, provenance, private-data/identity checks | Existing sanitized history and notices | Final staged content and metadata inspected; published models/evidence preserved |

## Development requirements and scoring coverage

The PDF requires an end-to-end perception/language/action workflow. Its examples include pouring and supporting a mug, but it does not state that every possible manipulation must be implemented. We retain the user's broader ambitions below instead of silently declaring them out of scope.

| ID | Capability / criterion | Verified starting point | Work still required |
| --- | --- | --- | --- |
| D1 | Multi-step table setting; task/bimanual score 30 | Six programmed physical skills in seed 42 | Learned control across bottle, plate, mug, drawer, fork and spoon; combined validation |
| D2 | Camera + instruction + maintained task context; reasoning score 20 | Bounded grammar; initial bottle RGB centroid; neural trajectory | Scene observations influence skill selection/progress; safe sequencing and supported combinations |
| D3 | Broader bottle starting positions; robustness score 15 | 10/10 small-jitter cases; 1/3 exposed wider cases | Repair demonstrated grasp coverage and fit; freeze new evaluation before checkpoint selection |
| D4 | Continuous visual correction | Unfinished; current model fixes visual context at start | Current camera observations must measurably affect execution, including disturbance tests; visual checks alone are not claimed as corrective control |
| D5 | Learned placement without reset between skills | Unfinished; success can precede arm parking | Finish safe learned retreat/parking, sequence next skill, verify prior objects stay placed |
| D6 | Two-arm handoff / complementary action | Programmed left-to-right bottle relay through the table | Integrate coordinated workflow and verify shared-space behavior; retain precise relay description |
| D7 | Reverse/other-object/direct handoffs | Unfinished | Implement and physically evaluate separately; do not relabel table relay as direct hand-to-hand |
| D8 | Pouring / one arm supports mug | Unfinished; no liquid dynamics | Explicit example/extension; require physical and semantic validation before any claim |
| D9 | Arbitrary reachable positions/destinations | Unfinished | Define measured workspace and unreachable/ambiguous refusal; finite tests cannot establish universal coverage |
| D10 | Other poses, masses, friction, shapes, light/background | Some bottle perturbation evidence | Evaluate final multi-skill system with disclosed perturbations and separate failure categories |
| D11 | Speed and live responsiveness | Full programmed sequence roughly four simulated minutes; live query timing variable | Profile and improve within physical regression constraints; report wall and simulation time separately |
| D12 | OpenVINO optimization; score 20 | Original upright model exported, historical CPU/iGPU benchmarks | Export promoted components; measure accuracy before/after precision changes and actual Intel runtime |
| D13 | Speechmatics integration | **Human microphone trial succeeded on PC**; synthetic tests also preserved | Rehearse final multi-step commands; request microphone only when needed; key stays in local `.env` |
| D14 | Reproducibility/technical quality 10; innovation/demo 5 | Baseline application/training tests pass, preserved assets | Final relevant tests, clean release, honest limitations and readable demonstration |

## Ordered milestones

1. **Requirements and access.** Complete this audit; inspect authenticated Intel AI PC catalog with the user. A normal 2–3-business-day request cannot be assumed available tomorrow. Prepare portable Intel verification while local work continues.
2. **Six-skill learning data.** Record compact, physical teacher demonstrations for each skill, including retreat/parking. Independently replay saved action endpoints. Keep failed attempts and split membership. Initial diagnostic seed 42 is an exposed development case, not a holdout.
3. **Multi-skill learned controller.** Train on the RTX 4070 in bounded batches. Preserve original models. Use image inputs and motor feedback at deployment; simulator poses and teacher stages may label training/score outcomes but must not secretly generate learned actions. First prove each skill on development cases, then composed execution.
4. **Visual feedback and coverage.** Add observable scene progress and correction; test image/scene perturbations. Address the old teacher reach failure as a new experiment, preserving the stopped coverage-v1 protocol and results. Promote only physically verified changes.
5. **Language and bimanual execution.** Expand supported commands, context and chaining; explicitly refuse unsupported actions. Demonstrate at least the verified table relay as part of a coordinated workflow and pursue additional manipulations with separate evidence.
6. **Final evaluation and Intel deployment.** Declare fresh evaluation seeds before selection. Run all final trials, preserve the denominator, compare optimized inference quality, and record actual Intel hardware execution.
7. **Package, publish and submit.** Produce versioned assets, validate links/form fields, inspect credentials/metadata, publish sanitized release, prepare the user's upload and record submission confirmation.

## Acceptance and storage rules

- Every promoted skill needs physical replay and learned execution evidence, not just low training loss or an animation.
- Report learned, programmed, classical-vision and privileged monitoring components separately.
- No attachments, teleportation or hidden forces during manipulation. Setup/reset writes are explicit.
- Check the actual destination drive before every dataset, episode and export, and during long writes; retain 10 GiB plus expected writes. September 13 starting free space: about 535 GiB.
- Never overwrite the original published models, training input, videos, presentation or failure evidence.
- Account access, qualifying Intel hardware, judge access and actual submission are open dependencies until confirmed. The full goal remains active while independent work can proceed.

## Progress log

- September 13: requirements re-read; full goal created; branch `codex/final-submission`; Intel catalog sign-in and enrollment/laptop availability requested. Six-skill demonstration/replay pipeline is the next implementation step.
- User confirms enrollment and Intel laptop availability tomorrow; cannot access an Intel cloud account. Historical laptop evidence identifies i7-10850H plus Intel UHD and Quadro T2000: it is not Core Ultra. Prepare final CPU/iGPU execution and retain the written/spoken eligibility discrepancy.
- Seed 42: all six teacher skills and independent motor-command replays succeed, including release and parked arms. The next batch replays the exact float32 endpoints saved for training. These are demonstration results, not learned six-skill claims.
- The preserved bottle model passes the new learned completion condition on seed 42: stable release plus both arms parked, 1.12 mm placement error, no unexpected contacts. This removes the early-success/reset limitation for that tested execution path; multi-skill learned chaining still requires verification.
- New training batch: 54/54 exact saved-endpoint replays passed across nine training scenes. About 140 MiB, with over 534 GiB free afterward.
- Five new neural skill models trained on the RTX 4070; all six learned skills pass their individual seed-42 starting states. The complete learned six-skill sequence passes seed 42 without resets or teacher action generation. The first refined candidate passed only 1/6 development sequences; failure reports are retained. It is not promoted as robust.
- Visual diagnosis: parked wrist views often face away from tableware; whole-frame PCA is sensitive to unrelated scene changes. Revision v2 uses calibrated fixed workspace views and explicit RGB geometry for plate, mug and cutlery, retraining on the same nine training scenes. This remains initial-image conditioning; continuous correction is still open. The ten evaluation seeds remain unexposed.
- Geometry v2 completes 4/6 development sequences; seed 42 and two development cases still fail placement precision. No acceptance tolerances were relaxed. The next candidate refines the same training data; all failed reports are retained.
- Licensing rechecked: keep MIT for original Talos contributions and the existing Apache asset notices. User requested a delegated pretrained-policy feasibility/license review; no model has been downloaded or incorporated by that review.
- [Pretrained-policy review](../robotics/PRETRAINED_POLICY_REVIEW.md) completed: SmolVLA is the plausible 12 GB adaptation candidate; exact weight licenses and dual-arm data adaptation matter. No pretrained controller is claimed integrated.
- Prepared [portable Intel verification](../robotics/INTEL_FINAL_VERIFICATION.md), including device names, OpenGL renderer, precision, throughput, parity and a physical run. Its PC smoke test passes physics and correctly fails the strict Intel hardware flag.
- The expanded data protocol declares 32 additional training scenes without touching evaluation seeds. Separate approach-only visual-feedback experiments retain both successful trials and tipping/ambiguity failures; no full continuous-correction claim.
- The 32-scene collection finished with 188 passing replays from 190 attempted skills. Two fork plans failed cabinet clearance; two following spoon episodes were not collected. The combined data has 41 eligible episodes for bottle/plate/mug/drawer and 39 for fork/spoon. All failures remain preserved.
- The larger v4 fit failed all six complete development sequences; seed 42 also failed spoon orientation/release. More data alone has not fixed robustness. Revision v5 tests sustained gripper labels and a focused drawer-handle RGB detector, with a physical replay gate before fitting.
- Both directions of table-supported bottle relay have 13 training examples per leg. All 52 repaired aligned-input replays pass. Earlier forward neural fits failed all four exposed/development trials; repaired-label neural testing is in progress. These input replays are not learned execution results.
- The live browser completed all six neural skills on exposed seed 42 with the first Adam candidate, in 246.34 simulated seconds. The current automated checks pass 53 application tests and 10 training-contract tests. The 41 original protected files remain byte-identical.
- [Public access check](PUBLIC_DEMO.md): the repository is currently private, correcting the earlier public-access assumption. A static evidence page is prepared; a live backend and actual submission remain open.
