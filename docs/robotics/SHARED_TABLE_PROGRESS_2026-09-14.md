# Shared-table progress — 2026-09-14

This development checkpoint covers terminal whole-table runs V12–V19. V20 seed4004 has no completed `report.json` at the cutoff and remains pending. The machine-readable inventory is [`shared-table-followup-inventory-v3.json`](evidence/shared-table-followup-inventory-v3.json); public diagnostic reports are [`table-regrasp-diagnostic-v1.json`](evidence/table-regrasp-diagnostic-v1.json) and [`table-edge-regrasp-diagnostic-v1.json`](evidence/table-edge-regrasp-diagnostic-v1.json).

V12–V19 contain 298 physical entries, 939,801 frames, and 1,308 observations: 25 physical successes and 273 failures, plus 619 planning failures retained. Eleven actions were accepted into the continuously evolving scene, totaling 85,334 frames. Every physical entry and accepted action replays exactly. No run completes the randomized seven-item table, and the final per-item results remain incomplete: V17 reaches plate and side plate; V18 and V19 reach no final item checks.

| Run | Entries / frames | Passing / failed | Planning failures | Accepted actions | Final task pass |
|---|---:|---:|---:|---:|---|
| V12 seed4001 | 26 / 43,569 | 2 / 24 | 26 | 2 | plate |
| V13 seed4001 | 27 / 67,018 | 2 / 25 | 46 | 2 | plate |
| V14 seed4002 | 25 / 63,629 | 4 / 21 | 52 | 0 | none |
| V15 seed4002 | 26 / 107,008 | 6 / 20 | 88 | 1 | none |
| V16 seed4002 | 39 / 133,699 | 3 / 36 | 91 | 1 | none |
| V17 seed4001 | 9 / 39,680 | 3 / 6 | 73 | 3 | plate, side plate |
| V18 seed4004 | 78 / 268,519 | 4 / 74 | 134 | 1 | none |
| V19 seed4002 | 68 / 216,679 | 1 / 67 | 109 | 1 | none |

The airborne spoon regrasp diagnostic is separate: 16 physical trials, 48,251 frames, zero successes, and exact independent replays. The edge diagnostic is reset-only: six stable supported overhangs across 20,400 support-physics frames; its final gate checks 288 receiver poses over 9 seeds each, with 286 IK-unsolved and 2 IK-solved-but-table-colliding poses. It delivered or regrasped nothing. The invalid-reset incident and correction remain retained. Neither diagnostic proves universal impossibility.

Finished glass-placement development is also separate repeat-start evidence: 12 attempts, 4 new physical action attempts, 8 planning-only attempts, 68,032 recorded frames including repeated prefixes, 20,014 new action motor steps, 48,006 repeated buffer-prefix steps, and 2 passing physical actions with exact replay. The V13 free-yaw continuation is a separate failed planning result (8,002 frames, 8,001-frame successful prefix); that prefix is counted once only.

V20 has since finished and is preserved separately in the [V20–V22 follow-up](evidence/shared-table-followup-v20-v22.json): 78 physical attempts, 300,908 frames, eight passing primitives and one accepted mug clearance. V21/V22 are planning-only bottle diagnoses in an unchanged joint scene. No final table passes. These later outcomes stay outside the frozen V12–V19 totals above. Planning, reset-only, physical, and repeated-start diagnostic totals remain separate.

The separate [V23 bottle-buffer diagnostic](evidence/shared-table-followup-v23.json) closes with 60 failed physical actions and 188,059 exactly replayed frames. A [measured-axis lift diagnosis](evidence/bottle-measured-lift-diagnostic-v1.json) finds two geometric paths after the same saved grasp, but its one 394-state physical continuation loses finger support and stops at the unchanged penetration guard. These outcomes add no bottle success.

The [glass bridge dependency](evidence/glass-bridge-development-v2.json) now passes both physical legs from the exposed post-mug-clearance state: temporary placement/park, then a horizontal regrasp to the unchanged final setting. The combined final trace has 15,085 states; the second leg lifts 6.58 cm, holds 1.805 s and finishes at 0.398 mm XY error with exact replay. Its 236 geometry-search entries and all failed bridge attempts remain. Generic alternate IK seeds pass source/goal checks without a scene-specific joint vector. This is one upright-glass dependency; shared whole-table integration and other glass orientations remain open.

The next dependency is integrating the measured intermediate-pose selection, horizontal grasp and checked retreat into the common teacher and executing them in a continuous joint scene. A shared visual policy still needs successful varied demonstrations and a fit. Handover remains September 14 at 18:00 CEST; the official deadline is September 16 at 20:30 CEST. GitHub and Hugging Face remain private, final Submit remains the user’s action, and all 41 protected published artifacts retain their original hashes.
