# Talos submission text — September 14 update

Saved in the submission draft on September 14 and verified by leaving the form, reopening it and checking all three steps. The form contains the 198-character summary, 1,856-character long description and 1,559-character additional information below. It updates the retained v6 draft with completed voice/Intel checks and the measured limits of the broader work. Existing v6 media and all earlier evidence remain unchanged. GitHub and Hugging Face remain private; only the user performs final Submit. The team page still reports “Submission draft in progress.”

**Title:** Talos: Learned Dinner-Table Robotics

**Short description:** Camera-based neural skills coordinate two SO-101 arms through six dinner-table tasks and bottle relays in MuJoCo, with Speechmatics voice input, OpenVINO inference and preserved physical evaluation.

**Mode:** Online  
**Category:** Home Automation  
**Selected track:** Intel Online (current form label; selection unchanged)  
**Selected platform technology:** Speechmatics api

## Long description

Talos turns supported spoken or typed instructions into dinner-table manipulation by two SO-101 arms in MuJoCo. Six learned skills place a bottle, plate and mug, open a passive drawer, and retrieve the fork and spoon. Speechmatics transcribes commands; a bounded language interpreter and classical RGB checks select actions. Neural motor trajectories use camera features and joint feedback. A separate simulator-state monitor stops or scores execution.

The original learned dinner system passed 8/10 frozen sequences. Table-supported bottle relays passed 5/5 frozen trials in each direction; a composed relay-plus-dinner workflow passed 8/10. All failures remain. Later stereo correction during late mug placement passed 20/20 narrow workflows, as did the original baseline; frozen corrective images passed 0/20. This establishes bounded visual feedback, not general workspace coverage.

A human “Set the table” voice command completed all six skills. The private hosted live-mug CPU demo also passed. Six-skill baseline execution is recorded on a legacy Intel laptop with Intel CPU/iGPU inference and UHD rendering. Core Ultra Series 2/3 eligibility remains unresolved; the latest live-mug profile needs separate Intel verification.

Broader work includes seven loose items, full tabletop yaw and stable alternative orientations. No complete jointly randomized seven-item scene has passed. Calibrated RGB-D and an adapted CLIPort runtime work on the RTX 4070, but no shared Talos spatial policy has been trained. Arbitrary reachable positions, general visual recovery, unrestricted language, pouring and airborne handoffs remain unfinished.

Original Talos contributions are MIT licensed; third-party components retain their own licenses. Source, compact training inputs, model provenance, protocols and successful/failed physical outcomes are preserved.

## Additional information

The demonstrated product is the six-skill learned dinner controller and table-supported bottle relays. Supported commands include “Set the table”; arbitrary instructions are refused rather than silently converted into programmed execution. Bounded late mug correction is selected in the current private demo.

Evidence separates learned execution, classical perception, privileged physical teachers, scratch geometry, explicit reset interventions and actual motor-driven actions. The wider-bottle application passed 10/10 paired final dinners versus 0/10 for the previous bottle controller, while the other objects retained narrow starting distributions. This is not jointly randomized seven-item coverage.

The broader shared-table goal remains unfinished. Separate scenes have passing plate/side-plate, bottle/glass, and glass/spoon subsets; these results cannot be combined into a complete workflow. The adapted CLIPort heads run within measured RTX 4070 memory, but have not been trained for Talos. Latest individual perception and grasp diagnoses are exposed development evidence, not fresh generalization results.

Legacy Intel execution and a human full-sequence voice rehearsal are verified. Qualifying Core Ultra Series 2/3 execution remains unresolved. The private live-mug hosted CPU sequence passed in 287.79 wall seconds; anonymous judge access remains a final release check. GitHub and the Space stay private until the agreed release time. All original models and evidence remain preserved, with neutral commit metadata and credentials excluded.

## Links and retained media

- Repository: https://github.com/jannissio/talos-ai-infra
- Hosted CPU demo: https://huggingface.co/spaces/jannis-sms/talos-dinner-robotics
- Existing cover, main video, slide PDF and all-trial videos: [v6 media inventory](../final-v6/PROJECT.md#media-and-links).
- Current requirements and remaining work: [completion checklist](../../docs/hackathon/FINAL_SUBMISSION_CHECKLIST.md).
- Human microphone and private live-mug CPU outcome: [evidence](../../docs/robotics/evidence/human-voice-dinner-v1/README.md).
- Intel baseline execution/rendering: [evidence](../../docs/robotics/evidence/intel-final-legacy-v3/README.md).
- Broader development status: [shared pipeline](../../docs/robotics/WHOLE_TABLE_PIPELINE.md).

## Remaining submission actions

- Review this saved text against any subsequent measured results. Final Submit has not been pressed.
- Update slides/video captions that still say the human rehearsal or legacy Intel run is pending; preserve the previous versions. Existing v6 videos show the earlier bounded system, not complete seven-item control.
- Resolve the Core Ultra eligibility requirement or explicitly disclose the limitation; no qualifying result is asserted.
- At the agreed release time, make the audited repository and demo public and verify anonymous access with a fresh judge-style CPU trial.
- Recheck all form fields and media, then leave final Submit to the user.
