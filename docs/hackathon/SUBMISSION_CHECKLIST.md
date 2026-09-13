# Submission checklist

**Current status, September 13:** use the [September 14 release plan](RELEASE_PLAN_2026-09-14.md) for the evidence-backed readiness table and remaining actions. The historical checklist below predates the packaged models, videos, fresh PC installation and human voice rehearsal; its unchecked boxes are not a current claim that those artifacts are missing. The user requires completion on September 14 and gives September 15 as the deadline; reconcile the exact cutoff without delaying that target.

Prepared 2026-09-08 for the online Intel track with the Speechmatics bonus. Unchecked boxes mean **not yet verified or completed**, not an assertion that an existing user account lacks the item.

September 11 livestream clarification: other Intel systems with integrated graphics are accepted in the spoken Q&A; Core Ultra Series 2/3 earns bonus points. This conflicts with a strict written-brief sentence. [Review and development implications](LIVESTREAM_REVIEW_2026-09-11.md). The deadline, ten-seed evidence and Speechmatics checklist remain unchanged.

## Enrollment

- [ ] Every member is independently registered and has checked their approval/enrollment status.
- [ ] Profiles and Discord connections are complete.
- [ ] A 1–5-person team exists on lablab.ai, including for solo participation.
- [ ] The team knows which member will finalize submission.
- [x] September 10 public recheck resolves the previous start/close conflict: kickoff 17:00 CEST September 10; deadline 20:30 CEST September 16. See [update](UPDATE_2026-09-10.md).

Sources: [event](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon), [FAQ](https://lablab.ai/guide).

## lablab submission form

| Check | Item | Preparation requirement |
| --- | --- | --- |
| [ ] | Project title | Clear; maximum 50 characters in the step-by-step guide |
| [ ] | Short description | At most 255 characters |
| [ ] | Long description | At least 100 words: problem, solution, users, technology, and current capabilities |
| [ ] | Primary track | Intel online bimanual VLA; only one primary track |
| [ ] | Technology/category tags | Include actual stack; explicitly identify Speechmatics |
| [ ] | Cover | PNG/JPG, 16:9 |
| [ ] | Presentation video | MP4, no longer than 5 minutes; keep below 300 MB per step-by-step guidance |
| [ ] | Slide deck | PDF |
| [ ] | Repository | Public GitHub URL with usable setup instructions |
| [ ] | Demo platform | Identify how the prototype is hosted |
| [ ] | Application URL | Judges can interact with the prototype |
| [ ] | Additional information | Relevant limits, reproducibility details, and evidence links |

Sources: [step-by-step guide](https://lablab.ai/ai-articles/hackathon-guidelines), [submission guide](https://lablab.ai/delivering-your-hackathon-solution), [rule book](https://lablab.ai/hackathon-rules). Recheck the actual form for upload/link mechanics and limits when available.

The rule book names Streamlit, Replit, and Vercel; the submission guide presents them as hosting options. Confirm the acceptable interface for a locally executing robotics simulation. Do not assume a recording alone replaces the application URL.

The generic submission guide currently contains an **IBM Bob report** instruction, although neither this event nor the Intel brief requires IBM Bob. This appears to be content intended for another event. It remains a clarification item, not a reason to add IBM Bob to our stack.

## Intel technical package

Source for this section: [Intel online brief, page 4](https://drive.google.com/file/d/1xSisqTQUAFQiLOpjLZrCVTCsQi4bMCpO/view), with deployment requirements on page 3.

- [ ] **Reproducible repository:** setup steps, dependencies, MuJoCo assets, training/fine-tuning code, evaluation code, inference code, and exact reproduction commands.
- [ ] **Simulation package:** two SO-101 arms and the dinner-table scenario, including randomization and evaluation configuration.
- [ ] **Benchmark script:** executes on the selected Intel target; reports exact hardware, latency, throughput, chosen device and precision. Reconcile the written Core Ultra wording with the livestream clarification before the final eligibility claim.
- [ ] **Demonstration video:** shows task execution for ten randomized environment seeds, with instructions, variations, and outcomes verifiable.
- [ ] **Technical README:** architecture, policy/model choice, coordination, training, robustness methods, optimization, and Intel device mapping.
- [ ] **Target execution:** final simulation and AI/VLA/VLM inference run on qualifying Intel hardware.
- [ ] **OpenVINO:** supported components are optimized and performance gains are evaluated against retained task quality.
- [ ] **Packaging recommendation:** consider an Intel-ready downloadable container; validate dependencies, assets and commands rather than merely supplying a Dockerfile.

The brief's suggested demo sequence includes scene/instruction, camera-based decisions, both arms acting with at least one hand-off or complementary action, final table state, a ten-seed success summary, and an Intel benchmark. Account for these in the five-minute presentation. Confirm whether a supplemental full run recording is allowed; retain the complete recordings locally regardless.

## Evidence to collect during implementation

These are our recommended records, not extra organizer-mandated file formats.

- [ ] Per-seed results: seed, randomized parameters, instruction, completion, failure reason, duration, and recording link.
- [ ] Benchmark record: exact CPU/model, device, runtime/model versions, precision, warmup/measurement method, latency, throughput, and task success before/after optimization.
- [ ] A dependency/model/asset attribution list and AI-assistance disclosure.
- [ ] Honest accounting of failures, unsupported commands, and remaining limitations.
- [ ] Reproduction from a clean environment or second machine using the written commands.

## Speechmatics bonus evidence

- [ ] Speechmatics genuinely processes speech in the project.
- [ ] The demo connects spoken input to useful robot behavior, with transcript/command visible.
- [ ] Architecture and submission text identify Speechmatics and its role.
- [ ] The relevant technology tag is selected; mention that the entry is also for Best Use of Speechmatics.
- [ ] Access, credit limits, and any additional bonus instructions have been checked with the sponsor.
- [ ] If claiming multilingual, accent, interruption, or latency benefits, show measured examples.

The event requires no separate bonus enrollment. The tagging, explicit wording, and evidence presentation above are our recommendations to make eligibility easy to assess. Source: [bonus section](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon).

## Final submission

- [ ] Freeze a clear project version and check all public links while signed out.
- [ ] Verify the demo runs on the intended hardware and the repository contains no credentials.
- [ ] Confirm the final time against organizer announcements.
- [ ] Submit using the platform's final submit action, not merely save a draft.
- [ ] Confirm the submitted status and save the submission URL/receipt.
- [ ] Keep a backup of the repo version, video, slides, and benchmark/evaluation outputs.

**Internal target: September 16 at 18:00 CEST.** The current event page and dashboard both give **20:30 CEST that day** as the submission cutoff. No automatic grace period is assumed.
