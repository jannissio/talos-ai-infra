# Preparation and development plan

**September 11:** dinner-table development has begun under the user's instruction. Use the [current submission build plan](BUILD_PLAN.md) for next actions and [dinner scene milestone](../robotics/DINNER_SCENE.md) for implemented behavior. The proposed schedule below is retained as September 8 preparation history.

Prepared 2026-09-08. This is **our proposed working plan**, not an official organizer schedule or a committed implementation. The [brief](BRIEF.md) and [open questions](OPEN_QUESTIONS.md) explain the constraints.

**September 10 update:** public kickoff is now 17:00 CEST September 10 and the deadline is 20:30 CEST September 16; the portal is live. The online challenge PDF is unchanged, and a new-to-our-notes Intel installation bundle is linked. See [the recheck](UPDATE_2026-09-10.md). The current learned-policy recommendation is in [AI control plan](../robotics/AI_CONTROL_PLAN.md); the September 8 preparation section below is historical.

Preparation update: the [robotics decision research](../robotics/RESEARCH.md) reviews work through late August 2026 and current software as of September 8. It identifies reusable MuJoCo/SO-101 assets, policy candidates, and experiments before selecting an architecture. Subsequently, at the user's explicit request, a [local chemistry learning sandbox](../../simulation_lab/README.md) was built on September 8 with manual robot control and actual MuJoCo physics. No learned policy has been selected or trained. The user is creating the team; enrollment completion has not been independently verified.

## Can we start on September 8?

**Yes, preparation can begin immediately.** The event encourages learning before kickoff, and teams can form early. Researching APIs, designing the architecture, finding reusable libraries/models, checking access, and preparing the environment are useful now.

**Eligibility of project-specific coding before kickoff is not confirmed.** The [general lablab guide](https://lablab.ai/guide/ai-hackathons) says libraries, starter templates, and earlier non-AI scaffolding are generally allowed, while most events expect core AI functionality to be built during the event. It expressly defers to event-specific rules. The [event](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon) tells online entrants to start at kickoff, without a complete policy on earlier code or training.

Recommendation: prepare now; begin the competition's robotics, policy, and voice integration after the organizer-confirmed start. If earlier work is allowed, record what was created when. An unanswered question is not permission. Documentation created on September 8 should remain identifiable as preparation.

## Proposed project

Working concept: **a voice-controlled table-setting assistant**.

1. A person speaks a table-setting instruction.
2. Speechmatics transcribes it.
3. A local language/vision component interprets the instruction and camera views.
4. A robotics policy coordinates both simulated arms.
5. A small interface shows the transcript, current action, simulation, and outcome.
6. Evaluation reruns the scenario with randomized conditions; benchmarks report Intel performance.

The principal robotics inference and MuJoCo simulation run on qualifying Intel hardware for the final demonstration. Clarify the allowed boundary for external Speechmatics transcription and any remote UI. Optional spoken responses or multiple languages come after the main task works.

This is an advanced robotics project. The major uncertainties are a usable dual-arm scene, policy adaptation/data, reliable coordination, and target hardware access. A speech frontend alone does not reduce those requirements. No claim is made yet that this local computer can train the eventual policy efficiently.

## Milestones

| Date, Berlin time | Action | Evidence / exit condition |
| --- | --- | --- |
| Sep 8–9 | Enrollment/team, hardware access, rule clarification, resource study, architecture and scope | Team status checked; a credible final-demo hardware path; questions sent by the user; preparation notes |
| Sep 10, before kickoff | Recheck announcements and submission fields | Agreed coding start and deadline; current brief version |
| Sep 10, after confirmed start | Initialize project code and reproduce a basic two-arm scene; test candidate inference conversion | Camera observations, both arms controllable, one object manipulated, one supported model component running on Intel |
| Sep 11 | Establish a reproducible policy/data workflow and a minimal multi-step table task | Training/fine-tuning path documented; measured baseline |
| Sep 12 | Add language-and-camera decisions and meaningful two-arm cooperation | Hand-off or complementary action; no reliance on one fixed initial scene |
| Sep 13 | Add the live Speechmatics command path | Spoken instruction visibly drives the task; errors are understandable |
| Sep 14 | Optimize and benchmark on the actual final hardware; run randomized evaluations | Performance and success measurements; the ten-seed configuration fixed |
| Sep 15 | Finish ten-seed evidence, demo interface, slides, video, and clean reproduction | Complete draft submission; final-demo machine already validated |
| Sep 16, by 18:00 | Submit and verify receipt/status | Public links, final version, uploaded media, confirmed submitted state |

Availability is unknown, so this is a milestone plan rather than an assumption of full-time work every day. Bring Intel deployment forward rather than leaving access or conversion until the last day.

## Implementation decisions to make after access and scope are clear

- Choose a small compatible pretrained policy and an achievable task-specific adaptation approach. Validate dual-arm action dimensions, camera inputs, licensing, and OpenVINO support before choosing between candidates named in the brief.
- Obtain or construct properly licensed SO-101 and table assets. A ready-to-run organizer scene or dataset was not supplied in the public brief; ask for starter resources.
- Define task success and randomization ranges before measuring improvements.
- Start with one full command sequence. Additional commands, interruptions, and languages are extensions, not substitutes for core manipulation.
- Keep simulation, inference, training, evaluation, benchmarking, and the speech interface independently runnable for debugging and reproduction.
- Record real measurements; avoid treating a model conversion or attractive UI as evidence of task success.

If the initial two-arm baseline or qualifying hardware access cannot be established, discuss a reduced eligible scenario with the organizers promptly. Do not silently substitute a generic chatbot or claim an unsupported platform meets the requirement.

## Suggested repository structure once implementation starts

```text
README.md                 # Project/run instructions; preserve links to research
docs/hackathon/            # This dated research package
src/                      # Speech, orchestration, perception/policy, control
simulation/               # Robot/table scene and licensed assets
training/                 # Data collection and training/fine-tuning
evaluation/               # Seeded scenarios and success checks
benchmarks/               # Intel inference benchmark
results/                  # Small evaluation/benchmark records
submission/               # Final text, deck, and media references
```

This tree is a proposal only. No empty code scaffold or dependency installation was performed during the research task. Large models, datasets, and recordings should have reproducible download/access instructions rather than automatically entering the source repository.
