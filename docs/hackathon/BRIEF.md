# Hackathon research brief

Checked **2026-09-08**. Scope: the publicly available event page, live dashboard, linked sponsor briefs, lablab rules and guides, and organizer/vendor sources. Private Discord announcements, enrollment-only material, and future changes are not included. Official requirements are distinguished from our recommendations and unresolved conflicts.

**September 10 recheck:** the online PDF is unchanged; the schedule is corrected below and practical Intel/Speechmatics resources have been added. See [the update record](UPDATE_2026-09-10.md). Other general-rule summaries retain their September 8 research date.

## Event and participation

| Item | Published information |
| --- | --- |
| Event | AI Infra Summit Hackathon |
| Organizers | lablab.ai and Kisaco Research |
| Theme | Working AI applications demonstrating modern infrastructure |
| Online build | September 10–16, 2026; worldwide, via lablab.ai and Discord |
| Onsite phase | September 15–17, 2026; invitation and track assignment required |
| Venue | Santa Clara Convention Center, 5001 Great America Pkwy, Santa Clara, CA |
| Entry | Free; complete enrollment and verify approval status |
| Team | 1–5 people; solo entrants need a one-person team |
| Track choice | Online requires no allocation; select one primary track at submission |

Each teammate registers independently, completes their profile, and connects Discord. General platform terms require age 18+ and impose sanctions-related eligibility restrictions. The event welcomes all experience levels, but the Intel online brief rates its challenge intermediate to advanced. The summit organizer says applications are subject to lablab approval: open online track selection should not be confused with automatic enrollment approval.

Sources: [event](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon), [platform FAQ](https://lablab.ai/guide), [summit organizer](https://www.ai-infra-summit.com/ai-infra-hackathon), [terms §3 and §11](https://lablab.ai/terms-of-use).

## Online challenge: what we actually need to build

**Intel: Bimanual VLA Manipulation with Multi-Modal Reasoning.** The published scenario is setting up a dinner table using two simulated SO-101 robot arms in MuJoCo. VLA means a model or policy connecting vision, language, and actions.

The system should turn a natural-language instruction and simulated camera observations into a sequence of coordinated manipulations. Examples in the brief involve opening a drawer, retrieving utensils, placing tableware, handing objects between arms, and having one arm support an object while the other performs a complementary action. The title calls table setting a challenge option, but the scoring explicitly assesses that workflow; do not assume a different scenario is eligible.

The technical objectives cover:

- Coordinated movement in a shared workspace, including hand-offs and collision-aware sequencing.
- Combining language with camera input to recognize objects, track progress, and choose subsequent actions.
- Robustness to changes in placement, mass, friction, shape, lighting, and background.
- Robotics policy training or fine-tuning in simulation, using LeRobot or compatible tooling. Named candidates include SmolVLA, Pi0.5, and ACT; these are examples rather than one prescribed model.
- OpenVINO optimization of supported model components, including conversion, precision/quantization choices, and appropriate CPU, iGPU, and/or NPU use without materially reducing task success.

**Final demonstration:** both MuJoCo and the AI/VLA/VLM inference pipeline must execute on an **Intel Core Ultra Series 2/3 system**. Page 3 also uses broader platform language and calls that hardware preferred, but its explicit deployment requirement is stricter. Our working interpretation follows the stricter requirement until Intel clarifies it.

Training may use local or cloud resources, but Intel explicitly does **not** provide training infrastructure. No physical SO-101 arms are required. The brief names optional/supporting resources including OpenVINO Physical AI, Intel Physical AI Studio, Open Edge Platform, Edge AI Suites for robotics, and Geti. Do not import the onsite requirement to use Anomalib into this online challenge.

The five Intel deliverables are a reproducible repository, a reproducible randomized simulation, an Intel inference benchmark script, a demonstration video covering **10 randomized seeds**, and an architecture/technical README. The detailed [submission checklist](SUBMISSION_CHECKLIST.md) maps these to evidence.

Source: [original Intel online PDF](https://drive.google.com/file/d/1xSisqTQUAFQiLOpjLZrCVTCsQi4bMCpO/view), pages 1–4; [official Drive link](https://drive.google.com/file/d/1xSisqTQUAFQiLOpjLZrCVTCsQi4bMCpO/view).

The track now links an [Intel setup guide and installer bundle](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/hackathon_resources.html). It targets Ubuntu/Intel deployment and does not contain a ready challenge simulation. See [saved package details](sources/2026-09-10/intel-install-package.json).

## Intel online scoring

| Criterion | Points | Evidence we should prepare |
| --- | ---: | --- |
| Task completion and two-arm manipulation | 30 | Completed sequence, coordinated motions, accurate placement |
| VLA / multimodal reasoning | 20 | Instruction and camera inputs affecting decisions and progress |
| Robustness and generalization | 15 | Ten seeded runs with recorded scene variations and outcomes |
| OpenVINO and Core Ultra optimization | 20 | Intel benchmark results and quality before/after optimization |
| Technical quality and reproducibility | 10 | Repeatable setup, evaluation, assets, and executable commands |
| Innovation and demonstration | 5 | Clear explanation and a useful technical contribution |
| Total | **100** | |

Source: [Intel online PDF, page 5](https://drive.google.com/file/d/1xSisqTQUAFQiLOpjLZrCVTCsQi4bMCpO/view). The evidence column is our interpretation of how to demonstrate the published criteria. General lablab criteria also cover presentation, business value, technology use, and originality; no formula combining them with Intel's 100 points is published.

## Speechmatics Bonus Award

**Best Use of Speechmatics** is an additional award available to online and onsite projects. It needs no separate signup or track assignment and can stack with the primary track award. The emphasis is meaningful speech-driven behavior: transcription feeding agents, automation, or multilingual/accent-diverse interactions. Both real-time and batch use cases are listed; a responsive spoken-command demonstration is our proposed approach.

| Place | Cash | Speechmatics API credits |
| --- | ---: | ---: |
| 1st | $500 | 1,000 per team |
| 2nd | $250 | 500 per team |
| 3rd | None | 250 |

The event labels these as API credits without stating a monetary conversion, expiration, or a detailed bonus scoring rubric. The award credits are prizes; they are not a confirmed allocation of development credits. Event-specific credit access should be checked with Speechmatics on its [Discord](https://discord.gg/speechmatics). Its [documentation](https://docs.speechmatics.com/) covers streaming and batch transcription, authentication, and SDKs.

Sources: [event bonus/prizes](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon), corroborated by the [organizer announcement reposted by Speechmatics](https://ca.linkedin.com/company/speechmatics).

## Award overview and other tracks

| Track | Mode | 1st | 2nd | 3rd |
| --- | --- | --- | --- | --- |
| Intel bimanual VLA | **Online** | $3,000 | $2,000 | $1,000 |
| Intel defect-detection robotics | Onsite | $5,000 | $3,000 | $2,000 |
| SiMa.ai Physical AI | Onsite | DevKit + webcam + robot-arm kit, listed $2,000 | DevKit + webcam, listed $1,600 | DevKit, listed $1,500 |
| Qualcomm Model-to-Device | Onsite | AI PC + $300 per participant | Uno Q + $300 per participant | $300 per participant |

Qualcomm lists $1,300 / $380 / $300 value per participant, or $6,500 / $1,900 / $1,500 for full five-person teams. Do not multiply the Intel or Speechmatics award figures by team size.

The advertised overall pool is **$31,650**, comprising **$21,250 cash** and **$10,400 hardware/devices**, plus **1,750 Speechmatics credits**. This includes onsite awards. There is a $100 inconsistency: SiMa.ai's three displayed bundle values sum to $5,100, while its headline says $5,000. Our chosen online prize amounts are unaffected by that arithmetic discrepancy. Winning both first places would mean a listed $3,500 gross cash plus 1,000 credits, subject to award and payout rules.

The onsite tracks provide context, not extra online categories:

- **Intel:** a physical SO-101 arm detects defects and responds using Physical AI Studio, Anomalib, OpenVINO, and Core Ultra Series 3. The exact task/objects are revealed on Challenge Day. Scoring: integrated solution 25, Anomalib 20, VLA/Studio 20, optimization 20, robot reliability 10, innovation/demo 5. [Original brief](https://drive.google.com/file/d/1qhfJETwHLxMyyQdhtf1BXrd72d5e0RgH/view).
- **SiMa.ai:** an application consumes real-world input and produces useful output using Modalix and Palette Neat. At least one meaningful AI workload runs locally. Teams must extend/combine examples. Starter apps, external libraries, and AI coding tools are explicitly allowed in this onsite brief. DevKits and mentors are available onsite. [Original brief](https://drive.google.com/file/d/1KBC3rrfKreEmH47-LAAUieLZu1iH3M4w/view).
- **Qualcomm:** Snapdragon X Elite and Arduino UNO Q hardware, with GenieX deploying models from Hugging Face or Qualcomm AI Hub. The event says the fuller brief is still being finalized; no separate full PDF was linked at inspection.

Onsite teams submit an idea and preferences through the emailed confirmation/application form; organizers assign one track. Travel and accommodation are not covered. The form link/capacity remain marked TBD in the page. These allocation steps do not apply to an online-only team.

Source for awards, Qualcomm, and allocation: [event](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon).

## Current timetable (rechecked September 10)

All local times below are **Europe/Berlin, CEST (UTC+2)**. See the [September 10 schedule evidence](sources/2026-09-10/schedule-observed.json); the [September 8 observation](sources/schedule-observed.json) is retained as history.

| Detailed event schedule | Berlin date/time |
| --- | --- |
| Kickoff | Sep 10, 17:00 |
| Opening remarks | Sep 10, 17:05 |
| Challenge introduction | Sep 10, 17:10 |
| Hackathon guide | Sep 10, 17:15 |
| Discord Q&A | Sep 10, 18:00 |
| Onsite doors / welcome | Sep 15, 18:00 / 19:30 |
| Onsite doors close | Sep 16, 02:00 |
| Onsite doors open | Sep 16, 18:00 |
| Submissions end | **Sep 16, 20:30** |
| Onsite doors close | Sep 17, 02:00 |
| Onsite winners ceremony | Sep 17, 22:30 |
| Onsite doors close | Sep 18, 01:00 |

The main page and [dashboard](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon/live) now agree on kickoff **September 10, 15:00 UTC / 17:00 CEST** and submission close **September 16, 18:30 UTC / 20:30 CEST**. The fully loaded dashboard shows Live / Submissions open. The previous conflicting timestamps are superseded. A generic onsite allocation notice still contains TBD wording; it does not replace the explicit deadline.

**Planning recommendation:** target submission by **Sep 16, 18:00 CEST**, 2.5 hours before the currently published cutoff. Verify enrollment status and follow official announcements. An online/bonus winner-announcement time was not separately verified; the listed ceremony is onsite.

Listed under speakers/mentors/judges: Andrea Marazzi, Founder & CCO, and Pawel Czech, CEO, both NativelyAI. Specific track judge assignments and a full online mentoring timetable were not found. Use [Twitch](https://www.twitch.tv/lablabai) and [lablab Discord](https://discord.gg/lablabai) for kickoff, recordings, help, and updates.

## Rules, ownership, and payout

- Submit original, open-source, MIT-compliant work unless the event specifies an exception. Keep third-party attribution and license notices; do not assume all models/assets can be relicensed.
- You retain ownership of your content, while granting the platform broad, continuing rights to use and distribute submitted content.
- Awards depend on eligibility and sponsors; organizers reserve the ability to change/cancel terms and awards.
- Cash is paid in USD to individuals. Winners must provide the payout form, tax documentation, photo ID, and bank verification within 90 calendar days of notification or forfeit the prize. Non-US recipients are directed to W-8BEN and international wire payment; the policy describes withholding and fees. Confirm personal payout details with the organizer rather than treating advertised cash as a guaranteed net amount.
- Distribution may take up to 90 days after winners are announced. The overview FAQ's older team/split wording differs from the specific individual payout policy.

Source: [terms §§4, 16–17](https://lablab.ai/terms-of-use). These are summaries of the organizer's policy, not an independent interpretation of tax law.

Plagiarism, vote manipulation, cheating, and other conduct undermining fairness can cause disqualification. A general manual-submission exception lasts up to six hours after the hackathon only for valid reasons with prior organizer/mentor approval; it is not an automatic extension. Organizers cannot win prizes, and participating mentors/organizers cannot judge. Source: [rule book](https://lablab.ai/hackathon-rules).

The [Code of Conduct](https://lablab.ai/code-of-conduct) requires respectful behavior, protection of confidential information, attribution, and disclosure of reliance on third-party AI tools. We should disclose coding-assistant and model use in the project documentation. A reference to unauthorized automation in the rule book is not a blanket ban on AI-assisted coding.

For submission formats, required fields, and conflicting generic instructions, use the [checklist](SUBMISSION_CHECKLIST.md) and [questions register](OPEN_QUESTIONS.md).
