# Open questions and conflicting information

Baseline checked September 8; schedule/resources rechecked **September 10**. No inquiry has been sent by the assistant. These are questions for organizers and sponsors, not missing facts to invent. The earlier kickoff/deadline conflict is now resolved in the public pages; see [update](UPDATE_2026-09-10.md).

## Resolve first

**September 11 scope decision:** the user chose to follow the dinner-table scenario. We no longer need chemistry-substitution approval to proceed. Starter assets, minimum required manipulations, and pouring approximation remain useful clarification questions.

| Priority | Issue | Evidence | Working decision |
| --- | --- | --- | --- |
| Critical | Final hardware access | Intel online PDF page 3 explicitly requires Core Ultra Series 2/3 for final simulation and inference. This PC has an i7-10850H. Broader wording elsewhere on the same page calls Core Ultra preferred. | Arrange qualifying access or obtain a written exception. Ask about loan/remote availability and proof of execution. |
| High | Pre-event code/training | General guide permits reuse/scaffolding in most events but reserves core AI development for the event window; the event has no complete specific policy. | Prepare now; clarify project code, model fine-tuning, datasets, starter templates, and disclosure requirements. |
| High | Starter environment / accepted task scope | Brief names two SO-101 arms, MuJoCo, and table setting; no organizer-provided starter repository or dataset link is publicly verified. Title calls the scenario an option, rubric names table setting. | Request scene/assets, baselines, data, allowed task variations, minimum required sequence, and accepted pouring approximation. |
| High | Demo format | Intel asks for ten randomized seeds; lablab presentation video is capped at five minutes. Platform asks for an interactive URL while Intel requires local execution. | Ask about a compressed main video plus full supplementary runs, and the accepted remote interface/reproduction arrangement. |

Sources: [main event](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon), [live dashboard](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon/live), [Intel online PDF](https://drive.google.com/file/d/1xSisqTQUAFQiLOpjLZrCVTCsQi4bMCpO/view), [general reuse guide](https://lablab.ai/guide/ai-hackathons), [submission guide](https://lablab.ai/delivering-your-hackathon-solution).

## Other clarification points

The [September 8 robotics research](../robotics/RESEARCH.md) found official reusable SO-101 models and community simulation tools. Their existence does not establish an organizer-supplied environment. The current event's online track still links to the same Intel PDF, rechecked against the saved original.

- **Speechmatics:** build-credit allocation, claim process, quotas/expiry, exact meaning of prize credits, bonus rubric, and any proof or tag requirements beyond actual integration. Confirm whether cloud speech transcription can front the locally running Intel robotics pipeline.
- **Registration:** confirm actual approval/enrollment. The dashboard has entered the live phase; a visible Join/Sign-up button alone does not verify account eligibility.
- **Six versus five people:** the generic getting-started guide permits six; this event specifies 1–5 twice. Use **five maximum**.
- **IBM Bob report:** the shared submission guide unexpectedly requires an IBM Bob task report; the event and Intel brief do not. Ask whether this is an unrelated template requirement. Do not infer mandatory IBM tooling for this event.
- **Hosting:** the rule book says Streamlit/Replit/Vercel, while the submission guide describes those as choices. Confirm another host or remote simulation interface if needed.
- **Judging/results:** weighting between the Intel rubric and general lablab criteria, bonus scoring, online/bonus winner date, whether remote finalists must attend any live session, and demo availability duration are not stated clearly.
- **Payout:** confirm how a team award is allocated among individuals. The FAQ's team/split description differs from terms §17, which specifies individuals only.
- **Onsite-only inconsistencies:** the summit site focuses on Sep 15–16 while lablab describes Sep 15–17; the SiMa.ai bundle figures sum to $5,100 against a $5,000 heading. These do not change our online project selection.

The live dashboard now names the tracks. Its generic category labels do not override the event's online/onsite distinctions. The newly linked Intel bundle is software setup material; online hardware access and a dedicated challenge scene remain unconfirmed.

Sources: [getting started](https://lablab.ai/getting-started-guide), [submission guide](https://lablab.ai/delivering-your-hackathon-solution), [rule book](https://lablab.ai/hackathon-rules), [terms](https://lablab.ai/terms-of-use), [FAQ](https://lablab.ai/guide), [summit](https://www.ai-infra-summit.com/ai-infra-hackathon), and the event sources above.

## Organizer inquiry draft

Suggested recipient: the hackathon's [lablab Discord](https://discord.gg/lablabai) support/mentor channel, or coordination@lablab.ai. This is a draft for the user to send; it has not been posted or emailed.

> Hello! We are preparing an online Intel Bimanual VLA entry for the AI Infra Summit Hackathon and intend to integrate Speechmatics for the Bonus Award. Could you please clarify:
>
> 1. The updated public schedule now gives September 10 at 15:00 UTC for kickoff and September 16 at 18:30 UTC for submissions. Where can we find the kickoff recording and track-specific Q&A announcements?
> 2. Before kickoff, may we write project code, fine-tune models, collect data, or prepare a scaffold? What pre-existing work and AI-assistance disclosures are required?
> 3. Must both simulation and robotics inference run on Core Ultra Series 2/3 for the final demo? Is a remotely accessed machine acceptable, and is loaner/remote access available? Does the broader Intel CPU/iGPU wording allow any other hardware?
> 4. Will you provide a dual SO-101 MuJoCo scene, baseline policy, dataset, or evaluation configuration? Is table setting mandatory, and which manipulations are the minimum expected?
> 5. How should ten seeded runs fit the five-minute video limit, and what interactive application URL is acceptable for a locally running simulation? Are supplementary recordings allowed?
> 6. Can Speechmatics cloud transcription feed the local Intel pipeline? How do we claim development credits, and is any bonus-specific evidence required?
> 7. Does the IBM Bob report instruction on the shared submission guide apply to this event?
>
> Thank you. Written clarification would help us keep the entry compliant and plan the build window correctly.
