# Source register

Checked **2026-09-08**. Links are to organizer/vendor sources. The final documents are factual notes and our analysis, not replacements for the organizers' rules.

**Targeted September 10 recheck:** [update and findings](UPDATE_2026-09-10.md), [revised schedule](sources/2026-09-10/schedule-observed.json), [identical online-PDF check](sources/2026-09-10/brief-comparison.json), and [Intel installer archive metadata](sources/2026-09-10/intel-install-package.json). General platform terms/guides below retain their original research date.

Additional resources identified: [Intel Hack-a-thon Resources](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/hackathon_resources.html), its [installer bundle](https://amrdocs.intel.com/downloads/hackathon_install.zip), and [Speechmatics model comparison](https://docs.speechmatics.com/speech-to-text/models). The bundle was downloaded and inspected, not executed.

## Event and rules

| Source | Used for |
| --- | --- |
| [Event page](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon) | Active join/about/track/award/guideline sections, detailed schedule, speakers |
| [Live dashboard](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon/live) | September 10 live/submissions-open status, named tracks and corrected schedule; September 8 conflicting evidence retained separately |
| [AI Infra Summit organizer](https://www.ai-infra-summit.com/ai-infra-hackathon) | Co-organized format, sponsor context, application approval |
| [Intel event announcement](https://newsroom.intel.com/artificial-intelligence/intel-at-ai-infra-summit-2026) | Independent sponsor confirmation of Sep 10–16 virtual and Sep 15–16 onsite challenges |
| [Hackathon Rule Book](https://lablab.ai/hackathon-rules) | Mandatory submission components, fair play, conditional manual submission, conflicts of interest |
| [Submission Guidelines](https://lablab.ai/delivering-your-hackathon-solution) | Media, descriptions, demo URL, and the unexplained IBM Bob report wording |
| [Step-by-step guidelines](https://lablab.ai/ai-articles/hackathon-guidelines) | Enrollment/form workflow, title/description/video limits; dated February 25, 2026 |
| [Getting Started](https://lablab.ai/getting-started-guide) | Discord, team/mentor workflow; generic six-person limit conflicts with event-specific five |
| [Guide / FAQ](https://lablab.ai/guide) | Individual enrollment, solo teams, free entry, recordings, support |
| [General AI hackathon guide](https://lablab.ai/guide/ai-hackathons) | General distinction between reusable scaffolding and core work during the event |
| [Terms of Use](https://lablab.ai/terms-of-use) | Eligibility, ownership/licensing, awards and payout; page displays revision July 1, 2025 |
| [Code of Conduct](https://lablab.ai/code-of-conduct) | Attribution, AI-use disclosure, respectful behavior; page displays July 28, 2023 |

## Original sponsor briefs

These PDFs were downloaded from the links in the active event's Tracks section and retained unchanged. They are third-party reference material with their own rights and notices, not our original project code.

For repository sharing, downloaded PDFs/installer archives remain local and are excluded from Git. The links below open their official sources; JSON retrieval/hash records are included. The original local copies remain under `docs/hackathon/sources/` on the preparation PC.

| Source | Local original | Pages |
| --- | --- | ---: |
| [Intel online, Drive](https://drive.google.com/file/d/1xSisqTQUAFQiLOpjLZrCVTCsQi4bMCpO/view) | [intel-online-challenge.pdf](https://drive.google.com/file/d/1xSisqTQUAFQiLOpjLZrCVTCsQi4bMCpO/view) | 5 |
| [Intel onsite, Drive](https://drive.google.com/file/d/1qhfJETwHLxMyyQdhtf1BXrd72d5e0RgH/view) | [intel-onsite-challenge.pdf](https://drive.google.com/file/d/1qhfJETwHLxMyyQdhtf1BXrd72d5e0RgH/view) | 4 |
| [SiMa.ai onsite, Drive](https://drive.google.com/file/d/1KBC3rrfKreEmH47-LAAUieLZu1iH3M4w/view) | [sima-onsite-challenge.pdf](https://drive.google.com/file/d/1KBC3rrfKreEmH47-LAAUieLZu1iH3M4w/view) | 5 |

See [file manifest](sources/manifest.json) for retrieval timestamps, sizes, and SHA-256 hashes. Qualcomm's full challenge brief was still marked forthcoming; no separate linked file was found.

## Technical and community resources

- Intel: [robotics](https://www.intel.com/content/www/us/en/products/details/robotics.html), [OpenVINO](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html). Further software names are listed in the online PDF; they are not all mandatory components.
- Speechmatics: [docs](https://docs.speechmatics.com/), [developer portal](https://portal.speechmatics.com/), [academy/code](https://github.com/speechmatics/speechmatics-academy), [Discord](https://discord.gg/speechmatics), [lablab technology overview](https://lablab.ai/tech/speechmatics), [award announcement repost](https://ca.linkedin.com/company/speechmatics).
- SiMa.ai: [developer setup](https://developer.sima.ai/software/getting-started/), [examples](https://developer.sima.ai/examples/), [applications repository](https://github.com/sima-neat/apps), [developer center](https://developer.sima.ai/).
- Qualcomm: [AI Hub](https://aihub.qualcomm.com/), [GenieX docs](https://geniex.aihub.qualcomm.com/), [GenieX repository](https://github.com/qualcomm/GenieX), [Device Cloud](https://qdc.qualcomm.com/), [developer home](https://www.qualcomm.com/developer).
- Event support: [lablab Discord](https://discord.gg/lablabai), [Twitch kickoff](https://www.twitch.tv/lablabai), [Twitch schedule](https://www.twitch.tv/lablabai/schedule), [YouTube](https://www.youtube.com/@lablabai), coordination@lablab.ai, community@lablab.ai. Payout questions: prize@lablab.ai.
- Community partners listed: [The Hype News](https://thehype.news/) and [Founders Bay](https://www.foundersbay.com/).

Technology links supplied by the event are a reading queue, not a claim that their examples have been installed, tested, or confirmed compatible with this project.

## Verification and limitations

- The initial web text extraction omitted most dynamically rendered event content. The rendered browser page was inspected after loading, confirming the active sections and detailed timetable. Draft/duplicate HTML sections embedded in page data were not treated as current rules.
- All five online PDF pages were extracted and visually inspected, including the hardware wording and 100-point rubric. Onsite PDFs were read for contextual summaries and retained intact.
- The rule book, submission guide, getting-started guide, terms, conduct, and live dashboard needed rendered-browser inspection because initial text fetches returned little or no body content.
- [Schedule data](sources/schedule-observed.json) preserves the detailed schedule's source timestamps and UTC/Berlin/Santa Clara conversions, together with the conflicting live-dashboard values. No calendar event or reminder has been created.
- Local hardware was inspected read-only using Windows system information. No drivers, packages, accounts, or application code were changed as part of the research.
- Registration status of the user, private Discord messages, mentor bookings, API account credits, undisclosed starter assets, and future updates remain unverified. No enrollment, team creation, organizer message, paid service, or external submission was performed.
- Before kickoff and submission, recheck the event, live dashboard, briefs, and official announcements. Resolve disagreements using organizer clarification, not assumptions based on marketing text or other teams' project ideas.
