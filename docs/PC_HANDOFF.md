# Continue Talos on another Windows PC

**Active goal (September 13):** finish the broader hackathon system and submission by September 14. Follow [the final-submission checklist](hackathon/FINAL_SUBMISSION_CHECKLIST.md), which supersedes the narrow release scope below. User has confirmed a successful microphone command on this PC; credentials remain local.

**September 13 evening:** the packaged [v6 architecture](robotics/FINAL_ARCHITECTURE.md) now has six learned skills (8/10 frozen full sequences) and both learned bottle relay directions (5/5 each). Five dinner models, four relay models, exact compact inputs and every outcome are packaged and privately pushed on `codex/final-submission`. Updated media and all three submission-form steps are saved as a draft, verified after reload; **Submit has not been pressed**. The private Hugging Face Space now passes a fresh complete six-skill sequence on OpenVINO CPU: 247.655 simulated / 273.03 wall seconds on exposed seed 42. All 41 original protected files remain unchanged; current checks pass 56 application and 13 training-contract tests. Follow [active RGB-feedback work](robotics/RGB_SERVO_PROGRESS.md), not just the submission baseline. Final Intel execution, human six-skill voice rehearsal, final public release and actual submission remain open.

**Later September 13:** the separate RGB correction experiment completes all 48 frozen trials: 2/12 nominal live, 1/12 nominal frozen, 3/12 pushed live, 0/12 pushed frozen. Seven seeds are refused before movement. The observer passes its fresh synthetic gate, but physical coverage remains poor; the candidate is **not promoted**. [All outcomes and frozen sources](robotics/evidence/rgb-servo-v1/README.md), models and compact training reproduction inputs are packaged, with a labeled comparison video. Latest checks pass **61 application and 13 training tests**. Do not tune on those final seeds. The user has reiterated: continue preparation, but **never click the final submission button**; they will review and submit personally.

September 13 PC follow-through: see [RTX 4070 baseline verification](robotics/PC_BASELINE.md) for fresh installations, physical regression results and a successful human-triggered voice trial. The [coverage experiment](robotics/RTX4070_COVERAGE_EXPERIMENT.md) stopped at its declared data gate and retained the published model. Follow the [September 14 release plan](hackathon/RELEASE_PLAN_2026-09-14.md): the user requires completion tomorrow and gives September 15 as the deadline. Live timing, final Intel execution and judge-access details remain open. The original handoff below is retained as historical context.

Clone https://github.com/jannissio/talos-ai-infra and open the checkout as a Codex project. Current final-submission development is on `codex/final-submission`; use that branch until the reviewed final release is merged. The old laptop's absolute path is not required.

## Read first

- `README.md`
- `docs/robotics/SUBMISSION_VERIFICATION.md`
- `docs/robotics/VISUAL_BOTTLE_POLICY.md`
- `models/bottle_visual/README.md`
- `training/bottle_visual/README.md`
- `docs/robotics/TABLE_RELAY.md`
- `docs/robotics/STORAGE_POLICY.md`
- `docs/hackathon/LIVESTREAM_REVIEW_2026-09-11.md`
- `submission/PROJECT.md`

## What transfers through Git

The repository includes simulation code and robot/dinner assets, both deployed bottle models, the exact compact 22-episode upright training input, training/evaluation scripts, published metrics, the presentation, cover and two HD videos. This is enough to run the packaged demonstration and retrain the compact upright model.

Excluded on purpose: `.env`, virtual environments, `.run` with raw images/recordings and experimental checkpoints, private source downloads/transcript and machine diagnostics. The complete experiment archive remains on the laptop. Do not assume those private paths exist on the PC. New demonstrations can be generated with the collectors; copying the old raw archive is optional for further analysis and should be selective after checking disk space. Never upload the raw archive or credentials indiscriminately.

Create a new local `.env` on the PC if voice is needed: `SPEECHMATICS_API_KEY=your-key`. Matching quotes and spaces around `=` are supported. Obtain the key privately from the account or transfer the local file securely; do not paste it into chat or commit it. No account credit purchase is authorized.

## First goal on the PC

Establish a working baseline on the RTX 4070 (12 GB VRAM) before changing models:

1. Inspect installed clean CPython 3.12, GPU/driver information and actual free disk space. Create new `.venv` and `.venv-training` environments using the pinned requirements in the README. Do not copy Windows environments from the laptop or use Anaconda. Verify CUDA availability in the training environment; avoid an accidental CPU-only Torch installation.
2. Run the documented application and training-contract tests. Physically evaluate `models/bottle_visual` on seed 42 before retraining. These are exposed regression cases, not new holdouts. Preserve the packaged models unchanged.
3. Start `start-lab.ps1 -Learned`, verify the browser cameras and controls, and let the user rehearse actual microphone input. Speechmatics synthetic-audio API tests passed on the laptop; unattended human microphone capture was not tested.
4. Report any environment differences. OpenVINO CPU/iGPU measurements are specific to the laptop; do not reuse them as measurements of this PC. Core Ultra hardware eligibility still needs confirmation against the written brief and livestream clarification.

Then propose a bounded broader-position learning goal: retain the current baseline, inspect the failed wider-position cases/protocols, collect a compact curriculum rather than a large image archive, train on the RTX 4070, and declare fresh evaluation seeds before checkpoint selection. More GPU capacity alone does not solve grasp precision or dataset coverage. Consider continuous visual feedback as a subsequent architectural improvement, not an already implemented feature.

## Current measured scope

- Programmed exact-state physical skills place bottle, plate and mug, open the passive drawer, and retrieve fork/spoon. The six-skill sequence passed on seed 42.
- The programmed left-to-right bottle relay releases onto a shared table area before regrasping with the other arm. It is not an airborne handoff; reverse direction and other objects are not supported by this relay.
- The default upright model uses a **classical overhead RGB centroid detector**, a neural trajectory and motor-feedback progress checks. It passed 10/10 declared new task-preset scenes, with small bottle-position jitter, and the same ten exposed scenes with stricter collision monitoring. Wider-position development was 1/3. This is not full reachable-workspace coverage or continuous visual correction.
- The original three-view model remains for familiar sideways practice. Its earlier broader ten-scene test was 0/10; keep that failure evidence.
- Learned actions never use exact simulator object positions, a programmed teacher, action retrieval, attachments or teleportation. A separate privileged-state monitor can stop/score execution. Learned success verifies release before the working arm necessarily finishes parking; reset before another learned trial.
- Other dishes and arbitrary destinations are not learned skills. Unsupported learned instructions must be refused explicitly, without a silent programmed fallback.
- The laptop passed 40 application tests and 9 training-contract tests from a clean source checkout. Dependencies were reused there, so setup on this PC is a new installation check.

## Persistent instructions

**Public-release timing (September 13 correction):** keep the GitHub repository private until immediately before the final submission. The user explicitly reversed the early public visibility change. The repository is private again, anonymous access returns 404, and GitHub Pages plus its publishing workflow are disabled. Push sanitized development commits privately; do not make the repository or a duplicate source repository public early. Prepare and test hosting privately/local first.

Check the destination drive before **every** new training batch/episode or export and periodically during writing. Preserve at least **10 GiB free in addition to expected writes**. Never automatically delete existing data or checkpoints. Use compact state data and separate demo rendering where possible.

The user authorizes pushing needed, sanitized project changes to the existing private GitHub repository. Public visibility follows the timing rule above. Use a `codex/` branch for new work and a nonpersonal commit identity such as `Talos contributors <contributors@talos.invalid>`. Inspect staged content for credentials/private paths; do not push old laptop branches or backups containing the previous personal commit identity. All current remote history was sanitized.

The user will upload the hackathon submission. Materials are in `submission/`; no public hosted interactive simulator has been deployed. Keep claims aligned with measured results and distinguish programmed skills, classical vision and neural control.
