# Public demonstration access

Checked September 13, 2026. The [submission guide](https://lablab.ai/ai-articles/hackathon-guidelines) asks for a live application URL. A video page alone must not be described as a hosted simulator.

**User's release timing:** keep the repository private until immediately before final submission. An early visibility change was reversed at the user's request. The repository is private again (anonymous HTTP 404), the Pages site was disabled and its workflow was disabled. Development pushes remain authorized. Do not publish a duplicate source repository early as a hosting workaround.

## GitHub page

The prepared `site/` page now contains the current v6 main demonstration, all-ten dinner and relay recordings, all-ten integrated workflows and the labeled RGB experiment. It links the three complete outcome summaries, current presentation, hosted demo and full-suite launch instructions. Scope text retains failed placements and pending Intel/voice/public-access checks. `scripts/build_public_site.py` exports only an explicit allowlist of assets, including MIT/Apache licenses and notices, checks the 10 GiB reserve and refuses to replace existing builds.

The GitHub Actions workflow is prepared for a manual Pages deployment after the final release; it also refuses a private repository. It remains disabled remotely. Development pushes cannot trigger deployment. [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages) hosts HTML/CSS/JavaScript; it does not run Talos's Python/MuJoCo server.

The authenticated repository check found that the existing repository was **private**, despite earlier handoff descriptions of it as public. Anonymous access returned 404. Two remote branches, no tags, no issues, no pull requests and no releases were present. All nine reachable commit identities use `Talos contributors <contributors@talos.invalid>`. Public access remains open until visibility, deployment and anonymous access have actually been verified.

Preview locally in a fresh export directory:

```powershell
.\.venv\Scripts\python scripts/build_public_site.py --output .run/public-site-preview
.\.venv\Scripts\python -m http.server 8767 --bind 127.0.0.1 --directory .run/public-site-preview
```

September 14 local verification: the fresh v6 export contains **16 allowlisted files / 27,946,778 bytes**, plus its manifest and Pages marker. Every file is byte-identical to its source and its unauthenticated localhost HTTP response; all local asset/anchor links resolve. The [machine-readable result](evidence/site-v6-local.json) is a local check, not proof of public availability. Browser checks at the existing 1280-pixel viewport load all five videos without media errors (189.1, 73.0, 76.3, 82.9 and 50.4 seconds), with the corresponding captions, selected controls and downloads. The page has no horizontal overflow. Current local review: `http://127.0.0.1:8769/`.

Recheck a fresh export with `scripts/verify_public_site.py --build <export-folder> --base-url <served-base-url> --output <new-report.json>`. The same command can verify delivered public static bytes after the authorized release; it sends no authentication. Public repository access, hosted physics and the actual submission remain separate checks. Refresh the preparation-only access paragraph when public access is actually verified.

## Interactive backend

[Current Hugging Face documentation](https://huggingface.co/docs/hub/spaces-overview) requires a paid personal PRO or organizational plan to create ordinary Docker/Gradio compute Spaces. CPU Basic has no hourly charge, but that does not make account-plan access free. No subscription or credit purchase is authorized.

The separate [Gradio ZeroGPU free tier](https://huggingface.co/docs/hub/spaces-zerogpu) allows up to two Spaces for personal accounts older than 30 days with verified email. It has queue and usage limits and supports the Gradio SDK. The signed-in account `jannis-sms` has a **verified email**; its creation form offers **ZeroGPU Free** and **Private**. Ordinary CPU Basic/Docker creation is disabled for the current account plan. No paid upgrade is authorized.

`hosting/gradio_app.py` now provides a compact interface for fresh physical trials: instruction, controller, seed, bottle region, a selected camera, progress, cancellation and outcomes. Local browser checks completed learned bottle placement, learned forward relay and all six programmed skills. Cancellation followed by a new learned relay passes after isolating each renderer in a spawned worker process. It does not expose the full browser lab or microphone capture.

`scripts/build_hf_space.py` builds a fresh upload folder from application dependencies, tracked robot assets, licensed model files and explicit hosting requirements. No `.env`, private recordings, local environments or Git history are copied. The initial bottle/relay package was about 27 MiB; the six-skill package is about 34 MB. Linux OSMesa rendering is now verified by a successful hosted bottle trial. The current hosting inference path is described below.

```powershell
./.venv-hosting/Scripts/python.exe hosting/gradio_app.py
./.venv-hosting/Scripts/python.exe scripts/build_hf_space.py --output .run/hf-space-private-build
```

## Private deployment verified September 13

The user supplied the upload token privately in the ignored `.env`. The Space [jannis-sms/talos-dinner-robotics](https://huggingface.co/spaces/jannis-sms/talos-dinner-robotics) is **PRIVATE** and running. Current deployment revision: `c08e96f493e76f38f06143fa621c3cf01da29b25` (September 14 hosting time allowance update), 153 files, about 34 MB. All source commits use `Talos contributors <contributors@talos.invalid>`; no `.env`, bytecode, private recordings or personal Git history was uploaded.

An initial CPU-only application was rejected by the ZeroGPU runtime because it registered no GPU function. The app now offers real CUDA trajectory generation: each skill observes its initial image, requests fresh network predictions on ZeroGPU, then the CPU worker executes them with the existing joint-feedback guard. It does not retrieve demonstration actions. The **default is now OpenVINO CPU inference**, with shared GPU as an explicit option, because the small models do not need a queued GPU. Physics and OSMesa rendering remain on CPU. Local CPU and RTX 4070 worker regressions both complete all six skills.

The first cloud learned bottle trial passes: 38.355 simulated seconds, 43.67 wall seconds, physical release/parking and no hidden state writes or forces. Cloud cancellation is confirmed. A full GPU sequence completes bottle, plate and mug but fails when requesting inference at the drawer step. Subsequent diagnostics identify a service allocation failure: no GPU became available within 60 seconds. A later short GPU bottle trial passes. The current revision reports inference device and distinguishes service failures. The default CPU path now completes all six learned tasks in a fresh cloud trial: **247.655 simulated / 273.03 wall seconds**, release and parked arms verified, no unexpected collisions, state writes or hidden forces. See the [selected-field browser evidence](../robotics/evidence/learned-dinner-v6/cloud-cpu-seed42.json). Seed 42 is an exposed deployment regression, not an extra unseen test.

The free tier has account/anonymous GPU quotas and a shared queue. No purchase or paid upgrade is authorized. Keep both source repositories private until the final submission release, then verify anonymous access and actual judge usability.

September 14 private deployment checks also pass both learned bottle-relay directions on exposed seed 42. The [forward relay](../robotics/evidence/learned-relays-v3/cloud-cpu-forward-seed42.json) completes in 74.02 simulated / 81.36 wall seconds; the [reverse relay](../robotics/evidence/learned-relays-v3/cloud-cpu-reverse-seed42.json) completes in 74.01 / 80.06 seconds. Every leg verifies release and parked arms, with no unexpected collisions, state writes or hidden forces. The farther-left scene uses the ordinary `Place the bottle` instruction and the RGB planner selects the reverse relay. These are signed-in cloud regressions, not new randomized holdouts, anonymous tests or Intel measurements. These relay checks used deployment `f6e59689a6ca75684f45b8cf8aa023e373468ef5`; all models remain unchanged.

[Render's free compute](https://render.com/docs/compute-plans) provides only 0.1 CPU and 512 MB memory. [Its lifecycle limits](https://render.com/docs/free) include idle shutdown. These resources have not been shown sufficient for Talos's renderer and inference stack; do not promise a suitable live backend on that basis.

The user's Speechmatics key stays local. It must not be included in site assets, a container build context or public source. A public voice service would need a separately authorized configuration and usage controls.

## Completion checks

- Verify the GitHub repository can be read anonymously after the privacy/identity audit.
- Verify the actual Pages URL and every local asset after deployment.
- Label the evidence page separately from any live simulator.
- Verify a fresh command/seed and cancellation in the chosen hosted backend.
- Reconcile the resulting URL with the authenticated submission form before the user's upload.

September 14 combined deployment check: the ordinary `Set the table` command with **Upright, farther left** completes both reverse bottle-relay legs and all five remaining dinner tasks on exposed seed 42. The [selected browser evidence](../robotics/evidence/composed-dinner-v1/cloud-cpu-seed42.json) records 283.31 simulated / 297.31 wall seconds, all seven released and parked, zero unexpected collisions and no hidden assistance. This passed the old 300-second limit with only 2.69 seconds of reported margin. The private deployment now gives each trial 420 seconds; its controller, physical thresholds and models are unchanged. Local cancellation stops the child process and a subsequent learned bottle trial succeeds. Anonymous access remains unverified.
