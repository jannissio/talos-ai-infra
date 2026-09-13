# Public demonstration access

Checked September 13, 2026. The [submission guide](https://lablab.ai/ai-articles/hackathon-guidelines) asks for a live application URL. A video page alone must not be described as a hosted simulator.

**User's release timing:** keep the repository private until immediately before final submission. An early visibility change was reversed at the user's request. The repository is private again (anonymous HTTP 404), the Pages site was disabled and its workflow was disabled. Development pushes remain authorized. Do not publish a duplicate source repository early as a hosting workaround.

## GitHub page

The prepared `site/` page contains recordings, measured scope, outcome links, the published presentation and local launch instructions. It labels all recordings explicitly. `scripts/build_public_site.py` exports only an explicit allowlist of public assets, checks the 10 GiB reserve and refuses to replace existing builds.

The GitHub Actions workflow publishes this build through Pages. [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages) hosts HTML/CSS/JavaScript; it does not run Talos's Python/MuJoCo server.

The authenticated repository check found that the existing repository was **private**, despite earlier handoff descriptions of it as public. Anonymous access returned 404. Two remote branches, no tags, no issues, no pull requests and no releases were present. All nine reachable commit identities use `Talos contributors <contributors@talos.invalid>`. Public access remains open until visibility, deployment and anonymous access have actually been verified.

Preview locally in a fresh export directory:

```powershell
.\.venv\Scripts\python scripts/build_public_site.py --output .run/public-site-preview
.\.venv\Scripts\python -m http.server 8767 --bind 127.0.0.1 --directory .run/public-site-preview
```

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

The user supplied the upload token privately in the ignored `.env`. The Space [jannis-sms/talos-dinner-robotics](https://huggingface.co/spaces/jannis-sms/talos-dinner-robotics) is **PRIVATE** and running. Current deployment revision: `f6e59689a6ca75684f45b8cf8aa023e373468ef5`, 153 files, about 34 MB. All source commits use `Talos contributors <contributors@talos.invalid>`; no `.env`, bytecode, private recordings or personal Git history was uploaded.

An initial CPU-only application was rejected by the ZeroGPU runtime because it registered no GPU function. The app now offers real CUDA trajectory generation: each skill observes its initial image, requests fresh network predictions on ZeroGPU, then the CPU worker executes them with the existing joint-feedback guard. It does not retrieve demonstration actions. The **default is now OpenVINO CPU inference**, with shared GPU as an explicit option, because the small models do not need a queued GPU. Physics and OSMesa rendering remain on CPU. Local CPU and RTX 4070 worker regressions both complete all six skills.

The first cloud learned bottle trial passes: 38.355 simulated seconds, 43.67 wall seconds, physical release/parking and no hidden state writes or forces. Cloud cancellation is confirmed. A new full sequence completes bottle, plate and mug but fails when requesting GPU inference at the drawer step. Subsequent diagnostics identify a service allocation failure: no GPU became available within 60 seconds. A later short GPU bottle trial passes. The current revision reports inference device and distinguishes service failures. Full CPU cloud-sequence verification is underway; it is not yet described as passed.

The free tier has account/anonymous GPU quotas and a shared queue. No purchase or paid upgrade is authorized. Keep both source repositories private until the final submission release, then verify anonymous access and actual judge usability.

[Render's free compute](https://render.com/docs/compute-plans) provides only 0.1 CPU and 512 MB memory. [Its lifecycle limits](https://render.com/docs/free) include idle shutdown. These resources have not been shown sufficient for Talos's renderer and inference stack; do not promise a suitable live backend on that basis.

The user's Speechmatics key stays local. It must not be included in site assets, a container build context or public source. A public voice service would need a separately authorized configuration and usage controls.

## Completion checks

- Verify the GitHub repository can be read anonymously after the privacy/identity audit.
- Verify the actual Pages URL and every local asset after deployment.
- Label the evidence page separately from any live simulator.
- Verify a fresh command/seed and cancellation in the chosen hosted backend.
- Reconcile the resulting URL with the authenticated submission form before the user's upload.
