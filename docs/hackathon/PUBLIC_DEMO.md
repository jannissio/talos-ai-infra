# Public demonstration access

Checked September 13, 2026. The [submission guide](https://lablab.ai/ai-articles/hackathon-guidelines) asks for a live application URL. A video page alone must not be described as a hosted simulator.

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

The separate [Gradio ZeroGPU free tier](https://huggingface.co/docs/hub/spaces-zerogpu) allows up to two Spaces for personal accounts older than 30 days with verified email. It has queue and usage limits and supports the Gradio SDK. A compact interface for running fresh, isolated simulation trials is being considered; compatibility and the user's account eligibility are not yet verified. It would need explicit scope labels if it offers fewer controls than the complete local application.

[Render's free compute](https://render.com/docs/compute-plans) provides only 0.1 CPU and 512 MB memory. [Its lifecycle limits](https://render.com/docs/free) include idle shutdown. These resources have not been shown sufficient for Talos's renderer and inference stack; do not promise a suitable live backend on that basis.

The user's Speechmatics key stays local. It must not be included in site assets, a container build context or public source. A public voice service would need a separately authorized configuration and usage controls.

## Completion checks

- Verify the GitHub repository can be read anonymously after the privacy/identity audit.
- Verify the actual Pages URL and every local asset after deployment.
- Label the evidence page separately from any live simulator.
- Verify a fresh command/seed and cancellation in the chosen hosted backend.
- Reconcile the resulting URL with the authenticated submission form before the user's upload.
