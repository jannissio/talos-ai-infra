# Submission verification · September 12, 2026

The selected publication checkout passed **40 application tests** and **9 training-contract tests**. These cover physical lift/transfer, the complete programmed dinner sequence, the table-supported relay, loss of grasp, bounded retry, collision checks, cancellation, camera/controller timing separation, privileged-observation filtering and disk-reserve refusals. The application tests took 93.4 seconds on the local machine; training-contract tests took 2.5 seconds. Existing installed environments were reused: this was a clean source checkout, not a fresh dependency installation.

The packaged upright OpenVINO policy also completed a physical seed-42 rollout from that clean checkout: 6.87 cm peak lift, 1.12 mm final error, verified finger support and stable release. The original sideways model passed its familiar sideways-01 case with the current collision monitor. These are regressions on exposed examples, not new holdouts.

The ten-seed [upright evaluation](../../submission/evidence/ten-seed-upright-visual.json) remains the declared frozen test. A subsequent [collision-monitor regression](../../submission/evidence/upright-visual-collision-regression.json) passed the same ten exposed seeds. Historical initialized collision counters were removed from public learned-run reports because they were not measurements.

The final local HTTP check rejected three unsupported learned commands before movement and rejected cross-origin controls. A supported bottle command then completed with the current runtime. See [live API evidence](../../submission/evidence/final-live-check.json). Request/rendering load affects wall time; network-only OpenVINO benchmark numbers must not be presented as complete application latency.

Windows `start-lab.ps1 -Learned -NoBrowser` was checked for startup and reuse, and `stop-lab.ps1` stopped its verified process. The browser showed the released bottle, success, all six camera choices and the corrected controller-specific instructions. Human microphone capture was not tested unattended; recorded Speechmatics tests used labeled synthetic English speech.

The main MP4 fully decoded: 1280×720, 20 fps, 231.8 seconds, 16.52 MB, with audio and labeled playback speeds. The separate ten-seed MP4 fully decoded: 1280×720, 10 fps, 31.3 seconds. The eight-slide PPTX passed artifact finalization; slide/PDF renders and changed video scenes were visually inspected. The PDF contains raster slide pages; the PPTX is editable.

Selected files, structured PDF/PPTX metadata, notebook outputs and sanitized commit history were audited for credentials and personal local paths. The actual local Speechmatics key was absent. Private datasets, logs, the supplied transcript and `.env` remain outside the published tree. Third-party SO-101 attribution is retained. Filesystem checks preserve a 10 GiB reserve before data collection and exports.

Run the automated checks from the repository root:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv-training\Scripts\python -m unittest discover -s training_tests -v
.\.venv-training\Scripts\python scripts/verify_live_policy.py --output .run/local-verification.json
```

This is a bounded prototype: full reachable-area coverage, learned manipulation of every dish, continuous visual correction, direct airborne handoff and Core Ultra hardware verification remain uncompleted. The interactive application runs locally. The user still needs to rehearse with their microphone and upload the supplied materials to the hackathon submission page.
