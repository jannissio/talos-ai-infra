# RTX 4070 PC baseline — September 13, 2026

The packaged upright model passed physical seed-42 evaluation on this PC. This is an exposed regression case, not a fresh holdout or a retrained model. The original models, frozen training input and published submission/evidence were preserved; SHA-256 checks covered all 41 files in those folders.

## Installation and hardware

Fresh `.venv` and `.venv-training` environments were created with clean CPython **3.12.10**, using the repository requirements. The laptop documented 3.12.14; the installed PC patch version passed the checks below. Plain `python` resolved to an unrelated environment, so installation and execution used explicit interpreter paths.

This PC has an **AMD Ryzen 7 5800X**, approximately 32 GiB RAM, and an **RTX 4070 with 12 GB VRAM**, NVIDIA driver **591.86**. C: initially had **549.34 GiB free** and retained more than **535 GiB after installation**. Space was checked before new output; the 10 GiB reserve was maintained. No old datasets, checkpoints or evidence were deleted.

The training environment uses **PyTorch 2.8.0+cu128**, torchvision 0.23.0+cu128, NumPy 2.2.6, LeRobot 0.6.1, MuJoCo 3.12.0 and OpenVINO 2026.3.0. The application environment retains NumPy 2.5.3. Both environments passed `pip check`. The README now explicitly installs the official CUDA 12.8 wheels before the remaining pinned dependencies.

CUDA availability, device identity and actual forward/backward/optimizer execution were verified with three disposable motion-primitive optimizer steps. Parameters changed and gradients/losses were finite. No checkpoint was exported from this smoke test. OpenGL identified NVIDIA / RTX 4070. The packaged policy reported OpenVINO execution on **CPU**, which here means the AMD processor. This is not a new Intel CPU/iGPU, Core Ultra or NPU measurement, nor confirmation of hackathon hardware eligibility.

The supplied folder initially lacked `.git` metadata. Published `main` at `fcb5ba63bb51a31051203d25fa9322fcfbf7db8c` was fetched, the unchanged source matched it, and work proceeded on `codex/pc-baseline-rtx4070` with `Talos contributors <contributors@talos.invalid>` as the local commit identity.

## Verified behavior

| Check | Result on this PC |
| --- | --- |
| Application tests | **40/40**, 73.015 seconds |
| Training-contract tests | **9/9**, 1.111 seconds |
| Packaged upright seed 42 | **Pass**, 28.65 simulated seconds |
| Peak physical lift | **6.8734 cm** |
| Final placement error | **1.1155 mm** |
| Verified finger-supported hold | **9.66 seconds** |
| Stable release | **1.005 seconds** |
| Unexpected arm collisions / hidden forces / equality constraints | **0 / 0 / 0** |
| Learned controller writes to object integration state | **0** |
| Renderer after standalone completion | Closed |
| Live camera and control API | Six 960×540 views, JPEG stream, pause/step, resets, joint bounds and invalid-input checks passed |
| Dinner API | Seeded scenes, drawer inspection, target preset and scene switching passed |
| Unsupported learned instructions | Plate placement, full table setting and bottle relay refused with HTTP 400 before movement |
| Cross-origin control | HTTP 403; paused physics unchanged |

The live-camera checker iterates the returned camera map, which contains six cameras. Its old human-readable result string says “five”; the inspected map and loop covered all six.

## Browser startup correction and timing limit

`start-lab.ps1 -Learned` started the browser demo, but the first live learned request exceeded the engine's 15-second request deadline while the action nevertheless began. The run was explicitly cancelled at 16.39 simulated seconds; its diagnostic evidence was retained. Its reported median control-query time was approximately 258 ms. Standalone seed-42 execution had already passed.

The server now sets PyTorch's CPU thread limit before starting HTTP/physics worker threads, in addition to the existing setting on the physics thread. After restart, a live learned command was accepted in **0.622 seconds** and completed with the same physical success criteria. The browser button subsequently displayed **“Instruction accepted”** on a reset seed-42 trial, which also completed: 6.8734 cm peak lift, 1.1155 mm final error, stable release and zero unexpected collisions. The browser visibly showed success, then was reset for voice rehearsal.

This addresses the observed startup timeout, but is not a full performance investigation: sustained live query latency remained variable (the two completed runs after the change reported 90.56 and 105.08 ms medians), despite initially running near real time. Browser/rendering load and live-thread behavior still need a separate controlled profile. Do not reuse the laptop's network-only benchmarks or describe this PC as maintaining real-time learned control in every viewing configuration.

The current local address is `http://127.0.0.1:8765/`. Choose **Learned upright bottle · OpenVINO**, load **Task start / Closed / Seed 42**, and enter **place the bottle**. Success verifies bottle release before the working arm necessarily finishes parking; reset before another learned trial. Stop with `stop-lab.ps1`.

## Voice, evidence and next work

The user configured a local Speechmatics key, and Git confirmed `.env` is ignored. During the subsequent human rehearsal, the user reported pressing **Speak instruction** and seeing **“Place. The bottle”**. The live API independently verified the resulting learned task's success: 28.645 simulated seconds, 6.8734 cm peak lift, 1.1154 mm final error, one second of stable release and zero unexpected collisions. This is one user-reported human microphone trial with physically verified completion; the assistant did not independently record or listen to the microphone audio. No microphone audio was saved. The original published speech recordings remain labeled synthetic. Live timing was slow in this rehearsal (383.74 ms median control query); it is not a low-latency speech/control claim.

Installation logs, dependency freezes, hardware diagnostics, physical rollout reports and profiling probes are retained locally under `.run/pc-baseline-20260913/`; live scene checks are `.run/live-check.json` and `.run/dinner-live-check.json`. Autonomous outcomes, including the cancelled diagnostic trial, are retained in `.run/autonomy/`. These local diagnostics are excluded from Git.

The [next bounded coverage experiment](RTX4070_COVERAGE_EXPERIMENT.md) proposes at most 24 compact training attempts around the exposed tracking/support failures, one bounded CUDA optimization schedule, six development cases and 12 separately declared fresh evaluation cases. It has not been executed. Full placement coverage, continuous visual correction and learned manipulation of other dishes remain unfinished. The published 10/10 task-preset result and 1/3 wider-position development result retain their original scope.
