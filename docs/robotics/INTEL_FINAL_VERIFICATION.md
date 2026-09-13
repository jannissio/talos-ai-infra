# Final Intel verification

Prepared September 13 for the user's Intel laptop on September 14. The selected v6 suite and its complete physical evidence are packaged in `models/dinner_suite/suite.json` and privately pushed on `codex/final-submission` (baseline revision `c2e6d3d`). Preserve that baseline while RGB-feedback candidates remain experimental.

The user confirmed enrollment and access to the laptop, but cannot access an Intel cloud account. Historical laptop identification is **Core i7-10850H**, Intel UHD and Quadro T2000. That is not Core Ultra Series 2/3. The written Intel brief and September 11 spoken clarification differ; see [the cloud/access record](INTEL_CLOUD.md). No script can turn an older processor into a Core Ultra result or establish organizer approval.

## Procedure on the laptop

Use the final `codex/final-submission` revision and the existing clean Python 3.12 `.venv-training` environment. Install the pinned `requirements-training.txt` and `requirements-openvino.txt` only if missing; preserve the 10 GiB reserve and check available disk first. CUDA is unnecessary for this inference verification. Keep `.env` local; this procedure does not require a Speechmatics key or microphone.

Run the hardware inspection first, choosing a fresh evidence folder:

```powershell
.\.venv-training\Scripts\python.exe scripts/inspect_intel_target.py --inspect-only --output .run/intel-final-hardware.json
```

Check the actual OpenGL renderer. If it says NVIDIA, it does not establish Intel graphics execution. Select the Intel graphics adapter for this Python runtime through the laptop's supported graphics settings, restart the process, and record a new inspection. Preserve both results. Device availability in OpenVINO and the OpenGL renderer are separate facts.

The packaged baseline is ready for this run from the private development branch:

```powershell
.\.venv-training\Scripts\python.exe scripts/verify_intel_submission.py --suite models/dinner_suite/suite.json --output .run/intel-final-cpu --device CPU
```

The script copies/export models into the new report directory, records real device names, numerical parity, FP32 inference precision, median/p95 latency and synchronous chunk throughput, then executes all six physical skills on exposed seed 42 and captures actual states. It does not overwrite the release models. A separate `--device GPU` run can test the Intel iGPU if available and parity passes; use a new output directory.

Exit code 1 on the older laptop may mean the **strict Core Ultra check fails even when physics passes**. Read `verification.json`: physical success and strict hardware success are separate flags. Retain the failure flag and CPU identity. The `--diagnostic` flag only changes the process exit condition; it never marks nonqualifying hardware as qualifying.

Network benchmark throughput excludes rendering, vision preprocessing, motor-feedback logic and physics. It is not full-robot task throughput. Full physical wall and simulation times are reported separately.

## Verification performed on the RTX PC

The portable script passed an original-bottle smoke test here: export parity passed, physical release and arm parking passed, and `strict_hardware_and_physics_passed` correctly remained false for AMD/NVIDIA. Local reports are retained under `.run/final-goal/intel-verifier-pc-smoke-v1`. This is a tooling check, not final Intel evidence.

The actual laptop run, hardware interpretation and submission confirmation remain open until their evidence exists. The source and media draft are prepared; that does not replace device execution.
