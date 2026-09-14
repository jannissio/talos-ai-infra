# Compact Intel verification handoff

The [verification ZIP](talos-intel-verification.zip) is **19,526,128 bytes (19.5 MB)** and contains 319 files, including the exact selected models, source, robot assets, verification commands and licenses. Its uncompressed contents total 35,395,057 bytes. It contains no credentials or Python environment.

SHA-256: `6a9b1228b5212efe815d07cbb5aca96e9966e354557afbd42583e986edf22394`

The kit is prepared from source revision `57d6ff6185da832d49ae378de393f4a3e7c8cc6a`. Its embedded manifest records every original file hash. The archive uses fixed timestamps and neutral file permissions. It remains in the private repository until final release.

On the Intel laptop:

1. Check free disk space before downloading and extracting. Allow **1 GiB of new writes plus the 10 GiB reserve**; dependency installation, if needed, needs its own additional preflight.
2. Download the ZIP from the private repository and extract it into a **new** folder. Preserve the original checkout and existing environments.
3. Open PowerShell in the extracted folder. Reuse the existing clean Python 3.12 training environment with the pinned training/OpenVINO dependencies:

```powershell
$talosPython = Read-Host "Full path to the existing .venv-training\Scripts\python.exe"
& .\verify-final-on-intel.ps1 -Python $talosPython -Output ".run/intel-final-laptop-cpu"
```

Keep that output folder unused. The wrapper checks environment/space, exports and benchmarks the selected networks on actual devices, then runs the complete six-skill sequence on exposed seed 42. Retain the entire `.run/intel-final-laptop-cpu` folder for review. No microphone or Speechmatics key is needed, and this procedure does not upload or submit anything.

The historical i7-10850H laptop is **not Core Ultra Series 2/3**. A successful physical run and a failed strict hardware flag can occur together. Preserve both results and the measured CPU, graphics renderer and inference-device names. The [full Intel procedure](../../docs/robotics/INTEL_FINAL_VERIFICATION.md) explains the remaining eligibility question.

The ZIP was extracted into an unused folder and run with isolated Python (`-I`) on the RTX PC, using only the extracted source/models plus the existing dependency environment. **All six learned skills and all six CPU export-parity checks pass**, in **52.662 physical wall seconds**. Both arms park; no teacher updates, hidden forces or control-time physics-state writes occur. The strict hardware flag correctly remains false on AMD/NVIDIA. This is a portability check, not Intel execution evidence or another fresh generalization test.

[The complete PC check](pc-portability/audit.json) preserves hardware/benchmark reports, all physical outcomes and the full state trace. The portable scene changes only its asset path; the original scene is also retained. The user's actual laptop run, human full-sequence voice rehearsal, final public access and final submission remain open. The assistant must never press final Submit.
