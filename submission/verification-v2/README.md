# Live-mug component verification on the Intel laptop

The [24.3 MB kit](talos-intel-live-mug-verification.zip) contains the unchanged dinner/relay models plus the newly promoted stereo mug observer, local neural correction model and bound settings. It complements the earlier kit and verifies the changed components on two exposed workflows. The already completed Intel baseline checks do not need repeating with the old kit.

SHA-256: `c21ceba704aa2e5f2879d98d5043fad0f9e5671baf5234632f9d546f1902400c`

Source revision: `57566ca390fd95d69713a684062d916915756f9e`. All 377 archive members, CRC and manifest hashes pass independent verification. An isolated extraction on the RTX PC completes standard/farther-left workflows in 36.023/40.643 wall seconds, with 15 live corrections each and 10,623 retained frames. [The complete portability check](pc-portability/audit.json) is preserved. These AMD/NVIDIA runs establish portability, not actual Intel or wider-position coverage. The harmless initial extraction-harness allowlist error is also retained.

On the laptop, check at least **1 GiB plus the 10 GiB reserve** before download/extraction. Extract into a new folder and reuse the existing Python 3.12 training environment. Keep the Intel graphics preference already verified on that laptop. From the extracted folder:

```powershell
$talosPython = Read-Host "Full path to the existing training Python executable"
& $talosPython -I scripts/verify_visual_mug_submission.py --output .run/intel-live-mug-v1
```

Retain the entire output folder, including both cases, raw reports, console logs and state recordings. No microphone, key, package installation or upload is needed. The ordinary exit code remains 1 on the i7-10850H even when physics passes because it is not Core Ultra Series 2/3. Read `physical_passed`, measured rendering/inference identities and `strict_hardware_and_physics_passed` separately. The option uses OpenVINO CPU; the earlier kit already records separate CPU/GPU.0 baseline runs.

Keep this archive private until final release. Never overwrite a preceding run. Only the user presses final Submit.
