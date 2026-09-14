# Stopped differential motor inputs

This package contains all 8,192 training and 1,024 development attempts for the [bounded motor protocol](protocol.json). **No model was fitted.** Fixed corrections of up to 2 mm fail the 99% input-coverage requirement before training.

Arrays retain sampled episode/continuous phase, joint jitter, five joint coordinates, requested XY translation, offline differential matrix, endpoint label and acceptance. Every rejected attempt remains present. The [evidence package](../../docs/robotics/evidence/mug-correction-motor-v1/README.md) holds every forward-geometry score, the independent audits and original closed console logs.

To reproduce, install the root README's training environment, restore the frozen files from `source/` in a separate checkout, and use a new `.run/mug-correction-motor-v1` directory:

```powershell
.\.venv-training\Scripts\python scripts/mug_correction_motor_v1.py --mode collect --split training
.\.venv-training\Scripts\python scripts/audit_mug_correction_motor.py --split training
.\.venv-training\Scripts\python scripts/mug_correction_motor_v1.py --mode collect --split development
.\.venv-training\Scripts\python scripts/audit_mug_correction_motor.py --split development
```

Both coverage flags are false. Stop before `--mode train`, export or fresh evaluation; the code explicitly requires passing coverage and independent geometry audits. The reserved 2026100103 evaluation remains unexposed. Check disk before every dataset/export and preserve the 10 GiB reserve; the total raw/package budget is 256 MiB. Original model weights, physical monitors and the selected robot remain unchanged.
