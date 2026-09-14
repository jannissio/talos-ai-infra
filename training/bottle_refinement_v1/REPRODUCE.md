# Reproduce the stopped bottle-refinement experiment

Use the documented training environment in a fresh checkout with an unused `.run/bottle-refinement-v1` directory. These commands regenerate the original training/development distribution and repeat the declared fit. They do not create a fresh evaluation claim.

```powershell
.\.venv-training\Scripts\python -I scripts/bottle_refinement_experiment.py --protocol docs/robotics/experiments/bottle-refinement-v1.json --action collect --split training
.\.venv-training\Scripts\python -I scripts/bottle_refinement_experiment.py --protocol docs/robotics/experiments/bottle-refinement-v1.json --action collect --split development
.\.venv-training\Scripts\python -I scripts/train_bottle_refinement.py --protocol docs/robotics/experiments/bottle-refinement-v1.json
```

The collectors check the actual disk reserve and refuse existing output directories. The original training GPU is an RTX 4070; record numerical differences on another runtime instead of assuming identical checkpoint bytes. Keep all reproduction output local until it has been audited.

The exact original crop arrays can also be recovered directly, independently of image regeneration. For each of `prepared-training` and `prepared-development`, read `manifest.json`, load the ordered `shards`, concatenate each named array along axis zero, and verify its declared dtype, shape and logical-array SHA-256. `world_points` and `present` have one row per scene; the other arrays have six rows per scene. Original supervision includes visibility and image coordinates; world geometry is only used to score perception.

The [original results](../../docs/robotics/evidence/bottle-refinement-v1/README.md) retain both checkpoints, every development/fresh prediction, CPU export parity and the failed fresh gate. Their images and seeds are now exposed. Leave the original reserved physical stages untouched; the later exposed-grid diagnostic is a separate experiment.

This file supplements the original immutable package manifest.
