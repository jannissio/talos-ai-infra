# Bottle image refinement V1: stopped at fresh perception

The RTX 4070 fits a new 51,986-parameter local RGB refinement network after the frozen coarse bottle detector. The original coarse model is preserved. Training uses 4,096 wider and physical-context observation states, with 512 development states. The single 8,000-update fit takes 82.18 seconds and reserves 1.084 GiB of Torch GPU memory. Both checkpoints and every score are retained.

The selected step-8,000 CPU export passes development: **433/449 present**, **0/63 absent false accepts**, **1.322 mm p95 / 3.971 mm maximum**. CPU/export decisions match the training runtime on every development state.

The frozen fresh test **fails**: **433/456 present (94.956%)**, **0/56 absent false accepts**, **1.494 mm p95 / 4.427 mm maximum**. The unchanged limits require at least 95% acceptance, no absent false accepts, at most 2 mm p95 and 4 mm maximum. All 23 refusals and the largest error remain in the original result. This model is **not promoted**. The original six development and 24 final physical seeds remain unexposed under this protocol.

All 5,120 observation states pass the geometry/image-hash audits, with seven fixed states (21 views) independently re-rendered per split. A first read-only audit was stopped for repeated decompression; its source and logs are retained. Caching one shard at a time changes no verification assertion.

The separate exposed-grid physical diagnostic asks where this observer/controller actually works, using already exposed contexts that contributed training views. It does not convert this failed gate into a pass or establish fresh-scene generalization.

The model package contains both checkpoints, selected weights, the frozen coarse checkpoint and actual FP32 CPU exports. The training package retains exact fit/development crop inputs in small ordered shards, all scene/joint/light recipes, every label and original RGB hash. Full raw RGB shards remain locally preserved. Privileged geometry and masks are labels/scoring only; the observer input is three RGB images and fixed calibration.
