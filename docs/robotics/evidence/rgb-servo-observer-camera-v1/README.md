# Observer camera adaptation: training selection frozen

This separate [protocol](protocol.json) trains the existing observer on 1,024 new paired original/opposite-camera states, with 256 different paired development states. The generator/protocol were committed in `b06be89`; the strict trainer was committed in `a3a27fe` before fitting. The four datasets finish in 87.17 seconds and occupy about 270 MiB locally. Each camera pair uses identical sampled qpos and light states.

All **6,000 declared training updates** complete in **207.78 seconds**, using 2.703 GiB peak Torch-allocated / 2.912 GiB reserved GPU memory on the RTX 4070. The 3,000-update checkpoint is selected before fresh perception because it accepts more opposite-view development states than the 6,000-update checkpoint. Both pass their development gates; the predeclared ordering favors coverage before accepted p95 error.

| Checkpoint | Original accepted present | Opposite accepted present | Opposite error p95 | Opposite maximum | Absent false accepts, each layout |
| --- | --- | --- | --- | --- | --- |
| 3,000 updates, selected | 214/233 | 221/233 | 2.662 mm | 4.700 mm | 0/23 |
| 6,000 updates | 215/233 | 219/233 | 2.156 mm | 4.334 mm | 0/23 |

[Both checkpoints, all development observations and exact training sources](training-manifest.json) are frozen here. The selected SHA is `2d82fc20ee8bc7d209be57ca1436361c4865b3ee3fb1d49a176d2ca2df5f1109`. Data hashes are in [training.json](training.json); compact reproduction recipes and subsequent checks are separate additions. The original observer and V5 motor remain unchanged.

This records **training selection only**. OpenVINO export, CPU rescoring and fresh perception seed 2026098203 still require verification before a physical experiment. Reserved physical development/final scenes remain unexposed. No selected dinner controller or hosted model changes.
