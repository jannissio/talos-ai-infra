# Held-mug camera visibility diagnostic

All 144 saved placement states have at least two usable RGB views in each of the three declared camera layouts. The diagnostic selects `focused_45_225` by its declared coverage-then-pixel-area rule. This establishes visibility for further observer work, not accurate pose estimation or continuous control.

| Camera layout | States with two usable views | Median second-largest component | Centroid triangulation p95 / maximum |
| --- | ---: | ---: | ---: |
| Original dinner views | 144/144 | 188 pixels | 8.753 / 9.622 mm |
| Focused overhead, 45° and 225° | 144/144 | 1,642 pixels | 10.836 / 11.205 mm |
| Focused overhead, 135° and 315° | 144/144 | 1,331 pixels | 12.491 / 12.800 mm |

The unchanged teal color rule finds visible surface pixels. Their centroid is biased by the mug's handle, different surfaces and gripper occlusion; triangulating those centroids does not recover the body midpoint precisely. A separately trained and evaluated pose observer is needed before motor correction.

The [protocol](protocol.json) was committed as `88b8d68431ce2f9f75d98ca7cbcfc87454e4aee2` before rendering. It includes all 24 baseline traces from the completed spoon and mug release experiments, six declared phases each, and all 1,296 view observations. These scenes are already exposed. No new physical trial or training occurs. Rendering takes 32.511 seconds.

Every RGB mosaic, calibration, predicted mask, scoring-only segmentation mask, result and source hash is retained. [The audit](audit.json) independently re-renders all 1,296 RGB views and segmentation masks, reconstructs every body midpoint and recomputes all gates and selection, with zero discrepancies. The core manifest contains 150 files / 46,804,619 bytes. Published input traces and assets are referenced by hash, not duplicated.

Reproduce in the training environment from the repository root:

```powershell
.venv-training/Scripts/python.exe scripts/probe_mug_visual_visibility.py
.venv-training/Scripts/python.exe scripts/package_mug_visual_visibility.py
```

Both commands preserve existing output and enforce the 10 GiB reserve and 256 MiB cumulative raw/package budget. Existing local runs must be retained; use a separate clean checkout to reproduce. `--verify-only` on the packaging command independently checks an existing raw run and its package.

The selected dinner models, private hosted demo and submission claims are unchanged. Further pose learning and physical correction require separately declared data, selection and evaluation. GitHub/Hugging Face remain private, and the user retains the final Submit action.
