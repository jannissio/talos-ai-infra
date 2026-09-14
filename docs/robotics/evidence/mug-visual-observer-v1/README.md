# Mug RGB pose observer v1 — stopped at development

The one declared 12,000-update GPU fit completes, but neither checkpoint passes the unchanged pose-accuracy gate. No checkpoint is selected, no OpenVINO export or fresh evaluation follows, and this observer never controls the arm. The selected dinner suite and hosted demo are unchanged.

| Checkpoint | Present accepted | Absent false accepts | 3D error p95 / maximum | XY error p95 |
| --- | ---: | ---: | ---: | ---: |
| 6,000 updates | 448/448 | 0/64 | 8.931 / 26.395 mm | 6.703 mm |
| 12,000 updates | 448/448 | 0/64 | 8.105 / 23.172 mm | 6.266 mm |

The gate requires at least 95% present coverage, zero absent false accepts, at most 1.5 mm p95 3D/XY error, and at most 3 mm maximum 3D error. Both candidates fail accuracy. Complete scores include all 512 development cases for each checkpoint, including absent cases.

The [protocol](protocol.json) and data generator were committed as `71be44aefdc62a926000eddc4d293c07fb27a335` before generation. The exact trainer was committed as `497c5c97c4505a412facbaf9a46e73dcf9cff573` before fitting. The fit takes **32.564 seconds** on the **RTX 4070**, with **0.082 GiB peak Torch-reserved memory**. Training uses FP32 CUDA with TF32 disabled; development selection uses PyTorch CPU FP32. These are PC measurements, not Intel execution evidence.

The model combines explicit classical RGB component extraction with a 792→256→256→128→3 SiLU network. Each view contributes a centroid, bounding box, area and 16×16 normalized silhouette. It predicts the mug body midpoint directly. It is not an end-to-end RGB detector and uses no pretrained model.

There are 4,096 synthetic training states and 512 development states, derived from the original 41 mug demonstration inputs. Joints, object pose and exposure vary within the declared ranges. These are **posed renders without physics stepping**, not valid grasps or manipulation-success evidence. No exposed 920x/960x physical trace supplies training data. The saved recipes independently regenerate all **13,824 RGB views**, every feature and every geometric label, with zero discrepancy. Both checkpoints also reproduce every development score exactly on CPU.

Packages preserve all examples, both failed candidates, all outcomes and frozen source:

- [Exact compact inputs](../../../../training/mug_visual_observer_v1/README.md): 73 core files / 5,432,284 bytes.
- [Both checkpoints](../../../../models/mug_visual_observer_v1/README.md): 3 core files / 2,423,148 bytes.
- This evidence package: 6 core files / 426,976 bytes, plus its manifest and this README.

The remaining fresh synthetic seed **2026099803 is unexposed**. There is no exported IR and no new physical trial. Preserve this stopped protocol; further work must use a separate protocol and fresh selection data.

The next architectural hypothesis is to predict semantic image points and triangulate them with the known camera calibration, so the network need not learn the entire image-to-world geometry. It still needs a bounded fit, fresh perception tests, actual held-object checks and paired physical control trials before any promotion. More iterations of this failed direct-XYZ fit are not authorized by its protocol.

The full submission goal remains active. Intel execution/eligibility, the human full-sequence voice rehearsal, final public access and the user's submission remain open. GitHub/HF stay private; the assistant must never press final Submit.
