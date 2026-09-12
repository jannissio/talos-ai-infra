# Intel inference preparation

The [livestream clarification](../hackathon/LIVESTREAM_REVIEW_2026-09-11.md) makes the laptop's Intel CPU/iGPU a useful candidate. Qualifying final behavior and the written-brief discrepancy still require attention. NVIDIA training and graphics are reported separately from Intel model execution.

## Implemented export path

`scripts/export_act_openvino.py` loads our exact LeRobot ACT checkpoint, embeds its saved input/output normalization, freezes inference-time zero-latent behavior, traces it, and saves an FP32 OpenVINO IR. It checks outputs on 16 evenly spaced recorded observations against the CPU PyTorch model. It does not consume action labels or simulator object positions as model inputs.

The export contract is deliberately fixed: one 24-value motor-state vector and three 320×240 RGB images in [0,1], producing 20 twelve-joint target vectors in radians. Model input shapes are explicitly fixed after conversion because tracing emits warnings about iteration over image features. Different batch sizes, image resolutions or camera counts require a new export and validation. The external 0.25 Nm gripper adapter and actuator limits remain necessary.

The source is [Intel's ACT conversion guidance](https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/ai_resources/openvino/models/model_act.html), adapted to our LeRobot architecture and dimensions. No vendor example checkpoint is substituted for our trained model.

```powershell
.\.venv-training\Scripts\python.exe -m pip install -r requirements-openvino.txt
.\.venv-training\Scripts\python.exe scripts/export_act_openvino.py --checkpoint .run/act-matched-1000/step-001000 --output .run/act-openvino-1000-static
```

Use a fresh output folder. The final static-shape export completed CPU inference with a maximum target discrepancy of 1.97e-6 rad, below the declared 1e-4-rad tolerance. It passed all 16 observations with the exact declared input shapes; see [verification](act-openvino-export.json). No inference speedup, physical task success, quantization quality or NPU support is inferred from numerical agreement.

## Detected devices and limitations

OpenVINO 2026.3.0 reports CPU = Intel i7-10850H, GPU.0 = Intel UHD Graphics, and GPU.1 = NVIDIA Quadro T2000. This mixed-device system makes `GPU` or automatic selection ambiguous for proving Intel-only execution; record the actual selected device and full device name. The export test explicitly compiles for CPU with an FP32 hint and two inference threads.

The Torch `--device cpu` two-second MuJoCo smoke rollout used CPU inference but NVIDIA OpenGL rendering. It is not an all-Intel demonstration. The export parity test does not render cameras at all; it uses recorded images. Runtime policy/rendering integration and Intel graphics execution remain to be validated.

Training was active during these checks, so their wall times are not benchmark evidence. Measure clean warmup/latency/throughput after training, and evaluate the exported policy physically before claiming retained task quality. An OpenVINO model cannot fix the current policy's failed bottle manipulation.

Only optional OpenVINO packages were added to `.venv-training`; the running simulator's `.venv`, existing Torch/LeRobot/NumPy versions, drivers and OS graphics preferences were unchanged.
