# Talos · Dinner-table robotics

Two SO-101 arms, a physical dinner-table simulation, and spoken or typed instructions. Built for the [AI Infra Summit online Intel challenge and Speechmatics bonus](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon).

![Talos dinner scene](docs/robotics/dinner-task.jpg)

Talos combines MuJoCo, six camera views, Speechmatics transcription and camera-conditioned neural motor policies running through OpenVINO. The measured v6 baseline chains six learned dinner skills and supports table-supported bottle relays in both directions. The browser offers programmed and learned modes, progress, cancellation and explicit failures. See [the current architecture and evidence](docs/robotics/FINAL_ARCHITECTURE.md).

## Run locally

Use clean CPython 3.12. For the programmed simulator:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m simulation_lab.server
```

Open [the local application](http://127.0.0.1:8765/). Select **Task start / Closed / Seed 42**, then enter `set the table`. This mode places the bottle, plate and mug, opens the passive drawer, and retrieves the fork and spoon with programmed physical skills.

For learned bottle control, use the separate training/runtime environment:

```powershell
py -3.12 -m venv .venv-training
.\.venv-training\Scripts\python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-training\Scripts\python -m pip install -r requirements-training.txt -r requirements-openvino.txt
.\.venv-training\Scripts\python -m pip check
.\.venv-training\Scripts\python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
.\.venv-training\Scripts\python -m simulation_lab.server --learned
```

The explicit CUDA wheel installation is for NVIDIA training on Windows; it follows the [official PyTorch 2.8 installation matrix](https://pytorch.org/get-started/previous-versions/). The learned browser controller still uses OpenVINO on the CPU; CPU-only inference can omit the CUDA-specific installation and availability assertion. Use `py -3.12` and the environment paths above: plain `python` may resolve to another project's environment. Check destination-drive space before installation and every dataset/export, preserving the [10 GiB reserve](docs/robotics/STORAGE_POLICY.md) in addition to expected writes.

Select **Learned upright bottle · OpenVINO**, reset to seed 42, and enter `place the bottle`. The [upright model](models/bottle_visual/README.md) uses overhead RGB localization and motor feedback. The original model remains available for familiar sideways practice. Stop a foreground server with Ctrl+C.

For the expanded [six-skill suite](models/dinner_suite/README.md), run:

```powershell
.\start-lab.ps1 -Learned -DinnerSuite models/dinner_suite/suite.json
```

Choose **Learned dinner sequence**, reset to Task start, and enter `set the table`. Each skill must release and park before the next begins; no reset occurs between the six skills. The menu retains the word “candidate” to distinguish this finite evaluated baseline from broad workspace coverage. Use `pass the bottle to the right arm` at the standard start, or choose **Left reach practice** and use `place the bottle` for the reverse relay.

After installation, Windows users can instead run `.\start-lab.ps1 -Learned` to launch the complete runtime in the background and open the browser. Stop it with `.\stop-lab.ps1`. Both scripts accept `-Port` when the default port is occupied.

## Voice

Create a local `.env` containing `SPEECHMATICS_API_KEY=your-key`. Spaces around `=` and matching quotes are supported. The file is ignored by Git. The long-lived key stays on the server; the browser receives a temporary token. Press **Speak instruction**, speak English, then finish recording. Audio is sent to Speechmatics only during recording, for at most 20 seconds. Typed commands remain available.

## Measured behavior

- The v6 learned dinner suite completes **8/10** frozen randomized task-start sequences. The two failures are retained: mug placement error 14.59 mm and spoon error 13.61 mm, against the unchanged 8 mm threshold. Five new policies use 201 physically replayed training examples; the original bottle model is preserved.
- Four learned relay policies pass **5/5** frozen trials in each direction, using 52 replayed training examples. They release onto the shared table before the other arm grasps. These are bounded upright regions and fixed destinations.
- The [integrated camera-planned workflow](docs/robotics/evidence/composed-dinner-v1/README.md) completes **8/10** new frozen scenes from the left-reach bottle preset: both relay legs, then plate, mug, drawer, fork and spoon, without a reset. All ten pass every step through fork; two final spoon placements fail. Existing models remain unchanged.
- A human microphone bottle command succeeded on this PC. Final six-skill microphone rehearsal is still open.
- Real Speechmatics transcripts have driven the six-skill programmed sequence and a learned bottle movement, using explicitly labeled synthetic speech tests.
- The revised upright model completed **10/10 new task-preset scene seeds**, with **0.36–1.66 mm** placement errors. This covers small bottle-position changes, not the entire reachable workspace. Wider-position trials still include failures.
- The original three-view model supports familiar sideways practice but completed **0/10** broader scene seeds. Its results remain available separately.
- `pass the bottle to the right arm` runs a programmed table-supported relay: left release, park, right regrasp and placement. Seed 42 completed in **76.3 simulated seconds**, with **1.45 mm** final placement error.
- OpenVINO FP32 inference has been checked on an Intel CPU and Intel UHD iGPU. Current camera rendering uses NVIDIA.
- Videos can be reconstructed at 1280×720, 20 fps, with four simultaneous views, without changing training image resolution or physics.

No objects are welded to grippers or teleported during control. The selected learned controllers use initial visual features and motor feedback; an independent simulator-state monitor can stop execution but cannot generate actions. The language interpreter is a constrained grammar with RGB-grounded preparation steps. Direct airborne handoffs, pouring and arbitrary placements remain unfinished. Separate RGB correction experiments remain unpromoted: [V1](models/bottle_servo_v1/README.md) passes 2/12 undisturbed and 3/12 pushed physical trials; [V2 routing](docs/robotics/evidence/rgb-servo-v2/README.md) passes 6/12 and 5/12 on new final seeds. All failures and both frozen-image controls are preserved. These experiments do not replace the dinner workflow or establish arbitrary-position reliability.

See the [upright policy experiment](docs/robotics/VISUAL_BOTTLE_POLICY.md), [relay evidence](docs/robotics/TABLE_RELAY.md), [simulator controls](simulation_lab/README.md), and [storage safeguards](docs/robotics/STORAGE_POLICY.md). A [271 KB frozen training input](training/bottle_visual/README.md) supports local GPU retraining.

The original [submission folder](submission/PROJECT.md) is preserved. Updated materials are versioned under `submission/final-v6/`; [the final checklist](docs/hackathon/FINAL_SUBMISSION_CHECKLIST.md) tracks remaining work. A compact Hugging Face demo is under [private deployment verification](docs/hackathon/PUBLIC_DEMO.md). The repository and Space stay private until immediately before submission. Final-suite Intel execution, public judge access and the user's submission confirmation remain open.

Project code, model weights and original dinner assets use [MIT](LICENSE). SO-101 assets retain their [Apache-2.0 attribution](simulation_lab/NOTICE.md). See [development and AI-assistance provenance](docs/PROVENANCE.md).
