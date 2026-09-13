# Pretrained robotics policy review

September 13, 2026. Requested by the user and investigated in a delegated research subtask. No pretrained checkpoint was downloaded or incorporated during this review.

## Decision for Talos

Post-training a pretrained policy is feasible and could improve visual/language generalization. It still requires embodiment-specific demonstrations, continuous observations, action normalization and physical evaluation. The Intel challenge permits training or fine-tuning related robotics models; it does not prescribe one pretrained foundation model.

SmolVLA is the most plausible research candidate for the RTX 4070's 12 GB VRAM. Keep the current six-skill development active; a model change does not itself solve the observed grasp/release failures. A replacement controller becomes a release candidate only after actual physical and Intel tests.

The name “ISAC 0.5” was not verified as an exact public model. Two plausible references are Physical Intelligence's **pi0.5** and NVIDIA **Isaac GR00T N1.5**, with substantially different licenses.

## Verified source distinctions

| Candidate | Resource/deployment fit | Code and weight licensing |
| --- | --- | --- |
| SmolVLA | Official hardware guidance estimates about 10–16 GB at batch 8; lower batches and freezing the vision encoder can reduce memory. 12 GB is plausible, not yet measured here. | LeRobot code is Apache-2.0. The current base card lacks an explicit weights license. The official `HuggingFaceVLA/smolvla_libero` checkpoint and Intel's converted derivative explicitly declare Apache-2.0. |
| pi0.5 | OpenPI documents over 22.5 GB for LoRA and over 70 GB for full fine-tuning, with Ubuntu support. No verified 12 GB recipe was established for this project. | OpenPI code is Apache-2.0; `lerobot/pi05_base` lists Gemma terms for weights. Do not label the complete checkpoint unconditionally Apache. |
| Isaac GR00T N1.5 | An official SO-101 tutorial exists, but the default training estimate is about 25 GB; the documented stack is Linux/CUDA. | Version-pinned code is Apache-2.0. The actual N1.5 weights license restricts use to noncommercial research/evaluation, including derivatives. Do not infer weights terms from a newer repository license. |

Primary sources: [LeRobot hardware guide](https://huggingface.co/docs/lerobot/main/en/hardware_guide), [LeRobot license](https://github.com/huggingface/lerobot/blob/main/LICENSE), [SmolVLA base files](https://huggingface.co/lerobot/smolvla_base/tree/main), [official LIBERO checkpoint](https://huggingface.co/HuggingFaceVLA/smolvla_libero), [OpenPI requirements](https://github.com/Physical-Intelligence/openpi#requirements), [pi0.5 model card](https://huggingface.co/lerobot/pi05_base), [NVIDIA SO-101 tutorial](https://huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning), [N1.5 weight license](https://huggingface.co/nvidia/GR00T-N1.5-3B/blob/main/LICENSE), [N1.5 code license](https://github.com/NVIDIA/Isaac-GR00T/blob/n1.5-release/LICENSE).

Intel already publishes an [OpenVINO SmolVLA LIBERO model and CPU example](https://huggingface.co/OpenVINO/smolvla-libero-fp16-ov). [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio) and the [OpenVINO Physical AI runtime](https://github.com/openvinotoolkit/physicalai) provide a concrete export/deployment route. This does not establish compatibility or speed on our older Intel laptop.

Talos uses 12 motor actions for two arms. The LIBERO example has different cameras, state and action semantics. [SmolVLA's configuration](https://huggingface.co/lerobot/smolvla_base/blob/main/config.json) allows padded dimensions up to 32, but our dataset must still define camera order, joint units, normalization and action timing. Initial-image data alone is insufficient for reactive vision control.

## Bounded follow-on experiment

Use one difficult skill, mug placement, before a six-skill migration. Freeze new training/development splits while retaining the existing final ten seeds unseen. Collect roughly 50 physical demonstrations with current RGB and motor feedback, and independently replay saved endpoints. [Hugging Face's training guidance](https://huggingface.co/docs/lerobot/smolvla) recommends approximately 50 episodes per task; its 20,000-step A100 timing is not a prediction for this GPU.

Create an isolated environment. Run a small-batch training smoke test with frozen visual backbone, measuring peak VRAM and seconds per step. Set the training budget from that measurement. Compare physical outcomes with the existing neural mug controller, including unseen positions and disturbances. Export to OpenVINO, test output parity and physical success, then measure the actual Intel device. Retain failures and refuse promotion if any of these gates fail.

Keep original Talos contributions MIT and preserve exact upstream terms and provenance for any incorporated checkpoint. See the [submission licensing check](../hackathon/LICENSING_2026-09-13.md). A repository's Apache code license does not override a checkpoint's separate restrictions.
