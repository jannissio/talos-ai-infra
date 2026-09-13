# Proposed next goal: bounded upright bottle coverage

This is a proposed experiment, not a completed training result. Keep both published models and all existing evaluations unchanged. The RTX 4070 provides 12 GB VRAM for training; the deployed baseline remains CPU OpenVINO inference with NVIDIA camera rendering. Extra VRAM does not by itself correct grasp precision or coverage.

## Evidence and hypothesis

The [wider-position development report](../../submission/evidence/upright-visual-development.json) records a pass for `train-017`, a tracking timeout without lift for `train-019`, and loss of support after a 6.83 cm lift for `train-025`. The [frozen input](../../training/bottle_visual/retrieval.json) includes `train-025` but does not include `train-019`. A failure on an included demonstration means that collecting more examples alone may be insufficient.

The original 32-attempt collector's deterministic protocol can reconstruct candidate specifications using RNG seed 617131. Its implied upright starts are:

| Case | X (m) | Y (m) | Yaw (rad) | Historical result |
| --- | --- | --- | --- | --- |
| train-017 | -0.004839 | -0.071144 | 0.280990 | Pass |
| train-019 | -0.073407 | -0.121573 | 0.309641 | Tracking timeout |
| train-025 | 0.013686 | -0.097940 | -0.209740 | Lost support |

These specifications are inferred from the current collector with `count=32`; the old raw manifests and integration states are absent on this PC. Newly reconstructed episodes must be labeled reconstructions and checked for physical teacher success and replay parity. They cannot be claimed as byte-identical copies of the laptop experiments.

Hypothesis: a small upright-only curriculum around these positions, with physically verified demonstrations and consistent stage alignment, can improve local coverage while preserving the task-preset baseline. Measure detection, grasp, tracking and support failures separately.

## Bounded protocol

1. Reconstruct the three exposed development starts first. Record detector centroids, joint tracking error, finger support and failure time during physical baseline execution. Do not relax the monitor or supply privileged state to policy actions. If the reconstruction cannot produce valid teacher/replay demonstrations, stop and investigate before training.
2. Predeclare a maximum of **24 new training attempts**: the three reconstructed starts, 15 upright starts within 15 mm of them, and six task-preset anchors. Keep camera calibration, object appearance, destination, controller architecture and torque limits fixed. Use new training scene seeds 2026091401–2026091424 and a fixed pose RNG seed 407024. Save every attempted specification and outcome, including rejected cases. The existing collector needs an explicit-specification input for this curriculum; merely changing `--count` changes its distribution and is not this protocol.
3. Save compact initial integration states, 20 Hz joint endpoints, stage labels and three initial 320×240 RGB images only. Require teacher success and replay checks at 200 Hz and 20 Hz before training eligibility. Preserve the frozen 22-episode input; prepare a separate input combining eligible new demonstrations with baseline anchors, using one stage-alignment calculation across the combined raw examples. Regenerating anchors requires fresh compact episodes because the frozen input does not include their raw stages/images.
4. Before training or checkpoint selection, freeze a separate six-case development set (scene seeds 2026091501–2026091506) and **12 fresh evaluation cases** (2026091601–2026091612). Use three task-preset evaluation anchors and nine offset positions around the three development regions. Fix pose RNG seeds 407006 and 407012, respectively, and write the exact poses and scene settings to a protocol file before any candidate rollout. Verify that no evaluation pose duplicates a training/development pose. Evaluate all declared cases; do not remove hard cases after seeing outcomes. These seed ranges were absent from the checked repository on September 13, 2026; local/private use must also be checked when starting the experiment.
5. Train one candidate family on CUDA: 16,000 Adam steps, followed by at most 400 LBFGS refinement steps, using the documented global visual scaling. Consider only the Adam endpoint and the 400-step refinement endpoint. Select on the six development cases using physical success first and placement error second; training loss is diagnostic. Record peak VRAM and wall time on this PC. Stop on nonfinite loss or resource failure; do not silently extend the budget.
6. Freeze the selected checkpoint and runtime, then compare it with the unchanged packaged model on all 12 declared evaluation cases. Also rerun seed 42 and the ten exposed task-preset regression seeds. Proposed promotion gate: at least 10/12 fresh successes, a strictly higher fresh success count than the packaged model, and no regression on the 11 exposed baseline cases. Require the existing physical lift/support/release criteria and collision monitor. Publish every outcome, denominator and failure reason. If the gate fails, retain the baseline and report the candidate as experimental.

## Resource and stopping limits

Budget at most 1 GiB of new compact data, checkpoints and evaluation evidence, plus the mandatory 10 GiB free-space reserve. Check actual destination-drive space before every batch, episode and export and periodically during writes. Never delete old data to make the run fit. Render a separate demonstration video only after evaluation and another disk check. No large image archive or model download is needed for this motion primitive.

If grasp/support failures persist despite adequate fit, the next separate architectural experiment is continuous visual feedback. It requires new observation sequences and a revised policy contract; the current initial-image-conditioned network does not implement it. Broader upright placement, learned dish skills, arbitrary destinations and airborne handoffs remain unverified.
