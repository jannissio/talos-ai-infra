# Talos submission build plan

Updated September 11, 2026. This is an engineering plan, not an organizer requirement. The target remains the online Intel dinner-table challenge plus the Speechmatics Bonus Award. [Fresh challenge recheck](UPDATE_2026-09-11.md).

The subsequent [livestream review](LIVESTREAM_REVIEW_2026-09-11.md) confirms this direction and clarifies that other Intel systems with integrated graphics can qualify, with Core Ultra 2/3 receiving bonus points. This differs from the strict written-brief sentence; preserve the discrepancy and seek written confirmation. The local Intel CPU/iGPU is now a candidate deployment target, not yet a verified final demo.

An initial [ACT OpenVINO export](../robotics/INTEL_INFERENCE.md) now passes numerical checks on the Intel CPU. Camera rendering, physical policy success, clean performance benchmarks and final-system integration remain separate requirements.

## Physical milestone implemented

The original environment and its licensed SO-101 assets now support a contact-only six-skill teacher:

1. Grasp the bottle neck, lift more than 5 cm, hold without external support, place and release.
2. Place the shallow plate and mug using rim grasps.
3. Pull the passive drawer open by its handle.
4. Retrieve and place the fork.
5. Retrieve, rotate and place the spoon.

Each skill verifies finger contacts, retention, placement and the parked arm. The browser exposes the full sequence, individual goals, progress, outcomes and cancellation. Exact-state evaluation explicitly checks that the controller does not write object positions or apply hidden forces. [Implementation, modified physical dimensions and measured reports](../robotics/DINNER_SCENE.md).

This establishes a teacher; it is not yet a trained robotics policy, language/vision control, Intel deployment or genuinely coordinated bimanual solution. The side plate/glass remain staging assets and liquid flow is not simulated. Keep the earlier preparation disclosure and model attribution in the submission.

## Next steps

September 12 user constraints: reuse existing training images where possible, retain a 10 GiB free-space reserve, and budget new experiments before collecting data. The [small inference diagnostic](../robotics/ACT_INFERENCE_DIAGNOSTIC.md) uses one existing episode and one checkpoint. Faster scripted movements are a later submission improvement under the [measured speed plan](../robotics/MOVEMENT_SPEED_PLAN.md); current training timing remains unchanged.

Use the [evidence-based training plan](../robotics/TRAINING_RESEARCH_PLAN.md) for the concrete learning sequence and acceptance gates. Begin dataset/replay and export checks before large-scale collection; keep genuine coordination as an essential milestone. The [mjbatch assessment](../robotics/MJBATCH_REVIEW.md) explains why it remains an optional isolated benchmark rather than a dependency of the Windows simulator.

1. **Improve reach/robustness, then add one coordinated two-arm action.** Keep whole-sequence randomized evaluation. Add a hand-off or complementary action, such as one arm holding the mug while the other positions/tilts the bottle. Define the pouring approximation explicitly rather than claiming liquid transfer without a model.
2. **Record demonstrations and train camera-based control.** Pin a LeRobot observation/action contract and exporter. Collect varied successful episodes; retain failed episodes separately. Train ACT as a focused baseline; evaluate SmolVLA if language-conditioned control is feasible. Hold out whole seeds/episodes and measure policy-only success without hidden teacher corrections. Existing chemistry recordings do not train a dinner policy.
3. **Language and Speechmatics.** Add real transcription and a constrained instruction-to-goal layer. Show recognized text and verified execution. Speech alone does not meet the Intel visual-control requirement.
4. **Intel deployment in parallel.** Test the local Intel CPU/iGPU route clarified in the livestream, run a supported OpenVINO export/inference check, and measure actual inference/simulation performance. Newer Core Ultra access remains valuable for optimization points. Colab and the user's RTX 4070/12 GB can train; they do not replace actual Intel execution evidence.
5. **Package the submission.** Reproducible setup, scene assets, training/evaluation/inference commands, hardware benchmark, ten-seed video evidence, technical README, slides/platform fields and provenance/AI disclosures.

The currently published submission deadline is **September 16, 20:30 CEST**; retain **18:00 CEST** as the internal target. Prioritize a narrow complete challenge demonstration over extra objects or visual polish. See [submission checklist](SUBMISSION_CHECKLIST.md).
