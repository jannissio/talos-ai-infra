# Talos submission build plan

Updated September 11, 2026. This is an engineering plan, not an organizer requirement. The target remains the online Intel dinner-table challenge plus the Speechmatics Bonus Award. [Fresh challenge recheck](UPDATE_2026-09-11.md).

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

1. **Improve reach/robustness, then add one coordinated two-arm action.** Keep whole-sequence randomized evaluation. Add a hand-off or complementary action, such as one arm holding the mug while the other positions/tilts the bottle. Define the pouring approximation explicitly rather than claiming liquid transfer without a model.
2. **Record demonstrations and train camera-based control.** Pin a LeRobot observation/action contract and exporter. Collect varied successful episodes; retain failed episodes separately. Train ACT as a focused baseline; evaluate SmolVLA if language-conditioned control is feasible. Hold out whole seeds/episodes and measure policy-only success without hidden teacher corrections. Existing chemistry recordings do not train a dinner policy.
3. **Language and Speechmatics.** Add real transcription and a constrained instruction-to-goal layer. Show recognized text and verified execution. Speech alone does not meet the Intel visual-control requirement.
4. **Intel deployment in parallel.** Secure Core Ultra Series 2/3 access, run an early supported OpenVINO export/inference check, and measure actual inference/simulation performance. Colab and the user's RTX 4070/12 GB can train; they do not replace the final Intel requirement.
5. **Package the submission.** Reproducible setup, scene assets, training/evaluation/inference commands, hardware benchmark, ten-seed video evidence, technical README, slides/platform fields and provenance/AI disclosures.

The currently published submission deadline is **September 16, 20:30 CEST**; retain **18:00 CEST** as the internal target. Prioritize a narrow complete challenge demonstration over extra objects or visual polish. See [submission checklist](SUBMISSION_CHECKLIST.md).
