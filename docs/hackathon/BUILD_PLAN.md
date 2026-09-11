# Talos submission build plan

Updated September 11, 2026. This is our engineering plan, not an organizer requirement or a promise of completion. The user chose the dinner-table task. Target: online Intel challenge plus a Speechmatics voice interface for the bonus.

## Completed foundation

- Dinner scene and seven original tableware models, with a passive cutlery drawer.
- Two licensed SO-101 arms, camera views, manual controls and reproducible resets.
- Initial object position, yaw, mass, friction and lighting variation.
- Physical stability audit and a bottle candidate reachable by either arm under the existing IK orientation.
- Existing chemistry teacher and recording infrastructure preserved, with preparation history disclosed.
- Repository connected to the user-created GitHub project, retaining its existing MIT license.

See [measured scene results](../robotics/DINNER_SCENE.md). Scene settling and candidate reach are not manipulation success.

## Next: one successful dinner manipulation

Implement **pick up the amber bottle, lift it 5 cm, hold, and place it upright in a specified clear serving position**. Start with one arm and park the other. Use exact simulator state to establish a teacher with explicit physical contact, support, collision and placement checks. Reuse the existing controller ideas while separating object-specific geometry and success conditions from the tube/rack code.

The bottle has a 22 mm neck and clear isolated approach/grasp/lift configurations for either arm at seed 42. We still need an entire collision-free trajectory, measured finger closure, sustained support, failure recovery and randomized task evaluation. A successful IK pose alone is not enough.

## Then: the actual multi-step dinner workflow

1. **Drawer and dish skills.** Pull the passive drawer by its handle; retrieve a utensil; pick/place a plate and mug. Add grasp orientations appropriate to rims, handles and horizontal cutlery. Current upright-finger tube IK cannot pick all these items from their initial heights/positions. Adjust fixtures and staging poses only where physically justified and document the final evaluation scene.
2. **One coordinated two-arm behavior.** Implement and evaluate a hand-off or complementary action, such as one arm holding the mug while the other positions and tilts the bottle. Define the pouring approximation explicitly; no current liquid-transfer result exists.
3. **Demonstrations and camera-based learning.** Generalize the recorder to dinner object/task metadata, choose a consistent image/action timebase, and create a pinned LeRobot export plus Colab notebook. Train ACT as a focused baseline; assess SmolVLA for language-conditioned control. The current chemistry recordings do not train a dinner policy by themselves. Hold out whole seeds/episodes and measure policy-only success without teacher corrections.
4. **Spoken goals.** Feed Speechmatics transcription into a local instruction/vision layer or the selected language-conditioned policy. Show the recognized instruction, current action and verified outcome. Speech transcription does not substitute for visual robot control.
5. **Intel deployment and final evaluation.** Export supported components to OpenVINO, benchmark on the actual Core Ultra Series 2/3 machine, and report ten randomized task runs. Measure inference speed, simulation speed, success and recovery separately.
6. **Submission package.** Reproducible repository, setup/training/inference/evaluation commands, Intel benchmark, demo video, slides and platform fields. Preserve asset attribution and AI/preparation disclosures.

## Work to bring forward

**Intel hardware access and a small model export/inference check should happen alongside physical skill development.** Training access is available through Colab now and the user's RTX 4070 / 12 GB from Sunday September 13, but final qualifying Intel hardware remains unconfirmed. Do not leave that dependency to the final day.

The September 10 public check gives a submission deadline of **September 16 at 20:30 CEST**; our internal target is **18:00 CEST**. With this short window, prioritize one reproducible dinner sequence and one meaningful coordinated action before adding more objects, commands or visual polish. See [submission checklist](SUBMISSION_CHECKLIST.md) and [dated source check](UPDATE_2026-09-10.md).
