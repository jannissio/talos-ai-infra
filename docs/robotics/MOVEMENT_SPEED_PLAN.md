# Faster submission movements

The user reported slow scripted movements and requested that speed be improved for the submission. Current training trajectories remain unchanged: they are verified physical references, and changing speed while diagnosing learning would alter observations, contact dynamics and targets together.

## Measured baseline

The existing upright-01 matched replay takes **38.165 simulated seconds**. The [per-stage measurement](teacher-speed-baseline.json) is extracted from 200 Hz recorded actions, not estimated from video playback. Close takes 5.30 s, release 4.305 s, align 4.305 s, lift 3.80 s and verified hold 1.805 s. Other phases account for the remainder. Hold duration includes controller settling; the required supported hold itself remains about 1.5 s.

`simulation_lab/autonomy.py` also clamps requested movement duration using a 0.8 rad/s nominal path-speed bound; simply reducing every stage's time would not necessarily make every motion faster. `_reached` waits for endpoint accuracy and an additional settling interval. These mechanisms explain part of the deliberate pace; they are not proof that the browser renders at real-time speed.

## Proposed separate demonstration profile

1. Measure both simulated task duration and actual wall time, alongside physics/render rates. Distinguish slow physical trajectories from a simulator falling behind real time.
2. Start with empty-hand approach, retract and parking. Try approximately 1.25× requested movement speed while retaining the existing joint-speed limit; measure achieved speed, acceleration/tracking error and collisions. A higher speed limit requires separate validation.
3. Replace excessive fixed gripper waits with bounded, stable contact/release checks where feasible. Do not shorten the required supported hold or final placement verification to inflate success.
4. Test loaded transport and rotation separately, preserving torque limits and checking slips, contact gaps, object displacement and upright release. Trial 1.5× only after the smaller change is reliable.
5. Compare the same seeded full sequences against the baseline. Adopt a faster profile only with retained physical success and no new disturbances; report failures and real task times. No claim that a proposed multiplier already works.
6. If faster execution becomes a training target, recollect/version the affected demonstrations and their timing contract. Do not play old policy targets faster and assume the learned controller remains valid.

Keep the slow verified profile available for training and debugging. A faster video playback may help presentation but must be labeled; it is not evidence of faster robot execution. This document records the submission improvement for later implementation, rather than changing the current diagnostic's physics.
