# Local mug correction: stop before training

September 14, 2026. The separately declared [protocol](protocol.json) samples local right-arm geometry around the 41 original mug trajectories. It proposes XY corrections of up to 2 mm with preservation of the current gripper axis. Label generation uses a scratch robot Jacobian offline. Deployment was intended to use a new neural differential map and a forward-only refusal guard.

The complete input gate fails:

| Split | All attempts | Accepted | Coverage | Required |
| --- | --- | --- | --- | --- |
| Training | 8,192 | 6,853 | 83.65% | ≥99% |
| Development | 1,024 | 857 | 83.69% | ≥99% |

All 1,339 training and 167 development rejections exceed the declared 0.08 rad joint-step limit. Sixty-four and seven respectively also miss the 0.15 mm endpoint-error limit. The maximum requested joint changes are 0.347 and 0.297 rad. No rejection is removed from the denominator; all points, matrices, endpoints, scores and sources are preserved.

Independent checks verify every saved endpoint using a separate Torch implementation of the robot tree: maximum translational disagreement is below 6×10⁻¹⁶ m. Central finite differences check 64 evenly spaced Jacobians per split, with maximum coefficient disagreement below 6.7×10⁻⁶ rad/m. This establishes that the failed labels reflect the geometry; it does not make them acceptable commands.

**Stopped:** no GPU fit, candidate, OpenVINO export, fresh seed 2026100103 or new physical controller trial follows. The passing [stereo observer](../mug-keypoint-observer-v1/README.md) remains a separate perception result. The selected dinner system and its published evidence are unchanged.

The next hypothesis is to limit correction size by required joint movement instead of requiring every 2 mm step. This needs a new bounded protocol and all-point evaluation, followed by live-versus-frozen-image contact-physics tests; no successful correction is claimed here. [Exact numeric inputs and reproduction](../../../../training/mug_correction_motor_v1/README.md) are packaged separately.
