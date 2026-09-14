# Shared spatial action pipeline for the complete table

The [single authoritative goal](../PC_HANDOFF.md#current-goal---authoritative-after-every-continuation) governs this work. All seven loose items must be considered in jointly randomized tabletop scenes, with full yaw and explicitly addressed stable alternative orientations. This implementation plan replaces further disconnected preset optimization. The existing private dinner demonstration remains a fallback.

## Selected reference and compatibility

Use **CLIPort's parallel-gripper spatial pick/place methodology** as the shared policy reference: current calibrated visual observation and instruction produce a pick location/rotation and conditioned placement prediction. Its [official implementation](https://github.com/cliport/cliport) reports single-GPU operation using about 8.5–9.5 GB and exposes a PyTorch implementation. The [paper, Appendix D](https://arxiv.org/html/2109.12098v1) describes crop-based pick rotation for parallel grippers. Its standard policy is spatially planar; full 3D grasping and dynamic correction are limitations, not capabilities Talos inherits automatically.

This is a compatibility choice, **not a trained or measured Talos CLIPort policy yet**. The reviewed upstream revision is `2be5c47b5bb9bb7040ad90693288b87b1e18e7ad`. No upstream model weights have been downloaded or executed. The local source review found Torch 2.8/CUDA and torchvision available; several upstream utility/training dependencies are absent. A measured forward/backward memory check must precede the fit.

The repository contains Apache-2.0 code, MIT-derived CLIP code and an explicitly GPL-3.0 U-Net helper. The default decoder imports that helper. Do not blindly vendor the entire repository into the MIT Talos submission. Retain upstream notices and source provenance for selected permissive modules; use an independently authored standard upsampling helper, or another verified permissive implementation, for that interface. Do not relabel copied third-party code as MIT. The reviewed GPL helper itself has not been downloaded.

RVT-2 offers a stronger native 3D reference, but its [official code and pretrained model terms](https://github.com/NVlabs/RVT#license), hardware setup and runtime integration add substantial migration work. It is not a parallel rewrite in this plan. No RVT/SmolVLA weights are integrated.

## One coherent implementation milestone

1. **Joint scene and observation interface.** All seven loose objects share the tabletop sampling domain. Sample full yaw and declared alternative orientation families, settle using ordinary physics, and record geometric proposal rejection, instability and changed resting orientation. No learned success box determines positions. Export calibrated camera observations suitable for spatial action prediction. If RGB-D is used, it is an explicit camera sensor input; exact body poses/segmentation remain demonstration/scoring data, not hidden policy observations. Previous RGB-only model claims remain unchanged.
2. **Physical demonstration and execution interface.** Search multiple grasp contacts, approach directions and arms using the robot model and physical pinching. Add side plate, glass and tabletop cutlery explicitly. Test transfers to intended place settings in the actual joint obstacle scene. Preserve planning failures as unsolved cases. Add temporary-clearance actions and task ordering where destinations/routes are blocked. Successful contact, supported lifting, release and final placement must be measured; hypothetical IK alone is insufficient. Exact-state teachers may generate demonstration labels, never be misreported as learned control.
3. **Shared visual pick/place policy.** Reuse the reference's spatial prediction structure and language features. Supervise from successful physical demonstrations over the shared scene distribution. Record gripper pose/orientation/closure keyframes and recovery states, rather than asking a time index to reproduce one scene's joint trajectory. The parallel-gripper method handles yaw; height/tilt and alternative grasps require explicitly evaluated adaptations, not an upright-only redefinition of the goal. Keep the useful bottle observer/local motor components where compatible.
4. **Observe–execute–verify–recover loop.** Predict from the current scene, execute a short segment, observe again and verify progress. Collision-aware robot-model planning is a distinct hybrid execution option; if used, label the controller accordingly instead of claiming every joint target is neural. Plans must use camera-derived obstacle estimates online. Preserve torque/contact safeguards and do not attach or teleport objects. Re-plan arm/order/grasp after a failed attempt within a declared recovery budget.
5. **Complete randomized table evaluation.** Evaluate all requested items together from fresh scene seeds, record all valid-start failures and intended final placements, and report each item's orientation/position coverage plus full-table completion. A sampler or isolated grasp success does not count as table-setting success. Keep the hard cases visible and preserve a fresh final set separate from training/selection.

These steps form one end-to-end deliverable. Do not spend the remaining work on unrelated media, repeated unchanged baseline tests or another succession of narrow model fits. If complete randomized table setting is unfinished at handover, state that plainly and keep the verified fallback accurately labeled.

## Current evidence and missing behaviors

The joint randomizer's geometry tests pass. Its frozen scene experiment produces stable full arrangements in **14/16 development** and **63/64 final** scenes. The three invalid scenes remain; these are not manipulation trials. Every item spans the shared tabletop and all declared orientation families are represented. A conservative robot-chain bound excludes points outside its outer limit; it does **not** certify a feasible grasp or collision-free path.

The prior wider-bottle application comparison finishes **10/10 complete dinners versus 0/10** for the older controller from paired starts. Other items in those trials still use narrow starting distributions. All 36 app outcomes, 99,579 state frames and 8,996 bottle RGB observations are packaged. This is a useful reusable component, not the clarified whole-table goal. No new browser/cloud promotion is needed before the shared pipeline work.

| Item or fixture | Useful existing component | Required broader behavior |
| --- | --- | --- |
| Bottle | Live RGB approach/grasp, neural motor and two destination routes | Full workspace, yaw/sideways/inverted starts, obstacle-aware routing and recovery |
| Plate | Narrow learned placement and physical rim grasp | Workspace-wide rim selection, both faces, varied routes |
| Mug | Narrow placement with bounded late visual correction | Broad handle/body grasps and stable alternative orientations |
| Fork and spoon | Narrow learned drawer retrieval | Tabletop starts, either face, full yaw, arm/route choice |
| Side plate | Procedural physical asset and intended destination | Verified grasp/transfer demonstrations and learned control |
| Glass | Procedural physical asset and intended destination | Verified grasp/transfer demonstrations and learned control |
| Drawer | Narrow learned opening; configuration teacher diagnosis | Observed state, clearance, partial-opening/changed-configuration support and task integration |

The drawer clearance diagnostic recovers all **29** earlier plate-obstruction failures with unchanged contact control after an explicit reset-only plate relocation. Twelve grasp-pose planning failures remain. It establishes the cause of those failures, not autonomous obstruction clearing or a newly learned drawer policy.
