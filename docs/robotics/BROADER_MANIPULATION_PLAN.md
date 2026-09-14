# Broader camera-driven manipulation

The active priority, corrected by the user on September 14, is useful manipulation beyond reset presets. The private submission baseline stays available, but submission readiness does not complete this development goal.

## First measurable milestone

Measure the current learned controllers across explicit source, orientation, destination and drawer-configuration variations. Compare with a separate exact-state physical controller to identify arrangements that are demonstrably feasible but fail under learned control. A failed planner does not establish unreachability. Keep every invalid reset, refusal and failure visible.

The initial diagnostic holds object geometry, appearance, arm bases and unrelated scene objects fixed. Positions are world metres, with the table at Z = 0.760 m. These are **candidate coverage targets**, not claims that every point is reachable:

| Skill | Source/configuration target | Orientation | Requested destinations |
| --- | --- | --- | --- |
| Upright bottle | X [-0.14, 0.16], Y [-0.18, -0.06] | Yaw +/-0.60 rad | (0.10,-0.115), (0.16,-0.06), (0.20,0.05) |
| Plate | X [-0.18,-0.04], Y [-0.05,0.09] | Yaw +/-0.60 rad | (-0.06,-0.025), (-0.10,-0.025), (-0.06,-0.075) |
| Mug | X [0.17,0.31], Y [-0.14,0.01] | Handle yaw +/-0.60 rad | (0.265,0.005), (0.215,0.005), (0.265,0.055) |
| Fork | Open-drawer local X [-0.038,-0.008], Y [-0.04,0.01] | Yaw +/-0.35 rad | (-0.16,-0.045), (-0.20,-0.045), (-0.16,-0.095) |
| Spoon | Open-drawer local X [0.008,0.038], Y [-0.04,0.01] | Yaw +/-0.35 rad | (-0.02,-0.155), (-0.06,-0.155), (-0.02,-0.105) |
| Drawer | Cabinet origin X [-0.32,-0.24], Y [0.14,0.22]; opening 0/0.04/0.08/0.12 m | Cabinet yaw +/-0.26 rad | Open to at least 0.105 m; fully open is a semantic no-op case |

Exclude overlapping, tipped, unsupported and off-table reset arrangements from valid-start physical success, while retaining them in the total planned-case table. Do not exclude solver or perception failures. This first grid varies one factor at a time; combined random arrangements, different obstacle layouts and the full Cartesian product remain untested.

The bottle gets a 24-position grid; other objects get nine-position grids, plus separate orientation and destination probes. Six known-context anchor checks run first to detect harness errors. Each diagnostic skill starts after the preceding skills in one saved seed-42 context; this is neither a new full-sequence result nor randomized-scene generalization. The learned and exact-state comparisons start from identical resets. Exact-state comparison actions never enter the learned controller. The current mug option is measured as an isolated diagnostic skill, separately from its public full-sequence interface restriction.

## Development steps

- [x] Complete the declared coverage map: [all 200 rollouts](evidence/manipulation-coverage-v1/README.md), with 100 identical paired starts and 88,760 state frames. Excluding six anchors: 11/83 valid learned successes versus 43/83 programmed successes; 11 invalid arrangements are also retained.
- [x] Identify the first target: wider bottle manipulation. Its 24-position grid has zero learned successes (two invalid starts) versus 15 programmed successes. Perception support and extrapolated motion are the dominant gaps; partially open/moved drawers also remain unsupported.
- [x] Freeze [bottle image refinement V1](experiments/bottle-refinement-v1.json): a new RGB crop-refinement network after the frozen coarse detector, 4,096 broad/physical-context training states and 512 fresh development states. One 8,000-update RTX fit, two checkpoints. Current status: training and CPU export pass development (433/449 present, 0/63 absent false accepts, 1.322 mm p95 / 3.971 mm maximum); all 4,608 preceding data states are audited. The selected step-8,000 weights are frozen before the 512-state fresh perception test. Physical integration is prepared but remains gated.
- [ ] Require the new observer to pass development, FP32 export and 512 fresh perception cases, then integrate the existing target-conditioned neural route/motor controller and run six development plus 24 fresh three-condition physical cases. No broader success is claimed before those tests.
- [ ] Require improved success beyond the narrow presets, preserved baseline regressions and unchanged physical criteria before promotion.
- [ ] Extend the successful method across skills, then evaluate combined configurations and full voice-command sequences.

Continuous visual correction during approach/grasp and placement, target-conditioned neural motion, and recovery are candidate methods. More fitting around old presets is not itself a coverage improvement. The recent mug live/baseline 20/20 comparison demonstrates bounded feedback and does not establish broader success.

## Current coverage checklist

- [x] Six learned skills and both table-supported bottle relay directions in narrow trained regions.
- [x] Bounded live stereo correction during late mug placement.
- [x] Human six-skill microphone rehearsal; actual Intel CPU/iGPU baseline execution with Intel UHD rendering.
- [ ] Robust coverage of the wider regions above, combined configurations and varied destinations.
- [ ] General drawer opening from partial states and changed cabinet position/orientation.
- [ ] Broad continuous grasp/placement correction and recovery after observed errors.
- [ ] Other-object or airborne handoffs, sideways objects and pouring.

The working package remains the deadline fallback. Keep media changes proportionate while substantive development proceeds. GitHub and Hugging Face remain private until the agreed release timing; only the user presses final Submit. Check both unused output and log paths and retain a 10 GiB reserve before every data write/export.
