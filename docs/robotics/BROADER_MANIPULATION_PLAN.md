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

- [ ] Complete the declared coverage map, including both controller outcomes and valid-start/feasibility distinctions.
- [ ] Identify the highest-value failure pattern: perception support, grasp geometry, carrying/release, destination conditioning or drawer semantics.
- [ ] Freeze one bounded improvement with varied demonstrations, current-image action inputs and fresh development/evaluation cases where justified by that pattern.
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
