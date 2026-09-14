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
- [x] Complete [bottle image refinement V1](experiments/bottle-refinement-v1.json): the 51,986-parameter crop model passes development/export but **fails fresh perception** (433/456 present, 0/56 absent false accepts, 1.494 mm p95 / 4.427 mm maximum). The original V1 physical stages remain unexposed. All 5,120 observation recipes/labels, exact compact fit inputs, both candidates and results are packaged; 242 bound files verify and the packaged CPU runtime loads.
- [x] Finish [exposed physical diagnostic V2](evidence/bottle-refinement-exposed-physics-v2/README.md): all 29 prior bottle cases / 58 live-frozen rollouts, identical original starts and unchanged V5 motor/V2 R1 routes. Live succeeds 25/27 valid versus frozen 9/27 and original learned 3/27. The position grid improves from 0/24 planned to 21/24 (22 valid). All 40,396 frames and 20,194 RGB observations verify. Training-exposed success cannot promote the candidate or pass the failed perception protocol.
- [x] Complete [wider physical V1](evidence/bottle-wide-physical-v1/README.md): development 8/8 live versus 1/8 baseline; final **32/32 live, 5/32 frozen, 0/32 baseline**, 16/16 live per destination and all six exposed regressions pass. All 118 outcomes, 94,611 frames and 31,363 RGB observations are retained; all 46 initial states independently regenerate exactly. The standalone perception gate stays failed and the middle destination remains unsupported.
- [ ] Complete [guarded runtime deployment](experiments/bottle-wide-deployment-v1.json) before promotion: all six exposed comparisons now preserve every recorded motor/physical value exactly (4,152 frames); separately declared fresh app/full-sequence checks follow. The wider physical improvement and preset regressions are established; browser and hosted integration remain unverified.
- [ ] Extend the method across skills, then evaluate combined configurations and full voice-command sequences. [Drawer configuration diagnosis](evidence/drawer-configuration-diagnostic-v1/README.md) preserves 15 starts and 3,995 frames: five programmed physical openings versus two originally, plus one separate no-op. The [80-case compact collection](experiments/drawer-demonstrations-v1.json) stops before fitting: 31/64 training and 8/16 development exact-replay-eligible episodes against a 32/8 minimum. All failures remain; opening the relocated drawer can push the already-placed plate. Collision-free access planning and camera/motor support require a new bounded experiment.

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
