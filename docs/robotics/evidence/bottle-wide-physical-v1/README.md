# Wider bottle physical evaluation V1

The unchanged candidate passes **32/32** fresh live-camera trials, compared with **5/32** using a frozen image and **0/32** using the original preset controller. There are **32 valid starts**; all invalid starts remain in planned denominators. Live successes by destination are **[16, 16]** out of 16 each. The frozen final gate is **PASS**.

Development was **8/8 live** versus **1/8 baseline**. All six exposed regression outcomes remain alongside the final comparison. All **118** physical outcomes, **94,611** state frames, **31,363** RGB observations and frozen sources are preserved; the independent audit reproduces all **46** initial states exactly and recomputes both gates.

Sources span X [-0.14,0.16], Y [-0.18,-0.06] metres and yaw +/-0.6 radians. Requested destinations are [0.10,-0.115] and [0.20,0.05]. Fresh original scene seeds vary other-object starts, mass +/-8%, friction +/-10% and lighting 0.92..1.06. The middle destination [0.16,-0.06] remains unsupported. This finite experiment does not certify every reachable position, combined changed cabinet arrangements, unfamiliar objects, or arbitrary instructions.

The original coarse/refined RGB observer, V5 neural motor and V2 R1 route controller stay fixed throughout. Live images update approach/descent and the closure-time carry offset; subsequent transport uses motor feedback. A route may include table-supported release, parking and regrasping by the other arm. The frozen condition refreshes its initial image only between completed legs. No exact object pose, teacher action, inverse solver or hidden force supplies learned targets.

The preceding standalone synthetic perception protocol remains **failed** (94.956% present acceptance; 4.427 mm maximum), and its original physical seeds remain unexposed. This separate complete-task result does not change those stricter perception limits. Passing this physical experiment permits separately guarded production integration and regressions; this package alone does not change the browser or hosted controller.

Reproduction: use the archived source and declared seeds in `protocol.json`. The existing exact observer inputs and both candidates are in `training/bottle_refinement_v1` and `models/bottle_refinement_v1`; the motor, routing and original model hashes are bound by `physical/integration.json`. No new fitting occurred in this experiment.
