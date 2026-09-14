# Spoon placement diagnosis

This is a read-only diagnosis of existing training data and five previously exposed integrated-workflow traces, declared in `70bb61d`. It runs no new physical trial and changes no model. Both the original fixed-wrist-landmark report and the calibrated spoon-grasp-point follow-up (`95c1a10`) are preserved. The latter uses the actual `[-0.003, 0, -0.100]` metre spoon grasp reference in the programmed demonstration generator; these reference points must not be conflated.

The selected spoon network's fit is measured on all 42,003 existing training endpoints from 39 episodes. With the calibrated point, p95 position error is 1.027 mm during closure, 0.716 mm during lowering and 0.688 mm during release. The already-trained L-BFGS comparison is slightly more accurate on these labels but is not selected or deployed by this diagnostic. Training fit alone cannot establish physical improvement.

Both failed integrated traces place the spoon within approximately 1 mm of its destination at the end of alignment. The object then shifts during table contact/lowering and release, ending at 10.094 and 14.254 mm error. The camera features lie near existing training examples. The demonstration generator opens the fingers while retreating sideways by 10 mm. These observations motivate a separately gated release-clearance/order experiment; they do not prove that retraining alone, or that intervention, will fix the failures.

The [manifest](manifest.json) retains both reports, all initial camera strips, each exact script/protocol and the existing L-BFGS checkpoint. The [independent geometry audit](audit.json) recomputes all 60 retained tool/object probes from the original synchronized state arrays with zero discrepancy. The original physical traces, selected models and final-test denominators remain unchanged.

No additional training, no new generalization test, no Intel measurement and no model promotion occurs here. All five inspected integrated scenes are exposed and cannot serve as fresh selection or final-test scenes for the follow-up.
