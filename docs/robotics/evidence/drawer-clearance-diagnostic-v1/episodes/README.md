# Drawer path clearance diagnosis

All **42** exposed cases are retained: one passing anchor, all **29** prior plate-obstruction failures and all **12** pose-planning failures. Moving only the plate during the declared reset to [-0.035,-0.155] recovers all 29 obstruction cases. The anchor remains successful; the 12 planning failures remain. No thresholds, teacher or torque cap changed.

The audit exactly reconstructs every original and intervention reset and all **30** successful saved-action replays. It retains **22,930** teacher and **18,773** replay frames, full-rate actions, all images, labels, reports and failures.

This is an exposed causal diagnosis with an explicit reset intervention. The robot has not learned to clear the plate, and the earlier 80-case demonstration input gate stays failed. No fitting, fresh manipulation claim or promotion follows from this package. The result motivates obstacle-aware task order and deliberate clearing in the shared whole-table pipeline.
