# Drawer configuration input gate V1 — stopped before fitting

All **80 combined configurations** are retained. Exact saved-action replay accepts **31/64 training** and **8/16 development** episodes. The frozen gate required at least **32/64** and **8/16**, so this protocol **fails before any neural fit**. No model or physical final evaluation is produced by this collection.

Candidate cabinet coordinates are X [-0.29,-0.235], Y [0.17,0.19] metres, yaw [-0.05,0.28] radians and initial opening [0,0.065] metres, sampled independently. These ranges are not asserted feasible everywhere. Each scene seed varies the original mass/friction/light settings; other objects retain the exposed preceding-task coordinates. This is isolated drawer preparation, not full-sequence generalization.

Every original teacher failure is retained. Many openings push the already-placed blue plate, revealing a drawer-path/scene-planning conflict. Other failures come from pose/path planning. A failed planner is not proof of unreachability. Neither failure type is reclassified as an invalid reset to improve a denominator.

The unchanged privileged contact-only teacher supplies training demonstrations, never learned runtime inputs. Only passing demonstrations with passing float32-action replay are marked eligible. All initial RGB images and camera calibrations, training-only handle/roof point labels, full-rate teacher actions, compact 20 Hz targets, initial integration states and physical traces are in `training/drawer_configurations_v1`.

The independent package audit exactly reproduces all **39 eligible physical replays**, with **38,431 teacher** and **24,460 replay** state frames retained. The next experiment must address collision-free drawer access and movement/perception support under a separately declared protocol; this stopped gate must not be resumed or weakened.
