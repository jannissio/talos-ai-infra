# Drawer configuration feasibility diagnosis

All **15 exposed starts** are retained and exactly match the original diagnostic. The cabinet-aligned, remaining-stroke programmed controller completes **5 physical openings**, compared with **2** for the original controller. This includes the anchor and its duplicate standard cabinet case, plus three recovered configurations: cabinet X=-0.24 m, cabinet yaw=+0.26 rad, and initial opening=0.04 m.

The initial 0.12 m opening passes a separate **already-open semantic no-op** check. It is not counted as a physical opening. The 0.08 m start and remaining displaced/rotated configurations still fail planning. A failed planner is not proof that a configuration is unreachable.

This teacher uses exact simulator state and inverse kinematics to produce physical-contact demonstrations. It is **not learned control** and is not installed in the selected browser. Its only changes rotate grasp/approach/pull geometry with the cabinet and pull the remaining distance to 0.116 m. Original torque, collision, contact, release and parking checks remain in force; there is no drawer actuator, state overwrite, attachment or hidden force.

The independent audit verifies all outcomes and **3,995** saved state frames. Every failure, source, portable scene and original comparison hash remains available. A separate compact demonstration, image-observer and learned-motion experiment is needed before any new learned drawer claim.
