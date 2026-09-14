# Joint whole-table scene generation

All **80** declared scenes and **969** geometric proposals are retained. Stable complete arrangements: **14/16 development**, **63/64 final**. All seven items share the tabletop sampling domain: bottle, both plates, mug, glass, fork and spoon. Requested yaw spans the full circle. Vessels include upright/inverted/sideways starts; plates and cutlery include both faces. Physics determines the realized resting orientation.

Every saved reset and all **4,880** settling frames independently replay exactly. Invalid stability cases remain in the planned denominator. The sampler conditions only on table/object geometry, initial non-intersection and a conservative robot-chain length bound. That bound does not certify a grasp or route, and successful scene generation is not successful table setting.

The next dependency is physical grasp/route demonstrations and a shared spatial action policy evaluated on jointly randomized complete workflows. Side plate, glass, broad orientations and tabletop cutlery remain manipulation work. See the single authoritative goal in `docs/PC_HANDOFF.md`. The running demonstration and private fallback are unchanged.
