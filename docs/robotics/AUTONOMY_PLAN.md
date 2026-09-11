# From manual control to goal-driven autonomy

Prepared September 8, 2026 for the local chemistry learning sandbox. **Lift-and-return, single-arm rack transfer and demonstration recording are now implemented.** Read [Lift & return](LIFT_AND_RETURN.md) and [Transfer and recording](TRANSFER_AND_RECORDING.md) for the demonstrations. Vision, learned policies and coordinated two-arm hand-off remain future work.

## Recommendation

Build a **goal-driven controller using known simulator state and inverse kinematics first**, then add visual perception and learnable skills. Keep task planning, motion execution, and success verification as separate components. This gives us a useful autonomous baseline on the current PC and a way to generate demonstrations for a later policy.

Our first goal should be:

> Lift one reachable tube approximately 5 cm, hold it briefly, and return it to its original slot.

Keep the other arm parked for this first experiment. The current default seed has all 18 slots occupied, so a later transfer task must deliberately provide an empty destination slot. After a single-arm grasp works, add transfers, shared-workspace scheduling, and a genuine hand-off.

## The main strategies

These approaches can be combined; they are not mutually exclusive categories.

| Strategy | How it produces behavior | Main advantage | Main cost or limitation | Role for us |
| --- | --- | --- | --- | --- |
| Classical task and motion planning | A state machine or behavior tree chooses steps; inverse kinematics and trajectories move the gripper using object poses | No training required; interpretable and relatively light to run | Needs reliable poses, reachable grasps and explicit failure handling | First autonomous baseline |
| Camera-based perception / visual servoing | Images estimate object/gripper positions and update the controller as the scene changes | Responds to visual changes and bridges toward real observation | Calibration, occlusion and pose estimation become difficult | Replace exact object state after grasping works |
| Imitation learning | A policy learns actions from demonstrations, for example with ACT or a diffusion policy | Can learn contact-rich skills that are awkward to hand-code | Requires suitable demonstrations and evaluation outside the training cases | Learn pick/place skills from our baseline or teleoperation |
| Pretrained vision-language-action policy | A model such as SmolVLA, EVO1 or MolmoAct2 processes images, language and robot state and predicts action chunks | Reuses learned representations and potentially supports broader goals | Our two-arm action convention, cameras, tasks and hardware still need adaptation | Compare a compact candidate after we have data and a baseline |
| Reinforcement learning | A policy improves through trial, rewards and repeated simulation | Can optimize recovery or difficult contact behavior | Many rollouts, reward design and failure modes; expensive from scratch | Later improvement, preferably starting from imitation |
| Hierarchical hybrid | A supervisor chooses goals/skills; classical or learned controllers execute them; observations verify progress | Combines understandable task logic with adaptable skills | Requires clear component interfaces and coordination | Preferred direction for the eventual system |

Mink supplies relevant IK tasks and constraints, but does not by itself solve grasp selection or global collision-free path planning. Nexus illustrates a demonstration → behavior cloning → reinforcement-learning workflow. Recent Gemini Robotics systems illustrate a split between high-level reasoning and motor-control models. These are supporting examples, not proof that our specific task will succeed. Sources: [Mink tasks](https://kevinzakka.github.io/mink/api/tasks.html), [Mink limits](https://kevinzakka.github.io/mink/api/limits.html), [SO101-Nexus](https://github.com/johnsutor/so101-nexus), [Gemini Robotics 2](https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/). Model-specific evidence and caveats are in the [research report](RESEARCH.md).

## What a goal needs to mean

An instruction must resolve to an object, a desired final state, and constraints. For example, a transfer goal would identify the source tube, a free target slot, allowed arms, and completion conditions. Natural-language input can be added later; a structured goal selector is enough for the first controller.

Completion should mean the tube is in the intended slot, sufficiently upright and stable after release, with the gripper withdrawn. Merely reaching a target gripper position or closing the fingers is not proof of successful manipulation.

The control loop is:

**Goal → observe → choose the next skill → plan motion → execute → verify → continue, recover, or report failure.**

## Proposed implementation stages

### 1. Verify the manipulation geometry

Check reachable grasp positions, gripper opening, clearance above the rack, tube/rack contact geometry and a feasible approach direction. Our arms have five arm joints plus a gripper, so arbitrary six-dimensional gripper poses cannot generally be imposed. Start with achievable position and partial-orientation targets. Adjust the learning scene if necessary; especially the back rack may be outside a useful manipulation workspace.

Use the exact MuJoCo object state for this stage. That is deliberate simulator ground truth for debugging, not a claim of camera-based intelligence. The free tubes should move through physical grasp/contact forces; keep any artificial attachment experiments separately labeled.

### 2. Implement reusable movement and grasping skills

Create tool-position control using Mink or a comparable IK solver, joint/velocity limits, and short collision-checked trajectories. Initially use simple approach/lift/transit/retract waypoints; introduce a global planner if those cannot navigate the environment.

Useful skills are: move above an object, approach a grasp, open/close the gripper, verify a grasp, lift, move to a destination, lower, release and retract. Recompute targets from the current scene, rather than recording one sequence of joint angles for one seed.

A finite-state controller should advance only after each stage's condition passes. On failure, it can stop, retreat, retry a bounded number of times or report an unreachable goal. A behavior tree can become useful when recovery branches grow; neither framework requires an LLM.

### 3. Add two-arm coordination

First let one scheduler assign independent reachable tasks to the two arms, with the inactive arm parked or a shared region reserved. Then add a hand-off with explicit stages: giver reaches a transfer pose, receiver grasps, receiver support is verified, giver releases, and both withdraw.

Running two unrelated single-arm policies simultaneously does not establish coordinated manipulation. Shared objects, collision checks, arm roles and hand-off timing need a common coordinator or a jointly trained policy.

### 4. Add perception and language

Replace ground-truth object poses with camera-based estimates while preserving the motion-controller interface. Start with simple identifiable objects and controlled views; use overhead/wrist observations to reduce occlusion. Test what changes when the estimates are imperfect.

An LLM or VLM can later resolve a command into grounded objects and allowed skills. It should choose operations such as pick, place or hand-off through validated interfaces. Continuous joint commands belong in the motion controller or robot policy. Speechmatics can provide the spoken-input layer once this goal interface works.

### 5. Learn and compare

Record synchronized images, joint state, instructions, actions, seeds and outcomes from successful and failed attempts. Use these to compare a compact VLA or imitation policy against the classical baseline on the same held-out tasks. Add corrective demonstrations for failures. Consider RL or residual corrections after a reliable starting policy and task evaluator exist.

Ground-truth state may remain in an independent evaluator even when the tested policy only sees images and proprioception. Document that distinction explicitly. A simulator-state baseline is useful preparation but does not alone demonstrate the final hackathon's vision/language/policy requirements.

## Engineering priorities for this sandbox

- Separate physics/control timing from camera rendering. Slow browser visualization should not set the rate of the autonomous control loop.
- Keep the 12-action joint/gripper convention, units, limits and left/right ordering explicit.
- Log the active task stage, goal, observed state, failure reason and success conditions.
- Evaluate full success, grasp retention, placement, collisions, retries, elapsed time and sensitivity to randomized layouts.
- Begin with reachable, well-separated objects. Introduce rack rotation, visual clutter and more difficult coordination progressively.
- Measure learned-model memory and latency on qualifying Intel hardware before committing to a policy architecture.

The first two milestones are complete: **verified physical lift-and-return and rack-to-rack transfer**, with progress, cancellation, bounded failure handling and synchronized demonstration recording. Next, expand beyond the narrow practice layouts and establish a shared transfer region for a coordinated two-arm hand-off; perception can then replace exact simulator object state behind the same goal interface.
