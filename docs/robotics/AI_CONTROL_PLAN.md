# From the working simulator to learned arm control

**Current decision:** [Evidence-based dinner training plan](TRAINING_RESEARCH_PLAN.md), September 11, 2026, supersedes the next-training priorities below. It specifies data contracts, ACT/SmolVLA experiments, corrective collection, bimanual training, Intel gates and recent literature. The physical dinner teacher is now implemented. See also the [mjbatch source and compatibility review](MJBATCH_REVIEW.md).

**September 11 scope update:** the user selected the dinner-table scenario. The [dinner scene](DINNER_SCENE.md) now exists; the next physical teacher skill is bottle grasp/lift/place. References to tube tasks below describe the preparation experiment, not the chosen submission task. ACT/SmolVLA and Colab/RTX 4070 remain candidate training paths; use [the updated build plan](../hackathon/BUILD_PLAN.md) for sequencing.

Decision note, September 8, 2026. This is a proposed next phase; no policy has been trained or deployed by this note.

## Recommendation

Use the working controller as a **teacher**, then train **ACT** as the first camera-based movement policy. Keep **SmolVLA** as the leading compact language-conditioned alternative for the hackathon. Choose the final policy after testing task success and Intel deployment, rather than committing to a large model before those tests.

**Hardware update from the user:** Google Colab is available now, and an RTX 4070 with 12 GB VRAM will be available at home from Sunday, September 13. This makes SmolVLA adaptation a practical target to test. ACT remains the quick baseline; SmolVLA is the main language-aware candidate. These are separate policies that can use the same prepared demonstrations, not successive versions of one trained model.

The broader architecture is a supervisor plus movement skills: interpret the request, select a task, execute a learned skill, and check the result. The first ACT experiment should learn one clearly defined transfer. Stock ACT does not itself interpret arbitrary language or select an arbitrary named tube/slot.

## What changes

BenchLab currently calculates movements using programmed logic and exact object coordinates from MuJoCo. It is autonomous, but those movements are not produced by a trained neural policy. The existing demonstrations show that the mechanics and recording pipeline work; they do not demonstrate learned visual control.

For learned control, the policy receives camera images and robot joint angles and predicts a short sequence of joint/gripper targets. It executes part of that sequence, observes again, and updates its movements. This repeated observation matters when a grasp or object position differs from the demonstration.

Training teaches the model from examples. Inference means using the trained model during the live simulation. These have different hardware requirements. The policy does not need to make 200 separate image-based decisions each second: the low-level simulator/controller retains its 5 ms steps while policy updates run at a separately measured rate.

## Options and feasibility

| Option | Meaning | Assessment for this project |
| --- | --- | --- |
| AI perception/planning plus our current motion controller | Recognize objects or interpret instructions, then call programmed pick/place skills | Feasible incremental route and useful comparison. The arm movement is still supplied by the classical controller; by itself this does not establish a learned manipulation policy. |
| ACT imitation learning | Learn the relation between camera views, joint state and future movements from demonstrations | Recommended first learned motor policy. Small and focused; initially use one task. Natural-language and arbitrary-target support require a separate interface or additional conditioning. |
| Compact pretrained VLA, starting with SmolVLA | Adapt a model that combines pictures, language and robot state to produce actions | Strong candidate for the language-driven system, with more training and deployment work. A pretrained checkpoint still needs adaptation to our scene and action convention. |
| Diffusion policy | Learn possible movement sequences by repeatedly refining a prediction | A reasonable imitation-learning alternative if ACT struggles with more varied actions. Adds computational cost; not the first comparison to implement. |
| Reinforcement learning | Improve through attempts, rewards and failures in simulation | Useful later for recovery or improving an existing skill. Starting from scratch also requires reward design, many rollouts and protection against exploiting simulator shortcuts. |
| Larger VLAs, such as MolmoAct2 or Pi0.5 | Adapt a more expensive pretrained robot model | Poor initial fit for this laptop and the current small dataset. Reconsider only if task evidence and available hardware justify the extra work. |

These feasibility judgments are engineering recommendations, not measured policy results from BenchLab.

ACT is an older, established baseline. Current LeRobot documentation describes an approximately 80-million-parameter model that maps images and joint positions to action chunks. That makes it a useful first experiment without needing language learning at the same time. [LeRobot ACT documentation](https://huggingface.co/docs/lerobot/act).

SmolVLA's current documentation describes a pretrained 450-million-parameter model using images, language and robot state. Its guide recommends task-specific fine-tuning and varied episodes; a ready checkpoint is not proof of reliable behavior in our dual-SO101 laboratory. [LeRobot SmolVLA documentation](https://huggingface.co/docs/lerobot/smolvla).

EVO1 remains a secondary compact-VLA candidate. Its current LeRobot implementation supports image/text/state inputs and staged adaptation, but we have not established an Intel export path for it. It should not create a third full training project before ACT and SmolVLA have been assessed. [LeRobot EVO1 documentation](https://huggingface.co/docs/lerobot/evo1).

For a concrete larger-model comparison, the documented MolmoAct2 LeRobot configuration reports 12.1 GiB for bf16 inference and higher memory for fine-tuning. Those are reference measurements on its specified NVIDIA setup, not results on this PC or on Intel hardware. [LeRobot MolmoAct2 documentation](https://huggingface.co/docs/lerobot/molmoact2).

## Hardware decision

Local read-only inspection confirmed **Quadro T2000 Max-Q, 4096 MiB GPU memory**. The earlier system inspection recorded an i7-10850H and 16 GiB system RAM.

The user additionally has **RTX 4070 / 12 GB VRAM at home from Sunday, September 13**, and **Google Colab access immediately**. We do not need to wait until Sunday to prepare data or begin the first training experiments.

LeRobot's current rough training guide groups light behavior-cloning policies including ACT at approximately 2–6 GB VRAM under its reference settings, diffusion policies at 8–14 GB, and SmolVLA at 10–16 GB. These are configuration-dependent envelopes, not promises for our multi-camera setup. [LeRobot compute hardware guide](https://huggingface.co/docs/lerobot/hardware_guide).

Our updated working decisions:

- Keep simulation, demonstration collection, data preparation and initial inference experiments on this PC.
- Prepare a reproducible Colab notebook for the first ACT training run and a SmolVLA memory/inference check. Inspect the assigned GPU and memory before choosing batch size and precision.
- Use the 12 GB RTX 4070 for ongoing ACT training and tests at home. SmolVLA fine-tuning is plausible with reduced batches and frozen components, but the exact camera/model configuration must be measured first. The 12 GB limit is not a promise that every recipe fits.
- Save resumable checkpoints so Colab runs can continue on the home machine. ACT weights do not become SmolVLA weights; dataset preparation and evaluation infrastructure are shared.
- Benchmark the selected policy on the final Core Ultra machine early. The existing Intel brief requires Core Ultra Series 2/3 for the final simulation and robotics inference; training can use other resources. [Saved Intel challenge summary](../hackathon/BRIEF.md).

Intel's current Physical AI Studio library lists ACT and SmolVLA, and shows an ACT-to-OpenVINO export path. Its runtime supports exported-policy inference. That is a useful integration lead, not proof that our exact model, quantization or selected CPU/GPU/NPU target already works. [Physical AI Studio library](https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/README.md), [Physical AI Runtime](https://github.com/openvinotoolkit/physicalai).

Colab GPU types, availability and runtime limits vary. A checkpoint/resume workflow is therefore part of the training plan. No specific Colab GPU allocation or paid subscription is assumed. The RTX 4070 provides training resources; it does not replace the separate final Intel deployment requirement. [Official Colab FAQ](https://research.google.com/colaboratory/faq.html).

## Concrete next experiment

1. **Define a first visual task.** Start with one arm transferring one visually identifiable tube to a clearly defined destination. Simulator IDs such as A2 are not printed in the camera images; arbitrary ID-based language cannot be inferred from identical-looking objects without an association mechanism.
2. **Collect diverse demonstrations.** A first pilot target is 50–100 successful episodes, with repeated examples across meaningfully different reachable positions and orientations. This is a starting experiment, not a guaranteed sufficient quantity. The current folder contains only a handful of successful recorded episodes; hundreds of correlated frames do not replace independent task examples. Current practice jitter of a few millimetres/degrees is too narrow to establish broad visual generalization.
3. **Prepare a consistent training dataset.** Convert the local recording schema to a pinned LeRobot-compatible format. The recordings contain 200 Hz actions, 20 Hz states and 5 Hz exported images. Choose a policy timebase explicitly; regenerate images from saved states at the required rate, rather than mixing mismatched timestamps or using future images. Train nominal joint targets and retain the actuator/gripper limiting layer.
4. **Train and test ACT on the fixed task.** Give the policy images and robot joint state, while keeping exact object coordinates in the teacher/evaluator. Test on held-out episodes and layouts, not neighboring frames from the same recording. Add changed-object-position tests to distinguish visual control from memorized motion.
5. **Run an Intel export/inference check early.** Confirm graph conversion, action agreement, memory and end-to-end latency before expensive training. Select CPU/GPU/NPU according to measured support and speed.
6. **Add language and coordination after the visual skill works.** For ACT, a separate supervisor can select learned skills; flexible target selection needs explicit conditioning and data. Compare SmolVLA when language-conditioned action learning is valuable and its hardware test is viable. Then add coordinated two-arm demonstrations and evaluate joint behavior.

A starting pilot size of this order is consistent with LeRobot's SmolVLA data-collection guidance, which stresses examples per variation. The actual amount needed for our tubes and goals must be established experimentally. [SmolVLA data guidance](https://huggingface.co/docs/lerobot/smolvla).

The existing classical controller remains valuable for collecting demonstrations, diagnosing failures and providing a separately labeled fallback. Learned-policy success must not secretly include the teacher correcting the movement. Ground-truth contact and outcome checks can remain part of the evaluator, with their privileged access documented.

## Language and the hackathon

Speechmatics would turn speech into text for the bonus. A local instruction/vision layer would ground a command in visible objects and select a task. A learned motor policy would execute the movement. SmolVLA can combine language and visual motor conditioning, while stock ACT covers the motor-learning part and needs that higher-level interface.

The chemistry scene remains a learning exercise. The published online challenge calls for dinner-table behavior, meaningful vision/language input, a trained/fine-tuned policy and coordinated two-arm performance on qualifying Intel hardware. A fixed ACT transfer alone is an intermediate milestone, not the complete competition entry. ACT is explicitly listed among the brief's policy examples, so a hierarchy with a learned ACT skill and a real instruction/vision layer is worth evaluating; final organizer clarifications still govern the entry. [Saved challenge brief](../hackathon/BRIEF.md).

The next development priority should therefore be **diverse data, dataset conversion and a Colab training notebook**, followed by the first camera-driven ACT experiment and a SmolVLA/Intel feasibility check. Continue on the RTX 4070 when available. Coordinated hand-off remains useful, but does not by itself answer whether we can learn to control from images.
