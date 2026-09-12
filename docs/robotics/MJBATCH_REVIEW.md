# mjbatch relevance and source review

## Assessment

**Relevant experimental infrastructure; no obvious malicious behavior found in the inspected core source and build configuration. Not certified malware-free, and not suitable for installation into BenchLab's current Windows environment.**

This static assessment was performed September 11, 2026 against commit [`e211c84a88d191039403150eb6a220e23d4a2ad5`](https://github.com/kevinzakka/mjbatch/tree/e211c84a88d191039403150eb6a220e23d4a2ad5). Repository code was not imported, built, installed or executed. No dependency packages, pretrained weights or example binary assets were downloaded. Forty-three text files were downloaded into the ignored `.run/research/mjbatch/` directory for inspection.

The conclusion applies to the reviewed snapshot, not future commits, dependency releases, or independently distributed wheels. No antivirus scan, dependency vulnerability audit, native-memory-safety audit or binary-to-source provenance verification was performed.

All 43 downloaded text files were checked against their Git blob hashes from the pinned repository tree. A local SHA-256 inventory is saved at `.run/research/mjbatch/download-manifest.json`. This confirms snapshot consistency; it does not establish that the contents are safe.

## Provenance and maturity

The repository belongs to `kevinzakka`, the account linked from [Kevin Zakka's research website](https://kzakka.com/). His published work and software include MuJoCo Playground, Menagerie and Mink. This is a meaningful provenance signal, but reputation does not establish that every current artifact is uncompromised.

The [GitHub repository API](https://api.github.com/repos/kevinzakka/mjbatch) reported creation on **September 10, 2026 at 02:19 UTC**. The inspected main commit is dated **September 11 at 01:14 UTC**. GitHub reports that commit as unsigned; that is not evidence of malware, but it provides no cryptographic author signature. The repository has a short five-commit history at inspection.

GitHub reports [CI success for the inspected commit](https://github.com/kevinzakka/mjbatch/actions/runs/34550188246). Its [workflow](https://github.com/kevinzakka/mjbatch/blob/e211c84a88d191039403150eb6a220e23d4a2ad5/.github/workflows/ci.yml) tests Linux/macOS, pins referenced actions by commit, and configures release provenance attestation and PyPI publishing. A configured attestation step is not proof that a particular downloaded wheel was verified. CI tests functional behavior; they do not certify absence of malware.

The [PyPI metadata](https://pypi.org/pypi/mjbatch/json) reports version 0.1.0, uploaded September 10, with Linux/macOS wheels and a source archive. The published files predate the inspected main commit; do not assume their source is identical. License: Apache-2.0.

## What was inspected

| Area | Observation | Interpretation |
|---|---|---|
| Build entry points | `pyproject.toml`, `CMakeLists.txt`, Makefile | Conventional scikit-build/nanobind C++ extension; no custom `setup.py` payload in the tree |
| CMake commands | Locates nanobind and the installed MuJoCo wheel; adjusts macOS library linkage | Expected build operations, but building imports dependencies and executes build tools |
| Python import path | Small dataclass/accessor wrapper imports MuJoCo, NumPy and `_bindings` | No network calls, credential access, persistence or shell execution found in this wrapper |
| Native core | Read `batch.h`, `bindings.cpp` and `threadpool.h` | Simulation state copying, parameter binding, thread scheduling and error handling; no obvious exfiltration or filesystem modification logic found |
| Dependency declarations | MuJoCo 3.11.0 and NumPy at runtime; scikit-build-core and nanobind for building | Dependencies remain separate trust surfaces; some version ranges are open-ended |
| Lockfile URLs | PyPI/Pythonhosted and PyTorch distribution domains | No unexpected package host found in the URL scan; transitive package code was not audited |
| Examples/developer scripts | Searched executable text for network/process execution, unsafe deserialization and destructive filesystem operations; reviewed matching locations | Normal simulation code and local example output, with specific caveats below |

Pinned source: [package/build configuration](https://github.com/kevinzakka/mjbatch/blob/e211c84a88d191039403150eb6a220e23d4a2ad5/pyproject.toml), [CMake](https://github.com/kevinzakka/mjbatch/blob/e211c84a88d191039403150eb6a220e23d4a2ad5/CMakeLists.txt), [Python wrapper](https://github.com/kevinzakka/mjbatch/blob/e211c84a88d191039403150eb6a220e23d4a2ad5/src/mjbatch/__init__.py), [native core](https://github.com/kevinzakka/mjbatch/tree/e211c84a88d191039403150eb6a220e23d4a2ad5/src/mjbatch/csrc).

### Concrete caveats

The native extension installs a process-wide MuJoCo log handler on import. It uses raw pointers and memory copies, and exposes writable simulation buffers. These are understandable implementation choices, but malformed inputs, memory bugs or misuse could crash a process. No exploitability or complete memory-safety determination was made.

The [Go1 example](https://github.com/kevinzakka/mjbatch/blob/e211c84a88d191039403150eb6a220e23d4a2ad5/examples/go1_joystick.py) loads a `.pt` checkpoint through `torch.load` without explicitly specifying `weights_only`. Its pinned PyTorch is 2.9.0, whose default restricted loading behavior should be preserved and verified if that example is ever used. The checkpoint itself was not downloaded or inspected. Other examples load NumPy archives or write local plans/checkpoints. Those examples are unnecessary for our training plan.

The developer [stub postprocessor](https://github.com/kevinzakka/mjbatch/blob/e211c84a88d191039403150eb6a220e23d4a2ad5/typings/postprocess_stubs.py) deletes named generated backend-stub files/directories under its own `typings/mujoco` location. This is not a broad user-file deletion operation in the inspected source. It was not run. The Makefile's formatting targets intentionally edit repository source files.

PyTorch's [version 2.9 serialization documentation](https://docs.pytorch.org/docs/2.9/notes/serialization.html#torch-load-with-weights-only-true) describes the restricted-loading default introduced in 2.6. Restricted loading reduces code-execution exposure; it is not a general safety certificate for arbitrary model files.

Default thread selection uses all logical CPUs up to the simulation count. A large batch can make a laptop unresponsive or consume substantial memory. That would be resource saturation, not by itself evidence of a virus. Any future benchmark should start with small, explicitly limited thread and batch counts.

## Compatibility with BenchLab

There are two immediate, verified incompatibilities:

1. The project explicitly excludes Windows wheel builds (`*-win*`) and documents its native-library linking limitation. Its advertised OS classifiers and CI are Linux/macOS.
2. Both build and runtime dependencies require **MuJoCo == 3.11.0**. BenchLab is verified on **3.12.0**. Installing it into the active environment would attempt a downgrade or produce a resolver conflict. The native module also checks compiled/runtime MuJoCo version agreement.

Do not remove the version pin and assume ABI compatibility. An isolated Linux environment, such as an existing Colab runtime, would be a more sensible experiment than altering the current Windows setup. A virtual environment prevents package conflicts but is not a security sandbox. If stronger isolation is desired for execution, use a disposable VM/runtime without project secrets or mounted personal storage.

## Relevance to our training strategy

The [README](https://github.com/kevinzakka/mjbatch/blob/e211c84a88d191039403150eb6a220e23d4a2ad5/README.md) describes CPU-parallel MuJoCo stepping with a C++ thread pool and per-simulation parameter variation. Potential uses are:

- Generate or evaluate many physical rollouts under mass/friction changes.
- Search short action sequences with model-predictive control.
- Run targeted reinforcement-learning experiments after a useful starting policy exists.

It does **not** replace ACT/SmolVLA, supply a dinner-task reward, or automatically batch our Python IK/state machine and camera renderer. A claim that a Go1 learns locomotion quickly is an example-specific performance report, not evidence for the cost of learning contact-rich two-arm manipulation from pixels.

The core batches copies of one model topology; expanded parameters are useful for randomization, but arbitrary different object/mesh topologies are not automatically interchangeable. Derived fields may lag the integrated state after stepping unless `forward=True` is selected. This matters for aligned camera/contact observations and must be accounted for in an adapter. Physics speedup alone may leave IK, rendering or video encoding as the dominant cost.

## Recommended decision

**Continue first with the existing MuJoCo engine and the imitation-learning plan.** Profile demonstration generation. If physics is the bottleneck, compare a small native-process worker pool, MuJoCo's built-in [threaded rollout facility](https://mujoco.readthedocs.io/en/stable/python.html#rollout), and an isolated `mjbatch` experiment. Built-in rollout is particularly useful for supplied control sequences; it does not automatically execute our closed-loop teacher either.

A later `mjbatch` benchmark should use a pinned snapshot, independently checked package provenance and dependencies, batches 1/8/32 and limited threads. Compare physical outcomes, memory, total accepted demonstrations per wall-clock minute, and rendering-inclusive throughput. Because testing 3.11.0 against our 3.12.0 baseline also changes physics version, separate version effects from batching effects. Retain the normal single-scene runtime for the browser and final demonstration.

**No installation or runtime change was made during this review.** The source review supports considering an isolated experiment; it does not support calling the repository malware-free or our Windows integration validated.
