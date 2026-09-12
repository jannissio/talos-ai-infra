# Talos submission text

**Title:** Talos: Spoken Dinner-Table Robotics

**Short description:** Two SO-101 arms set a simulated dinner table from spoken or typed instructions, with Speechmatics, contact-verified physical skills, a two-arm bottle relay and an experimental OpenVINO bottle policy.

**Primary track:** Intel online challenge

**Bonus:** Best Use of Speechmatics

**Technologies:** MuJoCo, SO-101, Python, PyTorch, OpenVINO, Speechmatics, FastAPI, JavaScript

## Project description

Talos explores how spoken instructions can become physically verified robot actions. A browser connects to a local MuJoCo dinner-table simulation with two SO-101 arms, plates, vessels and a passive cutlery drawer. Users can type an instruction or use Speechmatics streaming transcription. A constrained language interpreter selects supported actions and reports progress, cancellation and failures.

The programmed baseline places the bottle, plate and mug, opens the drawer and retrieves the fork and spoon. A table-supported relay lets the left arm release a bottle in a shared area before the right arm regrips and places it farther right. Objects move through actuator forces and finger contacts, without attachments or teleportation.

An experimental neural bottle primitive uses initial camera images and motor feedback, with FP32 OpenVINO inference verified on an Intel CPU and Intel UHD graphics. The revised upright model completed ten of ten new task-preset scene tests, with placement errors below 1.7 mm. An earlier three-view model completed zero of ten broader scene tests, and those failures remain documented. General placement coverage, continuous visual correction and learned dish manipulation remain development targets. These limits are part of the evidence.

Talos provides six camera views, four-view HD recordings, reproducible scripts and explicit model and asset provenance. It targets developers learning to build, evaluate and improve language-guided manipulation systems. Speechmatics supplies actual speech recognition in the command-to-action path; end-to-end recorded tests use synthetic English speech.

## Application and evidence

- Repository: https://github.com/jannissio/talos-ai-infra
- Application: runs locally with the repository setup commands. A public hosted interactive simulation has not been deployed. A localhost URL is not a public demo link.
- Presentation: `Talos.pdf`; editable source: `Talos.pptx`.
- Cover: `cover.png` (16:9).
- Demonstration: `Talos-demo.mp4`. Accelerated programmed segments carry speed labels; learned motion uses real simulation timing.
- Ten-seed recording: `Talos-ten-seeds.mp4`, showing all ten recorded physical trajectories together at their original simulation speed.
- Metrics: `evidence/`, including every outcome of the ten-seed bottle evaluation.

## Scope disclosures

The language interpreter is a bounded grammar. Programmed skills use simulator positions, while the learned primitive uses visual features and motor feedback. Its independent privileged-state monitor only stops or scores execution. The relay uses the table between grasps and is verified left to right. No simultaneous airborne handoff, liquid pouring or physical robot deployment is claimed. Current rendering uses NVIDIA, and Core Ultra hardware has not been tested. Pre-event infrastructure reuse and AI assistance are disclosed in `docs/PROVENANCE.md`.
