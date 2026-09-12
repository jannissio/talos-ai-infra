# Live learned bottle control

**Historical baseline:** the measurements below describe the original three-view model, now selected as **Original bottle model · sideways practice**. The default is now **Learned upright bottle · OpenVINO**; see [current model, ten-seed results and scope](VISUAL_BOTTLE_POLICY.md). Both modes use the same live task and independent safety monitor. The current monitor also checks unexpected arm contacts; the old initialized collision counter was not a measurement.

The main application accepts typed/Speechmatics commands selecting an explicit learned bottle mode. Start the training environment with `python -m simulation_lab.server --learned`; select Learned bottle and use `place the bottle` after a seed-42 reset.

PyTorch is imported before accepting requests; OpenVINO compilation happens on the physics thread that owns inference. Loading native dependencies during a request produced a timeout. Precompiling a throwaway model on a different thread produced large live inference overhead. The corrected path accepted a request in 1.39 seconds, processed policy queries at a median of about 7.6 ms and sustained a simulation/real-time ratio of about 1.0 in a live sample. These are local observations, not universal hardware guarantees.

## Boundaries

`LearnedBottleTask` uses 200 Hz physics, 20 Hz target endpoints and 5 Hz neural queries. Policy cameras retain their trained 320×240 resolution and are independent of the browser display. Initial images condition a trajectory; measured joints control progress. There is no continuous visual correction.

An independent simulator-state monitor checks finger support, a >=5 cm lift, hold, release, stable placement, other-object disturbance and the parked arm. It can stop execution but never provides action targets. No teacher updates, action retrieval, state restoration or attachment occurs during learned control. Cancellation and failure pause physics.

Unsupported learned requests—including plate movement, arbitrary destinations and full-table sequences—return explicit errors. API checks confirmed that these refusals leave physical and task state unchanged. Programmed mode remains separate.

## Evidence

| Test | Result |
| --- | --- |
| Familiar upright, live task class | Passed; 6.84 cm lift, 1.73 mm placement error |
| Familiar sideways, same class | Passed; 9.34 cm lift, 0.24 mm placement error |
| Default seed-42 scene | Passed; 6.89 cm lift, 0.86 mm placement error |
| Browser application | Visible success; 27.5 simulated seconds; released bottle; explicit OpenVINO mode |
| Cancellation | Paused without state writes; renderer closed |
| Blank images | Rejected before movement |
| Speechmatics to learned task | Passed from a 1.78-second synthetic English sample; transcript `Place. The bottle.`; only punctuation/spacing normalized |
| Ten new full-scene seeds, 2026091201–2026091210 | **0/10** completed; every outcome retained |

New-seed failures included missed grasps, lost support and timeout. The eleven-demonstration model overfits its original scene distribution. These seeds are now exposed; future selection must use separate development cases and newly declared evaluation seeds.

## Reproduction

`scripts/verify_live_policy.py` tests the actual live controller class. `scripts/evaluate_live_seeds.py` declares seeds and source/checkpoint hashes before evaluation and saves every result with compact physical states. `scripts/render_state_video.py` renders four views at 720p/20 fps without changing the original rollout.

`scripts/collect_compact_bottle.py` collects broader training-only data with three initial images and a compact trajectory per episode. Both 200 Hz and interpolated 20 Hz physical replays must pass for training eligibility. This is an initial-image dataset, not continuous visual-feedback data. Writers preserve a 10 GiB free-space reserve. Logs, raw data, credentials and private audit originals remain outside Git.
