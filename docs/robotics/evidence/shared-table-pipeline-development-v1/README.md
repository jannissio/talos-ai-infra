# Shared table pipeline: development evidence

Completed exploratory development attempts, not a frozen generalization evaluation. Repeated seeds are exposed. The shared visual policy has not been trained.

Retains 15 completed runs, 118 physical attempts and 300,126 state frames. There are 1 successful primitives and 0 completed randomized tables. These runs include explicitly labeled narrow positive controls; they are not broad success rates.

The adapted CLIPort pick-location and 36-rotation placement heads pass one numerical optimizer step each on a real 256×320×6 calibrated observation: 5.49 GiB peak Torch reservation. No robotics policy checkpoint is trained or exported. `spatial-runtime/report.json` binds the exact tested source.

Every attempted physical motor trace, failed search and run source snapshot is retained. 118/118 saved motor-command replays independently reproduce their recorded physics exactly; passing actions are additionally replayed into the continuously evolving main scene. The teacher uses privileged geometry and physics lookahead to generate demonstrations. It is not the learned controller and does not establish online recovery.

Original raw RGB-D archives remain unchanged locally. This compact export retains canonical array hashes, calibration/scene/state recipes and every preview; raw images are not duplicated.

See `summary.json` for every run, the invalid starts, unfinished local jobs and counts. Original outputs are never overwritten. GitHub/Hugging Face stay private; final Submit remains the user’s action.
