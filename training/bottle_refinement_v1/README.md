# Exact compact bottle-refinement input

`prepared-training` contains the exact 24,576 FP16 crops used to fit the network, grouped by 64 scene states. `prepared-development` contains all 3,072 development crops. Concatenate each named array along axis zero in manifest order to recover the original arrays; verify their dtypes, shapes and logical SHA-256 digests. Every shard was compared element-for-element with the original run. No single shard exceeds the repository's file-size limit.

`training`, `development` and `evaluation` retain all 5,120 scene/joint/light recipes, projection/visibility/geometry labels and every original RGB image hash, plus representative images and portable scenes. Regeneration uses the repository assets, the fixed camera protocol and the frozen collector's rendering settings. Labels and masks are privileged offline supervision/scoring only. Full RGB archives remain local and are not duplicated here.

Training source and original fit settings are preserved in [the evidence package](../../docs/robotics/evidence/bottle-refinement-v1/README.md). The model failed its fresh perception gate; its final split is exposed and must not be described as a new test when reproduced. Original physical V1 seeds remain unexposed. License: MIT.
