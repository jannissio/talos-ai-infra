# Exposed mug placement diagnosis

All twelve baseline mug traces from the completed spoon-release development comparison are inspected: six declared seeds in both starting presets, including every success and failure. This is recorded-state diagnosis, with **no new training or physical trial**.

The two failing seeds show different behavior. Seed 2026099201 is already 8.35 mm off target at the alignment probe, before lowering, and finishes at 8.52 mm. Seed 2026099204 is within 2.57 mm before release but shifts to approximately 8.01 mm after release. Both presets reproduce these patterns. This supports investigating both the trajectory fit and the opening/withdrawal sequence; a release-only explanation is insufficient.

The preserved mug network's training-endpoint tool error reaches p95 **1.517 mm during alignment, 1.762 mm during lowering and 1.875 mm during release**, using the actual right-gripper mug grasp point `[-0.006, 0, -0.092]` m. These errors describe agreement with teacher action labels, not physical placement accuracy. All 41 original training episodes remain eligible, with their prior physical replay gate intact.

Initial feature distances to the nearest training example range from 0.065 to 0.168 in normalized feature units. Successful cases include larger distances than either failure, so this diagnostic does not establish a simple out-of-support explanation. Measured grasp offsets and mug rotation remain in [the full report](diagnostic.json).

- [Protocol](dinner-mug-diagnostic-v1.json), [exact diagnostic source](diagnose_dinner_mug.py), twelve initial RGB composites and [package manifest](package-manifest.json).
- [Independent audit](audit.json): all 72 geometric probes reconstructed from the original published state arrays, with zero discrepancy.
- The initial diagnostic invocation failed on an incorrect single-geom goal name after saving one initial image. The corrected code reads the actual forty-segment mug marker. Its source, image and sanitized console output are retained under `first-attempt/`; no model, physical rollout or scored outcome was replaced.

The [separately declared next experiment](../../experiments/mug-release-v1.json) tests a weighted warm fit and a stationary finger-opening interval. Every revised demonstration must pass physical replay before fitting, followed by new paired full-workflow development/final gates. This diagnosis itself selects no candidate and changes no submitted or hosted model. Preserve private access, the 10 GiB reserve and the user's sole control of Submit.
