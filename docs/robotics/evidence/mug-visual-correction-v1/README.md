# mug-visual-correction-v1: complete physical comparison

All 24 experimental variants fail before the first movement because renderer replacement produces black camera frames. The unchanged baseline passes all 12 complete workflows. No mug correction was attempted. The independent renderer diagnostic uses an exposed saved state, with physics stepping disabled: closing the old renderer before creating the replacement exactly restores all three RGB images and the initial neural output. That setup repair requires a separate protocol; it does not repair or promote this stopped result.

| Mode | Standard workflows | Left-reach workflows | Standard mugs | Left-reach mugs |
| --- | --- | --- | --- | --- |
| baseline | 6/6 | 6/6 | 6/6 | 6/6 |
| live | 0/6 | 0/6 | 0/6 | 0/6 |
| frozen | 0/6 | 0/6 | 0/6 | 0/6 |

The latest completed split is **development**: 36 attempted workflows, 63,734 retained state frames, 0 visual correction queries. Its gate **fails**. Failed checks: paired_prefixes_before_correction, minimum_live_mug_success, no_mug_regression_against_baseline, no_workflow_regression_against_baseline, live_feedback_benefit.

The [protocol](protocol.json), [development results](development-audit.json), every scene/report/state trace, exact frozen runtime sources and console provenance are retained. Original XML files are stored beside the portable scenes. Package verification recounts outcomes and compares prefixes before the first correction, preserving mismatches as a failed criterion. Earlier missing-test-runner evidence is kept where applicable; no test ran during that failed launcher.

Models remain in [the stereo observer package](../../../../models/mug_keypoint_observer_v1/README.md) and [the limited motor package](../../../../models/mug_correction_motor_v2/README.md). Their manifests and the selected dinner suite are frozen in development-freeze.json. This experiment creates no replacement weights. The selected dinner/browser/hosted baseline remains unchanged pending a separate promotion decision.

Run the package-only integrity and paired-result check from the repository:

```powershell
.\.venv-training\Scripts\python scripts/package_mug_visual_correction.py --protocol docs/robotics/experiments/mug-visual-correction-v1.json --verify-only
```

No broad workspace, arbitrary-language, Intel-hardware or human-voice claim follows from this finite comparison. GitHub and Hugging Face remain private until the final release. Final Submit is exclusively the user's action.
