# Submission licensing check

Checked September 13, 2026 against the rendered event page and current terms. Talos keeps its existing MIT license for original code, simulation-derived weights, training arrays and dinner assets. No repository license has been changed.

## Organizer requirement

The [event page](https://lablab.ai/ai-hackathons/ai-infra-summit-hackathon), in the prizes section, requires original, MIT-compliant submissions. [Terms of Use section 16](https://lablab.ai/terms-of-use#16-participation-terms), showing revision July 1, 2025, states:

> All submissions by participants must be original work, open source, and compliant with the MIT License unless specified otherwise.

No Intel-track exception to this requirement was found in the five-page online challenge brief. The phrase “MIT-compliant” does not explicitly define how the organizer treats every possible third-party model license. We should not replace Talos's MIT license with an Apache-only project license based on an assumed exception.

## Apache dependencies and model weights

Our practical approach is to license original Talos contributions under MIT and preserve each third-party component's own license. This is a licensing assessment, not organizer approval of a particular pretrained model.

[Apache-2.0 section 4](https://www.apache.org/licenses/LICENSE-2.0#redistribution) permits redistribution under its conditions: provide the Apache license, retain applicable attribution and copyright notices, identify modifications, and preserve any required NOTICE material. It allows different terms for modifications or the combined work while retaining the original obligations. An MIT top-level LICENSE does not erase Apache obligations or turn third-party weights into MIT weights.

The current SO-101 assets already follow this separation: [asset notice](../../simulation_lab/NOTICE.md), [Apache license](../../simulation_lab/assets/so101/LICENSE), and pinned provenance. Original Talos code and weights use [MIT](../../LICENSE).

For any pretrained policy, inspect the exact checkpoint's model card and license as well as the implementation. Apache-licensed training code alone does not establish that its pretrained weights are Apache-licensed. Record the pinned upstream revision, license, base-model dependencies, modifications and redistribution conditions before adding a model to the public release. Noncommercial or custom-restricted weights need separate scrutiny; do not call them MIT/Apache merely because source code is available.

## Release action

- Retain Talos's existing MIT license and all current asset notices.
- Include exact model provenance and upstream licenses for any newly incorporated pretrained weights; do not publish experimental downloads indiscriminately.
- Keep this check in the final submission checklist. If the authenticated form imposes a stricter condition, reconcile it before the user's final upload.

The public event page also now explicitly displays the submission cutoff as September 16, 20:30 CEST. Our user-requested completion and handover target remains September 14.
