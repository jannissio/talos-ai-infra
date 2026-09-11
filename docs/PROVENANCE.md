# Development history and AI assistance

- **September 8, 2026:** research and the user-requested BenchLab chemistry learning experiment. Its reusable infrastructure includes the two-arm MuJoCo composition, browser viewer, physical tube lift/return and transfer teacher, fixed-step engine, independent renderer, and demonstration recording. No learned policy was trained.
- **September 10:** public challenge/schedule/resource recheck. Saved sponsor downloads are retained locally as reference material, excluded from Git, and not relicensed as project assets.
- **September 11:** user selected the published dinner-table task and supplied `https://github.com/jannissio/talos-ai-infra`. The existing remote initial commit and MIT license were retained. Dinner assets, scene presets, inventory, passive drawer, scene-specific API behavior, physical checks, and submission build plan were added.

Codex assisted with research, implementation, model geometry, debugging, documentation and verification. The project is work in progress; these contributions do not imply a trained policy, physical-hardware validation, or completed challenge.

SO-101 robot models originate from Google DeepMind MuJoCo Menagerie, pinned to commit `ac6b2b09983786f3036cab1000221017fa2193b4`. The original Apache-2.0 license, source files and provenance hashes are retained. Visual simplification and runtime composition are described in [NOTICE](../simulation_lab/NOTICE.md).

The dinner models are original procedural geometry under this repository's MIT license. No downloaded dinnerware model pack or generated image textures were used. Simulator mass/inertia/friction values are documented engineering approximations, not measurements of physical dishes.

Pre-event preparation reuse has not received organizer-specific approval. Preserve this dated disclosure in the eventual submission; do not describe all infrastructure as newly built during the event. The user's authorization to develop locally does not establish competition eligibility.


September 11 physical milestone: Codex-assisted implementation of the dinner teacher, drawer pull, rim and cutlery grasps, six-skill sequencing, browser controls and physical evaluation. The plate rim, utensil handles, drawer dimensions and object staging were revised after contact/clearance failures; these are original engineering approximations. No robot geometry changes, object attachments, external-force manipulation or learned-policy claims were introduced. See `robotics/DINNER_SCENE.md` and the dated challenge recheck.
