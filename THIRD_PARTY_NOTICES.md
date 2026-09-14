# Third-party notices

Original Talos code and original procedural assets use the repository's MIT license. Separately licensed components retain their own terms:

- SO-101 robot assets: Apache-2.0, with source revision and original notices in [simulation_lab/NOTICE.md](simulation_lab/NOTICE.md).
- Selected CLIPort spatial-policy source: Apache-2.0, with the pinned source manifest, modifications and license in [third_party/talos_cliport](third_party/talos_cliport/README.md).
- OpenAI CLIP implementation, tokenizer, vocabulary and optional RN50 representation: MIT; the upstream license is retained as [LICENSE-CLIP](third_party/talos_cliport/LICENSE-CLIP). Downloaded weights remain outside Git and are verified against their published SHA256.

CLIPort's GPL-3.0 U-Net helper is not included. The decoder interface uses an independently authored Talos helper, as explained in the component README. This adaptation is not an exact reproduction of the original architecture or evidence of a pretrained Talos robotics capability. Installed Python dependencies retain their upstream licenses.
