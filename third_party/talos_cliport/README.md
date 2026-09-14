# Selected CLIPort reference components

These source modules come from [CLIPort](https://github.com/cliport/cliport) at the revision and individual source hashes in `UPSTREAM.json`. Keep `LICENSE-CLIPORT` (Apache-2.0) and `LICENSE-CLIP` (MIT) with redistributions. Talos's original code license does not replace these notices.

The CLIP implementation/tokenizer and published RN50 representation originate in OpenAI CLIP. The RN50 download is explicit, SHA256 verified and stored outside Git. It is a pretrained vision/language representation, **not a pretrained robotics policy**.

The default upstream decoder imports a GPL-3.0 U-Net helper. That file was neither downloaded nor copied. `support.UpsampleMerge` is an independently authored MIT implementation of the required standard expansion/skip/conv interface. Other adaptations are recorded in the provenance manifest. A single frozen CLIP backbone is shared by the streams, stays in evaluation mode, and never downloads weights implicitly. This is an adapted reference architecture, not an exact upstream reproduction.

Only the two selected lateral streams and their dependencies are included. Training uses ordinary PyTorch; the broader upstream simulation, dataset and Lightning/Hydra stack are not required. Install `requirements-spatial.txt` into Talos's documented CUDA training environment.
