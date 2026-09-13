# Source at the frozen dinner evaluation

The three Python files here match the SHA-256 values in the parent `freeze.json` and were preserved before adding the hosted GPU policy factory. They are evidence copies, not modules imported by the application.

The subsequent application change adds an optional `policy_factory`; the normal CPU path still defaults to `PrimitivePolicy`. A full exposed seed-42 CPU regression passed after that change (247.655 simulated seconds, 34.36 wall seconds). A separate RTX 4070 worker check passed all six skills with six fresh CUDA trajectory calls. These regressions do not replace or relabel the frozen ten-seed results.

The frozen benchmark describes the exact evaluated source and models. New hosting error messages and experimental RGB observer modules are outside that frozen physical evaluation.
