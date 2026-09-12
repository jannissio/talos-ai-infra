# Table-supported bottle relay

In programmed mode, enter **“pass the bottle to the right arm”** after loading Task start, seed 42. The command expands into two contact-verified skills: the left arm places the bottle at the shared table location `(0, -0.10)` m and parks, then the right arm regrips and places it at `(0.20, 0.05)` m.

The default physical test completes both legs in **76.325 simulated seconds**. The lifts reach **6.41 cm** and **6.55 cm**, with at least **1.8 seconds** of unsupported, two-finger contact at each hold. Final placement error is **1.45 mm**. The additional scene seed 2026091101 also passes. This is a programmed simulator-state controller, not a learned handoff policy.

The controller does not write object poses, apply external forces or create equality constraints. One arm parks before the other starts. The task checks occupied destinations, carrying clearance, joint tracking, finger support, disturbances and stable release. Cancellation pauses physics. Unsupported objects and the unverified reverse direction are refused before execution.

```powershell
.\.venv\Scripts\python scripts/run_command_demo.py --text "pass the bottle to the right arm" --output .run/relay-check --seconds 130
.\.venv\Scripts\python -m unittest tests.test_command_task
```

Use a fresh output folder. `submission/evidence/programmed-relay.json` records both legs. Reverse-direction development trials found occupied targets, drawer collisions and useful-reach limits. Those failures do not prove physical impossibility; further planning is needed. The verified relay is a table release and regrasp, not direct airborne exchange or general bimanual manipulation.
