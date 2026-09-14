# Refined bottle control: exposed physical coverage

All **58 live/frozen rollouts across 29 bottle cases** finish with every outcome retained. There are **27 valid starts** per condition. Live control succeeds in **25**, frozen images in **9**, the preserved preset controller in **3**, and the separate programmed comparison in **18**. All starts exactly match the original comparison traces.

| Case family | Planned | Valid | Live successes | Frozen successes | Preset successes | Programmed successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| anchor | 1 | 1 | 1 | 0 | 1 | 1 |
| destination | 2 | 2 | 1 | 0 | 0 | 0 |
| orientation | 2 | 2 | 2 | 0 | 2 | 2 |
| position | 24 | 22 | 21 | 9 | 0 | 15 |

The controller uses the frozen coarse/refined RGB observer, the V5 neural motor map and unchanged V2 R1 route selection. It may use one arm or a table-supported release/park/regrasp transfer. Live images update approach/descent; carrying follows the closure-time observed offset and motor feedback. The frozen condition uses the initial image separately for each leg. No exact object pose, teacher action, inverse solver or hidden force generates learned motor targets.

These are **training-exposed diagnostic cases**, with one recorded preceding-task context and one varied factor at a time. They are not new-scene generalization, combined arrangements or a promotion test. The preceding synthetic perception gate remains failed; its reserved physical stages remain unexposed. A new bounded physical protocol must be frozen before any fresh success claim.

The independent audit retains **40,396 state frames**, **20,194 RGB observations** and **40 successful route legs**. All invalid starts and failures remain in the planned denominators. The stopped V1 harness (implicit destination, before control) and its source are retained alongside V2; the repaired destination binding was checked against all 29 original monitors. Sources, model hashes, portable scenes, original reports and representative views are available in the manifest. No selected browser model is changed by this diagnosis.
