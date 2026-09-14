# Wider manipulation coverage: first diagnostic

All **100 cases / 200 paired rollouts** finish with no harness errors. The six learned anchor checks pass before broader exposure. Both controllers receive identical initial physical states in every case. All source versions, reports, failures and **88,760 recorded frames** are retained.

The following totals exclude the six anchors. There are **94 diagnostic cases**, **83 valid starts**, and **11 invalid arrangements**. The learned system succeeds in **11**; the separate programmed controller demonstrates physical success in **43**. There are **32** cases that are physically demonstrated by the programmed controller but fail under learned control.

| Skill | Planned | Valid starts | Learned successes | Programmed successes | Programmed pass / learned fail |
| --- | ---: | ---: | ---: | ---: | ---: |
| bottle | 28 | 26 | 2 | 17 | 15 |
| plate | 13 | 11 | 3 | 7 | 4 |
| mug | 13 | 12 | 3 | 8 | 5 |
| drawer | 14 | 14 | 1 | 1 | 0 |
| fork | 13 | 10 | 1 | 5 | 4 |
| spoon | 13 | 10 | 1 | 5 | 4 |

[The frozen protocol](protocol.json) contains every exact coordinate, orientation, destination and drawer configuration. [The independent audit](audit.json) contains every paired outcome and failure category. The [current broader goal](../../BROADER_MANIPULATION_PLAN.md) gives the operational target and next development steps.

These are isolated skill tests in one preceding-task context, with one factor changed at a time. They do not test arbitrary clutter combinations or independent randomized full sequences. Invalid resets are retained and reported separately; perception and planner refusals stay in the valid denominator. A failed exact-state controller does not prove a point unreachable. The mug uses the selected live correction in this isolated diagnostic harness; its public interface still permits that option only for full table setting. Destinations passed to physical scoring are not secretly supplied to an unchanged neural policy.

This result confirms that the narrow submission demonstration does not satisfy wider manipulation coverage. The existing models and demo are preserved as the fallback. There is no candidate promotion from this measurement alone.
