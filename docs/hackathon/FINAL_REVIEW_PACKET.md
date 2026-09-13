# Final review and handover

Prepared September 13. Internal completion target: **September 14, 18:00 CEST**, with time afterward for the user's review and corrections. The assistant must **never click the final submission button**. The user will submit personally.

## Ready for review

- [Saved submission text and media](../../submission/final-v6/PROJECT.md): all three form steps and required attachments are saved. Supplementary dinner and relay videos include every frozen trial. The form's filled-fields percentage is not submission confirmation.
- Selected system: six learned dinner skills, 8/10 frozen full sequences; table-supported bottle relays, 5/5 in each direction. Release/parking checks allow chaining without resets.
- Private hosted CPU demo: all six learned tasks pass on exposed seed 42. Full local browser at `http://127.0.0.1:8768/`; hosted Space at `https://huggingface.co/spaces/jannis-sms/talos-dinner-robotics`.
- [Experimental RGB correction](../robotics/evidence/rgb-servo-v1/README.md): all 48 outcomes, selected experimental weights, compact generation inputs and a labeled comparison video are saved. It remains unpromoted because nominal physical coverage is only 2/12.
- [V2 routing experiment](../robotics/evidence/rgb-servo-v2/README.md): another 48 frozen trials finish at 6/12 nominal live and 5/12 pushed live, versus 2/12 and 1/12 with images frozen per leg. It also misses its promotion gate. Both revisions, all failures and exact state recordings are retained; the baseline is unchanged.
- Latest full checks: 67 application tests, 13 training tests, dependency checks, package hashes, exact compact arrays and video decoding pass. The 41 original protected artifacts are unchanged. Credentials remain in local `.env`; commit metadata is nonpersonal.
- Development is on `codex/final-submission`, including preserved V1 evidence revision `1a27b0b` and the subsequent V2 routing/Intel-verifier preparation. Use the latest privately pushed revision of this branch. The hosted baseline remains unchanged.

## Required before the public release

1. **Run on the Intel laptop.** Follow [the tested procedure](../robotics/INTEL_FINAL_VERIFICATION.md), using a fresh output folder. Record CPU SKU, OpenGL renderer, OpenVINO device, FP32 parity/latency/throughput and the complete physical sequence. The procedure has passed on this PC, but AMD/NVIDIA is not Intel evidence. The available historical i7-10850H laptop is not Core Ultra Series 2/3; preserve that distinction and any organizer clarification. Do not turn a successful legacy run into a qualifying-hardware claim.
2. **Human voice rehearsal.** On the full local browser, select the learned dinner sequence, reset Task start / Closed / seed 42, press Speak instruction and say “Set the table.” Check the transcript and all six outcomes. A human bottle command already passed; the full voice sequence remains pending. The assistant must not capture the microphone unattended. No new key is needed while the existing local `.env` works.
3. **Refresh only changed claims.** Add actual Intel results, retain failed hardware checks where applicable, and update the current deck/video/text with measured results. Preserve older versions. The experimental RGB comparison stays supplementary and cannot imply broad coverage.
4. **Review the final revision.** Recheck the [completion checklist](FINAL_SUBMISSION_CHECKLIST.md), source/model provenance and public file/commit metadata. Merge the reviewed development branch into `main` before public release so the default repository page exposes the final code and README. Re-run checks only for intervening changes or unresolved failures.
5. **Release at the agreed time.** Keep GitHub and the Space private until immediately before the user's final submission. Then make the reviewed revision and demo public, verify both without authentication, and run a judge-style fresh CPU trial. A signed-in private trial does not verify anonymous access.
6. **User review and submission.** Reopen all form steps, confirm title, track, links, required video/PDF and accurate limitations. Leave the final button to the user. Record confirmation only after the user actually submits.

## Still unfinished in the robot

Broad bottle-position reliability, continuous visual correction after grasp, arbitrary object/destination commands, other-object or airborne handoffs, and pouring remain unfinished. These are development gaps, not missing uploads. The written brief's examples do not require implementing every possible motion, but their absence must remain explicit. Core Ultra eligibility, actual final Intel execution and anonymous judge access are separate unresolved submission dependencies.

The broader goal stays active. Prepared media, a complete draft or a stopped experiment alone do not complete it.
