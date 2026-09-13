# Talos v6 submission draft

All three form steps were saved and verified after a full page reload on September 13. The entry is **not submitted**. The repository and Space remain private until the final release. Update pending checks below before the user's final Submit action.

**Title (36/50):** Talos: Learned Dinner-Table Robotics

**Short description (232/255):** Talos turns supported spoken or typed instructions into physical dinner-table manipulation by two SO-101 arms. Camera-conditioned neural skills, motor feedback and OpenVINO inference coordinate six tasks and bottle relays in MuJoCo.

**Mode:** Online  
**Category:** Home Automation  
**Track:** Intel Bimanual VLA Manipulation with Multi-Modal Reasoning  
**Selected platform technology:** Speechmatics api  
**Demo platform:** Other

## Long description saved in the form

Talos is a reproducible dual-arm robotics prototype for dinner-table assistance. Two SO-101 arms manipulate a bottle, plate, mug, drawer, fork and spoon through real MuJoCo contacts. Supported spoken commands use Speechmatics; typed commands use the same bounded grammar and scene-aware planner.

Five new neural skill models complement the preserved bottle model. Classical RGB features condition learned motor trajectories, while joint feedback regulates progress. A separate simulator-state monitor checks contacts, release, placement and parked arms; it does not generate learned actions. The six-skill sequence runs without scene resets. The frozen OpenVINO CPU evaluation completed 8 of 10 randomized dinner scenes; the two failures were mug and spoon placement errors. Learned table-supported bottle relays passed 5 of 5 frozen trials in each direction. All outcomes, including failures, are retained.

Training runs on an RTX 4070. Models are exported to OpenVINO with numerical parity checks. Historical Intel CPU and integrated-GPU bottle measurements are preserved; final-suite Intel verification remains pending. The repository includes compact training inputs, physical evaluation scripts, assets and model provenance.

The demonstrated scope is finite starting regions and fixed task destinations. Arbitrary reachable positions, continuous visual grasp correction, pouring and airborne handoffs remain unfinished. The private hosted CPU demo completes all six tasks on exposed seed 42; anonymous access remains to be checked at release. Local voice input has passed a human microphone bottle trial.

## Media and links

| Item | File or URL | Verified state |
| --- | --- | --- |
| Repository | https://github.com/jannissio/talos-ai-infra | Private; release timing remains the user's instruction |
| Interactive demo | https://huggingface.co/spaces/jannis-sms/talos-dinner-robotics | Private; all six learned tasks pass on OpenVINO CPU, exposed seed 42 |
| Cover | `cover-v6.png` | 1280×720, full 16:9 crop saved in draft |
| Main video | `Talos-demo-v6.mp4` | 1280×720, 20 fps, 189.1 seconds, 8,998,287 bytes; saved in draft |
| Presentation PDF | `Talos-v6.pdf` | Eight checked slides; saved in draft |
| Editable deck | `Talos-v6-r2.pptx` | Eight checked slides, nonpersonal author metadata |
| All dinner trials | `Talos-dinner-ten-v6.mp4` | All ten, 4× labeled playback, 8/10 outcome |
| All relay trials | `Talos-relays-ten-v3.mp4` | All ten, 1× playback, 5/5 each direction |
| Experimental RGB comparison | `rgb-feedback-experiment.mp4` | 50.4 seconds, 1× playback, one labeled pair plus complete 48-trial counts; supplementary only |

Main-video instructions/explanations are captions; it has no audio track. The original synthetic-speech video remains preserved in the parent submission folder. A human microphone bottle trial passed on this PC; final six-skill voice rehearsal has been requested.

## Additional information saved in the form

Reproduction, architecture and complete physical outcomes are in the repository. The v6 baseline completes 8/10 frozen six-skill dinner sequences; the two placement failures are retained. Table-supported bottle relays pass 5/5 trials in each direction. Separate ten-seed videos are in submission/final-v6.

The selected dinner controller uses initial RGB features and motor feedback. Separate RGB experiments remain unpromoted: V1 passed 2/12 nominal and 3/12 pushed live scenes, versus 1/12 and 0/12 with frozen images. V2 routing completed 48 new frozen trials: live 6/12 nominal and 5/12 pushed, frozen-image 2/12 and 1/12. Only one V2 seed passed both undisturbed controls. All outcomes, failed revisions and reproduction inputs are packaged. Different final seeds prevent a controlled V1/V2 comparison. Arbitrary placements, pouring, airborne handoffs and unrestricted language remain unfinished.

Original Talos code, weights and procedural data are MIT licensed. Upstream SO-101 assets retain their Apache-2.0 notices. Development reuse and AI assistance are disclosed in docs/PROVENANCE.md.

The private hosted demo completed all six learned tasks on exposed seed 42 using OpenVINO CPU: 247.655 simulated / 273.03 wall seconds, with physical release and parking verified. Final-suite Intel laptop execution and anonymous judge access remain open. Historical Intel timings describe the original bottle model only. The GitHub repository and hosted Space remain private until immediately before final submission.

## Before final submission

Finish the [active checklist](../../docs/hackathon/FINAL_SUBMISSION_CHECKLIST.md). Replace pending Intel/hosting statements only with actual measured evidence. Publish the sanitized repository/Space at the authorized release time and verify anonymous links. Recheck every form step and let the user perform the final submission. The form's “100%” indicates filled fields, not verified eligibility, public access or submission confirmation.
