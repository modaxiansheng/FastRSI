# Project website assets

- `framework.png` and `framework.pdf`: FastRSI project method figure, `FastRSI_Framework_reference_v6_palette.pdf`, referenced by the manuscript snapshot dated 2026-10-26. The figure presents the research design.
- `step-1.png` to `step-4.png`: unchanged 256 × 256 LIBERO scene images from the project figure assets, respectively `C_01_drawer_closed.png`, `C_02_open_drawer.png`, `C_03_lift_bowl.png`, `C_04_place_bowl.png`. They illustrate a task sequence, not synchronized experimental playback.
- `overview.svg`: new conceptual layout using the first two project scene images.
- `readme-banner.svg`: new vector typography and graphics for this project.
- `results.json`: derived without changing counts from `../ORIGINAL_LIBERO_SNAPSHOT.json`, captured 2026-10-26. Percentages are computed as successes / completed cases.

Visual direction was inspired by [LLM4Robot-Bench](https://llm4robot.github.io/bench/). Its imagery, video, research claims, and implementation are not reused. Existing third-party asset terms continue to apply.

## Recorded video collection

`media/*.mp4` contains 12 full, original-speed dual-view executions from the PAOS/MemRSI LIBERO development archive, using a π0.5 LIBERO checkpoint. All source and published hashes, task instructions, execution flags, and parent-run scope are in [VIDEO_PROVENANCE.json](VIDEO_PROVENANCE.json). Source frames were transcoded to browser-compatible H.264 without changing duration or playback rate. Corresponding JPEG posters are direct source-video frames.

Ten clips report native physical-task completion; two stop after a completed first subgoal with the full task incomplete. Their parent memory-decision runs report `decision_failure`. These selected development recordings use manually designed initial memory guidance and are separate from the released benchmark snapshot; they do not establish learned FastRSI improvement. No new experiments were run to produce this gallery.
