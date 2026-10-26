# LIBERO resource expansion: eight native lanes

Snapshot: 2026-10-24T00:44:12.407827+08:00. Live status remains in control05/PROGRESS.json.

Original LIBERO Spatial, Object, Goal; four VLA backends; 500 episodes per model/suite; 6000 registered cases. Scientific case identities, per-attempt limits and model generation remain unchanged.

Concurrency increased from two lanes on GPU2/3 to eight independent policy/verifier/target lanes across node107 GPU2/3/6/7. Each GPU hosts two separate lanes. No workloads on108 or126 were stopped. GPU0/1/4/5 reservations were preserved.

| GPU | Used GiB | VRAM use |
|---|---:|---:|
| 2 | 52.12 | 81.5% |
| 3 | 58.63 | 91.6% |
| 6 | 52.2 | 81.6% |
| 7 | 59.56 | 93.1% |

Completed: 117/6000; physical and end-to-end successes: 116/117. This is partial progress, not a final success rate. All eight live lane processes, sixteen resident model services and eight unique target services were inspected. Each lane made successful native policy calls; 8/8 lanes have also closed a new full case in this snapshot.

Admission targets at most90% predicted VRAM; actual peaks can vary. The approximately two-minute mixed warmup window closed18 cases; it is not a matched speedup measurement or an ETA.

One earlier X-VLA case failed after three unsuccessful physical attempts because the verifier allowed five recovery children while the dispatcher still executed only three attempts. The runtime, supervisor and verifier are now consistent at six total attempts/five recovery children; the inference safety ceiling2000 covers the largest six-attempt physical budget. CPU tests verify six completed failures compile without changing success, and pending records are still rejected. Its raw failure remains immutable and counted as a failure; its root was not retried for quality. See CLOSED_RUNTIME_FAILURE_RECEIPT.json and CPU_CONTRACT_05.json.

resource05_overlay pins these compatibility edits by SHA-256 for every backend. Older controls, startup/transition diagnostics and all case costs remain separate. The independent native target scenes, resident RNG and complete500 denominators are preserved.

A bounded backfill owner waits for initial eight-lane acceptance and spare memory, then moves the already-registered OpenVLA Goal500 group into an available owned card. The same group lock prevents duplicate execution. It never allocates another GPU.

Key receipts: control05/NATIVE_RESOURCE_ACCEPTANCE_05.json; control05/RESOURCE_WINDOW_05.json; control05/OWNER_LAUNCH.json; RESOURCE05_BINDINGS.json.
