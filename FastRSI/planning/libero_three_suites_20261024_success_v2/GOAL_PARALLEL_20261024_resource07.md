# Original LIBERO Goal early parallel execution

Snapshot: 2026-10-24T01:19:01.769428+08:00

107 continues the existing eight Spatial/Object lanes.108 now owns audited GPU3/6 and runs Goal pi05, pi0 and OpenVLA in three separate native policy/verifier/target lanes. Goal XVLA is admitted on the first safe card or continues through the existing107 queue. Live status is control07/OWNER_STATUS.json and global progress is control05/PROGRESS.json.

| Model | Goal completed /500 | Physical successes | State |
|---|---:|---:|---|
| pi05 | 9/500 | 9 | RUNNING on108 GPU3 |
| pi0 | 4/500 | 4 | RUNNING on108 GPU3 |
| openvla | 3/500 | 3 | RUNNING on108 GPU6 |
| X-VLA | 0/500 | - | Waiting for safe memory or its existing107 owner |

Goal total 16/2000; global total 495/6000. This is partial progress, not final benchmark success. Unique complete episode identities checked across both nodes; zero duplicates.

Model weights, official original-suite assets, 50 initial states/task, per-attempt300-step Goal budget, maximum5 recovery children and success accounting are unchanged. The repaired6-attempt runtime overlay is inherited and hash-pinned in RESOURCE07_BINDINGS.json.

Resource audit preserved108 GPU0/1 paused workspaces, resident RoboCasa/FastEvo services on1/2/4/7, and the existing4/5 baseline queue reservation. The old FBA lease owners for3/6 and their central controller were absent, their dispatch windows had ended, and there were no active device holders. Exact old lease directories were atomically archived with matching SHA-256, then new scoped leases were created. See RETIRED_CLOSED_LEASES.json and FRESH_ADMISSION.json.

Cross-host group flock was tested: a lock held on108 was rejected on107. Existing owners will wait or reuse the completed500-case group rather than repeat it. Additional model admission checks predicted memory below90%; OpenVLA is kept on its own card to preserve headroom and throughput.

Current108 VRAM: GPU3 81.9%, GPU6 57.5%. Existing resident foreign container IDs, paused states, and unselected lease bytes were rechecked unchanged.
