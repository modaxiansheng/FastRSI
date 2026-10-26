# OpenPI-Compatible Policy Runtime

This directory contains the OpenPI-compatible policy wire integration.

## Components

- `client.py`: websocket client used by runtime skills for `openpi://` and
  `policyws://` endpoints.
- `lerobot_pi0_server.py`: standalone websocket policy server for LeRobot
  pi0-family checkpoints. The checkpoint `config.json` `type` selects the
  LeRobot policy class: `pi0`, `pi05`, or `pi0fast`.
- `native_openpi_server.py`: standalone websocket policy server for official
  OpenPI checkpoints. Use this for OpenPI-native checkpoints that contain
  `params/` or official OpenPI PyTorch checkpoints, without depending on
  LeRobot policy classes.
- `../msgpack_numpy.py`: numpy msgpack wire codec used by the OpenPI-compatible
  protocol.

## Install OpenPI Environment

The exported conda environment is stored at
`PhyAgentOS/runtime/policy/openpi/environment.yml`. Create it from the
repository root:

```bash
conda env create -f PhyAgentOS/runtime/policy/openpi/environment.yml
```

## Policy Endpoint

The runtime policy endpoint is:

```text
openpi://<policy-host>:8000
```

On connect, the server returns metadata including `policy_type`, `backend`,
`model_dir`, `chunk_size`, `n_action_steps`, and `action_dim`.

## Start An Official OpenPI Policy Server

Run this in the environment that has the official OpenPI package installed:

```bash
conda run --no-capture-output -n openpi python -m PhyAgentOS.runtime.policy.openpi.native_openpi_server \
  --policy-config pi05_libero \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero \
  --host 0.0.0.0 --port 8000
```

For the official OpenPI PI0 LIBERO checkpoint:

```bash
conda run --no-capture-output -n openpi python -m PhyAgentOS.runtime.policy.openpi.native_openpi_server \
  --policy-config pi0_libero \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi0_libero \
  --host 0.0.0.0 --port 8000
```

OpenPI-native checkpoints usually contain `params/`; LeRobot checkpoints contain
`config.json` and `model.safetensors`. Use `native_openpi_server.py` for the
former and `lerobot_pi0_server.py` for the latter.

## Official PI0.5 4-Suite LIBERO Evaluation

The target-native benchmark flow creates one PAOS session
per suite and lets the LIBERO target run the 10 tasks x 50 init states loop
internally against the PI0.5 policy endpoint. The LIBERO TargetWS server uses
the typed `target.benchmark.*` job RPCs for this path.

Start both servers from the repository root.

1. Start the LIBERO TargetWS server in the `libero` environment:

```bash
MUJOCO_GL=egl PYTHONWARNINGS=ignore \
conda run --no-capture-output -n libero python PhyAgentOS/runtime/targets/remote/libero/server.py \
  --host 0.0.0.0 --port 9022 \
  --camera-height 256 --camera-width 256 \
  --max-steps 300 --num-steps-wait 10 \
  --control-mode relative --seed 7
```

2. Start the official OpenPI PI0.5 policy server:

```bash
export CUDA_VISIBLE_DEVICES="#"
conda run --no-capture-output -n openpi python -m PhyAgentOS.runtime.policy.openpi.native_openpi_server \
  --policy-config pi05_libero \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi05_libero \
  --host 0.0.0.0 --port 8020
```

3. Generate one target-native benchmark workspace per suite. This still
   evaluates the full 2,000-episode protocol: 4 suites x 10 tasks x 50 initial
   states.

```bash
RUN_ROOT=tests/openpi/pi05/libero_target_benchmark_$(date -u +%Y%m%dT%H%M%SZ)
export RUN_ROOT
mkdir -p "$RUN_ROOT"

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  PYTHONPATH=$(pwd) conda run -n paos python scripts/prepare_libero_target_benchmark.py \
    --workspace "$RUN_ROOT/$SUITE" \
    --suite "$SUITE" \
    --policy-id pi05 \
    --target-endpoint targetws://127.0.0.1:9022 \
    --policy-endpoint openpi://127.0.0.1:8020 \
    --task-ids 0-9 \
    --init-state-ids 0-49 \
    --control-mode relative \
    --replan-every-steps 5 \
    --seed 7 \
    --retry-instruction-mode original \
    --force-init
done
```

4. Run the benchmark sessions. With a single LIBERO target server, the suites
   run one after another. If you start one target/policy server pair per suite,
   run one watchdog per suite in parallel instead.

```bash
PYTHONPATH=$(pwd) conda run --no-capture-output -n paos python \
  scripts/run_eval_watchdog.py \
  --run-root "$RUN_ROOT"
```

5. Inspect results:

```bash
conda run --no-capture-output -n paos python \
  scripts/summarize_eval_results.py \
  --run-root "$RUN_ROOT"
```

### Target-native Recovery Evaluation

Target-native recovery creates one PAOS session per
suite. The first attempt is preserved as the official score. When an episode
fails, the LIBERO target immediately sends that episode's initial/final RGB
evidence to the verifier and retries the same task/init-state inside the same
suite session when the verifier returns `replan`. The summary reports both
first-attempt and final-outcome rates.

`replan_every_steps` is the policy refresh cadence: at most that many actions
from one policy response are executed before requesting a new response. It is
independent of verification recovery. The Target's `retry_instruction_mode`
selects the recovery instruction: `original` (default) keeps the original task,
while `verifier_rewrite` uses the verifier's required nonempty
`replan_task_description`.

The Verification Service is started and supervised by `paos agent`; no fourth
terminal is required.

Configure the Agent verification settings and enable `libero_real_remote` in
`TARGETS.md` as described in the project README. With the LIBERO and PI0.5
servers running, use the Agent as the third terminal:

```bash
paos agent --workspace ~/.PhyAgentOS/workspace -m \
  "Evaluate PI0.5 on all four LIBERO suites with libero_real_remote and libero_target_benchmark. Use target_native execution, recovery verification, task ids 0-9, init-state ids 0-49, relative control, max_steps 300, replan_every_steps 5, and policy endpoint openpi://127.0.0.1:8020."
```

## PI0 Eval

PI0 uses the same LIBERO TargetWS server, OpenPI-native policy server,
relative control mode, session generator, watchdog, and result summarizer. Keep
`--replan-every-steps 5` and `--seed 7`, matching OpenPI's official LIBERO
eval script. To run PI0 instead of PI0.5, change only these fields:

| Location | PI0.5 value | PI0 value |
| --- | --- | --- |
| Policy server `--policy-config` | `pi05_libero` | `pi0_libero` |
| Policy server `--checkpoint-dir` | `gs://openpi-assets/checkpoints/pi05_libero` | `gs://openpi-assets/checkpoints/pi0_libero` |
| `RUN_ROOT` | `tests/openpi/pi05/libero_target_benchmark_...` | `tests/openpi/pi0/libero_target_benchmark_...` |
| `prepare_libero_target_benchmark.py --policy-id` | `pi05` | `pi0` |

For example, the PI0 policy server command is:

```bash
conda run --no-capture-output -n openpi python -m PhyAgentOS.runtime.policy.openpi.native_openpi_server \
  --policy-config pi0_libero \
  --checkpoint-dir gs://openpi-assets/checkpoints/pi0_libero \
  --host 0.0.0.0 --port 8020
```

And the PI0 workspace generation differences are:

```bash
RUN_ROOT=tests/openpi/pi0/libero_target_benchmark_$(date -u +%Y%m%dT%H%M%SZ)

# inside the suite loop:
    --policy-id pi0 \
```

Keep `--control-mode relative`, `--replan-every-steps 5`, `--seed 7`, the four
suite names, `--init-state-ids 0-49`, and the watchdog/result commands
unchanged.


### PAOS PI0/PI0.5 Agent-assisted Results

These recovery runs used `retry_instruction_mode: verifier_rewrite` and allowed
one retry after a failed episode.

#### PI0

| Suite | First attempt(original) | Final after agent retry |
| --- | ---: | ---: |
| `libero_spatial` | 484 / 500 = 96.8% | 488 / 500 = 97.6% |
| `libero_object` | 491 / 500 = 98.2% | 491 / 500 = 98.2% |
| `libero_goal` | 467 / 500 = 93.4% | 471 / 500 = 94.2% |
| `libero_10` | 413 / 500 = 82.6% | 413 / 500 = 82.6% |
| Overall | 1855 / 2000 = 92.8% | 1863 / 2000 = 93.2% |

#### PI0.5

| Suite | First attempt(original) | Final after agent retry |
| --- | ---: | ---: |
| `libero_spatial` | 497 / 500 = 99.4% | 498 / 500 = 99.6% |
| `libero_object` | 494 / 500 = 98.8% | 494 / 500 = 98.8% |
| `libero_goal` | 486 / 500 = 97.2% | 490 / 500 = 98.0% |
| `libero_10` | 464 / 500 = 92.8% | 473 / 500 = 94.6% |
| Overall | 1941 / 2000 = 97.0% | 1955 / 2000 = 97.8% |
