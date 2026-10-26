# OpenVLA Policy Server

This directory contains a PAOS-compatible websocket server for OpenVLA LIBERO
evaluation. It serves the same msgpack protocol consumed by
`OpenPIClientPolicyWrapper`, so runtime sessions can keep using
`openpi://<host>:<port>` policy endpoints.

## Install OpenVLA Environment

No complete local OpenVLA conda environment is exported in this repository.
Create the policy environment from the official OpenVLA installation flow, then
install the PAOS websocket runtime dependencies:

```bash
# Create and activate conda environment.
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch for your CUDA platform. This is the official sample command;
# adjust pytorch-cuda if your machine requires a different CUDA version.
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Clone and install OpenVLA.
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# Install FlashAttention 2, matching the official OpenVLA setup.
pip install packaging ninja
ninja --version; echo $?
pip install "flash-attn==2.5.5" --no-build-isolation

# Extra dependencies required by the PAOS OpenVLA websocket server.
pip install msgpack websockets
```

The official OpenVLA LIBERO instructions also install LIBERO and
`experiments/robot/libero/libero_requirements.txt` for the upstream
`run_libero_eval.py` script. In the PAOS workflow below, LIBERO runs in the
separate `libero` environment through the TargetWS server, so the OpenVLA
policy environment only needs to load and serve the model.

## Start OpenVLA For LIBERO

Run this in an environment with OpenVLA, Transformers, PyTorch, Pillow,
`msgpack`, and `websockets`. Start from the repository root:

```bash
conda run --no-capture-output -n openvla python -m PhyAgentOS.runtime.policy.openvla.libero_server \
  --model-path openvla/openvla-7b-finetuned-libero-spatial \
  --unnorm-key libero_spatial \
  --host 0.0.0.0 --port 8000 \
  --device cuda \
  --center-crop \
  --seed 7
```

Use a LIBERO-finetuned OpenVLA checkpoint that matches the selected suite, and
set the selected suite as `--unnorm-key`, for example `libero_object`,
`libero_goal`, or `libero_10`. The base `openvla/openvla-7b` checkpoint is not
a LIBERO benchmark checkpoint.

The server returns `[T, 7]` float32 actions. OpenVLA typically returns one
action at a time for LIBERO, so `T` is usually `1`.

## Official 4-Suite LIBERO Evaluation

The target-native benchmark flow creates one PAOS session
per suite and lets the LIBERO target run the 10 tasks x 50 init states loop
internally against the suite-specific OpenVLA policy endpoint through the typed
`target.benchmark.*` job RPCs.

The official OpenVLA LIBERO protocol evaluates 4 suites x 10 tasks x 50
initial states = 2,000 episodes. OpenVLA provides one finetuned checkpoint per
suite, so a fully parallel 4-suite run needs four policy servers and enough GPU
memory for four OpenVLA models. If memory is limited, run the same commands one
suite at a time.

Start from the repository root.

1. Start the LIBERO TargetWS server in the `libero` environment:

```bash
MUJOCO_GL=egl PYTHONWARNINGS=ignore \
conda run --no-capture-output -n libero python PhyAgentOS/runtime/targets/remote/libero/server.py \
  --host 0.0.0.0 --port 9032 \
  --camera-height 256 --camera-width 256 \
  --max-steps 300 --num-steps-wait 10 \
  --control-mode relative --seed 7
```

2. Start one OpenVLA policy server per suite:

```bash
mkdir -p tests/openvla/logs

declare -A POLICY_PORT=(
  [libero_spatial]=8030
  [libero_object]=8031
  [libero_goal]=8032
  [libero_10]=8033
)

declare -A MODEL_PATH=(
  [libero_spatial]=openvla/openvla-7b-finetuned-libero-spatial
  [libero_object]=openvla/openvla-7b-finetuned-libero-object
  [libero_goal]=openvla/openvla-7b-finetuned-libero-goal
  [libero_10]=openvla/openvla-7b-finetuned-libero-10
)

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  PYTHONPATH=$(pwd) conda run --no-capture-output -n openvla python -m \
    PhyAgentOS.runtime.policy.openvla.libero_server \
    --model-path "${MODEL_PATH[$SUITE]}" \
    --unnorm-key "$SUITE" \
    --host 0.0.0.0 --port "${POLICY_PORT[$SUITE]}" \
    --device cuda \
    --center-crop \
    --seed 7 \
    > "tests/openvla/logs/policy_${SUITE}.log" 2>&1 &
done
```

3. Generate one target-native benchmark workspace per suite. This still
   evaluates the full 2,000-episode protocol.

```bash
RUN_ROOT=tests/openvla/libero_target_benchmark_$(date -u +%Y%m%dT%H%M%SZ)
export RUN_ROOT
mkdir -p "$RUN_ROOT"

declare -A POLICY_PORT=(
  [libero_spatial]=8030
  [libero_object]=8031
  [libero_goal]=8032
  [libero_10]=8033
)

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  PYTHONPATH=$(pwd) conda run -n paos python scripts/prepare_libero_target_benchmark.py \
    --workspace "$RUN_ROOT/$SUITE" \
    --suite "$SUITE" \
    --policy-id openvla \
    --target-endpoint targetws://127.0.0.1:9032 \
    --policy-endpoint "openpi://127.0.0.1:${POLICY_PORT[$SUITE]}" \
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

Configure Agent verification and the LIBERO Target registry as described in
the project README. For each suite-specific policy endpoint, ask the Agent to
create and wait for the target-native recovery Session:

```bash
declare -A POLICY_PORT=(
  [libero_spatial]=8030
  [libero_object]=8031
  [libero_goal]=8032
  [libero_10]=8033
)

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  paos agent --workspace ~/.PhyAgentOS/workspace -m \
    "Evaluate OpenVLA on LIBERO suite ${SUITE} with libero_real_remote and libero_target_benchmark. Use target_native execution, recovery verification, task ids 0-9, init-state ids 0-49, relative control, replan_every_steps 5, and policy endpoint openpi://127.0.0.1:${POLICY_PORT[$SUITE]}."
done
```

### PAOS OpenVLA Agent-assisted Results

PAOS target-native OpenVLA evaluation over 10 tasks x 50 init states per suite.
The first-attempt score is the original OpenVLA attempt; the final score allows
one agent-assisted retry after a failed episode. These recovery runs used
`retry_instruction_mode: verifier_rewrite`.

| Suite | First attempt(original) | Final after agent retry |
| --- | ---: | ---: |
| `libero_spatial` | 422 / 500 = 84.4% | 429 / 500 = 85.8% |
| `libero_object` | 432 / 500 = 86.4% | 438 / 500 = 87.6% |
| `libero_goal` | 373 / 500 = 74.6% | 376 / 500 = 75.2% |
| `libero_10` | 263 / 500 = 52.6% | 267 / 500 = 53.4% |
| Overall | 1490 / 2000 = 74.5% | 1510 / 2000 = 75.5% |
