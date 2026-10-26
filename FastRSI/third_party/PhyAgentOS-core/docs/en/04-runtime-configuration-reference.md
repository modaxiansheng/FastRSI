# Runtime Configuration Reference

PhyAgentOS Runtime configuration has four layers:

1. `~/.PhyAgentOS/config.json` controls process lifecycle and global verification.
2. `TARGETS.md` declares available Targets and their capabilities.
3. `SKILLRUNTIME.md` declares executable Policy and Builtin SkillRuntimes.
4. `SESSIONS.md` records the concrete Target, SkillRuntime, routing, execution,
   benchmark, and verification choices for each task.

The Agent selects from the registries and writes a Session. Preflight verifies
that the selected fields agree; it does not replace an incompatible Target,
SkillRuntime, interface, or execution mode.

## Global Runtime Configuration

The following fields belong under `runtime` in `~/.PhyAgentOS/config.json`.
JSON uses the camelCase names shown here.

| Field | Default | Meaning |
|:--|:--|:--|
| `enabled` | `true` | Enables Runtime workspace preparation and Session execution. |
| `workspace` | `null` | Runtime workspace path. `null` uses the workspace selected by the CLI. |
| `autostartWatchdog` | `true` | Starts and supervises the Watchdog with `paos agent` or `paos gateway`. |
| `watchdogPollIntervalS` | `1.0` | Interval in seconds for scanning pending Sessions. |
| `targetEnabled` | `{}` | Global Target enable overrides keyed by Target ID. An override takes precedence over `TARGETS.md`. |

Example:

```json
{
  "runtime": {
    "enabled": true,
    "workspace": null,
    "autostartWatchdog": true,
    "watchdogPollIntervalS": 1.0,
    "targetEnabled": {}
  }
}
```

## Global Verification Configuration

Verification settings belong under `agents.verification`. They are global and
cannot be overridden by a Session. A Session only selects a
`verification_profile`.

| Field | Default | Meaning |
|:--|:--|:--|
| `serviceEnabled` | `true` | Starts the Agent-owned Verification Service child process. |
| `provider` | `null` | Verification provider. `null` reuses the Agent default provider. |
| `model` | `null` | Verification model. `null` reuses the Agent default model. |
| `timeoutS` | `180` | Timeout in seconds for one verification request. Must be greater than zero. |
| `evidenceRetention` | `none` | SessionVerifier RGB retention: `all`, `failed`, or `none`. Target-native episode evidence is temporary and is always deleted after verification. |
| `maxReplansPerEpisode` | `2` | Maximum additional recovery attempts for one episode. `0` disables recovery retries. |
| `maxVerifierCallsPerRun` | `50` | Verifier-call budget shared by one target-native benchmark run. `0` disables verifier calls. |
| `serviceHost` | `127.0.0.1` | Verification Service bind address. |
| `servicePort` | `8100` | Verification Service HTTP port. |
| `remoteTargetUrl` | `null` | Verification Service URL reachable by a Target on another host. |

```json
{
  "agents": {
    "verification": {
      "serviceEnabled": true,
      "provider": null,
      "model": null,
      "timeoutS": 180,
      "evidenceRetention": "none",
      "maxReplansPerEpisode": 2,
      "maxVerifierCallsPerRun": 50,
      "serviceHost": "127.0.0.1",
      "servicePort": 8100,
      "remoteTargetUrl": null
    }
  }
}
```

Verification profiles have path-specific behavior:

| Profile | `policy_loop` | `target_native` |
|:--|:--|:--|
| `strict` | No verifier call. | No verifier call. |
| `audit` | SessionVerifier validates the completed Session without recovery. | The benchmark job validates failed episodes without recovery. |
| `recovery` | SessionVerifier may create a child Session from the required verifier rewrite. | The benchmark job may continue the failed episode; the Target configuration selects whether the next attempt uses the original task or the verifier rewrite. |

Both policy-loop Session verification and target-native episode verification use
the Agent-owned Verification Service for the model-backed verdict and strict
response normalization. A target-native root Session is not passed through
SessionVerifier again.

A valid response has a supported verdict, a nonempty array of nonempty evidence
strings, and a nonempty lesson. `failure` also requires a nonempty
`failure_reason`; `replan` also requires a nonempty
`replan_task_description`. A response that violates this contract is recorded
as `failure` with `verifier_status: invalid_response`. Model timeouts, provider
errors, service HTTP errors, and connection failures remain verification
infrastructure errors and are never converted into `replan`.

## `TARGETS.md`

`TARGETS.md` uses `version: runtime_target_registry_v1` and contains a
`targets` list.

| Field | Required | Meaning |
|:--|:--:|:--|
| `id` | yes | Unique Target ID used by `target://<id>` references. |
| `target_class` | yes | `local` or `remote`. |
| `target_kind` | yes | `game`, `debug`, `simulation`, or `real_robot`. |
| `embodiment` | no | Optional embodiment identifier. |
| `enabled` | no | Makes the Target selectable unless a global `targetEnabled` override changes it; defaults to `true`. |
| `workspace` | yes | Target-specific workspace path. |
| `supported_skillruntimes` | no | Exact SkillRuntime IDs accepted by this Target; defaults to an empty list. |
| `runtime.target_runtime` | yes | Runtime implementation or remote proxy class. |
| `runtime.target_endpoint` | no | Remote Target endpoint, normally `targetws://host:port`. |
| `runtime.target_adapter` | yes | Target adapter reference. |
| `runtime.runtime_contract_ref` | yes | Runtime contract file. |
| `observation.observation_type` | no | Observation shape category used for compatibility checks; defaults to `multimodal`. |
| `observation.empty_observation_allowed` | no | Whether the Target may legally return an empty observation; defaults to `false`. |
| `perception.enabled` | no | Enables the Target perception pipeline; defaults to `false`. |
| `perception.strict_preflight` | no | Defaults to and must remain `true`; required perception inputs are checked before execution. |
| `perception.sensor_config_ref` | no | Sensor configuration file. |
| `perception.perception_config_ref` | no | Perception configuration file. |
| `perception.artifact_dir` | no | Perception artifact directory. |
| `config` | no | Target-specific parameters such as dimensions, limits, scene, control mode, and reset behavior; defaults to an empty map. |

The LIBERO target-native benchmark reads these evaluation controls from the
Target `config` map:

| Field | Default | Meaning |
|:--|:--|:--|
| `control_mode` | `relative` | LIBERO action convention; use the convention required by the selected policy. |
| `seed` | `0` | Environment seed applied to every benchmark episode. |
| `retry_instruction_mode` | `original` | Recovery instruction behavior. `original` keeps the original task; `verifier_rewrite` replaces it with the verifier's nonempty `replan_task_description`. |

### Benchmark capability

`benchmark_capabilities` is optional for ordinary tasks and required when the
Target participates in structured benchmarking.

| Field | Meaning |
|:--|:--|
| `benchmark_id` | Benchmark family, for example `libero` or `behavior1k`. |
| `suites` | Suite IDs supported by the Target. An empty list means no suite restriction. |
| `execution_modes[].mode` | `policy_loop` or `target_native`. |
| `execution_modes[].interface` | `rollout_episode_v1` for `policy_loop`; `target_benchmark_job_v1` for `target_native`. |
| `execution_modes[].reset_owner` | `session_runner` for `policy_loop`; `skillruntime` for `target_native`. |

## `SKILLRUNTIME.md`

`SKILLRUNTIME.md` uses `version: runtime_skill_registry_v1` and contains a
`skillruntimes` list.

| Field | Required | Meaning |
|:--|:--:|:--|
| `id` | yes | Unique ID used by `skillruntime://<id>` references. |
| `runtime` | yes | Concrete SkillRuntime implementation. |
| `runtime_kind` | yes | `policy` for a policy loop or `builtin` for controlled direct Target interfaces. |
| `loop_mode` | yes | Runtime-specific loop mode. |
| `agent_exposure` | no | `none`, `target_tools`, or `constrained_target_tools`; defaults to `none`. This controls interactive TargetTool exposure, not builtin access through `TargetSessionHandle`. |
| `supported_target_kinds` | yes | Target kinds accepted by the runtime. |
| `policy.policy_client` | policy only | Policy protocol client. |
| `policy.policy_adapter` | policy only | Policy adapter reference. |
| `policy.supports_chunk` | policy only | Whether the policy can return action chunks. |
| `observation_contract` | no | Observation requirements matched against the Target; defaults to a multimodal nonempty contract. |
| `supports_chunk` | no | Whether the SkillRuntime consumes action chunks; defaults to `false`. |
| `default_replan_every` | no | Default policy refresh cadence; defaults to `1`. |
| `input_contract` / `output_contract` | no | Explicit policy inputs and action outputs; each defaults to an empty map. |
| `adapter_requirements` | no | Allowed bridges and forbidden implicit conversions; defaults to an empty map. |
| `requires` | no | Sensor, environment output, geometry, confidence, and strict-contract requirements; defaults to strict requirements with empty dependency lists. |
| `target_tool_policy` | tool exposure only | Allowed/forbidden operations and validation requirements. |

### Benchmark capability

A benchmark-capable SkillRuntime declares one `benchmark` block:

| Field | Meaning |
|:--|:--|
| `benchmark_id` | Must match the Target capability and Session benchmark. |
| `execution_mode` | `policy_loop` or `target_native`. |
| `target_interface` | Must match the Target interface for that mode. |
| `result_schema` | `benchmark_execution_result_v1`. |
| `reset_owner` | Must match the Target reset owner. |

Target-native is not a generic SkillRuntime. Each benchmark supplies its own
builtin, such as `LiberoBenchmarkSkillRuntime` or a future
`B1KBenchmarkSkillRuntime`.

## `SESSIONS.md`

`SESSIONS.md` uses `version: runtime_sessions_v1`. A Session is the immutable
selection of Target, SkillRuntime, routing, and execution semantics plus its
mutable lifecycle and result.

### Identity and lifecycle

| Field | Default | Meaning |
|:--|:--|:--|
| `session_id` | required | Unique Session ID. |
| `parent_session_id` | `null` | Parent for a replan child Session. |
| `replan_attempt` | `0` | Replan generation, starting at zero. |
| `goal_id` / `parent_goal_id` | `null` | Optional goal lineage. |
| `horizon` | `null` | Optional `short_term` or `long_term` classification. |
| `target_ref` | required | `target://<target-id>`. |
| `skillruntime_ref` | required | `skillruntime://<runtime-id>`. |
| `task_description` | required | Text delivered to the selected execution path. |
| `verification_profile` | `strict` | `strict`, `audit`, or `recovery`. |
| `status` | `pending` | Session lifecycle state; the Agent creates work as `pending`. |
| `priority` | `normal` | `low`, `normal`, or `high`. |
| `depends_on` | `[]` | Dependency metadata. The current scheduler records it but does not block execution on it. |

`created_at`, `updated_at`, `claimed_by`, `claim_token`, `retry.attempted`, and
`result` are maintained by Runtime. Users and the Agent should not fabricate
claim data or execution results. Runtime advances `status` through claim,
Preflight, execution, finalization, verification, and a terminal state.

### Timeouts and retry

| Field | Default | Meaning |
|:--|:--|:--|
| `timeouts.queue_timeout_s` | `30` | Maximum time waiting to be claimed. |
| `timeouts.preflight_timeout_s` | `20` | Maximum Preflight duration. |
| `timeouts.execute_timeout_s` | `300` | Maximum execution duration. A suite benchmark must allow for the full suite. |
| `timeouts.policy_timeout_s` | `5` | Timeout for one policy request. |
| `retry.max_retries` | `0` | Runtime execution retries; separate from verification replan. |
| `retry.attempted` | `0` | Retry counter maintained by Runtime. |

### Routing and execution

| Field | Default | Meaning |
|:--|:--|:--|
| `routing.target_endpoint` | `null` | Session endpoint override for the selected Target. Prefer the Registry value unless the Session intentionally routes elsewhere. |
| `routing.policy_endpoint` | `null` | Policy endpoint used by PolicySkillRuntime or passed to a target-native benchmark job. |
| `routing.adapter_resolution` | `strict_auto` | `strict_auto` or `strict_override`; both require a fully valid adapter plan. |
| `routing.adapter_overrides` | `null` | Explicit adapter override map used with `strict_override`. |
| `execution.max_steps` | `600` | Maximum execution steps. |
| `execution.control_hz` | `null` | Optional control frequency. |
| `execution.replan_every` | `8` | Legacy/default policy refresh count used when a step-specific value is absent. |
| `execution.replan_every_steps` | `null` | Maximum action steps consumed from one policy response before requesting a new response. This policy refresh cadence is independent of verification `replan`. |
| `execution.action_chunk_mode` | `chunk_buffer` | `chunk_buffer`, `open_loop`, or `single_step`. |
| `execution.chunk_switch_mode` | `hard_switch` | `hard_switch` or `soft_blend`. |
| `execution.reset_policy` | `session_runner` | `session_runner` for normal/policy-loop Sessions; `skillruntime_managed` for target-native benchmark Sessions. |
| `execution.steps` / `timeline` | `null` | Optional explicit builtin step or timeline payloads. |

### Runtime hints and safety

| Field | Default | Meaning |
|:--|:--|:--|
| `runtime_hints.perception_queries` | `[]` | Structured perception or benchmark selection hints. |
| `runtime_hints.force_environment_refresh` | `false` | Requests an environment refresh before execution. |
| `runtime_hints.preferred_replan_every_steps` | `null` | Preferred value for the same policy refresh cadence supplied to planning. |
| `safety_profile.profile` | `default` | Named safety profile. |
| `safety_profile.workspace_bounds` | `null` | Workspace bound reference. |
| `safety_profile.stop_on_policy_timeout` | `true` | Stops execution when the policy times out. |

### Benchmark metadata

The `benchmark` block is required for benchmark Sessions.

| Field | Meaning |
|:--|:--|
| `benchmark_id` | Benchmark family matched against Target and SkillRuntime. |
| `suite_id` | Suite selected from the Target capability. |
| `execution_mode` | `policy_loop` or `target_native`. |
| `task_name`, `task_index`, `instance_id` | Optional episode identity for policy-loop Sessions. |
| `policy_id` | Optional benchmark policy identity. |
| `run_id` | Benchmark run lineage. |

For `policy_loop`, one episode is one root Session and
`execution.reset_policy` is `session_runner`. For `target_native`, one suite is
one root Session and `execution.reset_policy` is `skillruntime_managed`.

## Endpoint and Host Rules

- A remote Target uses `targetws://<host>:<port>`.
- A PAOS policy service uses its supported scheme, such as
  `openpi://<host>:<port>`.
- The Agent must reach the Target endpoint.
- For target-native benchmarking, the Target host must reach the policy
  endpoint because the Target runs the policy attempts.
- For a remote Target that uses verification, bind the service with
  `serviceHost: "0.0.0.0"` and set `remoteTargetUrl` to an HTTP URL reachable
  from the Target.
- `paos agent` still owns the Verification Service; no user-managed fourth
  process is required.

## LIBERO PI0.5 Example

The project README contains the three-terminal commands. The corresponding
minimal configuration is:

```json
{
  "runtime": {
    "enabled": true,
    "workspace": null,
    "autostartWatchdog": true
  },
  "agents": {
    "verification": {
      "serviceEnabled": true,
      "provider": null,
      "model": null,
      "timeoutS": 180,
      "evidenceRetention": "none",
      "maxReplansPerEpisode": 2,
      "maxVerifierCallsPerRun": 50,
      "serviceHost": "127.0.0.1",
      "servicePort": 8100,
      "remoteTargetUrl": null
    }
  }
}
```

In the existing `libero_real_remote` Target entry, apply these fields while
retaining its other contracts and adapter configuration:

```yaml
enabled: true
supported_skillruntimes:
  - pi05_libero_remote
  - libero_target_benchmark
runtime:
  target_endpoint: targetws://127.0.0.1:9002
benchmark_capabilities:
  - benchmark_id: libero
    suites: [libero_spatial, libero_object, libero_goal, libero_10]
    execution_modes:
      - mode: policy_loop
        interface: rollout_episode_v1
        reset_owner: session_runner
      - mode: target_native
        interface: target_benchmark_job_v1
        reset_owner: skillruntime
config:
  control_mode: relative
  max_steps: 300
  seed: 7
  retry_instruction_mode: original
```

The README service commands use these example-specific parameters:

| Command field | Value | Meaning |
|:--|:--|:--|
| Target `--host` / `--port` | `0.0.0.0` / `9002` | Listens for TargetWS connections. |
| `--camera-height` / `--camera-width` | `256` / `256` | Sets LIBERO observation image dimensions. |
| Target `--max-steps` | `300` | Limits policy steps in one attempt. |
| `--num-steps-wait` | `10` | Runs scene-settling steps after each episode reset. |
| Target `--control-mode` | `relative` | Matches the PI0.5 LIBERO action convention. |
| Target `--seed` | `7` | Matches the official OpenPI LIBERO evaluation seed. |
| Policy `--policy-config` | `pi05_libero` | Selects the official PI0.5 LIBERO policy configuration. |
| Policy `--checkpoint-dir` | `gs://openpi-assets/checkpoints/pi05_libero` | Loads the official PI0.5 checkpoint. |
| Policy `--host` / `--port` | `0.0.0.0` / `8000` | Exposes the same-host endpoint `openpi://127.0.0.1:8000`. |
| Agent `--workspace` | `~/.PhyAgentOS/workspace` | Selects the Runtime registry, Session state, and artifact workspace. |

The Agent request selects:

| Selection | Value |
|:--|:--|
| Target | `target://libero_real_remote` |
| SkillRuntime | `skillruntime://libero_target_benchmark` |
| Benchmark / suite | `libero` / `libero_spatial` |
| Execution mode | `target_native` |
| Reset policy | `skillruntime_managed` |
| Verification profile | `recovery` |
| Policy endpoint | `openpi://127.0.0.1:8000` |
| Task / init-state IDs | `0-9` / `0-49` in `runtime_hints.perception_queries` |
| Policy refresh cadence | `execution.replan_every_steps: 5` |

The Target resets once per logical episode. Every `replan` verdict retains a
nonempty `replan_task_description`. With the default
`retry_instruction_mode: original`, the next attempt keeps the original task;
set it to `verifier_rewrite` to use that description as the policy instruction.
Either mode continues from the failed environment state without another reset.
Artifacts preserve first-attempt and recovery-final scores separately.

## Related Documentation

- [User Manual](02-user-manual.md)
- [Developer Manual](03-developer-manual.md)
- [Project README](../../README.md)
