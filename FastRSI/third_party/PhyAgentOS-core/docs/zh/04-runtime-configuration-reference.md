# Runtime 参数配置参考

PhyAgentOS Runtime 配置分为四层：

1. `~/.PhyAgentOS/config.json` 管理进程生命周期与全局 verification。
2. `TARGETS.md` 声明可用 Target 及其能力。
3. `SKILLRUNTIME.md` 声明可执行的 Policy 与 Builtin SkillRuntime。
4. `SESSIONS.md` 记录每项任务实际选择的 Target、SkillRuntime、routing、
   execution、benchmark 和 verification。

Agent 从 Registry 中完成选择并写入 Session。Preflight 校验这些字段是否
一致，不会替换不匹配的 Target、SkillRuntime、接口或 execution mode。

## Runtime 全局配置

以下字段位于 `~/.PhyAgentOS/config.json` 的 `runtime` 下。JSON 使用表中的
camelCase 名称。

| 字段 | 默认值 | 含义 |
|:--|:--|:--|
| `enabled` | `true` | 启用 Runtime workspace 准备和 Session 执行。 |
| `workspace` | `null` | Runtime workspace 路径；`null` 表示使用 CLI 选择的 workspace。 |
| `autostartWatchdog` | `true` | 随 `paos agent` 或 `paos gateway` 启动并管理 Watchdog。 |
| `watchdogPollIntervalS` | `1.0` | 扫描 pending Session 的时间间隔，单位为秒。 |
| `targetEnabled` | `{}` | 以 Target ID 为键的全局 enable 覆盖；优先级高于 `TARGETS.md`。 |

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

## Verification 全局配置

Verification 参数位于 `agents.verification`。它们是全局配置，Session
不能覆盖；Session 只选择 `verification_profile`。

| 字段 | 默认值 | 含义 |
|:--|:--|:--|
| `serviceEnabled` | `true` | 将 Verification Service 作为 Agent 子进程启动。 |
| `provider` | `null` | verification provider；`null` 表示复用 Agent 默认 provider。 |
| `model` | `null` | verification model；`null` 表示复用 Agent 默认 model。 |
| `timeoutS` | `180` | 单次 verification 请求超时，单位为秒，必须大于零。 |
| `evidenceRetention` | `none` | SessionVerifier RGB evidence 保留方式：`all`、`failed` 或 `none`。target-native episode evidence 是临时文件，校验后始终删除。 |
| `maxReplansPerEpisode` | `2` | 单个 episode 允许的额外 recovery attempt 数；`0` 禁止 recovery retry。 |
| `maxVerifierCallsPerRun` | `50` | 单个 target-native benchmark run 共享的 verifier 调用预算；`0` 禁止 verifier 调用。 |
| `serviceHost` | `127.0.0.1` | Verification Service 监听地址。 |
| `servicePort` | `8100` | Verification Service HTTP 端口。 |
| `remoteTargetUrl` | `null` | Target 位于其他主机时可访问的 Verification Service URL。 |

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

不同 verification profile 的行为如下：

| Profile | `policy_loop` | `target_native` |
|:--|:--|:--|
| `strict` | 不调用 verifier。 | 不调用 verifier。 |
| `audit` | Session 完成后由 SessionVerifier 校验，不触发 recovery。 | benchmark job 校验失败 episode，不触发 recovery。 |
| `recovery` | SessionVerifier 可以根据 verifier 必须返回的 rewrite 创建子 Session。 | benchmark job 可以从失败状态继续 episode；下一次 attempt 使用原任务还是 verifier rewrite，由 Target 配置决定。 |

policy-loop Session 与 target-native episode 的模型校验和严格响应规范化都由
Agent 管理的 Verification Service 完成。target-native root Session 不会再次进入
SessionVerifier。

有效响应必须包含受支持的 verdict、由非空字符串组成的非空 evidence 数组，
以及非空 lesson；`failure` 还必须包含非空 `failure_reason`，`replan` 还必须
包含非空 `replan_task_description`。违反该规范的响应记录为
`failure/verifier_status: invalid_response`。模型超时、provider 错误、service
HTTP 错误和连接故障仍属于 verification 基础设施错误，不会转换为 `replan`。

## `TARGETS.md`

`TARGETS.md` 使用 `version: runtime_target_registry_v1`，主体为 `targets`
列表。

| 字段 | 必需 | 含义 |
|:--|:--:|:--|
| `id` | 是 | 唯一 Target ID，由 `target://<id>` 引用。 |
| `target_class` | 是 | `local` 或 `remote`。 |
| `target_kind` | 是 | `game`、`debug`、`simulation` 或 `real_robot`。 |
| `embodiment` | 否 | 可选 embodiment 标识。 |
| `enabled` | 否 | 允许选择该 Target；默认值为 `true`，全局 `targetEnabled` 可覆盖它。 |
| `workspace` | 是 | Target 专属 workspace 路径。 |
| `supported_skillruntimes` | 否 | 该 Target 接受的准确 SkillRuntime ID；默认为空列表。 |
| `runtime.target_runtime` | 是 | Target runtime 实现或 remote proxy 类。 |
| `runtime.target_endpoint` | 否 | Remote Target endpoint，通常为 `targetws://host:port`。 |
| `runtime.target_adapter` | 是 | Target adapter 引用。 |
| `runtime.runtime_contract_ref` | 是 | Runtime contract 文件。 |
| `observation.observation_type` | 否 | 用于兼容性校验的 observation 类型；默认为 `multimodal`。 |
| `observation.empty_observation_allowed` | 否 | Target 是否允许返回空 observation；默认为 `false`。 |
| `perception.enabled` | 否 | 是否启用 Target perception pipeline；默认为 `false`。 |
| `perception.strict_preflight` | 否 | 默认且必须保持 `true`，执行前校验所需 perception 输入。 |
| `perception.sensor_config_ref` | 否 | Sensor 配置文件。 |
| `perception.perception_config_ref` | 否 | Perception 配置文件。 |
| `perception.artifact_dir` | 否 | Perception artifact 目录。 |
| `config` | 否 | Target 专属参数，例如维度、限制、场景、控制模式和 reset 行为；默认为空 map。 |

LIBERO target-native benchmark 从 Target 的 `config` map 读取以下评测参数：

| 字段 | 默认值 | 含义 |
|:--|:--|:--|
| `control_mode` | `relative` | LIBERO action 约定，应与所选 policy 的控制方式一致。 |
| `seed` | `0` | 应用于每个 benchmark episode 的环境 seed。 |
| `retry_instruction_mode` | `original` | Recovery 指令行为：`original` 保持原任务；`verifier_rewrite` 使用 verifier 返回的非空 `replan_task_description` 替换原任务。 |

### Benchmark capability

普通任务可以不声明 `benchmark_capabilities`；参与结构化 benchmark 的
Target 必须声明。

| 字段 | 含义 |
|:--|:--|
| `benchmark_id` | Benchmark 类型，例如 `libero` 或 `behavior1k`。 |
| `suites` | Target 支持的 suite ID；空列表表示不限制 suite。 |
| `execution_modes[].mode` | `policy_loop` 或 `target_native`。 |
| `execution_modes[].interface` | `policy_loop` 使用 `rollout_episode_v1`；`target_native` 使用 `target_benchmark_job_v1`。 |
| `execution_modes[].reset_owner` | `policy_loop` 使用 `session_runner`；`target_native` 使用 `skillruntime`。 |

## `SKILLRUNTIME.md`

`SKILLRUNTIME.md` 使用 `version: runtime_skill_registry_v1`，主体为
`skillruntimes` 列表。

| 字段 | 必需 | 含义 |
|:--|:--:|:--|
| `id` | 是 | 唯一 ID，由 `skillruntime://<id>` 引用。 |
| `runtime` | 是 | 具体 SkillRuntime 实现。 |
| `runtime_kind` | 是 | `policy` 表示 policy loop；`builtin` 表示通过受控 Target 接口执行。 |
| `loop_mode` | 是 | Runtime 专属 loop mode。 |
| `agent_exposure` | 否 | `none`、`target_tools` 或 `constrained_target_tools`，默认为 `none`；它控制交互式 TargetTool 暴露，不控制 builtin 通过 `TargetSessionHandle` 访问 Target。 |
| `supported_target_kinds` | 是 | Runtime 接受的 Target kind。 |
| `policy.policy_client` | 仅 policy | Policy 协议 client。 |
| `policy.policy_adapter` | 仅 policy | Policy adapter 引用。 |
| `policy.supports_chunk` | 仅 policy | Policy 是否返回 action chunk。 |
| `observation_contract` | 否 | 与 Target 匹配的 observation 要求；默认为非空 multimodal contract。 |
| `supports_chunk` | 否 | SkillRuntime 是否消费 action chunk；默认为 `false`。 |
| `default_replan_every` | 否 | 默认 policy 刷新频率；默认为 `1`。 |
| `input_contract` / `output_contract` | 否 | 明确的 policy 输入和 action 输出；各自默认为空 map。 |
| `adapter_requirements` | 否 | 允许的 bridge 和禁止的隐式转换；默认为空 map。 |
| `requires` | 否 | Sensor、environment output、geometry、confidence 和 strict contract 要求；默认使用空依赖列表和 strict contract。 |
| `target_tool_policy` | 仅工具暴露 | 允许/禁止的操作和校验要求。 |

### Benchmark capability

支持 benchmark 的 SkillRuntime 声明一个 `benchmark` block：

| 字段 | 含义 |
|:--|:--|
| `benchmark_id` | 必须与 Target capability 和 Session benchmark 一致。 |
| `execution_mode` | `policy_loop` 或 `target_native`。 |
| `target_interface` | 必须与对应 mode 的 Target interface 一致。 |
| `result_schema` | `benchmark_execution_result_v1`。 |
| `reset_owner` | 必须与 Target reset owner 一致。 |

target-native 不是通用 SkillRuntime。每种 benchmark 使用自己的 builtin，
例如 `LiberoBenchmarkSkillRuntime` 或未来的 `B1KBenchmarkSkillRuntime`。

## `SESSIONS.md`

`SESSIONS.md` 使用 `version: runtime_sessions_v1`。Session 是 Target、
SkillRuntime、routing 和 execution 语义的确定选择，同时记录生命周期和结果。

### 标识与生命周期

| 字段 | 默认值 | 含义 |
|:--|:--|:--|
| `session_id` | 必填 | 唯一 Session ID。 |
| `parent_session_id` | `null` | replan 子 Session 的父 Session。 |
| `replan_attempt` | `0` | replan 代数，从零开始。 |
| `goal_id` / `parent_goal_id` | `null` | 可选 goal lineage。 |
| `horizon` | `null` | 可选 `short_term` 或 `long_term`。 |
| `target_ref` | 必填 | `target://<target-id>`。 |
| `skillruntime_ref` | 必填 | `skillruntime://<runtime-id>`。 |
| `task_description` | 必填 | 发送给选定执行路径的任务文本。 |
| `verification_profile` | `strict` | `strict`、`audit` 或 `recovery`。 |
| `status` | `pending` | Session 生命周期状态；Agent 使用 `pending` 创建任务。 |
| `priority` | `normal` | `low`、`normal` 或 `high`。 |
| `depends_on` | `[]` | 依赖元数据；当前 scheduler 会记录该字段，但不会据此阻塞执行。 |

`created_at`、`updated_at`、`claimed_by`、`claim_token`、`retry.attempted`
和 `result` 由 Runtime 维护。用户和 Agent 不应伪造 claim 信息或执行结果。
Runtime 负责推动 `status` 经过 claim、Preflight、execution、finalization、
verification 并进入终态。

### Timeout 与 retry

| 字段 | 默认值 | 含义 |
|:--|:--|:--|
| `timeouts.queue_timeout_s` | `30` | 等待 claim 的最长时间。 |
| `timeouts.preflight_timeout_s` | `20` | Preflight 最长时间。 |
| `timeouts.execute_timeout_s` | `300` | 执行最长时间；suite benchmark 必须覆盖整个 suite。 |
| `timeouts.policy_timeout_s` | `5` | 单次 policy 请求超时。 |
| `retry.max_retries` | `0` | Runtime 执行重试次数，与 verification replan 不同。 |
| `retry.attempted` | `0` | 由 Runtime 维护的 retry 计数。 |

### Routing 与 execution

| 字段 | 默认值 | 含义 |
|:--|:--|:--|
| `routing.target_endpoint` | `null` | 所选 Target 的 Session endpoint 覆盖；除非需要明确改道，否则使用 Registry 值。 |
| `routing.policy_endpoint` | `null` | PolicySkillRuntime 使用的 endpoint，或传入 target-native benchmark job 的 endpoint。 |
| `routing.adapter_resolution` | `strict_auto` | `strict_auto` 或 `strict_override`，两者都要求完整合法的 adapter plan。 |
| `routing.adapter_overrides` | `null` | 与 `strict_override` 配套的 adapter 覆盖。 |
| `execution.max_steps` | `600` | 最大执行 step 数。 |
| `execution.control_hz` | `null` | 可选控制频率。 |
| `execution.replan_every` | `8` | 没有 step 专属值时使用的默认 policy 刷新计数。 |
| `execution.replan_every_steps` | `null` | 一次 policy 响应最多执行多少个 action step 后重新请求 policy；该刷新频率与 verification `replan` 相互独立。 |
| `execution.action_chunk_mode` | `chunk_buffer` | `chunk_buffer`、`open_loop` 或 `single_step`。 |
| `execution.chunk_switch_mode` | `hard_switch` | `hard_switch` 或 `soft_blend`。 |
| `execution.reset_policy` | `session_runner` | 普通任务/policy-loop 使用 `session_runner`；target-native benchmark 使用 `skillruntime_managed`。 |
| `execution.steps` / `timeline` | `null` | 可选的 builtin step 或 timeline payload。 |

### Runtime hints 与 safety

| 字段 | 默认值 | 含义 |
|:--|:--|:--|
| `runtime_hints.perception_queries` | `[]` | 结构化 perception 或 benchmark 选择提示。 |
| `runtime_hints.force_environment_refresh` | `false` | 请求执行前刷新 environment。 |
| `runtime_hints.preferred_replan_every_steps` | `null` | 提供给 planning 的同一 policy 刷新频率偏好。 |
| `safety_profile.profile` | `default` | Safety profile 名称。 |
| `safety_profile.workspace_bounds` | `null` | Workspace bounds 引用。 |
| `safety_profile.stop_on_policy_timeout` | `true` | Policy 超时时停止执行。 |

### Benchmark metadata

Benchmark Session 必须包含 `benchmark` block。

| 字段 | 含义 |
|:--|:--|
| `benchmark_id` | 与 Target 和 SkillRuntime 匹配的 benchmark 类型。 |
| `suite_id` | 从 Target capability 中选择的 suite。 |
| `execution_mode` | `policy_loop` 或 `target_native`。 |
| `task_name`、`task_index`、`instance_id` | policy-loop Session 可选的 episode 标识。 |
| `policy_id` | 可选 benchmark policy 标识。 |
| `run_id` | Benchmark run lineage。 |

`policy_loop` 中，一个 episode 对应一个 root Session，
`execution.reset_policy` 为 `session_runner`。`target_native` 中，一个 suite
对应一个 root Session，`execution.reset_policy` 为 `skillruntime_managed`。

## Endpoint 与主机规则

- Remote Target 使用 `targetws://<host>:<port>`。
- PAOS policy service 使用实现支持的 scheme，例如
  `openpi://<host>:<port>`。
- Agent 必须能够访问 Target endpoint。
- target-native benchmark 由 Target 执行 policy attempt，因此 Target 主机
  必须能够访问 policy endpoint。
- Remote Target 需要 verification 时，将 `serviceHost` 设为 `"0.0.0.0"`，
  并将 `remoteTargetUrl` 设置为 Target 可访问的 HTTP URL。
- Verification Service 仍由 `paos agent` 管理，不需要用户维护第四个进程。

## LIBERO PI0.5 示例

项目 README 给出了三个终端的启动命令。对应的最小全局配置为：

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

在已有的 `libero_real_remote` Target 条目中应用以下字段，其余 contract
和 adapter 配置保持不变：

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

README 的服务命令使用以下示例参数：

| 命令参数 | 值 | 含义 |
|:--|:--|:--|
| Target `--host` / `--port` | `0.0.0.0` / `9002` | 监听 TargetWS 连接。 |
| `--camera-height` / `--camera-width` | `256` / `256` | 设置 LIBERO observation 图像尺寸。 |
| Target `--max-steps` | `300` | 限制单个 attempt 的 policy step 数。 |
| `--num-steps-wait` | `10` | 每次 episode reset 后执行场景稳定 step。 |
| Target `--control-mode` | `relative` | 匹配 PI0.5 LIBERO action 约定。 |
| Target `--seed` | `7` | 与 OpenPI 官方 LIBERO 评测 seed 保持一致。 |
| Policy `--policy-config` | `pi05_libero` | 选择官方 PI0.5 LIBERO policy 配置。 |
| Policy `--checkpoint-dir` | `gs://openpi-assets/checkpoints/pi05_libero` | 加载官方 PI0.5 checkpoint。 |
| Policy `--host` / `--port` | `0.0.0.0` / `8000` | 提供同机 endpoint `openpi://127.0.0.1:8000`。 |
| Agent `--workspace` | `~/.PhyAgentOS/workspace` | 选择 Runtime registry、Session 状态和 artifact workspace。 |

Agent 请求选择以下值：

| 选择项 | 值 |
|:--|:--|
| Target | `target://libero_real_remote` |
| SkillRuntime | `skillruntime://libero_target_benchmark` |
| Benchmark / suite | `libero` / `libero_spatial` |
| Execution mode | `target_native` |
| Reset policy | `skillruntime_managed` |
| Verification profile | `recovery` |
| Policy endpoint | `openpi://127.0.0.1:8000` |
| Task / init-state ID | `runtime_hints.perception_queries` 中的 `0-9` / `0-49` |
| Policy 刷新频率 | `execution.replan_every_steps: 5` |

Target 对每个逻辑 episode 只 reset 一次。每个 `replan` verdict 都会保留非空
`replan_task_description`。默认的 `retry_instruction_mode: original` 使下一次
attempt 继续使用原任务；设为 `verifier_rewrite` 时才将该描述作为 policy 指令。
两种模式都从失败环境状态继续且不再次 reset。Artifact 分别保留
first-attempt score 和 recovery-final score。

## 相关文档

- [用户手册](02-user-manual.md)
- [开发者手册](03-developer-manual.md)
- [项目 README](../../README_zh.md)
