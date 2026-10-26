# PhyAgentOS 通信架构

> 版本：v0.1.6 · [English](COMMUNICATION_en.md)

## 1. 三个通信边界

PhyAgentOS 不使用一条总线承载所有职责，而是分为：

1. Agent 消息边界：Channel ↔ MessageBus ↔ AgentLoop。
2. Agent/Runtime 边界：Markdown + YAML 工作区协议。
3. Runtime/Target/Policy 边界：本地调用或 msgpack-over-WebSocket RPC。

```text
Channel ─ MessageBus ─ AgentLoop
                         │
                         │ TARGETS / SKILLRUNTIME / SESSIONS
                         ▼
                  WatchdogSupervisor
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       targetws:// RPC       policy endpoint RPC
```

## 2. Agent 消息边界

Channel 把外部消息转换为 `InboundMessage` 并发布到 MessageBus。AgentLoop 按 session key 处理上下文、模型与 Tool Call，再通过 `OutboundMessage` 返回 Channel。CLI 单轮路径可直接调用 `process_direct`，但 Agent 内部闭环相同。

消息通道不能直接编辑 Runtime Session 状态，也不能直接调用 Target。

## 3. 工作区协议边界

| 文件 | 主要作者 | 主要读者 |
|---|---|---|
| `RUNTIME.md` | RuntimeWorkspaceManager | Agent |
| `TARGETS.md` | 用户/集成开发者 | Agent、Scheduler、Preflight |
| `SKILLRUNTIME.md` | 用户/集成开发者 | Agent、Scheduler、Preflight |
| `SESSIONS.md` | Agent、Registry、Verifier | Agent、Watchdog、Verifier |
| `ENVIRONMENT.md` | Runtime/Perception | Agent、Skill |
| `LOG.md` | ResultWriter | Agent、Benchmarking、运维 |
| `LESSONS.md` | Runtime/Verifier | Agent、开发者 |

`TARGETS.md` 和 `SKILLRUNTIME.md` 是能力声明，`SESSIONS.md` 是队列与状态，`ENVIRONMENT.md` 是事实快照；三者不能相互代替。

## 4. 原子性与所有权

- `SessionRegistry` 用 `SESSIONS.md.lock` 和原子写保护 claim/状态变更。
- `EnvironmentWriter` reconcile 后原子写 Environment v2。
- Agent 追加 pending Session 时保留已有 Session 和 Result。
- Runtime 不修改 Agent 的任务规划；Agent 不伪造 Runtime 终态。
- Verification review 只追加审计信息，不改变既有终态。

## 5. Remote Target Envelope

```yaml
version: phyagentos.runtime_rpc.v2
type: target.action_chunk
session_id: sess_001
target_id: target_001
skillruntime_id: skill_001
episode_id: ep_001
seq: 17
timestamp_ns: 1779030000000000000
trace_id: trace_001
payload: {}
```

当前 Transport：

- Remote Target：WebSocket + msgpack
- Local Target：进程内调用，保持相同生命周期语义
- Policy：按 Endpoint Scheme 构造独立 Client

Response 必须匹配请求的 seq、session、target 与 skillruntime。`target.observe` 的响应类型为 `target.observation`；`agent_tool.call` 的响应类型为 `agent_tool.result`。

## 6. Target 消息集合

当前 Proxy/Server 生命周期使用：

```text
target.describe
target.configure_session
target.start_session
target.reset
target.observe / target.observation
target.action_chunk
target.execution_status
agent_tool.call / agent_tool.result
target.cancel
target.close
runtime.error
```

`SessionRunner` 在 Preflight accepted 后才能进入 Target lifecycle。Skill Runtime 后续调用必须经过 `TargetSessionHandle`。

## 7. Policy Endpoint

| Scheme | 语义 |
|---|---|
| `dummy://local` | 本地确定性 Dummy Policy |
| `openpi://host:port` | OpenPI WebSocket Client |
| `policyws://host:port` | OpenPI-compatible Client |
| `b1k-ws://host:port` | BEHAVIOR-1K Policy Client |

Policy Payload 由 PolicyAdapter 生成，Target 原始 Observation 不应绕过 Adapter 直接发送。

## 8. 数据放置规则

- 小型结构化控制数据放 RPC Payload。
- 大型图像/Mask/Depth/Point Cloud/Trace 写 Artifact，并用 Path/URI 引用。
- ENVIRONMENT 只保存 compact state 与 Artifact 引用。
- `episode.json` 保存单次执行事实；`LOG.md` 保存历史索引。
- Verification Bundle 保存任务、环境、历史和 RGB 路径。

## 9. 错误边界

`runtime.error` 应包含稳定 `error_code` 和可操作 `message`。连接错误、协议不匹配、Preflight rejection、Policy failure、Target failure、timeout 与 semantic failure 是不同错误层，不应折叠为同一 task failure。

## 10. 当前限制与演进

v0.1.6 尚未完整实现远端在线 Healthcheck、Runner heartbeat RPC、Session 依赖调度和真机 Operator Override 协议。HAL v3 将继续收敛统一 Envelope、严格 Contract 与 Target-side safety，但不会让 Runtime RPC 绕过工作区 Session 状态机。

## 相关文档

- [集成开发指南](README.md)
- [开发者手册](../zh/03-developer-manual.md)
- [Runtime 感知说明](../../PhyAgentOS/runtime/perception/README.md)
