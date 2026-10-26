"""Standalone TargetWS server for Unitree Go2 builtin commands.

Run this in an environment that has unitree_sdk2py installed. The server speaks
the same msgpack-over-WebSocket TargetWS protocol used by the PhyAgentOS runtime.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
import traceback
from typing import Any

import msgpack

RPC_VERSION = "phyagentos.runtime_rpc.v2"
logger = logging.getLogger(__name__)

DEFAULT_LIMITS = {
    "vx": (-0.5, 0.5),
    "vy": (-0.2, 0.2),
    "vyaw": (-0.5, 0.5),
    "duration_s": (0.1, 1.0),
}
SDK_OK = 0


class TargetProtocolError(Exception):
    """Raised when an RPC or command payload is invalid."""


def packb(payload: Any) -> bytes:
    return msgpack.packb(payload, use_bin_type=True)


def unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, raw=False)


def make_response(request: dict[str, Any], response_type: str, payload: dict[str, Any]) -> bytes:
    return packb(
        {
            "version": RPC_VERSION,
            "type": response_type,
            "session_id": request.get("session_id"),
            "target_id": request.get("target_id"),
            "skillruntime_id": request.get("skillruntime_id"),
            "episode_id": request.get("episode_id"),
            "seq": int(request.get("seq", 0)),
            "timestamp_ns": time.time_ns(),
            "trace_id": request.get("trace_id"),
            "payload": payload,
        }
    )


class Go2SportBackend:
    """Thin wrapper around Unitree SportClient."""

    def __init__(self, *, network_interface: str, robot_ip: str, dry_run: bool = False):
        self.network_interface = network_interface
        self.robot_ip = robot_ip
        self.dry_run = dry_run
        self._client = None
        self._initialized = False
        self.command_log: list[dict[str, Any]] = []

    def connect(self) -> None:
        if self._initialized:
            return
        if self.dry_run:
            self._initialized = True
            self.command_log.append({"command": "connect", "dry_run": True})
            return
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.go2.sport.sport_client import SportClient
        except ModuleNotFoundError as exc:
            raise TargetProtocolError(
                "unitree_sdk2py/cyclonedds is required outside --dry-run; "
                "install it in the Go2 TargetWS environment"
            ) from exc
        ChannelFactoryInitialize(0, self.network_interface)
        client = SportClient()
        client.SetTimeout(10.0)
        client.Init()
        self._client = client
        self._initialized = True

    def stand_up(self) -> Any:
        self.connect()
        return self._call("StandUp")

    def balance_stand(self) -> Any:
        self.connect()
        return self._call("BalanceStand")

    def recovery_stand(self) -> Any:
        self.connect()
        return self._call("RecoveryStand")

    def stand_down(self) -> Any:
        self.connect()
        return self._call("StandDown")

    def damp(self) -> Any:
        self.connect()
        return self._call("Damp")

    def stop(self) -> Any:
        self.connect()
        return self._call("StopMove")

    def move(self, *, vx: float, vy: float, vyaw: float, duration_s: float, control_hz: float) -> dict[str, Any]:
        self.connect()
        interval_s = 1.0 / max(1.0, float(control_hz))
        steps = max(1, int(math.ceil(duration_s / interval_s)))
        started = time.monotonic()
        responses = []
        try:
            for _ in range(steps):
                response = self._call("Move", vx, vy, vyaw)
                responses.append(_safe_response(response))
                _raise_for_sdk_error("Move", response)
                remaining = duration_s - (time.monotonic() - started)
                if remaining <= 0:
                    break
                time.sleep(min(interval_s, remaining))
        finally:
            stop_response = self.stop()
        _raise_for_sdk_error("StopMove", stop_response)
        return {
            "move_steps": len(responses),
            "elapsed_s": round(time.monotonic() - started, 3),
            "responses": responses,
            "stop_response": _safe_response(stop_response),
        }

    def close(self) -> None:
        if self._initialized:
            try:
                self.stop()
            except Exception:
                logger.debug("Go2 stop during close failed", exc_info=True)

    def _call(self, method_name: str, *args: Any) -> Any:
        record = {"command": method_name, "args": list(args), "dry_run": self.dry_run}
        self.command_log.append(record)
        if self.dry_run:
            return {"dry_run": True, "method": method_name, "args": list(args)}
        if self._client is None:
            raise TargetProtocolError("Go2 SportClient is not initialized")
        method = getattr(self._client, method_name)
        response = method(*args)
        record["response"] = _safe_response(response)
        logger.info("Go2 SDK call %s args=%s response=%s", method_name, list(args), record["response"])
        return response


class Go2BuiltinRuntime:
    AGENT_TOOLS = [
        {
            "name": "execute_step",
            "description": "Execute one constrained Unitree Go2 builtin command",
            "parameters": {
                "step": {
                    "type": "object",
                    "commands": [
                        "stand_up",
                        "balance_stand",
                        "recovery_stand",
                        "stand_down",
                        "squat",
                        "damp",
                        "stop",
                        "move",
                    ],
                }
            },
        }
    ]

    def __init__(
        self,
        *,
        network_interface: str,
        robot_ip: str,
        dry_run: bool,
        control_hz: float = 10.0,
    ):
        self.network_interface = network_interface
        self.robot_ip = robot_ip
        self.dry_run = dry_run
        self.control_hz = float(control_hz)
        self.backend = Go2SportBackend(
            network_interface=network_interface,
            robot_ip=robot_ip,
            dry_run=dry_run,
        )
        self.session_id: str | None = None
        self.step_idx = 0
        self._last_obs = self._empty_observation()
        self._last_status = self._base_status(message="idle")

    def describe(self) -> dict[str, Any]:
        return {
            "runtime": "Go2BuiltinTargetRuntime",
            "robot_id": "unitree_go2",
            "robot_ip": self.robot_ip,
            "network_interface": self.network_interface,
            "dry_run": self.dry_run,
            "observation_schema": {"type": "empty"},
            "action_contract": {
                "id": "go2_builtin_command_v1",
                "accepted_representations": ["builtin_command"],
                "shape": [1, 1],
                "dtype": "object",
                "control_hz": self.control_hz,
            },
            "agent_tools": self.AGENT_TOOLS,
            "safety_limits": self._limits_payload(),
        }

    def configure_session(self, ctx: dict[str, Any]) -> dict[str, Any]:
        self._merge_ctx(ctx)
        return {"configured": True, "session_id": self.session_id, "go2": self._metadata()}

    def start_session(self, ctx: dict[str, Any]) -> dict[str, Any]:
        self._merge_ctx(ctx)
        if not self.dry_run:
            self.backend.connect()
        return {"started": True, "session_id": self.session_id, "go2": self._metadata()}

    def reset(self, ctx: dict[str, Any]) -> dict[str, Any]:
        self._merge_ctx(ctx)
        self.step_idx = 0
        self._last_obs = self._empty_observation()
        self._last_status = self._base_status(message="ready")
        return self._last_obs

    def observe(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        del payload
        return self._last_obs

    def action_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        del chunk
        raise TargetProtocolError("Go2 builtin target does not accept action chunks; use execute_step")

    def execution_status(self) -> dict[str, Any]:
        return dict(self._last_status)

    def describe_agent_tools(self) -> dict[str, Any]:
        return {"tools": self.AGENT_TOOLS}

    def call_agent_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name != "execute_step":
            raise TargetProtocolError("unknown agent tool: %s" % tool_name)
        step_def = dict(arguments.get("step") or {})
        result = self._execute_step(step_def)
        self.step_idx += 1
        message = str(result.get("message", "ok"))
        success = bool(result.get("success", True))
        self._last_obs = self._empty_observation()
        self._last_status = self._base_status(message=message, success=success, command=result)
        return {
            "tool_name": tool_name,
            "result": {
                "success": success,
                "message": message,
                "reward": 1.0 if success else 0.0,
                "info": result,
                "step_idx": self.step_idx,
            },
        }

    def cancel(self, reason: str) -> dict[str, Any]:
        try:
            self.backend.stop()
        finally:
            self._last_status = self._base_status(message="cancelled: %s" % reason, success=False)
        return {"cancelled": True, "reason": reason}

    def close(self) -> dict[str, Any]:
        self.backend.close()
        return {"closed": True}

    def _execute_step(self, step_def: dict[str, Any]) -> dict[str, Any]:
        command = _normalize_command(step_def)
        params = dict(step_def.get("params") or {})
        if command == "stand_up":
            response = self.backend.stand_up()
            return _step_result("stand_up", response)
        if command == "balance_stand":
            response = self.backend.balance_stand()
            return _step_result("balance_stand", response)
        if command == "recovery_stand":
            response = self.backend.recovery_stand()
            return _step_result("recovery_stand", response)
        if command in {"stand_down", "squat"}:
            response = self.backend.stand_down()
            return _step_result("stand_down", response)
        if command == "damp":
            response = self.backend.damp()
            return _step_result("damp", response)
        if command == "stop":
            response = self.backend.stop()
            return _step_result("stop", response)
        if command == "move":
            if "params" not in step_def or not isinstance(step_def.get("params"), dict):
                raise TargetProtocolError("Go2 move command requires params.vx/params.vy/params.vyaw/params.duration_s")
            clipped = _clip_move_params(params)
            move_result = self.backend.move(control_hz=self.control_hz, **clipped)
            return {
                "success": True,
                "message": "move",
                "params": clipped,
                "move": move_result,
            }
        raise TargetProtocolError("unsupported Go2 command: %s" % command)

    def _merge_ctx(self, ctx: dict[str, Any]) -> None:
        self.session_id = ctx.get("session_id", self.session_id)

    def _empty_observation(self) -> dict[str, Any]:
        return {
            "observation_id": "go2_obs_%d" % self.step_idx,
            "go2": self._metadata(),
        }

    def _metadata(self) -> dict[str, Any]:
        return {
            "robot_ip": self.robot_ip,
            "network_interface": self.network_interface,
            "dry_run": self.dry_run,
            "step_index": self.step_idx,
        }

    def _base_status(
        self,
        *,
        message: str,
        success: bool = True,
        command: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "accepted": True,
            "buffered_steps": 0,
            "executed_steps": self.step_idx,
            "target_step_index": self.step_idx,
            "need_replan": False,
            "safety_status": "ok" if success else "stopped",
            "success": success,
            "done": True,
            "reward": 1.0 if success else 0.0,
            "message": message,
            "obs": self._last_obs,
            "go2": self._metadata(),
            "command": command or {},
        }

    @staticmethod
    def _limits_payload() -> dict[str, list[float]]:
        return {key: [float(value[0]), float(value[1])] for key, value in DEFAULT_LIMITS.items()}


def _normalize_command(step_def: dict[str, Any]) -> str:
    if "command" in step_def:
        return str(step_def["command"]).strip().lower()
    if "text" in step_def:
        text = str(step_def["text"]).strip().lower()
        aliases = {
            "stand up": "stand_up",
            "stand": "stand_up",
            "站起": "stand_up",
            "起立": "stand_up",
            "balance stand": "balance_stand",
            "balanced stand": "balance_stand",
            "平衡站立": "balance_stand",
            "recovery": "recovery_stand",
            "recovery stand": "recovery_stand",
            "恢复站立": "recovery_stand",
            "stand down": "stand_down",
            "squat": "squat",
            "蹲下": "stand_down",
            "趴下": "stand_down",
            "stop": "stop",
            "停止": "stop",
            "damp": "damp",
            "阻尼": "damp",
        }
        if text in aliases:
            return aliases[text]
    if "mode" in step_def and str(step_def["mode"]).strip().lower() == "move":
        return "move"
    raise TargetProtocolError("Go2 step requires command/text/mode")


def _clip_move_params(params: dict[str, Any]) -> dict[str, float]:
    values = {
        "vx": _float_param(params, "vx", 0.0),
        "vy": _float_param(params, "vy", 0.0),
        "vyaw": _float_param(params, "vyaw", _float_param(params, "omega", 0.0)),
        "duration_s": _float_param(params, "duration_s", _float_param(params, "duration", 1.0)),
    }
    return {name: _clip(value, *DEFAULT_LIMITS[name]) for name, value in values.items()}


def _float_param(params: dict[str, Any], name: str, default: float) -> float:
    value = params.get(name, default)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TargetProtocolError("Go2 move parameter %s must be numeric" % name) from exc
    if not math.isfinite(result):
        raise TargetProtocolError("Go2 move parameter %s must be finite" % name)
    return result


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _safe_response(response: Any) -> Any:
    if response is None or isinstance(response, (bool, int, float, str)):
        return response
    if isinstance(response, (list, tuple)):
        return [_safe_response(item) for item in response]
    if isinstance(response, dict):
        return {str(key): _safe_response(value) for key, value in response.items()}
    return repr(response)


def _step_result(message: str, response: Any) -> dict[str, Any]:
    _raise_for_sdk_error(message, response)
    return {"success": True, "message": message, "response": _safe_response(response)}


def _raise_for_sdk_error(command: str, response: Any) -> None:
    if isinstance(response, int) and response != SDK_OK:
        raise TargetProtocolError("Go2 SDK command %s returned error code %d" % (command, response))


def _dispatch(runtime: Go2BuiltinRuntime, request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    rtype = request.get("type")
    payload = request.get("payload") or {}
    if rtype == "target.describe":
        return rtype, runtime.describe()
    if rtype == "target.configure_session":
        return rtype, runtime.configure_session(payload)
    if rtype == "target.start_session":
        return rtype, runtime.start_session(payload)
    if rtype == "target.reset":
        return rtype, runtime.reset(payload)
    if rtype == "target.observe":
        return "target.observation", runtime.observe(payload)
    if rtype == "target.action_chunk":
        return rtype, runtime.action_chunk(payload)
    if rtype == "target.execution_status":
        return rtype, runtime.execution_status()
    if rtype == "agent_tool.describe":
        return rtype, runtime.describe_agent_tools()
    if rtype == "agent_tool.call":
        tool_name = str(payload.get("tool_name", ""))
        arguments = dict(payload.get("arguments") or {})
        return "agent_tool.result", runtime.call_agent_tool(tool_name, arguments)
    if rtype == "target.cancel":
        return rtype, runtime.cancel(str(payload.get("reason", "cancelled")))
    if rtype == "target.close":
        return rtype, runtime.close()
    raise TargetProtocolError("unsupported target RPC type: %s" % rtype)


def _handle_request(runtime: Go2BuiltinRuntime, raw: bytes) -> bytes:
    request = unpackb(raw)
    try:
        rtype, payload = _dispatch(runtime, request)
        return make_response(request, rtype, payload)
    except Exception as exc:
        logger.warning("Go2 TargetWS request failed: %s", exc, exc_info=True)
        return make_response(
            request,
            "runtime.error",
            {
                "error_code": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


def serve_blocking(runtime: Go2BuiltinRuntime, host: str, port: int) -> None:
    from websockets.sync.server import serve as sync_serve

    def handle(websocket: Any) -> None:
        peer = getattr(websocket, "remote_address", None)
        logger.info("Go2 TargetWS client connected: %s", peer)
        try:
            for raw in websocket:
                if isinstance(raw, str):
                    websocket.send(
                        make_response(
                            {"seq": 0},
                            "runtime.error",
                            {"error_code": "BAD_PAYLOAD", "message": "expected binary msgpack"},
                        )
                    )
                    continue
                websocket.send(_handle_request(runtime, raw))
        except Exception as exc:
            if type(exc).__name__ != "ConnectionClosed":
                logger.info("Go2 TargetWS client disconnected: %s (%s)", peer, exc)

    with sync_serve(handle, host, port, max_size=None) as server:
        print("Go2 TargetWS server listening on targetws://%s:%d" % (host, port), flush=True)
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Unitree Go2 TargetWS server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--network-interface", default="enp4s0")
    parser.add_argument("--robot-ip", default="192.168.123.161")
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    runtime = Go2BuiltinRuntime(
        network_interface=args.network_interface,
        robot_ip=args.robot_ip,
        dry_run=bool(args.dry_run),
        control_hz=float(args.control_hz),
    )
    try:
        serve_blocking(runtime, args.host, args.port)
    except KeyboardInterrupt:
        print("\n[go2/server] stopped", flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
