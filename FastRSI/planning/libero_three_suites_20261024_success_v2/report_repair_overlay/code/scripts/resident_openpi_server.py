#!/usr/bin/env python3
"""Opt-in experimental OpenPI server. Never modifies the frozen native server.

After metadata, admin sockets send {"__fastrsi_resident__": {"op": "begin",
"session_id": "...", "request_id": "...", "max_inferences": 104}}. The admin
socket cannot infer. Wait for receipt, THEN connect one normal native client.
Close that client before sending end on an admin socket. For another native
recovery attempt in the SAME case, use op=handoff with session_id, request_id,
expected_generation. It rotates the owner binding WITHOUT resetting RNG or
the cumulative inference count. Connect the next client after its receipt.
An end receipt is a barrier; create the next case with a new ID. status is read-only.
"""
import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import traceback

from fastrsi.resident_policy import ADMIN_KEY, ResidentPolicy, ResidentProtocolError


async def serve(policy, *, host, port, receipt_dir):
    import websockets
    import websockets.asyncio.server as websocket_server
    from PhyAgentOS.runtime.policy.msgpack_numpy import packb, unpackb
    from PhyAgentOS.runtime.policy.openpi.native_openpi_server import _health_check

    async def handler(socket):
        identity = str(socket.id)
        role = None
        policy.connect(identity)
        try:
            await socket.send(packb(policy.metadata))
            while True:
                message = await socket.recv()
                if isinstance(message, str):
                    raise ResidentProtocolError("binary msgpack required")
                data = unpackb(message)
                if not isinstance(data, dict):
                    raise ResidentProtocolError("request must be a map")
                is_admin = ADMIN_KEY in data
                wanted = "admin" if is_admin else "infer"
                if role not in (None, wanted):
                    raise ResidentProtocolError("cannot mix admin and inference on one connection")
                role = wanted
                if is_admin:
                    if set(data) != {ADMIN_KEY} or not isinstance(data[ADMIN_KEY], dict):
                        raise ResidentProtocolError("invalid admin envelope")
                    result = await asyncio.to_thread(policy.admin, data[ADMIN_KEY])
                    if data[ADMIN_KEY].get("op") in {"begin", "end", "handoff"}:
                        name = hashlib.sha256(data[ADMIN_KEY]["request_id"].encode()).hexdigest()
                        path = receipt_dir / (name + ".json")
                        content = json.dumps({"command": data[ADMIN_KEY], "receipt": result}, indent=2) + "\n"
                        if path.exists():
                            if path.read_text() != content:
                                raise ResidentProtocolError("receipt path collision")
                        else:
                            path.write_text(content)
                else:
                    result = await asyncio.to_thread(policy.infer, data, connection_id=identity)
                await socket.send(packb(result))
        except websockets.ConnectionClosed:
            pass
        except Exception:
            try:
                await socket.send(traceback.format_exc())
                await socket.close(code=1011, reason="Resident policy error")
            finally:
                logging.exception("resident request rejected")
        finally:
            policy.disconnect(identity)

    async with websocket_server.serve(handler, host, port, compression=None, max_size=None,
            process_request=_health_check, ping_interval=None, ping_timeout=None) as server:
        print(f"[resident-openpi] listening {host}:{port}; explicit begin required", flush=True)
        await server.serve_forever()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", default="pi05_libero")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--pytorch-device", default="cuda:0")
    parser.add_argument("--disable-torch-compile", action="store_true", required=True)
    parser.add_argument("--torch-seed", type=int, default=7)
    parser.add_argument("--experimental-action-horizon", type=int, choices=(10, 20), default=None,
        help="Opt in to a fixed experimental prediction horizon; omitted keeps the native loader")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-sessions", type=int, default=64)
    parser.add_argument("--max-inferences", type=int, default=1024)
    parser.add_argument("--receipt-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.experimental_action_horizon is not None:
        if args.policy_config != "pi05_libero" or args.default_prompt is not None:
            parser.error("experimental horizon requires pi05_libero and no default prompt")
    return args


def create_policy(args):
    horizon = args.experimental_action_horizon
    if horizon is not None:
        # Validate before importing/loading the model; never silently ignore flags.
        if (type(horizon) is not int or horizon not in (10, 20)
                or args.policy_config != "pi05_libero" or args.default_prompt is not None
                or args.disable_torch_compile is not True):
            raise ValueError("experimental horizon requires eager pi05_libero H10/H20 without default prompt")
        from fastrsi.experimental_horizon_policy import ExperimentalHorizonPolicy
        return ExperimentalHorizonPolicy(checkpoint_dir=args.checkpoint_dir,
            action_horizon=horizon, pytorch_device=args.pytorch_device, torch_seed=args.torch_seed)
    from PhyAgentOS.runtime.policy.openpi.native_openpi_server import NativeOpenPIPolicy
    return NativeOpenPIPolicy(policy_config=args.policy_config,
        checkpoint_dir=args.checkpoint_dir, default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device, disable_torch_compile=args.disable_torch_compile,
        torch_seed=args.torch_seed)


def main():
    args = parse_args()
    native = create_policy(args)
    # No warmup/inference between policy construction and snapshot. Explicit H10
    # uses the same native load; its extra metadata/checks draw no random numbers.
    resident = ResidentPolicy(native, max_sessions=args.max_sessions, max_inferences=args.max_inferences)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    startup = args.receipt_dir / "STARTUP.json"
    with startup.open("x") as handle:
        json.dump({"metadata": resident.metadata, "rng_hashes": resident.rng.hashes,
                   "policy_state": resident.guard.reset(), "argv": {k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(args).items()}}, handle, indent=2)
    print(f"[resident-openpi] captured post-load RNG {resident.rng.digest}", flush=True)
    asyncio.run(serve(resident, host=args.host, port=args.port, receipt_dir=args.receipt_dir))


if __name__ == "__main__":
    main()
