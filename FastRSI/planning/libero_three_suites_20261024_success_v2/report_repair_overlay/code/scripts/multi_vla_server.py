"""Versioned PI0/PI05 resident server: unchanged pinned native policy + meters."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import time


def main():
    import numpy as np
    import torch
    from PhyAgentOS.runtime.policy.openpi.native_openpi_server import NativeOpenPIPolicy
    from fastrsi.resident_policy import ResidentPolicy, OpenPIStateGuard
    from resident_openpi_server import serve
    from pi_native_meter_v8 import install
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['pi0', 'pi05'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--receipt-dir', type=Path, required=True)
    # Kept only for the shared v7 launch interface; these models do not use it.
    parser.add_argument('--xvla-wire-chunk', type=int, choices=[1], default=1)
    args = parser.parse_args()
    random.seed(7); np.random.seed(7); torch.manual_seed(7); torch.cuda.manual_seed_all(7)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    ledger = args.receipt_dir.parent / 'POLICY_COST_EVENTS.jsonl'
    owner = None
    def event(kind, **fields):
        session = owner.active['session_id'] if owner and owner.active else None
        with ledger.open('a') as handle:
            handle.write(json.dumps({'event': kind, 'session_id': session,
                                    'mono_ns': time.monotonic_ns(), **fields}) + '\n')
    native = NativeOpenPIPolicy(policy_config=args.model + '_libero', checkpoint_dir=args.checkpoint,
        default_prompt=None, pytorch_device='cuda:0', disable_torch_compile=True, torch_seed=7)
    expected = 50 if args.model == 'pi0' else 10
    assert native.metadata['action_horizon'] == expected
    meter = install(native, event)
    # Snapshot after instrumentation is installed; no policy inference/warmup.
    guard = OpenPIStateGuard(native)
    infer = native.infer
    rpc = {'count': 0}
    def measured(observation):
        rpc['count'] += 1
        index = rpc['count']
        event('policy_rpc_begin', index=index)
        started = time.perf_counter()
        output = infer(observation)
        rows = output['actions']
        assert rows.shape == (expected, 7), rows.shape
        event('policy_rpc_end', index=index, returned_actions=len(rows),
              actions_sha256=hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest(),
              seconds=time.perf_counter() - started)
        return output
    native.infer = measured
    native.metadata.update(system_vla_model=args.model, native_meter=meter,
        checkpoint_config_sha256=hashlib.sha256((Path(args.checkpoint) / 'config.json').read_bytes()).hexdigest())
    owner = ResidentPolicy(native, guard=guard, max_sessions=1024, max_inferences=2000)
    with (args.receipt_dir / 'STARTUP.json').open('x') as handle:
        json.dump({'metadata': owner.metadata, 'rng_hashes': owner.rng.hashes,
                   'policy_state': guard.reset(), 'arguments': {k: str(v) for k, v in vars(args).items()}}, handle, indent=2)
    print('PI_RESIDENT_READY ' + args.model, flush=True)
    asyncio.run(serve(owner, host=args.host, port=args.port, receipt_dir=args.receipt_dir))


if __name__ == '__main__':
    main()
