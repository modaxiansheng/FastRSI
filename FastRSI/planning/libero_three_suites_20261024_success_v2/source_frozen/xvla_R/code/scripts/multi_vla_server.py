"""Exclusive resident serving for pinned VLA backends, with real generation costs."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time
import types
import numpy as np
import torch
from fastrsi.resident_policy import ResidentPolicy, OpenPIStateGuard
from resident_openpi_server import serve


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelStateGuard:
    def __init__(self, native, model_name):
        self.native = native
        self.model_name = model_name
        self.model = native.policy if model_name == 'xvla' else native.model
        self.baseline = self.fingerprint()
        self.digest = hashlib.sha256(repr(self.baseline).encode()).hexdigest()

    def fingerprint(self):
        assert all(not module.training for module in self.model.modules())
        return tuple((kind, name, id(tensor), tensor.data_ptr(), tuple(tensor.shape),
                      str(tensor.dtype), tensor._version)
                     for kind, items in [('parameter', self.model.named_parameters()),
                                         ('buffer', self.model.named_buffers())]
                     for name, tensor in items)

    def check(self):
        assert self.fingerprint() == self.baseline, 'VLA weights or buffers changed'

    def reset(self):
        self.check()
        cache_rows = 0
        if self.model_name == 'xvla':
            self.native.policy.reset()
            self.native._last_session_id = None
            cache_rows = len(self.native.policy._queues[self.native._action_key])
            assert cache_rows == 0
        return {'state_sha256': self.digest, 'action_cache_rows_after_reset': cache_rows,
                'scope': 'Tensor identity/version check and explicit native cache reset; not a full weight hash.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['pi0', 'xvla', 'openvla'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--receipt-dir', type=Path, required=True)
    parser.add_argument('--xvla-wire-chunk', type=int, choices=[1, 30], default=1)
    args = parser.parse_args()
    random.seed(7); np.random.seed(7); torch.manual_seed(7); torch.cuda.manual_seed_all(7)
    args.receipt_dir.mkdir(exist_ok=True, parents=True)
    ledger = args.receipt_dir.parent / 'POLICY_COST_EVENTS.jsonl'
    counter = {'rpc': 0, 'generation': 0, 'forward': 0}
    owner = None

    def event(kind, **fields):
        session = owner.active['session_id'] if owner and owner.active else None
        with ledger.open('a') as handle:
            handle.write(json.dumps({'event': kind, 'session_id': session,
                                    'mono_ns': time.monotonic_ns(), **fields}) + '\n')

    if args.model == 'pi0':
        from PhyAgentOS.runtime.policy.openpi.native_openpi_server import NativeOpenPIPolicy
        native = NativeOpenPIPolicy(policy_config='pi0_libero', checkpoint_dir=args.checkpoint,
                    default_prompt=None, pytorch_device='cuda:0', disable_torch_compile=True, torch_seed=7)
        guard = OpenPIStateGuard(native)
    elif args.model == 'xvla':
        module = load('pinned_xvla_server', '/vla-native/xvla_server_transport_v2.py')
        native = module.XVLALiberoPolicy(model_id=args.checkpoint, device='cuda:0',
                                        image_size=224, empty_camera_size=224)
        guard = ModelStateGuard(native, args.model)
        generate = native.policy._get_action_chunk
        def measured_generate(*values, **kw):
            counter['generation'] += 1
            event('action_generation_begin', index=counter['generation'])
            started = time.perf_counter()
            output = generate(*values, **kw)
            event('action_generation_end', index=counter['generation'], seconds=time.perf_counter()-started)
            return output
        native.policy._get_action_chunk = measured_generate
        if args.xvla_wire_chunk == 30:
            def batched(observation):
                sid = observation.get('session_id')
                if sid is not None and sid != native._last_session_id:
                    native.policy.reset(); native._last_session_id = str(sid)
                batch = native.preprocess(module._libero_observation_to_lerobot_frame(
                    observation, image_size=224, empty_camera_size=224))
                pending = len(native.policy._queues[native._action_key])
                count = pending or native.policy.config.n_action_steps
                assert 1 <= count <= 30
                actions = []
                with torch.inference_mode():
                    for _ in range(count):
                        action = native.postprocess(native.policy.select_action(batch))
                        action = native.env_postprocess({native._action_key: action})[native._action_key]
                        rows = module._action_to_numpy(action)
                        assert rows.shape == (1, 7), rows.shape
                        actions.append(rows)
                assert len(native.policy._queues[native._action_key]) == 0
                return {'actions': np.ascontiguousarray(np.concatenate(actions), dtype=np.float32),
                        'policy_meta': {'backend': 'xvla_lerobot', 'wire_chunk': count,
                                        'scope': 'Drain only the native cached action block, without regenerating it.'}}
            native.infer = batched
        native.metadata.update(action_horizon=1 if args.xvla_wire_chunk == 1 else 30,
                               xvla_wire_chunk=args.xvla_wire_chunk)
    else:
        module = load('pinned_openvla_server', '/vla-native/openvla_server_promptfix.py')
        native = module.OpenVLALiberoPolicy(model_path=args.checkpoint, unnorm_key='libero_10',
            device='cuda:0', torch_dtype='bfloat16', image_size=224, center_crop=True,
            load_in_8bit=False, load_in_4bit=False, attn_implementation='eager', seed=7,
            code_revision='47a0ec7fc4ec123775a391911046cf33cf9ed83f')
        guard = ModelStateGuard(native, args.model)
        native.metadata.update(action_horizon=1)

    def track_forward(model, method, scope):
        old_forward = getattr(model, method)
        def measured_forward(*values, **kw):
            counter['forward'] += 1
            index = counter['forward']
            event('neural_forward_begin', index=index, scope=scope)
            output = old_forward(*values, **kw)
            event('neural_forward_end', index=index, scope=scope)
            return output
        setattr(model, method, measured_forward)
    if args.model == 'xvla':
        track_forward(native.policy.model, 'forward_vlm', 'xvla_vision_language_encoding')
        track_forward(native.policy.model.transformer, 'forward', 'xvla_action_denoising')
    elif args.model == 'openvla':
        track_forward(native.model, 'forward', 'openvla_autoregressive_forward')
    infer = native.infer
    def measured_infer(observation):
        counter['rpc'] += 1
        if args.model != 'xvla':
            counter['generation'] += 1
            event('action_generation_begin', index=counter['generation'])
        event('policy_rpc_begin', index=counter['rpc'])
        before = time.perf_counter()
        output = infer(observation)
        if args.model != 'xvla':
            event('action_generation_end', index=counter['generation'], seconds=time.perf_counter()-before)
        rows = output['actions']
        event('policy_rpc_end', index=counter['rpc'], returned_actions=len(rows),
              actions_sha256=hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest(),
              seconds=time.perf_counter()-before)
        return output
    native.infer = measured_infer
    native.metadata.update(system_vla_model=args.model,
        checkpoint_config_sha256=hashlib.sha256((Path(args.checkpoint)/'config.json').read_bytes()).hexdigest())
    owner = ResidentPolicy(native, guard=guard, max_sessions=1024, max_inferences=2000)
    with (args.receipt_dir/'STARTUP.json').open('x') as handle:
        json.dump({'metadata':owner.metadata,'rng_hashes':owner.rng.hashes,'policy_state':guard.reset(),
                   'arguments':{k:str(v) for k,v in vars(args).items()}},handle,indent=2)
    print('VLA_RESIDENT_READY ' + args.model, flush=True)
    asyncio.run(serve(owner,host=args.host,port=args.port,receipt_dir=args.receipt_dir))


if __name__ == '__main__':
    main()
