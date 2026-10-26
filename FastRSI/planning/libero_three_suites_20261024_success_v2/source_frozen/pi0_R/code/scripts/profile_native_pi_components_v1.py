"""Paired passive component profiling of the exact frozen eager native policy.

Retained public inputs only; no target, robot steps, teacher updates or actor.
CUDA events measure component elapsed latency, not active-kernel utilization.
"""
import contextlib
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from fastrsi.resident_policy import RNGSnapshot, OpenPIStateGuard


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def behavior_hash(result):
    """Bind every actual response field except the two declared timing fields."""
    def encode(value):
        if isinstance(value, np.ndarray):
            return dict(dtype=str(value.dtype), shape=list(value.shape),
                        sha256=hashlib.sha256(value.tobytes()).hexdigest())
        if isinstance(value, dict):
            return {k:encode(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        return value
    payload = dict(result)
    timing = payload.pop('policy_timing')
    assert set(timing) == {'infer_ms'}
    payload['policy_meta'] = dict(payload['policy_meta'])
    payload['policy_meta'].pop('policy_latency_ms')
    return hashlib.sha256(json.dumps(encode(payload), sort_keys=True, allow_nan=False).encode()).hexdigest()


class PassiveTap:
    def __init__(self, native):
        self.native = native
        self.events = []
        self.features = {}
        self.stack = contextlib.ExitStack()

    def patch(self, obj, name, wrapper):
        own = name in vars(obj)
        original = getattr(obj, name)
        setattr(obj, name, wrapper(original))
        self.stack.callback(lambda: setattr(obj, name, original) if own else delattr(obj, name))

    def timed(self, name, original, args, kwargs, capture=None):
        first = torch.cuda.Event(enable_timing=True)
        last = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        first.record()
        result = original(*args, **kwargs)
        last.record()
        self.events.append((name, first, last, time.perf_counter()-start))
        if capture is not None:
            capture(result)
        return result

    def __enter__(self):
        m = self.native.policy._model
        try:
            self.patch(m, '_preprocess_observation', lambda f: lambda *a, **kw:
                       self.timed('preprocess', f, a, kw))
            def embed(original):
                def call(*args, **kwargs):
                    def capture(result):
                        self.features['prefix_embedding'] = result[0].detach()
                        self.features['prefix_mask'] = result[1].detach()
                    return self.timed('image_language_embedding', original, args, kwargs, capture)
                return call
            self.patch(m, 'embed_prefix', embed)
            def forward(original):
                def call(*args, **kwargs):
                    embeddings = kwargs['inputs_embeds']
                    if embeddings[1] is None:
                        return self.timed('prefix_transformer', original, args, kwargs,
                                          lambda result: self.features.update(prefix_hidden=result[0][0].detach()))
                    return original(*args, **kwargs)
                return call
            self.patch(m.paligemma_with_expert, 'forward', forward)
            self.patch(m, 'denoise_step', lambda f: lambda *a, **kw:
                       self.timed('denoise_step', f, a, kw))
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        self.stack.close()

    def measurement(self):
        torch.cuda.synchronize()
        return [dict(component=name, device_elapsed_ms=first.elapsed_time(last),
                     host_enqueue_seconds=host)
                for name, first, last, host in self.events]


def main():
    output = Path('/output')
    inputs = Path('/audit/inputs')
    registration = json.loads((inputs/'REGISTRATION.json').read_text())
    start = time.time()
    state = dict(status='LOADING', started_at=start, robot_episodes=0, new_policy_inferences=0)
    save(output/'STATUS.json', state)
    for name, digest in registration['input_manifest'].items():
        assert hashlib.sha256((inputs/name).read_bytes()).hexdigest() == digest
    for module, digest in registration['bound_module_sha256'].items():
        file = Path(inspect.getfile(importlib.import_module(module)))
        assert hashlib.sha256(file.read_bytes()).hexdigest() == digest, module
    from PhyAgentOS.runtime.policy.openpi.native_openpi_server import NativeOpenPIPolicy
    load_started = time.perf_counter()
    native = NativeOpenPIPolicy(policy_config='pi05_libero', checkpoint_dir='/model',
                                default_prompt=None, pytorch_device='cuda:0',
                                disable_torch_compile=True, torch_seed=7)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-load_started
    assert native.metadata['action_horizon'] == 10
    assert native.policy._model.config.pi05 is True
    guard = OpenPIStateGuard(native)
    rows = registration['samples']
    def observation(row):
        with np.load(inputs/row['file']) as data:
            return {**{k: data[k].copy() for k in data.files}, 'prompt': row['prompt']}
    initial_rng = RNGSnapshot()
    warmup = []
    for _ in range(2):
        torch.cuda.synchronize()
        t = time.perf_counter()
        native.infer(observation(rows[0]))
        torch.cuda.synchronize()
        warmup.append(time.perf_counter()-t)
        state['new_policy_inferences'] += 1
        guard.reset()
    initial_rng.restore()
    state['status'] = 'PAIRED_PROFILING'
    state['loading_seconds'] = load_seconds
    state['warmup_seconds'] = warmup
    save(output/'STATUS.json', state)
    results = []
    for index, row in enumerate(rows):
        prior = RNGSnapshot()
        pair = {}
        tap = None
        for condition in (('baseline','tapped') if index % 2 == 0 else ('tapped','baseline')):
            guard.reset()
            prior.restore()
            obs = observation(row)
            torch.cuda.synchronize()
            timer = time.perf_counter()
            if condition == 'tapped':
                tap = PassiveTap(native)
                with tap:
                    result = native.infer(obs)
            else:
                result = native.infer(obs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-timer
            state['new_policy_inferences'] += 1
            guard.reset()
            pair[condition] = dict(actions=result['actions'].copy(),
                                   behavior_sha256=behavior_hash(result),
                                   seconds=elapsed, rng=RNGSnapshot().hashes)
        assert np.array_equal(pair['baseline']['actions'], pair['tapped']['actions']), row['id']
        assert pair['baseline']['behavior_sha256'] == pair['tapped']['behavior_sha256'], row['id']
        assert pair['baseline']['rng'] == pair['tapped']['rng'], row['id']
        components = tap.measurement()
        assert sum(c['component']=='denoise_step' for c in components) == 10
        assert set(tap.features) == {'prefix_embedding','prefix_mask','prefix_hidden'}
        t = time.perf_counter()
        features = {key:value.float().cpu().numpy() for key,value in tap.features.items()}
        conversion_seconds = time.perf_counter()-t
        t = time.perf_counter()
        folder = output/row['id']
        folder.mkdir()
        files = {}
        for name, data in features.items():
            path = folder/(name+'.npy')
            np.save(path, data, allow_pickle=False)
            files[name] = dict(shape=list(data.shape), dtype=str(data.dtype),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
        np.save(folder/'new_inference_actions.npy', pair['baseline']['actions'], allow_pickle=False)
        write_seconds = time.perf_counter()-t
        item = dict(id=row['id'], actions_bit_equal=True, all_non_timing_output_equal=True, rng_equal=True,
                    baseline_seconds=pair['baseline']['seconds'], tapped_seconds=pair['tapped']['seconds'],
                    components=components, feature_conversion_seconds=conversion_seconds,
                    feature_write_and_hash_seconds=write_seconds, feature_files=files,
                    output_rng=pair['baseline']['rng'],
                    action_scope='New offline inference from saved input with controlled RNG; not historical action reproduction or executed outcome')
        results.append(item)
        save(folder/'MEASUREMENT.json', item)
        state['completed_pairs'] = len(results)
        save(output/'STATUS.json', state)
        del features, tap, pair
    state.update(status='COMPLETED', finished_at=time.time(), seconds=time.time()-start,
                 original_outputs_preserved=True, original_rng_preserved=True,
                 actor_or_controller_changed=False, new_robot_steps=0)
    save(output/'RESULT.json', dict(**state, metadata=native.metadata, pairs=results,
                                  timing_scope='Synchronized host end-to-end and component CUDA-event elapsed latency; nested component sums exclude duplication; not active GPU utilization. All 18 inferences and export costs retained.'))
    save(output/'STATUS.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        path = Path('/output/STATUS.json')
        failure = json.loads(path.read_text()) if path.exists() else {}
        failure.update(status='FAILED', finished_at=time.time(), error=traceback.format_exc())
        save(path, failure)
        raise
