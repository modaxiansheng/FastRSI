"""Offline A / zero-update reload / step200 audit. No optimizer or robot execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

PREFIXES = ('paligemma_with_expert.gemma_expert.', 'action_in_proj.',
            'action_out_proj.', 'time_mlp_in.', 'time_mlp_out.')
HOST_ROOT = Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_sample(row):
    path = Path('/project') / Path(row['path']).relative_to(HOST_ROOT)
    assert sha(path) == row['sha256'] and row['eligible']
    assert row['observation_step'] == row['action_index'] == row['label_start']
    with np.load(path, allow_pickle=False) as z:
        return {k: str(z[k].item()) if k == 'prompt' else z[k].copy() for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['A', 'B0', 'B200'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    started = time.perf_counter()
    out = a.output / a.stage
    out.mkdir(exist_ok=False)
    report = dict(status='RUNNING', stage=a.stage, training_steps=0, robot_executions=0,
                  sampling_steps=10, noise_seed_base=20261009,
                  scope='Saved development observations; not rollout or quality evidence')
    def save():
        report['wall_seconds'] = time.perf_counter() - started
        (out / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        from safetensors.torch import save_file
        from PhyAgentOS.runtime.policy.openpi import native_openpi_server as native
        from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
        dataset_path = Path('/project/runs/bowl_demo_collection_v1_20261009/DATASET.json')
        dataset = json.loads(dataset_path.read_text())
        # Six fixed observations, selected without inspecting model outputs.
        chosen = []
        for demo in ('demo_0', 'demo_7', 'demo_25'):
            eligible = sorted((r for r in dataset['samples'] if r['demo'] == demo),
                              key=lambda r: r['action_index'])
            assert len(eligible) >= 2
            chosen += [eligible[0], eligible[len(eligible) // 2]]
        report['dataset_sha256'] = sha(dataset_path)
        report['observations'] = chosen
        if a.stage != 'A':
            adapter = (a.output / 'A/zero_update.safetensors' if a.stage == 'B0' else
                       Path('/project/runs/bowl_action_adaptation_v2_20261009/training/action_adaptation.safetensors'))
            expected = (json.loads((a.output / 'A/RESULT.json').read_text())['zero_update']['sha256']
                        if a.stage == 'B0' else
                        '40039fb29f21f52ea900db74bd6eb7634a0f369d55c643539c23dfce620c4841')
            assert sha(adapter) == expected
            os.environ['FASTRSI_ACTION_ADAPTER'] = str(adapter)
            os.environ['FASTRSI_ACTION_ADAPTER_SHA256'] = expected
            os.environ['FASTRSI_POLICY_LOAD_RECEIPT'] = str(out / 'LOAD.json')
            # The same strict adapter initializer used by the actual service.
            import serve_action_adapted_pi05_v1  # noqa: F401
            report['adapter_sha256'] = expected
        policy = native.NativeOpenPIPolicy(policy_config='pi05_libero', checkpoint_dir='/model',
            default_prompt=None, pytorch_device='cuda:0', disable_torch_compile=True, torch_seed=7)
        model = policy.policy._model
        params = {n: p for n, p in model.named_parameters() if n.startswith(PREFIXES)}
        report['runtime'] = dict(torch=torch.__version__, hip=torch.version.hip,
            model_class=type(model).__name__, parameter_dtypes=sorted(set(str(p.dtype) for p in params.values())),
            sample_kwargs=policy.policy._sample_kwargs)
        report['transforms'] = dict(input=repr(policy.policy._input_transform), output=repr(policy.policy._output_transform))
        if a.stage == 'A':
            zero = out / 'zero_update.safetensors'
            save_file({n: p.detach().cpu().contiguous() for n, p in params.items()}, str(zero))
            report['zero_update'] = dict(path=str(zero), sha256=sha(zero), tensors=len(params),
                                        bytes=zero.stat().st_size, exact_native_dtypes=True)
            # Actual labels through the actual configured transforms, without NN.
            errors = []
            for row in dataset['samples']:
                data = read_sample(row)
                expected = data['actions'].copy()
                transformed = policy.policy._input_transform(data)
                restored = policy.policy._output_transform(dict(state=transformed['state'].copy(),
                    actions=transformed['actions'].copy()))['actions']
                assert restored.shape == expected.shape
                errors.append(np.max(np.abs(restored - expected), axis=0))
            coordinate_max = np.max(errors, axis=0)
            assert np.max(coordinate_max) < 1e-5
            report['label_transform_roundtrip'] = dict(samples=len(errors),
                coordinate_max_abs=coordinate_max.tolist(), passed=True,
                scope='Numerical round trip only; physical convention verified separately')
        original_sample = policy.policy._sample_actions
        capture = {}
        def sample(device, observation, **kwargs):
            kwargs['noise'] = torch.from_numpy(capture['noise'].copy()).to(device)[None, ...]
            kwargs['num_steps'] = 10
            tensor = original_sample(device, observation, **kwargs)
            capture['normalized_sampled_actions'] = tensor.detach().float().cpu().numpy()[0].copy()
            return tensor
        policy.policy._sample_actions = sample
        adapter = OpenPIPi05Adapter()
        report['outputs'] = []
        for i, row in enumerate(chosen):
            data = read_sample(row)
            labels = data.pop('actions')
            noise = np.random.default_rng(20261009 + i).standard_normal((10, 32)).astype(np.float32)
            capture['noise'] = noise
            result = policy.infer(data)
            command = adapter.from_policy_output(result, {'action_dim': 7})['actions']
            assert np.array_equal(command, result['actions'])
            normalized = capture['normalized_sampled_actions'].copy()
            assert np.isfinite(normalized).all() and np.isfinite(command).all()
            key = f'{i:02d}_{row["demo"]}_{row["action_index"]:04d}'
            target = out / f'{key}.npz'
            np.savez_compressed(target, noise=noise, normalized_sampled_actions=normalized,
                output_actions=result['actions'], controller_input=command, labels=labels)
            report['outputs'].append(dict(key=key, file=target.name, sha256=sha(target),
                noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
                observation_sha256=row['sha256'], role=row['role'], demo=row['demo'], action_index=row['action_index']))
            if i == 0:
                repeat = policy.infer(data)
                report['same_noise_repeat'] = dict(output_max_abs=float(np.max(np.abs(repeat['actions'] - command))),
                    normalized_max_abs=float(np.max(np.abs(capture['normalized_sampled_actions'] - normalized))))
            print(json.dumps(dict(stage=a.stage, observation=key, completed=True)), flush=True)
        report['status'] = 'OFFLINE_AUDIT_COMPLETED'
        save()
    except BaseException as e:
        report.update(status='OFFLINE_AUDIT_FAILED', error=repr(e))
        save()
        raise


if __name__ == '__main__':
    main()
