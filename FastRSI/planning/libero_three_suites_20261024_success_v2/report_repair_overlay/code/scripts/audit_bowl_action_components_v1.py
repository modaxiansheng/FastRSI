"""Fixed-observation parameter substitution audit, never a deployable score."""
import argparse
import collections
import json
from pathlib import Path
import time

import numpy as np
import torch

from audit_bowl_action_pipeline_v1 import PREFIXES, read_sample, sha

ROOT = Path('/project')
OLD = ROOT / 'runs/bowl_action_pipeline_audit_v2_20261009'
TRAIN = ROOT / 'runs/bowl_action_adaptation_v2_20261009/training'
STAGES = ('A', 'B200', 'EXPERT_ONLY', 'PROJECTIONS_TIME_ONLY', 'OUT_ONLY',
          'B200_BASE_ROTATION_OUT', 'RESTORED_A')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    started = time.perf_counter()
    result = dict(status='RUNNING', training_steps=0, robot_executions=0, stages=list(STAGES),
                  interpretation='Parameter interventions on saved observations, not closed-loop success or efficiency')
    def save():
        result['wall_seconds'] = time.perf_counter() - started
        (out / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    save()
    try:
        from safetensors.torch import load_file
        from PhyAgentOS.runtime.policy.openpi.native_openpi_server import NativeOpenPIPolicy
        old = json.loads((OLD / 'A/RESULT.json').read_text())
        rows = old['observations']
        assert sha(OLD / 'A/zero_update.safetensors') == old['zero_update']['sha256']
        assert sha(TRAIN / 'action_adaptation.safetensors') == '40039fb29f21f52ea900db74bd6eb7634a0f369d55c643539c23dfce620c4841'
        base = load_file(str(OLD / 'A/zero_update.safetensors'), device='cpu')
        updated = load_file(str(TRAIN / 'action_adaptation.safetensors'), device='cpu')
        assert set(base) == set(updated)
        protocol = dict(stages=list(STAGES), observation_manifest=rows, old_A_result_sha256=sha(OLD / 'A/RESULT.json'),
            base_sha256=old['zero_update']['sha256'], updated_sha256=sha(TRAIN / 'action_adaptation.safetensors'),
            sampling_steps=10, noise_seeds=[20261009 + i for i in range(len(rows))], no_training=True,
            stage_selection='Defined before component outputs; no candidate promotion')
        (out / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n')
        # Per-module update measurements; finite CPU chunks limit transient memory.
        deltas = {}
        for name, a in base.items():
            b = updated[name]
            assert a.shape == b.shape and a.dtype == b.dtype
            group = next(p for p in PREFIXES if name.startswith(p))
            g = deltas.setdefault(group, dict(elements=0, changed_elements=0, tensors=0,
                changed_tensors=0, base_squared_sum=0., delta_squared_sum=0., maximum_abs_delta=0.))
            count = 0
            for lo in range(0, a.numel(), 1_000_000):
                x = a.reshape(-1)[lo:lo + 1_000_000].float()
                y = b.reshape(-1)[lo:lo + 1_000_000].float()
                d = y - x
                count += int(torch.count_nonzero(d))
                g['base_squared_sum'] += float(torch.sum(x * x, dtype=torch.float64))
                g['delta_squared_sum'] += float(torch.sum(d * d, dtype=torch.float64))
                g['maximum_abs_delta'] = max(g['maximum_abs_delta'], float(torch.max(torch.abs(d))))
            g['elements'] += a.numel()
            g['changed_elements'] += count
            g['tensors'] += 1
            g['changed_tensors'] += int(count > 0)
        for g in deltas.values():
            g['relative_l2_update'] = float(np.sqrt(g['delta_squared_sum'] / g['base_squared_sum']))
            g['changed_element_fraction'] = g['changed_elements'] / g['elements']
        (out / 'PARAMETER_DELTAS.json').write_text(json.dumps(deltas, indent=2) + '\n')
        dataset_file = ROOT / 'runs/bowl_demo_collection_v1_20261009/DATASET.json'
        dataset = json.loads(dataset_file.read_text())
        assert sha(dataset_file) == old['dataset_sha256']
        label_rows = []
        for row in dataset['samples']:
            data = read_sample(row)
            actions = data['actions']
            label_rows.append(dict(demo=row['demo'], role=row['role'], index=row['action_index'],
                sample_sha256=row['sha256'], rotation_mean=actions[:, 3:6].mean(axis=0).tolist(),
                rotation_l2_mean=float(np.linalg.norm(actions[:, 3:6], axis=1).mean()),
                rotation_clip_fraction=float(np.mean(np.abs(actions[:, 3:6]) > 1))))
        usage = collections.Counter(json.loads(line)['sample_sha256'] for line in (TRAIN / 'TRAINING.jsonl').read_text().splitlines())
        assert sum(usage.values()) == 200
        selected_means = np.array([r['rotation_mean'] for r in label_rows if r['role'] == 'train'])
        weighted = np.array([r['rotation_mean'] for r in label_rows for _ in range(usage[r['sample_sha256']])])
        labels = dict(rows=label_rows, samples=len(label_rows), original_dataset_sha256=sha(dataset_file),
            training_uniform_chunk_rotation_mean=selected_means.mean(axis=0).tolist(),
            actual_200_draws_rotation_mean=weighted.mean(axis=0).tolist(),
            unique_training_chunks_used=len(usage),
            scope='Descriptive action supervision; sign similarity is not proof of a labeling error')
        (out / 'SUPERVISION.json').write_text(json.dumps(labels, indent=2) + '\n')
        policy = NativeOpenPIPolicy(policy_config='pi05_libero', checkpoint_dir='/model', default_prompt=None,
            pytorch_device='cuda:0', disable_torch_compile=True, torch_seed=7)
        model = policy.policy._model
        params = {n: p for n, p in model.named_parameters() if n.startswith(PREFIXES)}
        assert set(params) == set(base)
        original_sample = policy.policy._sample_actions
        captured = {}
        def sample(device, observation, **kwargs):
            kwargs['noise'] = torch.from_numpy(captured['noise'].copy()).to(device)[None, ...]
            kwargs['num_steps'] = 10
            y = original_sample(device, observation, **kwargs)
            captured['normalized'] = y.detach().float().cpu().numpy()[0].copy()
            return y
        policy.policy._sample_actions = sample
        reference = None
        result['outputs'] = {}
        for stage in STAGES:
            with torch.no_grad():
                for n, p in params.items():
                    use_updated = (stage in ('B200', 'B200_BASE_ROTATION_OUT') or
                        (stage == 'EXPERT_ONLY' and n.startswith(PREFIXES[0])) or
                        (stage == 'PROJECTIONS_TIME_ONLY' and not n.startswith(PREFIXES[0])) or
                        (stage == 'OUT_ONLY' and n.startswith('action_out_proj.')))
                    p.copy_((updated[n] if use_updated else base[n]).to(p.device))
                    if stage == 'B200_BASE_ROTATION_OUT' and n.startswith('action_out_proj.'):
                        p[3:6].copy_(base[n][3:6].to(p.device))
            model.eval()
            stage_outputs = []
            for i, row in enumerate(rows):
                data = read_sample(row)
                label = data.pop('actions')
                captured['noise'] = np.random.default_rng(20261009 + i).standard_normal((10, 32)).astype(np.float32)
                response = policy.infer(data)
                actions = response['actions']
                normalized = captured['normalized'].copy()
                assert np.isfinite(actions).all() and np.isfinite(normalized).all()
                name = f'{stage}_{i:02d}.npz'
                np.savez_compressed(out / name, actions=actions, normalized=normalized,
                    noise=captured['noise'], labels=label)
                stage_outputs.append(dict(file=name, sha256=sha(out / name), demo=row['demo'], action_index=row['action_index'],
                    rotation_mean=actions[:, 3:6].mean(axis=0).tolist(),
                    rotation_l2_mean=float(np.linalg.norm(actions[:, 3:6], axis=1).mean()),
                    rotation_label_rmse=float(np.sqrt(np.mean((actions[:, 3:6] - label[:, 3:6]) ** 2))),
                    command_label_rmse=float(np.sqrt(np.mean((actions - label) ** 2))),
                    normalized_padded_rms=float(np.sqrt(np.mean(normalized[:, 7:] ** 2)))))
                with np.load(OLD / ('B200' if stage == 'B200' else 'A') / old['outputs'][i]['file']) as z:
                    if stage in ('A', 'B200', 'RESTORED_A'):
                        assert np.array_equal(actions, z['output_actions']), (stage, i, 'old outputs differ')
                        assert np.array_equal(normalized, z['normalized_sampled_actions']), (stage, i, 'old normalized outputs differ')
            result['outputs'][stage] = stage_outputs
            save()
            print(json.dumps(dict(stage=stage, observations=len(stage_outputs), complete=True)), flush=True)
        result.update(status='COMPONENT_AUDIT_COMPLETED', base_and_trained_reproduced_exactly=True,
                      original_restored_exactly=True)
        save()
    except BaseException as e:
        result.update(status='FAILED', error=repr(e))
        save()
        raise


if __name__ == '__main__':
    main()
