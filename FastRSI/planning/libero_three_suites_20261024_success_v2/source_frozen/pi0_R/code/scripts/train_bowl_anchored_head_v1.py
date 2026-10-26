"""50-step native output-row adaptation with frozen features and base-output anchoring."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from audit_bowl_action_pipeline_v1 import PREFIXES, read_sample, sha


def tensor_sha(p):
    return hashlib.sha256(p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    out = a.output
    started = time.perf_counter()
    report = dict(status='RUNNING', training_steps=0, robot_executions=0, native_model_calls=0,
        fast_rsi_gain_proven=False, behavior_gain_proven=False)
    def save():
        report['full_wall_seconds'] = time.perf_counter() - started
        (out / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        from PhyAgentOS.runtime.policy.openpi.native_openpi_server import NativeOpenPIPolicy
        from openpi.models.model import Observation
        from safetensors.torch import save_file
        dataset_path = Path('/project/runs/bowl_demo_collection_v1_20261009/DATASET.json')
        dataset = json.loads(dataset_path.read_text())
        assert sha(dataset_path) == 'afd494cbf367718bd134646d0f4c51fb44e298ff84c7d92add030162c6c5ec93'
        train_rows = [r for r in dataset['samples'] if r['role'] == 'train']
        validation = [r for r in dataset['samples'] if r['role'] == 'validation']
        validation = [validation[int(i)] for i in np.linspace(0, len(validation) - 1, 8, dtype=int)]
        old = json.loads(Path('/project/runs/bowl_action_pipeline_audit_v2_20261009/A/RESULT.json').read_text())
        probes = old['observations']
        protocol = dict(steps=50, trainable='action_out_proj.weight/bias first seven rows only',
            frozen='All native feature modules, time modules and padded output rows', batch_size=16,
            optimizer='AdamW', learning_rate=2.5e-5, weight_decay=0., betas=[.9,.95], clip_norm=1.,
            anchor_coefficient=1., seed=19, feature_cache='One fixed noise/time per original training chunk',
            training_chunks=len(train_rows), validation=validation, regression_probes=probes,
            sampling_steps=10, selection='Final step only; no checkpoint or hyperparameter search',
            gates=dict(validation_sampled_normalized_mse_must_decrease=True,
                maximum_rotation_command_change=.02, zero_rotation_source_probe_norm_increase_limit=.005),
            scope='External demonstration output adaptation with retention regularization; not cyclic RSI')
        (out / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n')
        torch.manual_seed(19); torch.cuda.manual_seed_all(19)
        policy = NativeOpenPIPolicy(policy_config='pi05_libero', checkpoint_dir='/model', default_prompt=None,
            pytorch_device='cuda:0', disable_torch_compile=True, torch_seed=7)
        model = policy.policy._model
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        named = dict(model.named_parameters())
        frozen_before = {n: tensor_sha(p) for n,p in named.items() if not n.startswith('action_out_proj.')}
        base_weight = model.action_out_proj.weight.detach().clone()
        base_bias = model.action_out_proj.bias.detach().clone()
        features = []
        targets = []
        anchors = []
        captured = {}
        def hook(module, arguments, output):
            captured['features'] = arguments[0].detach().float().clone()
            captured['anchor'] = output.detach().float().clone()
        handle = model.action_out_proj.register_forward_hook(hook)
        rng = np.random.default_rng(19)
        t = time.perf_counter()
        for i, row in enumerate(train_rows):
            data = policy.policy._input_transform(read_sample(row))
            def device(v):
                if isinstance(v, dict):
                    return {k: device(x) for k,x in v.items()}
                return torch.as_tensor(np.asarray(v).copy()).unsqueeze(0).to('cuda:0')
            batch = device(data)
            actions = batch.pop('actions').float()
            noise = torch.from_numpy(rng.standard_normal((1,10,32)).astype(np.float32)).to('cuda:0')
            tau = torch.tensor([rng.uniform(.001,.999)], device='cuda:0', dtype=torch.float32)
            torch.manual_seed(19000 + i); torch.cuda.manual_seed_all(19000 + i)
            with torch.no_grad():
                model(Observation.from_dict(batch), actions, noise=noise, time=tau)
            report['native_model_calls'] += 1
            features.append(captured['features'][0])
            anchors.append(captured['anchor'][0,:,:7])
            targets.append((noise - actions)[0,:,:7])
            if (i + 1) % 30 == 0:
                print(json.dumps(dict(feature_chunks=i+1)), flush=True)
        handle.remove()
        features = torch.stack(features)
        anchors = torch.stack(anchors)
        targets = torch.stack(targets)
        report['feature_collection_seconds'] = time.perf_counter() - t
        # The only optimized tensors are 7x1024 plus seven biases; deployment merges into native rows.
        w = torch.nn.Parameter(base_weight[:7].float().clone())
        b = torch.nn.Parameter(base_bias[:7].float().clone())
        report['trainable_elements'] = w.numel() + b.numel()
        assert report['trainable_elements'] == 7175
        opt = torch.optim.AdamW([w,b], lr=2.5e-5, betas=(.9,.95), weight_decay=0.)
        with torch.no_grad():
            assert torch.allclose(F.linear(features, w, b), anchors, atol=1e-5, rtol=1e-5)
        t = time.perf_counter()
        for step in range(50):
            idx = torch.as_tensor(rng.integers(0,len(train_rows),size=16), device='cuda:0')
            predicted = F.linear(features[idx], w, b)
            demonstration_loss = F.mse_loss(predicted, targets[idx])
            retention_loss = F.mse_loss(predicted, anchors[idx])
            loss = demonstration_loss + retention_loss
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True); loss.backward()
            assert all(p.grad is None for p in model.parameters())
            gradient_norm = torch.nn.utils.clip_grad_norm_([w,b],1.)
            assert torch.isfinite(gradient_norm)
            opt.step()
            record = dict(step=step+1, loss=float(loss), demonstration_loss=float(demonstration_loss),
                retention_loss=float(retention_loss), samples=[train_rows[int(i)]['sha256'] for i in idx.cpu().tolist()])
            with (out / 'TRAINING.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            report['training_steps'] = step+1
        torch.cuda.synchronize()
        report['optimizer_seconds'] = time.perf_counter() - t
        original_sample = policy.policy._sample_actions
        capture = {}
        def sample(device, observation, **kwargs):
            kwargs['noise'] = torch.from_numpy(capture['noise'].copy()).to(device)[None,...]
            kwargs['num_steps'] = 10
            y = original_sample(device, observation, **kwargs)
            capture['normalized'] = y.detach().float().cpu().numpy()[0].copy()
            return y
        policy.policy._sample_actions = sample
        # Final check uses complete sampling; flow-field loss does not decide admission.
        eval_rows = validation + probes
        report['sampled_evaluation'] = {}
        for stage in ('A','HEAD50'):
            with torch.no_grad():
                model.action_out_proj.weight.copy_(base_weight)
                model.action_out_proj.bias.copy_(base_bias)
                if stage == 'HEAD50':
                    model.action_out_proj.weight[:7].copy_(w)
                    model.action_out_proj.bias[:7].copy_(b)
            metrics = []
            for i,row in enumerate(eval_rows):
                data = read_sample(row)
                transformed = policy.policy._input_transform({k: v.copy() if hasattr(v,'copy') else v for k,v in data.items()})
                data.pop('actions')
                noise_seed = 29000+i if i<len(validation) else 20261009+i-len(validation)
                capture['noise'] = np.random.default_rng(noise_seed).standard_normal((10,32)).astype(np.float32)
                response = policy.infer(data)
                report['native_model_calls'] += 1
                norm = capture['normalized']
                actions = response['actions']
                if stage == 'A' and i>=len(validation):
                    with np.load(Path('/project/runs/bowl_action_pipeline_audit_v2_20261009/A') /
                                 old['outputs'][i-len(validation)]['file']) as z:
                        assert np.array_equal(actions,z['output_actions']), 'Frozen feature extraction changed base sampling'
                name = f'{stage}_{i:02d}.npz'
                np.savez_compressed(out/name, normalized=norm, actions=actions,
                    normalized_labels=transformed['actions'], noise=capture['noise'])
                metrics.append(dict(file=name, sha256=sha(out/name), demo=row['demo'], action_index=row['action_index'],
                    validation=i<len(validation), mse=float(np.mean((norm[:,:7]-transformed['actions'][:,:7])**2)),
                    rotation_l2_mean=float(np.linalg.norm(actions[:,3:6],axis=1).mean())))
            report['sampled_evaluation'][stage] = metrics
            save()
        prior, after = [report['sampled_evaluation'][s] for s in ('A','HEAD50')]
        baseline_mse = float(np.mean([r['mse'] for r in prior if r['validation']]))
        candidate_mse = float(np.mean([r['mse'] for r in after if r['validation']]))
        max_rotation_change = 0.
        zero_source_change = None
        for i,(base_r,new_r) in enumerate(zip(prior,after)):
            with np.load(out/base_r['file']) as z:
                base_actions = z['actions'].copy()
            with np.load(out/new_r['file']) as z:
                new_actions = z['actions'].copy()
            max_rotation_change = max(max_rotation_change, float(np.max(np.abs(new_actions[:,3:6]-base_actions[:,3:6]))))
            if new_r['demo']=='demo_7' and new_r['action_index']==0:
                zero_source_change = new_r['rotation_l2_mean'] - base_r['rotation_l2_mean']
        assert zero_source_change is not None
        frozen_after = {n: tensor_sha(p) for n,p in named.items() if not n.startswith('action_out_proj.')}
        assert frozen_before == frozen_after
        assert torch.equal(model.action_out_proj.weight[7:],base_weight[7:])
        assert torch.equal(model.action_out_proj.bias[7:],base_bias[7:])
        (out/'FROZEN_PARAMETER_HASHES.json').write_text(json.dumps(frozen_before,indent=2)+'\n')
        passed = candidate_mse < baseline_mse and max_rotation_change <= .02 and zero_source_change <= .005
        adapter = out/'action_adaptation.safetensors'
        save_file({n:p.detach().cpu().contiguous() for n,p in named.items() if n.startswith(PREFIXES)},str(adapter))
        compact = out/'head_update.safetensors'
        save_file(dict(weight=w.detach().cpu().contiguous(),bias=b.detach().cpu().contiguous()),str(compact))
        report.update(status='ANCHORED_HEAD_OFFLINE_PASSED' if passed else 'ANCHORED_HEAD_OFFLINE_REJECTED',
            frozen_parameters_unchanged=True, padded_output_rows_unchanged=True,
            gates=dict(passed=passed, original_validation_sampled_mse=baseline_mse,candidate_validation_sampled_mse=candidate_mse,
                maximum_rotation_command_change=max_rotation_change,zero_source_rotation_norm_change=zero_source_change),
            adapter=dict(path=str(adapter),sha256=sha(adapter),bytes=adapter.stat().st_size),
            compact=dict(path=str(compact),sha256=sha(compact),bytes=compact.stat().st_size),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save(); print(json.dumps(dict(status=report['status'],gates=report['gates'])),flush=True)
    except BaseException as e:
        report.update(status='TRAINING_FAILED',error=repr(e));save();raise


if __name__ == '__main__':
    main()
