"""One isolated real-gradient update; not a behavior-training experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay', type=Path, required=True)
    ap.add_argument('--host-receipt', type=Path, required=True)
    ap.add_argument('--export', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status='RUNNING', purpose='One-step training implementation feasibility only',
                  robot_runs=0, quality_gain_proven=False, fast_rsi_gain_proven=False, training_steps=0)

    def save():
        report['full_wall_seconds'] = time.perf_counter() - started
        (args.output / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')

    try:
        replay = json.loads((args.replay / 'RESULT.json').read_text())
        host = json.loads(args.host_receipt.read_text())
        assert replay['status'] == 'NATIVE_SUCCESS' and replay['demo'] == 'demo_0' and host['host_exit_code'] == 0
        assert replay['intermediate_state_restores'] == 0 and replay['executed_actions'] > 10
        manifest = json.loads((args.export / 'MANIFEST.json').read_text())
        raw_path = args.export / 'replay.npz'
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == manifest['files']['replay.npz']['sha256']
        arrays = np.load(raw_path, allow_pickle=False)
        sample = {'observation/image': np.asarray(Image.open(args.replay / 'initial_agentview_image.png').convert('RGB')),
                  'observation/wrist_image': np.asarray(Image.open(args.replay / 'initial_robot0_eye_in_hand_image.png').convert('RGB')),
                  'observation/state': np.asarray(replay['initial_public_state'], dtype=np.float32),
                  'prompt': 'stack the right bowl on the left bowl and place them in the tray',
                  'actions': arrays['actions'][:10].astype(np.float32)}
        report['sample'] = dict(demo='demo_0', observation_step=0, source_action_indices=list(range(10)),
            alignment='Fresh native PRE-action observation, not raw HDF5 post-action obs[0]',
            replay_receipt_sha256=hashlib.sha256((args.replay / 'RESULT.json').read_bytes()).hexdigest(),
            image_sha256={name: hashlib.sha256((args.replay / name).read_bytes()).hexdigest()
                          for name in ('initial_agentview_image.png', 'initial_robot0_eye_in_hand_image.png')},
            source_npz_sha256=manifest['files']['replay.npz']['sha256'])
        from PhyAgentOS.runtime.policy.openpi.native_openpi_server import NativeOpenPIPolicy
        from openpi.models import model as model_module
        from safetensors.torch import load_file, save_file
        assert torch.cuda.is_available(), 'Real accelerator unavailable; no CPU fallback claim'
        report['runtime'] = dict(torch=torch.__version__, hip=torch.version.hip,
                                 device=torch.cuda.get_device_name(0))
        save()
        policy = NativeOpenPIPolicy(policy_config='pi05_libero', checkpoint_dir=str(args.checkpoint),
            default_prompt=None, pytorch_device='cuda:0', disable_torch_compile=True, torch_seed=7)
        model = policy.policy._model
        allowed = ('paligemma_with_expert.gemma_expert.', 'action_in_proj.', 'action_out_proj.',
                   'time_mlp_in.', 'time_mlp_out.')
        named = dict(model.named_parameters())
        for name, param in named.items():
            param.requires_grad_(name.startswith(allowed))
        train = {name: p for name, p in named.items() if p.requires_grad}
        frozen = {name: p for name, p in named.items() if not p.requires_grad}
        assert train and frozen and all(not p.requires_grad for p in model.paligemma_with_expert.paligemma.parameters())
        report['parameters'] = dict(trainable_elements=sum(p.numel() for p in train.values()),
            frozen_elements=sum(p.numel() for p in frozen.values()),
            trainable=[dict(name=n, shape=list(p.shape), dtype=str(p.dtype)) for n, p in train.items()])
        before_frozen = {n: tensor_sha(p) for n, p in frozen.items()}
        before_train = {n: tensor_sha(p) for n, p in train.items()}
        original_train = {n: p.detach().cpu().clone() for n, p in train.items()}
        model.train()
        model.paligemma_with_expert.paligemma.eval()
        model.gradient_checkpointing_enable()
        transformed = policy.policy._input_transform(sample)

        def to_device(value):
            if isinstance(value, dict):
                return {k: to_device(v) for k, v in value.items()}
            return torch.as_tensor(np.asarray(value).copy()).unsqueeze(0).to('cuda:0')

        batch = to_device(transformed)
        actions = batch.pop('actions').float()
        observation = model_module.Observation.from_dict(batch)
        assert actions.shape == (1, 10, 32)
        report['transformed_actions_shape'] = list(actions.shape)
        optimizer = torch.optim.SGD(train.values(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        stage_start = time.perf_counter()
        loss_all = model(observation, actions)
        loss = loss_all[:, :, :7].mean()
        assert torch.isfinite(loss)
        loss.backward()
        gradients = {n: dict(present=p.grad is not None, finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else None,
                            norm=float(p.grad.float().norm()) if p.grad is not None else None) for n, p in train.items()}
        assert all(v['finite'] for v in gradients.values() if v['present'])
        assert any(v['norm'] and v['norm'] > 0 for v in gradients.values() if v['present'])
        assert all(p.grad is None for p in frozen.values()), 'Frozen parameter acquired gradient'
        norm = torch.nn.utils.clip_grad_norm_(list(train.values()), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        report.update(training_steps=1, optimizer='SGD lr=0.001, clip_norm=1; feasibility, not tuned training',
                      loss=float(loss), loss_scope='7 real action coordinates only',
                      gradient_norm_before_clip=float(norm), gradients=gradients,
                      forward_backward_update_seconds=time.perf_counter() - stage_start)
        updated_train = {n: tensor_sha(p) for n, p in train.items()}
        changed = [n for n in train if updated_train[n] != before_train[n]]
        assert changed, 'No actual parameter changed'
        assert before_frozen == {n: tensor_sha(p) for n, p in frozen.items()}, 'Frozen parameters changed'
        report.update(changed_parameters=changed, frozen_parameter_hashes_unchanged=True)
        save()
        adapter = args.output / 'action_update.safetensors'
        save_file({n: p.detach().cpu().contiguous() for n, p in train.items()}, str(adapter))
        reloaded = load_file(str(adapter), device='cpu')
        assert set(reloaded) == set(train)
        with torch.no_grad():
            # Restore original action tensors, then apply the saved update.
            for n, p in train.items():
                p.copy_(original_train[n].to(device=p.device, dtype=p.dtype))
            assert before_train == {n: tensor_sha(p) for n, p in train.items()}
            for n, p in train.items():
                assert p.shape == reloaded[n].shape and p.dtype == reloaded[n].dtype
                p.copy_(reloaded[n].to(p.device))
        assert updated_train == {n: tensor_sha(p) for n, p in train.items()}
        assert before_frozen == {n: tensor_sha(p) for n, p in frozen.items()}
        report.update(status='REAL_UPDATE_AND_ADAPTER_RELOAD_PASSED', adapter_bytes=adapter.stat().st_size,
                      checkpoint_scope='Action-only tensors reloaded into same verified base model; no second full-model boot',
                      source_base_mounted_read_only=True, gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        report.update(status='UPDATE_CHECK_FAILED', error=repr(error))
        raise
    finally:
        save()
    print(json.dumps({k: report[k] for k in ('status', 'training_steps', 'full_wall_seconds')}))


if __name__ == '__main__':
    main()
