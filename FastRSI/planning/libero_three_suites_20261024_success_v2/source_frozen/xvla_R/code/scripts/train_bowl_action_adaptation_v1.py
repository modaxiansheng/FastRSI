"""Fixed-budget action-expert adaptation from native pre-action demonstrations."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch

ALLOWED=('paligemma_with_expert.gemma_expert.', 'action_in_proj.', 'action_out_proj.', 'time_mlp_in.', 'time_mlp_out.')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def tensor_sha(p):
    return hashlib.sha256(p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--previous-adapter',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(exist_ok=False)
    started=time.perf_counter();report=dict(status='RUNNING',training_steps=0,robot_evaluations=0,
        quality_gain_proven=False,fast_rsi_gain_proven=False,external_demonstrations=True)
    def save():
        report['full_wall_seconds']=time.perf_counter()-started
        (a.output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    def seed(n):
        random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
    try:
        from PhyAgentOS.runtime.policy.openpi.native_openpi_server import NativeOpenPIPolicy
        from openpi.models import model as model_module
        from safetensors.torch import load_file,save_file
        d=json.loads(a.dataset.read_text());rows=d['samples']
        for row in rows:
            row['source_host_path']=row['path']
            root=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
            row['path']=str(Path('/project')/Path(row['path']).relative_to(root))
        for row in rows:
            assert sha(row['path'])==row['sha256'] and row['eligible']
            assert row['observation_step']==row['action_index']==row['label_start']
            assert row['label_end_exclusive']-row['label_start']==10
        train_rows=[r for r in rows if r['role']=='train'];val_rows=[r for r in rows if r['role']=='validation']
        assert train_rows and val_rows
        val_rows=[val_rows[int(i)] for i in np.linspace(0,len(val_rows)-1,min(8,len(val_rows)),dtype=int)]
        protocol=dict(steps=200,batch_size=1,seed=7,optimizer='AdamW on FP32 master action tensors',
            learning_rate=2.5e-5,betas=[.9,.95],eps=1e-8,weight_decay=.01,warmup_steps=10,clip_norm=1.,
            validation_at=[0,200],validation_seed=1800,selection='Final step only; no best-checkpoint selection',
            trainable_prefixes=list(ALLOWED),action_loss_coordinates=7,base_graph_and_dtypes_preserved=True,
            dataset_sha256=sha(a.dataset),train_samples=len(train_rows),validation_samples_total=len([r for r in rows if r['role']=='validation']),
            validation_used=[dict(path=r['path'],sha256=r['sha256'],demo=r['demo']) for r in val_rows],
            training_demos=sorted(set(r['demo'] for r in train_rows)),validation_demos=sorted(set(r['demo'] for r in val_rows)))
        (a.output/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n');report['protocol']=protocol;save()
        assert torch.cuda.is_available()
        report['runtime']=dict(torch=torch.__version__,hip=torch.version.hip,device=torch.cuda.get_device_name(0))
        seed(7)
        policy=NativeOpenPIPolicy(policy_config='pi05_libero',checkpoint_dir=str(a.checkpoint),default_prompt=None,
            pytorch_device='cuda:0',disable_torch_compile=True,torch_seed=7)
        model=policy.policy._model
        named=dict(model.named_parameters())
        for n,p in named.items():p.requires_grad_(n.startswith(ALLOWED))
        train={n:p for n,p in named.items() if p.requires_grad};frozen={n:p for n,p in named.items() if not p.requires_grad}
        frozen_before={n:tensor_sha(p) for n,p in frozen.items()};original={n:p.detach().cpu().clone() for n,p in train.items()}
        train_before={n:tensor_sha(p) for n,p in train.items()}
        def read(row):
            with np.load(row['path'],allow_pickle=False) as z:
                return {k:(str(z[k].item()) if k=='prompt' else z[k].copy()) for k in z.files}
        expected='846aaa607018ab18de626d7385ba9feb19226df1978cd134fe558db192e4b894'
        assert sha(a.previous_adapter)==expected
        previous=load_file(str(a.previous_adapter),device='cpu');assert set(previous)==set(train)
        with torch.no_grad():
            for n,p in train.items():
                assert previous[n].shape==p.shape and previous[n].dtype==p.dtype;p.copy_(previous[n].to(p.device))
        probe=read(train_rows[0]);probe.pop('actions');t=time.perf_counter()
        out=policy.infer(probe);actions=np.asarray(out['actions'])
        assert actions.shape==(10,7) and np.isfinite(actions).all()
        report['fresh_process_previous_adapter_check']=dict(status='PASSED',sha256=expected,action_shape=list(actions.shape),
            policy_calls=1,seconds=time.perf_counter()-t,claim='Load and finite output only, not behavior quality')
        with torch.no_grad():
            for n,p in train.items():p.copy_(original[n].to(p.device))
        assert train_before=={n:tensor_sha(p) for n,p in train.items()};del previous,original
        seed(7)
        cache={r['path']:policy.policy._input_transform(read(r)) for r in train_rows+val_rows}
        def device(v):
            if isinstance(v,dict):return {k:device(x) for k,x in v.items()}
            return torch.as_tensor(np.asarray(v).copy()).unsqueeze(0).to('cuda:0')
        def loss_for(row):
            batch=device(cache[row['path']]);acts=batch.pop('actions').float()
            assert acts.shape==(1,10,32)
            return model(model_module.Observation.from_dict(batch),acts)[:,:,:7].mean()
        def validate(step):
            cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all();np_rng=np.random.get_state();py_rng=random.getstate()
            model.eval();vals=[];t=time.perf_counter()
            with torch.no_grad():
                for i,row in enumerate(val_rows):
                    seed(1800+i);value=loss_for(row);assert torch.isfinite(value);vals.append(float(value))
            torch.cuda.synchronize()
            result=dict(step=step,losses=vals,mean=float(np.mean(vals)),seconds=time.perf_counter()-t,
                scope='Fixed seed demonstration loss, not rollout success')
            model.train();model.paligemma_with_expert.paligemma.eval()
            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng);np.random.set_state(np_rng);random.setstate(py_rng)
            return result
        model.gradient_checkpointing_enable()
        model.train();model.paligemma_with_expert.paligemma.eval()
        report['validation']=[validate(0)];save()
        masters={n:torch.nn.Parameter(p.detach().float().clone()) for n,p in train.items()}
        optimizer=torch.optim.AdamW(masters.values(),lr=2.5e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
        rng=np.random.default_rng(7);trace=a.output/'TRAINING.jsonl'
        for step in range(200):
            row=train_rows[int(rng.integers(len(train_rows)))];model.zero_grad(set_to_none=True);optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();t=time.perf_counter();loss=loss_for(row);assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is None for p in frozen.values())
            for n,p in train.items():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(),n;masters[n].grad=p.grad.detach().float()
            norm=torch.nn.utils.clip_grad_norm_(list(masters.values()),1.);assert torch.isfinite(norm)
            lr=2.5e-5*min(1.,(step+1)/10)
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step()
            with torch.no_grad():
                for n,p in train.items():p.copy_(masters[n])
            torch.cuda.synchronize()
            record=dict(step=step+1,demo=row['demo'],sample_sha256=row['sha256'],action_index=row['action_index'],
                loss=float(loss),gradient_norm_before_clip=float(norm),lr=lr,seconds=time.perf_counter()-t)
            with trace.open('a') as f:f.write(json.dumps(record)+'\n')
            report['training_steps']=step+1
            if (step+1)%10==0:
                save();print(json.dumps(record),flush=True)
        report['validation'].append(validate(200))
        assert frozen_before=={n:tensor_sha(p) for n,p in frozen.items()}
        changed=[n for n,p in train.items() if train_before[n]!=tensor_sha(p)];assert changed
        adapter=a.output/'action_adaptation.safetensors';save_file({n:p.detach().cpu().contiguous() for n,p in train.items()},str(adapter))
        masterfile=a.output/'action_fp32_master.safetensors';save_file({n:p.detach().cpu().contiguous() for n,p in masters.items()},str(masterfile))
        torch.save(dict(optimizer=optimizer.state_dict(),step=200,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=rng.bit_generator.state,protocol=protocol),a.output/'OPTIMIZER.pt')
        report.update(status='FIXED_BUDGET_TRAINING_COMPLETED',frozen_parameters_unchanged=True,changed_tensors=len(changed),
            trainable_elements=sum(p.numel() for p in train.values()),frozen_elements=sum(p.numel() for p in frozen.values()),
            adapter=dict(path=str(adapter),bytes=adapter.stat().st_size,sha256=sha(adapter)),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save();print(json.dumps(dict(status=report['status'],steps=200,seconds=report['full_wall_seconds'])),flush=True)
    except BaseException as e:
        report.update(status='TRAINING_FAILED',error=repr(e));save();raise

if __name__=='__main__':main()
