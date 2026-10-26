"""One fixed rank-four action-expert candidate; native full-coordinate flow objective."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from bowl_expert_lora_v1 import install,enable,adapter_tensors
from audit_bowl_action_pipeline_v1 import read_sample,sha

def tensor_sha(p):return hashlib.sha256(p.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output
 started=time.perf_counter();r=dict(status='RUNNING',steps=0,behavior_gain_proven=False,fast_rsi_gain_proven=False)
 def save():r['seconds']=time.perf_counter()-started;(out/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save()
 try:
  from PhyAgentOS.runtime.policy.openpi.native_openpi_server import NativeOpenPIPolicy
  from openpi.models.model import Observation
  from safetensors.torch import save_file
  dpath=Path('/project/runs/bowl_demo_collection_v1_20261009/DATASET.json');assert sha(dpath)=='afd494cbf367718bd134646d0f4c51fb44e298ff84c7d92add030162c6c5ec93'
  d=json.loads(dpath.read_text());train=[x for x in d['samples'] if x['role']=='train'];val=[x for x in d['samples'] if x['role']=='validation'];val=[val[int(i)] for i in np.linspace(0,len(val)-1,8,dtype=int)]
  assert len(train)==108 and {x['demo'] for x in train}=={'demo_0','demo_2','demo_7','demo_12','demo_32'}
  protocol=dict(rank=4,alpha=4,targets=['q_proj','v_proj'],steps=400,batch_size=1,lr=1e-4,warmup=20,clip_norm=1.,
   betas=[.9,.95],weight_decay=.01,teacher_anchor=.1,action_coordinates=32,
   native_random_noise_time_per_update=True,paired_teacher_student_augmentation=True,
   base_all_frozen=True,train_demos=sorted({x['demo'] for x in train}),train_chunks=len(train),
   validation_used=val,source_rollout_roster=[0,7,25],final_checkpoint_only=True,
   scope='External demonstration candidate; not FastRSI or cyclic decision-model distillation')
  (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
  policy=NativeOpenPIPolicy(policy_config='pi05_libero',checkpoint_dir='/model',default_prompt=None,pytorch_device='cuda:0',disable_torch_compile=True,torch_seed=7)
  model=policy.policy._model;model.eval();base_named=dict(model.named_parameters());before={n:tensor_sha(p) for n,p in base_named.items()}
  raw=read_sample(train[0]);raw.pop('actions');noise=np.random.default_rng(941).standard_normal((10,32)).astype(np.float32)
  base_actions=policy.policy.infer(raw,noise=noise)['actions'].copy()
  torch.manual_seed(31);torch.cuda.manual_seed_all(31);modules=install(model);params=adapter_tensors(modules)
  zero_actions=policy.policy.infer(raw,noise=noise)['actions'].copy()
  assert np.array_equal(base_actions,zero_actions),'Zero adapter changes native sampled output'
  r['zero_update_exact']=True;r['trainable_elements']=sum(p.numel() for p in params.values());r['target_modules']=len(modules)
  assert all(not p.requires_grad for p in base_named.values())
  (out/'BASE_PARAMETER_HASHES.json').write_text(json.dumps(before,indent=2)+'\n')
  cache={x['path']:policy.policy._input_transform(read_sample(x)) for x in train+val}
  def device(v):
   if isinstance(v,dict):return {k:device(x) for k,x in v.items()}
   return torch.as_tensor(np.asarray(v).copy()).unsqueeze(0).to('cuda:0')
  def batch(row):
   data=device(cache[row['path']]);acts=data.pop('actions').float();return Observation.from_dict(data),acts
  capture={}
  def hook(module,args,output):capture['prediction']=output
  handle=model.action_out_proj.register_forward_hook(hook)
  opt=torch.optim.AdamW(params.values(),lr=1e-4,betas=(.9,.95),weight_decay=.01)
  model.gradient_checkpointing_enable();model.train();model.paligemma_with_expert.paligemma.eval()
  rng=np.random.default_rng(31);by_demo={demo:[x for x in train if x['demo']==demo] for demo in protocol['train_demos']}
  torch.manual_seed(31);torch.cuda.manual_seed_all(31)
  for step in range(400):
   pool=by_demo[protocol['train_demos'][step%len(by_demo)]];row=pool[int(rng.integers(len(pool)))];obs,acts=batch(row)
   noise=model.sample_noise(acts.shape,acts.device);tau=model.sample_time(acts.shape[0],acts.device)
   cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all()
   enable(modules,False)
   with torch.no_grad():
    model(obs,acts,noise=noise,time=tau);teacher=capture['prediction'].detach().clone()
   enable(modules,True);torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(gpu_rng)
   opt.zero_grad(set_to_none=True);loss_map=model(obs,acts,noise=noise,time=tau)
   native_loss=loss_map.mean();anchor=F.mse_loss(capture['prediction'],teacher);loss=native_loss+.1*anchor
   assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for p in base_named.values())
   norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.);assert torch.isfinite(norm)
   assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in params.values())
   for group in opt.param_groups:group['lr']=1e-4*min(1.,(step+1)/20)
   opt.step();r['steps']=step+1
   record=dict(step=step+1,demo=row['demo'],sample_sha256=row['sha256'],loss=float(loss),native_flow_loss=float(native_loss),teacher_anchor=float(anchor),gradient_norm=float(norm),time=float(tau[0]))
   with (out/'TRAINING.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   if (step+1)%20==0:save();print(json.dumps(record),flush=True)
  handle.remove();model.eval();enable(modules,True)
  assert before=={n:tensor_sha(p) for n,p in base_named.items()};r['base_parameters_unchanged']=True
  path=out/'expert_lora.safetensors';save_file({n:p.detach().cpu().contiguous() for n,p in params.items()},str(path))
  r['adapter']=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
  metrics=[]
  for i,row in enumerate(val):
   raw=read_sample(row);full=policy.policy._input_transform({k:v.copy() if hasattr(v,'copy') else v for k,v in raw.items()});label=full['actions'];raw.pop('actions')
   noise=np.random.default_rng(7200+i).standard_normal((10,32)).astype(np.float32)
   outputs={}
   for name,active in [('original',False),('lora400',True)]:
    enable(modules,active);actions=policy.policy.infer(raw,noise=noise)['actions'];outputs[name]=actions.copy()
    assert np.isfinite(actions).all() and actions.shape==(10,7)
   np.savez_compressed(out/('validation_%02d.npz'%i),**outputs,labels=read_sample(row)['actions'])
   metrics.append(dict(demo=row['demo'],action_index=row['action_index'],raw_command_mse={n:float(np.mean((x-read_sample(row)['actions'])**2)) for n,x in outputs.items()}))
  enable(modules,True);r.update(status='TRAINING_COMPLETED_PENDING_SOURCE_ROLLOUT',validation=metrics,
   peak_gpu_bytes=torch.cuda.max_memory_allocated(),training_native_forwards=800,sampled_diagnostic_calls=18)
  save();print(json.dumps({k:v for k,v in r.items() if k!='validation'}),flush=True)
 except BaseException as e:r.update(status='FAILED',error=repr(e));save();raise
if __name__=='__main__':main()
