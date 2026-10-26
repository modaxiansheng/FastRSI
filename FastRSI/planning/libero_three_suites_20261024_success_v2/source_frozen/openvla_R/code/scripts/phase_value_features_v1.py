"""Four sharded saved-request replays through unchanged frozen pi; no robot."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from PhyAgentOS.runtime.policy.openpi import native_openpi_server as native
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
from prefix_feature_capture_v1 import capture_prefix
P=Path('/project');R=Path('/check');D=P/'planning/phase_value_dataset_v1_20261010'
slot=int(os.environ['FEATURE_SLOT']);requests=json.loads((D/'REQUESTS.json').read_text())[slot::4]
started=time.perf_counter()
policy=native.NativeOpenPIPolicy(policy_config='pi05_libero',checkpoint_dir='/model',default_prompt=None,
    pytorch_device='cuda:0',disable_torch_compile=True,torch_seed=7)
model=policy.policy._model;versions={n:(id(p),p._version) for n,p in model.named_parameters()}
official=policy.policy.infer
noise=np.random.default_rng(20261010).standard_normal((10,32)).astype(np.float32)
policy.policy.infer=lambda value:official(value,noise=noise)
rows=[];R.joinpath('features').mkdir(exist_ok=False)
for i,ref in enumerate(requests):
    raw=(P/ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'] and len(raw)==ref['bytes']
    data=unpackb(raw);obs={k:data[k] for k in ('observation/image','observation/wrist_image','observation/state','prompt')}
    tick=time.perf_counter()
    with capture_prefix(model) as stats:policy.infer(obs)
    assert stats['prefix_passes']==1 and len(stats['features'])==1
    f=stats['features'][0];assert f.shape==(4096,) and np.isfinite(f).all()
    np.savez_compressed(R/'features'/(ref['sha256']+'.npz'),feature=f,state=obs['observation/state'])
    rows.append(dict(request_sha256=ref['sha256'],seconds=time.perf_counter()-tick,
        extraction_seconds=stats['feature_seconds'],prefix_passes=1,image_passes=stats['image_passes']))
    if i%25==0:print(json.dumps(dict(slot=slot,completed=i+1,total=len(requests))),flush=True)
assert versions=={n:(id(p),p._version) for n,p in model.named_parameters()}
report=dict(status='COMPLETED',slot=slot,requests=len(rows),native_inferences=len(rows),robot_executions=0,
    parameter_updates=0,wall_seconds=time.perf_counter()-started,rows=rows)
(R/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
