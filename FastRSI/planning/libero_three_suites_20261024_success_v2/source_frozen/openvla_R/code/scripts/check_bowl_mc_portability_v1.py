"""Validate saved NumPy scorer against torch with real saved observations/actions."""
import json
from pathlib import Path
import numpy as np
import torch
from bowl_mc_value_v1 import Predictor,features
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
P=Path('/project');R=Path('/check')
torch.set_num_threads(4)
predictor=Predictor(R/'training/value_model.npz')
folder=P/'runs/bowl_candidates_gpu0_v1_20261009/init42_random2'
raw=unpackb((folder/'000_request.msgpack').read_bytes())
obs={k:raw[k] for k in ('observation/image','observation/wrist_image','observation/state','prompt')}
candidates=np.asarray(unpackb((folder/'000_response.msgpack').read_bytes())['candidates'])
selected,meta=predictor.score(obs,candidates,400)
z=(np.stack([features(obs,a,400) for a in candidates])-predictor.arrays['mean'])/predictor.arrays['std']
outputs=[]
for m in (0,1):
    y=torch.from_numpy(z)
    for i in (0,1):y=torch.tanh(torch.nn.functional.linear(y,torch.from_numpy(predictor.arrays['m%d_w%d'%(m,i)]),torch.from_numpy(predictor.arrays['m%d_b%d'%(m,i)])))
    y=torch.nn.functional.linear(y,torch.from_numpy(predictor.arrays['m%d_w2'%m]),torch.from_numpy(predictor.arrays['m%d_b2'%m]))
    outputs.append(torch.stack([torch.sigmoid(y[:,0]),torch.clamp(y[:,1],0.,1.)],dim=-1).numpy())
torch_output=np.stack(outputs);numpy_output=np.asarray(meta['ensemble_scores'])
error=float(np.max(np.abs(torch_output-numpy_output)));assert error<1e-5,error
assert candidates.shape==(2,10,7) and selected in (0,1)
result=dict(status='PASSED_NUMPY_TORCH_SCORER',max_abs_error=error,selected_index=selected,
    model_sha256=predictor.sha256,scoring_seconds=meta['selection_seconds'],
    action_score_difference=abs(meta['success_scores'][0]-meta['success_scores'][1]),quality_evidence=False)
(R/'PORTABILITY_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
