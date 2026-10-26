"""One genuine feedback update; fixed representation, no new robot labels."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
from phase_iql_value_v1 import Predictor
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb

class AdaptivePredictor(Predictor):
    def value(self,feature,robot,budget):
        x=self.state(feature,robot,budget)
        for i in range(2):x=np.tanh(x@self.a[f'v_w{i}'].T+self.a[f'v_b{i}'])
        x=x@self.a['v_w2'].T+self.a['v_b2']
        return float(1./(1.+np.exp(-np.clip(x[0],-40,40))))

def main():
    import torch
    from torch import nn
    torch.set_num_threads(4);torch.manual_seed(11);tick=time.perf_counter()
    P=Path('/project');O=Path('/check/training');O.mkdir(exist_ok=False)
    old=P/'runs/phase_iql_training_v1_20261010/training';oldarrays=dict(np.load(old/'value_model.npz',allow_pickle=False))
    state=torch.load(old/'training_state.pt',map_location='cpu',weights_only=True)
    def network(dim):return nn.Sequential(nn.Linear(dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1),nn.Sigmoid())
    q=nn.ModuleList([network(143),network(143)]);target=copy.deepcopy(q);v=network(73)
    q.load_state_dict(state['q']);target.load_state_dict(state['target']);v.load_state_dict(state['v'])
    def arrays_for(q,v):
        arrays=dict(oldarrays)
        for m,net in enumerate(q):
            for i,layer in enumerate((net[0],net[2],net[4])):arrays[f'q{m}_w{i}']=layer.weight.detach().numpy().copy();arrays[f'q{m}_b{i}']=layer.bias.detach().numpy().copy()
        for i,layer in enumerate((v[0],v[2],v[4])):arrays[f'v_w{i}']=layer.weight.detach().numpy().copy();arrays[f'v_b{i}']=layer.bias.detach().numpy().copy()
        return arrays
    np.savez_compressed(O/'frozen_router.npz',**arrays_for(q,v));frozen=AdaptivePredictor(O/'frozen_router.npz')
    for k in oldarrays:assert np.array_equal(oldarrays[k],frozen.a[k])
    featurebank={}
    F=P/'planning/phase_value_features_v1_20261010'
    for slot in range(4):
        for f in (F/f'gpu{slot}/features').glob('*.npz'):featurebank[f.stem]=dict(np.load(f,allow_pickle=False))
    oldrows=list(map(json.loads,(P/'planning/phase_value_dataset_v1_20261010/TRANSITIONS.jsonl').read_text().splitlines()))
    def convert(r,feature,robot,next_feature,next_robot):
        s=frozen.state(feature,robot,r['budget']);a=(np.asarray(r['planned_actions'],np.float32).ravel()-oldarrays['action_mean'])/oldarrays['action_std']
        ns=np.zeros_like(s) if r['terminal'] else frozen.state(next_feature,next_robot,r['budget']-r['n'])
        return s,a,ns,r['reward']*.995**(r['n']-1),(0 if r['terminal'] else .995**r['n'])
    olddata=[]
    for r in oldrows:
        if r['role']!='train':continue
        z=featurebank[r['observation']['sha256']];nz=featurebank[r['next_observation']['sha256']] if r['next_observation'] else None
        olddata.append(convert(r,z['feature'],z['state'],nz['feature'] if nz else None,nz['state'] if nz else None))
    newdata=[];newrows=[];episodes=[]
    for slot in range(4):
        root=P/'runs'/f'phase_iql_rollout_gpu{slot}_v1_20261010'/f'init{28+slot}_iql2'
        result=json.loads((root/'RESULT.json').read_text());assert result['status']=='COMPLETED' and result['persistent_updates']==0
        episodes.append(dict(path=str(root.relative_to(P)),success=result['native_success'],actions=result['executed_actions']))
        for i,r in enumerate(result['rows']):
            requestraw=(root/f'{i:03d}_request.msgpack').read_bytes();response_raw=(root/f'{i:03d}_response.msgpack').read_bytes()
            obs=unpackb(requestraw);response=unpackb(response_raw);terminal=i==len(result['rows'])-1
            nz=unpackb((root/f'{i+1:03d}_response.msgpack').read_bytes()) if not terminal else None
            nobs=unpackb((root/f'{i+1:03d}_request.msgpack').read_bytes()) if not terminal else None
            command=np.load(root/f'{i:03d}_command.npz',allow_pickle=False)['planned_actions']
            assert np.array_equal(command,response['candidates'][r['selected_index']])
            rr=dict(episode=str(root.relative_to(P)),chunk=i,budget=400-r['step_before'],n=r['step_after']-r['step_before'],
                terminal=terminal,reward=int(terminal and result['native_success']),planned_actions=command.tolist(),
                request_sha256=hashlib.sha256(requestraw).hexdigest(),response_sha256=hashlib.sha256(response_raw).hexdigest(),
                counterfactual_label=None,feature_from_existing_policy_prefix=True)
            newrows.append(rr);newdata.append(convert(rr,response['prefix_feature'],obs['observation/state'],
                nz['prefix_feature'] if nz else None,nobs['observation/state'] if nobs else None))
    (O/'NEW_FEEDBACK.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in newrows))
    def tensors(data):return [torch.tensor(np.stack([r[i] for r in data]),dtype=torch.float32) for i in range(5)]
    oldt=tensors(olddata);newt=tensors(newdata);qopt=torch.optim.Adam(q.parameters(),lr=1e-4);vopt=torch.optim.Adam(v.parameters(),lr=1e-4)
    logs=[]
    for step in range(1000):
        # Fixed 3:1 replay/new-feedback mixture; no fitting to next evaluation.
        io=torch.randint(len(olddata),(96,));inew=torch.randint(len(newdata),(32,))
        s,a,ns,r,d=[torch.cat((x[io],y[inew]),0) for x,y in zip(oldt,newt)];r=r[:,None];d=d[:,None];sa=torch.cat((s,a),1)
        with torch.no_grad():tq=torch.minimum(target[0](sa),target[1](sa))
        diff=tq-v(s);vl=(torch.where(diff>0,.7,.3)*diff.square()).mean();vopt.zero_grad();vl.backward();vopt.step()
        with torch.no_grad():y=r+d*v(ns)
        ql=sum((net(sa)-y).square().mean() for net in q);qopt.zero_grad();ql.backward();nn.utils.clip_grad_norm_(q.parameters(),1.);qopt.step()
        with torch.no_grad():
            for dst,src in zip(target.parameters(),q.parameters()):dst.lerp_(src,.005)
        if step in (0,499,999):logs.append(dict(step=step+1,q_loss=float(ql.detach()),v_loss=float(vl.detach())))
    arrays=arrays_for(q,v);np.savez_compressed(O/'updated_router.npz',**arrays)
    torch.save(dict(q=q.state_dict(),v=v.state_dict(),target=target.state_dict()),O/'updated_training_state.pt')
    updated=AdaptivePredictor(O/'updated_router.npz')
    diffs=[];port=[]
    for i in (0,len(newdata)//2,len(newdata)-1):
        ss,aa,*_=newdata[i]
        def infer_np(arr):
            y=ss
            for j in range(2):y=np.tanh(y@arr[f'v_w{j}'].T+arr[f'v_b{j}'])
            y=y@arr['v_w2'].T+arr['v_b2'];return float(1./(1.+np.exp(-y[0])))
        with torch.no_grad():tv=float(v(torch.from_numpy(ss[None]))[0,0])
        error=abs(tv-infer_np(arrays));assert error<1e-5;port.append(error)
        diffs.append(dict(row=i,old_v=infer_np(frozen.a),new_v=infer_np(arrays)))
    report=dict(status='ACTUAL_FEEDBACK_UPDATED_EXTERNAL_Q_V',new_episodes=episodes,new_rows=len(newdata),old_replay_rows=len(olddata),
        updates=1000,old_new_batch_ratio=[96,32],pi_updates=0,new_feature_encodings=0,training_device='cpu',
        frozen_q_exact_to_previous_checkpoint=True,representation_and_normalization_frozen=True,
        frozen_sha256=frozen.sha256,updated_sha256=updated.sha256,feedback_examples=diffs,portable_v_max_error=max(port),
        q_parameter_change_norm=float(np.sqrt(sum(np.square(arrays[k]-oldarrays[k]).sum() for k in oldarrays if k.startswith('q')))),
        losses=logs,seconds=time.perf_counter()-tick,learning_gain_proven=False,phase_goal_complete=False,
        next_router='Generate first native candidate; generate second only if min Q(first)<learned V and fewer than20 extras used; execute original selected candidate',
        future_eval_initials=[24,25,26,27],future_noise_bases=[7101024,7101025,7101026,7101027],
        candidate_note='The Q/V comparison is a testable heuristic, not a certificate or a direct label of information value')
    (O/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
