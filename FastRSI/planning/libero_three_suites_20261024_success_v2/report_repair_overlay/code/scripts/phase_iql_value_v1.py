"""IQL-style Q/V adaptation for native action blocks, frozen pi, no trained actor.

Borrow expectile V and Bellman Q from official IQL. Native success discounted
per executed action is the only reward. Scores are not success probabilities.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np

class Predictor:
    def __init__(self,path):
        self.path=Path(path);self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.a=dict(np.load(self.path,allow_pickle=False))
    def state(self,feature,robot,budget):
        z=((np.asarray(feature,np.float32)-self.a['feature_mean'])/self.a['feature_std'])@self.a['projection']
        s=np.concatenate([z,np.asarray(robot,np.float32),np.array([budget/400.],np.float32)])
        return (s-self.a['state_mean'])/self.a['state_std']
    def score(self,feature,robot,candidates,budget):
        tick=time.perf_counter();state=self.state(feature,robot,budget)
        actions=(np.asarray(candidates,np.float32).reshape(-1,70)-self.a['action_mean'])/self.a['action_std']
        x=np.concatenate([np.repeat(state[None],len(actions),axis=0),actions],axis=1);scores=[]
        for m in range(2):
            y=x
            for i in range(2):y=np.tanh(y@self.a[f'q{m}_w{i}'].T+self.a[f'q{m}_b{i}'])
            y=y@self.a[f'q{m}_w2'].T+self.a[f'q{m}_b2'];y=1./(1.+np.exp(-np.clip(y,-40,40)))
            scores.append(y[:,0])
        twin=np.stack(scores);q=twin.min(0);selected=int(np.argmax(q))
        return selected,dict(q_scores=q.tolist(),twin_scores=twin.tolist(),selection_seconds=time.perf_counter()-tick,
            interpretation='Uncalibrated discounted native-success Q estimates; no guarantee for unseen actions')

def train(project,out):
    import torch
    from torch import nn
    torch.set_num_threads(4);torch.manual_seed(7);started=time.perf_counter();out.mkdir(exist_ok=False)
    D=project/'planning/phase_value_dataset_v1_20261010';F=project/'planning/phase_value_features_v1_20261010'
    rows=list(map(json.loads,(D/'TRANSITIONS.jsonl').read_text().splitlines()))
    assert json.loads((F/'RESULT.json').read_text())['status']=='COMPLETED'
    features={}
    for slot in range(4):
        for f in (F/f'gpu{slot}/features').glob('*.npz'):
            assert f.stem not in features;features[f.stem]=dict(np.load(f,allow_pickle=False))
    train=np.array([r['role']=='train' for r in rows]);assert train.sum()==1320
    raw=np.stack([features[r['observation']['sha256']]['feature'] for r in rows]);assert raw.shape==(len(rows),4096)
    robots=np.stack([features[r['observation']['sha256']]['state'] for r in rows]).astype(np.float32)
    fm=raw[train].mean(0);fs=np.maximum(raw[train].std(0),.05)
    normalized=(raw-fm)/fs
    # PCA fitted exclusively to training groups; no future test or validation fit.
    _,_,projection=torch.pca_lowrank(torch.from_numpy(normalized[train]),q=64,center=False,niter=3)
    projection=projection.numpy().astype(np.float32)
    state=np.concatenate([normalized@projection,robots,np.array([[r['budget']/400.] for r in rows],np.float32)],axis=1)
    sm=state[train].mean(0);ss=np.maximum(state[train].std(0),.05);state=(state-sm)/ss
    nxt=np.zeros_like(state)
    for i,r in enumerate(rows):
        if r['next_observation']:
            z=features[r['next_observation']['sha256']]
            a=np.concatenate([((z['feature']-fm)/fs)@projection,z['state'],np.array([(r['budget']-r['n'])/400.],np.float32)])
            nxt[i]=(a-sm)/ss
        else:assert r['terminal']
    actions=np.array([r['planned_actions'] for r in rows],np.float32).reshape(len(rows),70)
    am=actions[train].mean(0);ast=np.maximum(actions[train].std(0),.05);actions=(actions-am)/ast
    n=np.array([r['n'] for r in rows]);done=np.array([r['terminal'] for r in rows]);gamma=.995
    reward=np.array([r['reward'] for r in rows],np.float32)*gamma**(n-1)
    discount=(~done).astype(np.float32)*gamma**n
    assert np.all(discount[done]==0) and np.all(reward[~done]==0)
    S=torch.from_numpy(state[train]);A=torch.from_numpy(actions[train]);N=torch.from_numpy(nxt[train])
    RW=torch.tensor(reward[train],dtype=torch.float32)[:,None];DS=torch.tensor(discount[train],dtype=torch.float32)[:,None]
    def network(dim):return nn.Sequential(nn.Linear(dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1),nn.Sigmoid())
    q=nn.ModuleList([network(S.shape[1]+70),network(S.shape[1]+70)]);target=copy.deepcopy(q);v=network(S.shape[1])
    qopt=torch.optim.Adam(q.parameters(),lr=3e-4);vopt=torch.optim.Adam(v.parameters(),lr=3e-4)
    losses=[];updates=3000
    for step in range(updates):
        idx=torch.randint(len(S),(128,));s=S[idx];a=A[idx];sa=torch.cat((s,a),1)
        with torch.no_grad():tq=torch.minimum(target[0](sa),target[1](sa))
        diff=tq-v(s);vl=(torch.where(diff>0,.7,.3)*diff.square()).mean()
        vopt.zero_grad();vl.backward();nn.utils.clip_grad_norm_(v.parameters(),1.);vopt.step()
        with torch.no_grad():y=RW[idx]+DS[idx]*v(N[idx])
        ql=sum((net(sa)-y).square().mean() for net in q)
        qopt.zero_grad();ql.backward();nn.utils.clip_grad_norm_(q.parameters(),1.);qopt.step()
        with torch.no_grad():
            for dst,src in zip(target.parameters(),q.parameters()):dst.lerp_(src,.005)
        if step in (0,999,1999,2999):
            record=dict(step=step+1,q_loss=float(ql.detach()),v_loss=float(vl.detach()),q_target_mean=float(y.mean()))
            losses.append(record);print(json.dumps(record),flush=True)
    arrays=dict(feature_mean=fm,feature_std=fs,projection=projection,state_mean=sm,state_std=ss,action_mean=am,action_std=ast)
    for m,net in enumerate(q):
        for i,layer in enumerate((net[0],net[2],net[4])):arrays[f'q{m}_w{i}']=layer.weight.detach().numpy().copy();arrays[f'q{m}_b{i}']=layer.bias.detach().numpy().copy()
    np.savez_compressed(out/'value_model.npz',**arrays);torch.save(dict(q=q.state_dict(),v=v.state_dict(),target=target.state_dict()),out/'training_state.pt')
    pred=Predictor(out/'value_model.npz');portable=[];diagnostics=[]
    for role,mask in [('train',train),('diagnostic_validation',~train)]:
        with torch.no_grad():
            ts=torch.from_numpy(state[mask]);ta=torch.from_numpy(actions[mask]);sa=torch.cat((ts,ta),1)
            qq=torch.minimum(q[0](sa),q[1](sa))[:,0].numpy();vv=v(torch.from_numpy(nxt[mask]))[:,0].numpy()
        outcomes=np.array([r['observed_episode_success'] for r in rows])[mask]
        target_labels=reward[mask]+discount[mask]*vv
        shuffled=ta.roll(1,0)
        with torch.no_grad():shq=torch.minimum(q[0](torch.cat((ts,shuffled),1)),q[1](torch.cat((ts,shuffled),1)))[:,0].numpy()
        diagnostics.append(dict(role=role,rows=int(mask.sum()),td_mse=float(np.mean((qq-target_labels)**2)),
            q_mean_observed_success=float(qq[outcomes==1].mean()),q_mean_observed_failure=float(qq[outcomes==0].mean()),
            action_shuffle_mean_abs_score_change=float(np.abs(qq-shq).mean()),
            interpretation='Trajectory-group diagnostics, not true same-state ranking or deployment quality'))
    for i in (0,len(rows)//2,len(rows)-1):
        r=rows[i];z=features[r['observation']['sha256']]
        _,pp=pred.score(z['feature'],z['state'],[r['planned_actions']],r['budget'])
        with torch.no_grad():
            sa=torch.from_numpy(np.concatenate([state[i],actions[i]])[None]);expected=float(torch.minimum(q[0](sa),q[1](sa))[0,0])
        error=abs(pp['q_scores'][0]-expected);assert error<1e-5;portable.append(error)
    report=dict(status='TRAINED_IQL_STYLE_EXTERNAL_CRITIC',rows=len(rows),train_rows=int(train.sum()),validation_rows=int((~train).sum()),
        updates=updates,native_pi_parameter_updates=0,robot_executions=0,actor_training=False,goal_achieved=False,
        gamma_per_executed_action=gamma,expectile=.7,target_tau=.005,steps=losses,diagnostics=diagnostics,
        native_success_discounted_not_negative_failure_time=True,partial_terminal_rows_retained=int(sum(done&(n<10))),
        pca_fit_only_train_groups=True,portable_max_error=max(portable),model_sha256=pred.sha256,
        seconds=time.perf_counter()-started,source_transitions_sha256=hashlib.sha256((D/'TRANSITIONS.jsonl').read_bytes()).hexdigest(),
        adaptation='Official IQL Q/V objectives; bounded sigmoid heads, semi-MDP discount, frozen pi candidate selection; not full IQL/Cal-QL reproduction',
        upcoming_pilot_initials=[28,29,30,31],test_noise_bases=[6101028,6101029,6101030,6101031])
    (out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();train(args.project,args.output)
