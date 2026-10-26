"""Registered factual default-pi targets, small head and one real feedback update."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
from phase_iql_value_v1 import Predictor
from native_quality_cost_v1 import QualityCostPredictor

P=Path('/project');O=Path('/check/training')

def main():
    import torch
    from torch import nn
    torch.set_num_threads(4);start=time.perf_counter();O.mkdir(exist_ok=False)
    regpath=P/'planning/native_quality_cost_head_registration_v1_20261010/REGISTRATION.json'
    reg=json.loads(regpath.read_text());assert reg['status']=='REGISTERED_BEFORE_NEW_QUALITY_COST_HEAD_FITTING'
    data=P/'planning/native_default_continuations_v1_20261010'
    assert hashlib.sha256((data/'RESULT.json').read_bytes()).hexdigest()==reg['default_continuations_sha256']
    records=[json.loads(x) for x in (data/'RECORDS.jsonl').read_text().splitlines()]
    bank=dict(np.load(data/'SUPERVISION.npz',allow_pickle=False))
    X=bank['inputs'];Y=bank['targets'][:,:2];roles=[reg['group_roles'][r['group']] for r in records]
    groups={g:np.array([i for i,r in enumerate(records) if r['group']==g])
        for g in sorted({r['group'] for r in records}) if reg['group_roles'][g]=='train'}
    assert sum(len(a) for a in groups.values())==316 and len(groups)==10
    pairdir=P/'planning/native_quality_cost_pairs_v2_20261010'
    pairpath=pairdir/'PAIRS.json';assert hashlib.sha256(pairpath.read_bytes()).hexdigest()==reg['pair_supervision_sha256']
    pairs=json.loads(pairpath.read_text());raw=dict(np.load(pairdir/'INPUTS.npz',allow_pickle=False))
    rep=Predictor(P/'planning/native_paired_q_update_v1_20261010/training/paired_router.npz');PX=[];PY=[]
    for i,pair in enumerate(pairs):
        state=rep.state(raw[f'feature_{i}'],raw[f'robot_{i}'],float(raw[f'budget_{i}'][0]))
        action=(raw[f'candidates_{i}'].reshape(2,70)-rep.a['action_mean'])/rep.a['action_std']
        PX.append(np.concatenate([np.repeat(state[None],2,axis=0),action],axis=1))
        PY.append(np.stack([pair['observed_native_success'],np.array(pair['observed_remaining_actions'])/400.],axis=1))
    TX=torch.tensor(X,dtype=torch.float32);TY=torch.tensor(Y,dtype=torch.float32)
    TPX=torch.tensor(np.stack(PX),dtype=torch.float32);TPY=torch.tensor(np.stack(PY),dtype=torch.float32)
    rng=np.random.default_rng(101);groupnames=list(groups)
    def sample():
        return torch.tensor([int(rng.choice(groups[groupnames[int(rng.integers(len(groupnames)))]])) for _ in range(64)])
    def net(dim):return nn.Sequential(nn.Linear(dim,32),nn.Tanh(),nn.Linear(32,2),nn.Sigmoid())
    heads=[];state_heads=[]
    for seed in reg['members_seeds']:
        torch.manual_seed(seed);heads.append(net(143));torch.manual_seed(seed);state_heads.append(net(73))
    heads=nn.ModuleList(heads);state_heads=nn.ModuleList(state_heads)
    def losses(predicted,targets):
        quality=nn.functional.binary_cross_entropy(predicted[...,0],targets[...,0])
        mask=targets[...,0]>0
        cost=(predicted[...,1]-targets[...,1]).square()[mask].mean() if mask.any() else predicted.sum()*0
        return quality+cost
    def pair_loss(head,count,stateonly=False):
        values=TPX[:count,:,:73] if stateonly else TPX[:count]
        preds=head(values)
        # Balance groups rather than letting three points from one start dominate.
        by_group={}
        for i,pair in enumerate(pairs[:count]):by_group.setdefault(pair['development_group'],[]).append(i)
        allgroups=[]
        for indices in by_group.values():
            terms=[]
            for i in indices:
                term=losses(preds[i],TPY[i]);qp=pairs[i]['quality_preference'];cp=pairs[i]['cost_preference_given_both_success']
                if qp is not None:
                    sign=1 if qp else -1;term=term+torch.relu(.1-sign*(preds[i,1,0]-preds[i,0,0])).square()
                if cp is not None:
                    sign=1 if cp else -1;margin=abs(float(TPY[i,1,1]-TPY[i,0,1]))
                    term=term+torch.relu(margin-sign*(preds[i,0,1]-preds[i,1,1])).square()
                terms.append(term)
            allgroups.append(torch.stack(terms).mean())
        return torch.stack(allgroups).mean()
    opt=torch.optim.Adam(list(heads.parameters())+list(state_heads.parameters()),lr=reg['fit_learning_rate'])
    logs=[]
    for step in range(reg['fixed_fit_updates']):
        idx=sample();loss=0
        for head in heads:loss=loss+losses(head(TX[idx]),TY[idx])+pair_loss(head,6)
        for head in state_heads:loss=loss+losses(head(TX[idx,:73]),TY[idx])+pair_loss(head,6,True)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(list(heads.parameters())+list(state_heads.parameters()),1.);opt.step()
        if step in (0,499,999):logs.append(dict(stage='initial_fit',step=step+1,loss=float(loss.detach())))
    frozen=copy.deepcopy(heads);teacher=copy.deepcopy(heads);teacher.requires_grad_(False)
    opt=torch.optim.Adam(heads.parameters(),lr=reg['update_learning_rate'])
    for step in range(reg['fixed_feedback_updates']):
        idx=sample();loss=0
        for head,old in zip(heads,teacher):
            prediction=head(TX[idx]);loss=loss+losses(prediction,TY[idx])+pair_loss(head,10)
            with torch.no_grad():anchor=old(TX[idx])
            loss=loss+(prediction-anchor).square().mean()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(heads.parameters(),1.);opt.step()
        if step in (0,249,499):logs.append(dict(stage='feedback_update',step=step+1,loss=float(loss.detach())))
    base={k:v.copy() for k,v in rep.a.items() if k in ('feature_mean','feature_std','projection','state_mean','state_std','action_mean','action_std')}
    portable=[]
    models={}
    for name,ensemble in [('F',frozen),('R',heads)]:
        arrays={k:v.copy() for k,v in base.items()}
        for m,head in enumerate(ensemble):
            for i,layer in enumerate((head[0],head[2])):
                arrays[f'h{m}_w{i}']=layer.weight.detach().numpy().copy();arrays[f'h{m}_b{i}']=layer.bias.detach().numpy().copy()
        path=O/f'quality_cost_{name}.npz';np.savez_compressed(path,**arrays);models[name]=QualityCostPredictor(path)
        with torch.no_grad():expected=torch.stack([head(TPX.reshape(-1,143)) for head in ensemble]).numpy()
        error=float(np.max(np.abs(expected-models[name].outputs(np.stack(PX).reshape(-1,143)))));assert error<1e-5;portable.append(error)
    diagnostics=[]
    for role in sorted(set(roles)):
        idx=np.array([i for i,r in enumerate(roles) if r==role]);groupnames_role=sorted({records[i]['group'] for i in idx})
        for name,ensemble,stateonly in [('F',frozen,False),('R',heads,False),('state_only',state_heads,True)]:
            with torch.no_grad():values=torch.stack([h(TX[idx,:73] if stateonly else TX[idx]) for h in ensemble]).mean(0).numpy()
            groupmetrics=[]
            for group in groupnames_role:
                mask=np.array([records[i]['group']==group for i in idx]);yy=Y[idx][mask];vv=values[mask];success=yy[:,0]>0
                groupmetrics.append(dict(group=group,success=int(yy[0,0]),brier=float(np.mean((vv[:,0]-yy[:,0])**2)),
                    success_action_cost_mse=float(np.mean((vv[success,1]-yy[success,1])**2)) if success.any() else None))
            diagnostics.append(dict(role=role,model=name,episodes=len(groupmetrics),macro_brier=float(np.mean([g['brier'] for g in groupmetrics])),groups=groupmetrics))
    selections=[]
    for i,pair in enumerate(pairs):
        item=dict(pair=i,group=pair['development_group'],step=pair['step'],quality_preference=pair['quality_preference'],
            conditional_cost_preference=pair['cost_preference_given_both_success'])
        for name,model in models.items():
            chosen,score=model.score(raw[f'feature_{i}'],raw[f'robot_{i}'],raw[f'candidates_{i}'],float(raw[f'budget_{i}'][0]))
            item[name]=dict(selected=chosen,**score)
        selections.append(item)
    report=dict(status='ACTUAL_SMALL_QUALITY_COST_HEADS_FITTED_AND_FEEDBACK_UPDATED',
        initial_updates=1000,feedback_updates=500,pi_updates=0,llm_updates=0,robot_executions=0,
        representation_frozen=True,no_completion_discount=True,frozen_pairs=6,updated_pairs=10,
        train_episodes=10,train_rows=316,balanced_development_calibration_episodes=4,
        initial_parameter_count=sum(p.numel() for p in frozen.parameters()),
        model_sha256={k:v.sha256 for k,v in models.items()},portable_max_error=max(portable),
        diagnostics=diagnostics,development_pair_selections=selections,losses=logs,
        seconds=time.perf_counter()-start,registration_sha256=hashlib.sha256(regpath.read_bytes()).hexdigest(),
        independent_gain=False,goal_complete=False,
        caveat='Factual default-pi continuation scores, limited groups and uncalibrated member ranges; no native policy gain or repayment from fitting.')
    (O/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('development_pair_selections','diagnostics')}),flush=True)

if __name__=='__main__':main()
