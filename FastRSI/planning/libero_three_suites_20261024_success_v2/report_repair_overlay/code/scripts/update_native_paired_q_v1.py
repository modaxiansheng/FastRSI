"""Use actual paired default-pi continuations to update Q only, with old-Q retention.

These are observed Monte Carlo returns, not a guarantee for later learned-router
continuations. Development fit is not held-out policy gain. No teacher prose labels.
"""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
from phase_iql_feedback_update_v1 import AdaptivePredictor
P=Path('/project');O=Path('/check');GAMMA=.995

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())

def build_pairs():
    audit=load(P/'planning/native_value_branch_audit_v1_20261010/RESULT.json')
    assert audit['status']=='AUDITED_REAL_BRANCH_SUPERVISION' and audit['default_branch_reproduced_native_rgb_robot_and_actions']
    rows=[]
    for pair in audit['pairs']:
        initial=pair['initial'];plan=load(P/f'planning/native_value_branch_inputs_v1_20261010/init{initial}/PLAN.json')
        obs=unpackb((P/plan['source_request']).read_bytes());resp=unpackb((P/plan['source_response']).read_bytes())
        assert sha(P/plan['source_request'])==plan['source_request_sha256'] and sha(P/plan['source_response'])==plan['source_response_sha256']
        outcomes=[pair['arms'][str(j)] for j in (0,1)]
        rows.append(dict(initial=initial,step=10,feature=resp['prefix_feature'],robot=obs['observation/state'],candidates=resp['candidates'],outcomes=outcomes,
            request=plan['source_request'],request_sha256=plan['source_request_sha256'],response=plan['source_response'],response_sha256=plan['source_response_sha256']))
    campaign=load(P/'planning/native_value_branch_batch_v2_20261010/RESULT.json');assert campaign['status']=='COMPLETED'
    registration=load(P/'planning/native_value_branch_next_v1_20261010/PLAN.json')
    for slot,ctx in enumerate(registration['contexts']):
        root=P/f'runs/native_value_branch_gpu{slot}_v2_20261010';r=load(root/'RESULT.json')
        assert r['status']=='COMPLETED' and r['context']==ctx and r['default_candidate_exact'] and all(r['fork_inputs_exact'].values())
        assert all(not x['state']['Running'] and x['state']['Pid']==0 for x in load(root/'STOP_RECEIPT.json'))
        assert sha(P/ctx['full_actual_baseline_outcome'])==ctx['outcome_sha256']
        baseline=load(P/ctx['full_actual_baseline_outcome']);assert len(baseline['cases'])==1
        sessions=baseline['cases'][0]['sessions'];assert len(sessions)==1
        b=sessions[0]['result'];assert type(b['success']) is bool
        step=ctx['step'];index=step//10
        obs=unpackb((root/f'{index:03d}_request.msgpack').read_bytes());resp=unpackb((root/f'{index:03d}_response.msgpack').read_bytes())
        expected=unpackb((P/ctx['request']).read_bytes());ref=unpackb((P/ctx['response']).read_bytes())
        assert sha(P/ctx['request'])==ctx['request_sha256'] and sha(P/ctx['response'])==ctx['response_sha256']
        for key in ('observation/image','observation/wrist_image','observation/state'):assert np.array_equal(obs[key],expected[key])
        assert obs['prompt']==expected['prompt'] and np.array_equal(resp['candidates'][0],ref['actions'])
        checked=0
        for line in r['rows']:
            i=line['before']//10;req=unpackb((root/f'{i:03d}_request.msgpack').read_bytes());reply=unpackb((root/f'{i:03d}_response.msgpack').read_bytes())
            tag=req['_branch_baseline_v2'];assert tag['step']==line['before'] and tag['fork_step']==step
            selected=1 if line['before']==step else 0;assert line['selected_index']==selected
            assert np.array_equal(np.load(root/f'{i:03d}_command.npz',allow_pickle=False)['actions'],reply['candidates'][selected])
            for j in range(line['candidates']):
                noise=np.random.default_rng(np.random.SeedSequence([tag['noise_base'],i,j])).standard_normal((10,32)).astype(np.float32)
                assert hashlib.sha256(noise.tobytes()).hexdigest()==reply['branch_baseline']['noise_sha256'][j]
            checked+=1
        outcomes=[dict(native_success=b['success'],remaining_actions=b['num_steps']-step,reference=ctx['full_actual_baseline_outcome'],source_sha256=ctx['outcome_sha256']),
            dict(native_success=r['native_success'],remaining_actions=r['remaining_actions'],reference=str((root/'RESULT.json').relative_to(P)),source_sha256=sha(root/'RESULT.json'))]
        rows.append(dict(initial=ctx['initial'],step=step,feature=resp['prefix_feature'],robot=obs['observation/state'],candidates=resp['candidates'],outcomes=outcomes,
            request=str((root/f'{index:03d}_request.msgpack').relative_to(P)),request_sha256=sha(root/f'{index:03d}_request.msgpack'),
            response=str((root/f'{index:03d}_response.msgpack').relative_to(P)),response_sha256=sha(root/f'{index:03d}_response.msgpack'),checked_commands=checked))
    assert len(rows)==6
    return rows

def main():
    import torch
    from torch import nn
    torch.set_num_threads(4);torch.manual_seed(37);tick=time.perf_counter()
    (O/'training').mkdir(exist_ok=False);out=O/'training'
    source=P/'planning/phase_iql_feedback_v1_20261010/training/frozen_router.npz'
    frozen=AdaptivePredictor(source);a=frozen.a
    def network():return nn.Sequential(nn.Linear(143,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1),nn.Sigmoid())
    q=nn.ModuleList([network(),network()])
    with torch.no_grad():
        for m,net in enumerate(q):
            for i,layer in enumerate((net[0],net[2],net[4])):
                layer.weight.copy_(torch.from_numpy(a[f'q{m}_w{i}']));layer.bias.copy_(torch.from_numpy(a[f'q{m}_b{i}']))
    teacher=copy.deepcopy(q);teacher.requires_grad_(False)
    pairs=build_pairs();inputs=[];targets=[];records=[]
    for pair in pairs:
        state=frozen.state(pair['feature'],pair['robot'],400-pair['step'])
        actions=(np.asarray(pair['candidates'],np.float32).reshape(2,70)-a['action_mean'])/a['action_std']
        inputs.append(np.concatenate([np.repeat(state[None],2,axis=0),actions],axis=1))
        y=[float(GAMMA**(r['remaining_actions']-1)) if r['native_success'] else 0. for r in pair['outcomes']];targets.append(y)
        pref=None if y[0]==y[1] else int(y[1]>y[0])
        _,old=frozen.score(pair['feature'],pair['robot'],pair['candidates'],400-pair['step'])
        records.append({**{k:v for k,v in pair.items() if k not in ('feature','robot','candidates')},'observed_discounted_returns':y,
            'preferred_observed_candidate':pref,'frozen_scores':old['q_scores'],'development_only':True,
            'continuation_policy':'unchanged default pi; does not label learned-router future choices'})
    X=torch.from_numpy(np.stack(inputs));Y=torch.tensor(targets,dtype=torch.float32)
    featurebank={}
    for slot in range(4):
        for f in (P/f'planning/phase_value_features_v1_20261010/gpu{slot}/features').glob('*.npz'):
            featurebank[f.stem]=dict(np.load(f,allow_pickle=False))
    old=[]
    for r in map(json.loads,(P/'planning/phase_value_dataset_v1_20261010/TRANSITIONS.jsonl').read_text().splitlines()):
        if r['role']!='train':continue
        z=featurebank[r['observation']['sha256']];s=frozen.state(z['feature'],z['state'],r['budget'])
        ac=(np.asarray(r['planned_actions'],np.float32).ravel()-a['action_mean'])/a['action_std'];old.append(np.concatenate([s,ac]))
    OLD=torch.from_numpy(np.stack(old));assert len(OLD)==1320
    with torch.no_grad():anchor=torch.stack([net(OLD).squeeze(-1) for net in teacher])
    # Registered fixed update schedule, no selection on subsequent robot outcomes.
    opt=torch.optim.Adam(q.parameters(),lr=5e-5);logs=[]
    for step in range(300):
        io=torch.randint(len(OLD),(128,));ql=0.;rl=0.;rankloss=0.
        for m,net in enumerate(q):
            predicted=net(X.reshape(-1,143)).reshape(6,2)
            ql=ql+(predicted-Y).square().mean()
            rl=rl+(net(OLD[io]).squeeze(-1)-anchor[m,io]).square().mean()
            diff=Y[:,1]-Y[:,0];valid=diff!=0
            signed=(predicted[:,1]-predicted[:,0])*torch.sign(diff)
            rankloss=rankloss+torch.relu(torch.clamp(diff.abs(),max=.1)-signed)[valid].square().mean()
        loss=ql+5*rl+rankloss;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(q.parameters(),1.);opt.step()
        if step in (0,149,299):logs.append(dict(step=step+1,pair_loss=float(ql.detach()),old_retention_loss=float(rl.detach()),rank_loss=float(rankloss.detach())))
    arrays={k:v.copy() for k,v in a.items()}
    for m,net in enumerate(q):
        for i,layer in enumerate((net[0],net[2],net[4])):arrays[f'q{m}_w{i}']=layer.weight.detach().numpy().copy();arrays[f'q{m}_b{i}']=layer.bias.detach().numpy().copy()
    for k in arrays:
        if not k.startswith('q'):assert np.array_equal(arrays[k],a[k])
    np.savez_compressed(out/'paired_router.npz',**arrays);updated=AdaptivePredictor(out/'paired_router.npz')
    portable=[]
    for pair,record,x in zip(pairs,records,X):
        selected,new=updated.score(pair['feature'],pair['robot'],pair['candidates'],400-pair['step'])
        with torch.no_grad():expected=torch.stack([net(x).squeeze(-1) for net in q]).min(0).values.numpy()
        error=float(np.max(np.abs(expected-np.asarray(new['q_scores']))));assert error<1e-5;portable.append(error)
        record.update(updated_scores=new['q_scores'],updated_selection=selected)
    with torch.no_grad():retention=float((torch.stack([net(OLD).squeeze(-1) for net in q])-anchor).square().mean())
    (out/'PAIRED_LABELS.json').write_text(json.dumps(records,indent=2)+'\n')
    report=dict(status='ACTUAL_PAIRED_Q_UPDATE',pairs=6,independent_initial_groups=2,old_retention_rows=1320,
        updates=300,pi_updates=0,v_updates=0,representation_and_normalization_frozen=True,
        baseline_sha256=frozen.sha256,updated_sha256=updated.sha256,old_q_retention_mse=retention,losses=logs,
        portable_max_error=max(portable),seconds=time.perf_counter()-tick,
        input_development_pairs=records,learning_gain_proven=False,goal_complete=False,
        limitations=['Only two initial-context groups, no generalization claim','Observed default-pi continuations are single samples, not expected values',
            'Updated Q can affect the unchanged Q<V sampling heuristic; real extra inference costs must be measured'])
    (out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='input_development_pairs'}),flush=True)

if __name__=='__main__':main()
