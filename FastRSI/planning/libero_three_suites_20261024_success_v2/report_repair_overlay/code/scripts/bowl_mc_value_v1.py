"""Small action-conditioned Monte Carlo outcome baseline, not Cal-QL or optimal Q.

Portable NumPy inference; frozen base policy. Partial terminal prefixes excluded
from fitting so their post-execution lengths cannot become decision inputs.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
PROMPT='stack the right bowl on the left bowl and place them in the tray'

def features(observation,actions,budget):
    assert observation['prompt']==PROMPT and 0<budget<=400
    values=[]
    for key in ('observation/image','observation/wrist_image'):
        image=np.asarray(observation[key]);assert image.shape==(256,256,3) and image.dtype==np.uint8
        pooled=image.reshape(16,16,16,16,3).mean(axis=(1,3)).astype(np.float32)/255.
        values.append(pooled.ravel())
    state=np.asarray(observation['observation/state'],dtype=np.float32);assert state.shape==(8,)
    actions=np.asarray(actions,dtype=np.float32);assert actions.shape==(10,7)
    result=np.concatenate(values+[state,np.array([budget/400.],dtype=np.float32),actions.ravel()])
    assert result.shape==(1615,) and np.isfinite(result).all()
    return result

class Predictor:
    def __init__(self,path):
        self.path=Path(path);self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.arrays=dict(np.load(self.path,allow_pickle=False))
    def score(self,observation,candidates,budget):
        tick=time.perf_counter();x=np.stack([features(observation,a,budget) for a in candidates])
        z=(x-self.arrays['mean'])/self.arrays['std'];outputs=[]
        for m in (0,1):
            y=z
            for i in (0,1):y=np.tanh(y@self.arrays['m%d_w%d'%(m,i)].T+self.arrays['m%d_b%d'%(m,i)])
            y=y@self.arrays['m%d_w2'%m].T+self.arrays['m%d_b2'%m]
            probability=1./(1.+np.exp(-np.clip(y[:,0],-40,40)))
            outputs.append(np.stack([probability,np.clip(y[:,1],0.,1.)],axis=-1))
        output=np.stack(outputs);probability=np.min(output[:,:,0],axis=0);cost=np.mean(output[:,:,1],axis=0)
        selected=max(range(len(candidates)),key=lambda j:(float(probability[j]),-float(cost[j])))
        return selected,dict(success_scores=probability.tolist(),conditional_remaining_action_scores=cost.tolist(),
            ensemble_scores=output.tolist(),selection_seconds=time.perf_counter()-tick,
            interpretation='Observed behavior outcome/cost prediction; scores not calibrated safety probabilities or optimal Q')

def train(project,output):
    import torch
    from torch import nn
    from bowl_value_data_v1 import read_transition
    from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
    torch.set_num_threads(4);started=time.perf_counter();output.mkdir(exist_ok=False)
    train_rows=[];validation_rows=[];excluded=[];evidence=[]
    old=project/'planning/bowl_value_records_v2_20261009/TRANSITIONS.jsonl'
    for row in [json.loads(s) for s in old.read_text().splitlines()]:
        value=read_transition(project,row)
        if len(value['executed_actions'])!=10:
            excluded.append(dict(episode=row['episode_id'],chunk=row['chunk'],reason='Partial post-execution prefix; not an online input'));continue
        item=(features(value['observation'],value['executed_actions'],value['remaining_execution_budget']),
            float(value['labels']['observed_episode_return']),value['labels']['observed_remaining_actions']/400.)
        (validation_rows if row['initial_state_group']=='demo_25' else train_rows).append(item)
        evidence.append(dict(episode=row['episode_id'],chunk=row['chunk'],role='validation' if row['initial_state_group']=='demo_25' else 'train'))
    parent=project/'planning/bowl_candidates_v1_20261009/AGGREGATE.json'
    agg=json.loads(parent.read_text());assert agg['status']=='COMPLETED'
    for slot in (0,1):
        root=project/'runs'/('bowl_candidates_gpu%d_v1_20261009'%slot)
        for case in json.loads((root/'AGGREGATE.json').read_text())['cases']:
            folder=root/case['case'];result=json.loads((folder/'RESULT.json').read_text())
            assert result['status']=='COMPLETED'
            for row in result['rows']:
                index=row['chunk'];request=(folder/('%03d_request.msgpack'%index)).read_bytes();response=(folder/('%03d_response.msgpack'%index)).read_bytes()
                data=unpackb(request);proposed=np.asarray(unpackb(response)['candidates'])[row['selected_index']]
                n=row['step_after']-row['step_before']
                if n!=10:
                    excluded.append(dict(episode=case['case'],chunk=index,reason='Partial terminal prefix'));continue
                obs={k:data[k] for k in ('observation/image','observation/wrist_image','observation/state','prompt')}
                train_rows.append((features(obs,proposed,400-row['step_before']),float(result['native_success']),
                                   (result['executed_actions']-row['step_before'])/400.))
                evidence.append(dict(episode=case['case'],chunk=index,role='train',request_sha256=hashlib.sha256(request).hexdigest(),
                    response_sha256=hashlib.sha256(response).hexdigest(),selected_index=row['selected_index']))
    x=np.stack([r[0] for r in train_rows]);targets=np.array([r[1:] for r in train_rows],np.float32)
    mean=x.mean(axis=0);std=np.maximum(x.std(axis=0),.05);z=(x-mean)/std
    tensors=torch.from_numpy(z);y=torch.from_numpy(targets)
    vx=torch.from_numpy((np.stack([r[0] for r in validation_rows])-mean)/std)
    vy=torch.tensor([r[1:] for r in validation_rows],dtype=torch.float32)
    arrays={'mean':mean,'std':std};logs=[]
    for m,seed in enumerate((7,19)):
        torch.manual_seed(seed)
        model=nn.Sequential(nn.Linear(1615,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,2))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01)
        for step in range(300):
            indices=torch.randint(len(tensors),(64,));out=model(tensors[indices]);target=y[indices]
            classification=nn.functional.binary_cross_entropy_with_logits(out[:,0],target[:,0])
            mask=target[:,0]>0
            cost=nn.functional.mse_loss(out[mask,1],target[mask,1]) if mask.any() else out[:,1].sum()*0.
            loss=classification+.1*cost
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if step in (0,99,199,299):logs.append(dict(seed=seed,step=step+1,loss=float(loss.detach())))
        with torch.no_grad():
            pred=model(vx);bce=nn.functional.binary_cross_entropy_with_logits(pred[:,0],vy[:,0]);accuracy=((pred[:,0]>0)==(vy[:,0]>.5)).float().mean()
        logs.append(dict(seed=seed,diagnostic_validation_bce=float(bce),diagnostic_validation_accuracy=float(accuracy),
                         validation_scope='Exposed source demo25; not new-task generalization or action ranking'))
        for i,layer in enumerate((model[0],model[2],model[4])):
            arrays['m%d_w%d'%(m,i)]=layer.weight.detach().numpy().copy();arrays['m%d_b%d'%(m,i)]=layer.bias.detach().numpy().copy()
    checkpoint=output/'value_model.npz';np.savez_compressed(checkpoint,**arrays)
    predictor=Predictor(checkpoint)
    result=dict(status='TRAINED_MC_OUTCOME_BASELINE_NOT_OPTIMAL_Q',training_rows=len(train_rows),diagnostic_validation_rows=len(validation_rows),
        training_success_rows=int(targets[:,0].sum()),training_steps_per_member=300,ensemble_members=2,
        parameters_per_member=1615*128+128+128*64+64+64*2+2,
        native_policy_parameter_updates=0,heldout_generalization=False,partial_prefixes_excluded=excluded,
        value_model_sha256=predictor.sha256,seconds=time.perf_counter()-started,logs=logs,
        data_scope='Source demo0/7 plus completed normal init41-44, both arms; demo25 excluded from fitting',
        objective='MC observed episode outcome, conditional successful remaining actions; not Bellman/CQL/IQL training',
        quality_before_cost=True,default_promoted=False,learning_gain_proven=False)
    (output/'TRAINING_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'ROW_PROVENANCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    train(a.project,a.output)
