"""Post-first-run representation diagnostic, not a new independent efficacy test.

Same fitted head/optimizer, current field vs current-minus-initial field. All
arms of one initial state are held out together, including failed terminals.
Uses existing features only; no new neural backbone or robot inference.
"""
import argparse,json,time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(exist_ok=False);start=time.perf_counter()
    import numpy as np
    import torch
    torch.set_num_threads(4)
    features=np.load(a.run/'experiment/FEATURES.npz',allow_pickle=False)
    labels={r['id']:r for r in json.loads((a.run/'inputs/SUPERVISION.json').read_text())}
    rows=[labels[str(i)] for i in features['ids']];raw=torch.from_numpy(features['features']).float()
    groups=sorted({r['group'] for r in rows});initial={}
    for i,r in enumerate(rows):
        if r['source_step']==0:
            if r['group'] in initial:raise ValueError('Ambiguous initial memory')
            initial[r['group']]=i
    if set(initial)!=set(groups):raise ValueError('Missing initial image memory')
    delta=torch.stack([raw[i]-raw[initial[r['group']]] for i,r in enumerate(rows)])
    variants={'current':torch.nn.functional.normalize(raw,dim=-1),
              'initial_delta':torch.nn.functional.normalize(delta,dim=-1)}
    config=json.loads((a.run/'inputs/PROTOCOL.json').read_text())['fit']
    y=torch.tensor([r['native_complete'] for r in rows],dtype=torch.float32)
    predictions={};bindings={};fit_seconds={}
    for name,X in variants.items():
        predicted={};folds=[];tick=time.perf_counter()
        for group in groups:
            train=[i for i,r in enumerate(rows) if r['group']!=group];test=[i for i,r in enumerate(rows) if r['group']==group]
            assert not {rows[i]['group'] for i in train}&{rows[i]['group'] for i in test}
            torch.manual_seed(config['seed']);head=torch.nn.Linear(1024,1)
            torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
            opt=torch.optim.AdamW(head.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
            yy=y[train];xx=X[train];counts={g:sum(rows[i]['group']==g for i in train) for g in groups if g!=group}
            w=torch.tensor([1./counts[rows[i]['group']] for i in train])
            for c in (0,1):w[yy==c]/=w[yy==c].sum()
            w/=w.sum()
            for _ in range(config['iterations']):
                opt.zero_grad();loss=(torch.nn.functional.binary_cross_entropy_with_logits(head(xx)[:,0],yy,reduction='none')*w).sum()
                loss.backward();opt.step()
            with torch.no_grad():p=head(X[test])[:,0].sigmoid().tolist()
            predicted.update({rows[i]['id']:v for i,v in zip(test,p)})
            folds.append({'test_group':group,'training_groups':sorted(counts),'test_ids':[rows[i]['id'] for i in test]})
        predictions[name]=predicted;bindings[name]=folds;fit_seconds[name]=time.perf_counter()-tick
    (a.output/'PREDICTIONS.json').write_text(json.dumps(predictions,indent=2))
    (a.output/'FOLDS.json').write_text(json.dumps(bindings,indent=2))
    result={}
    for name,pred in predictions.items():
        records=[dict(id=r['id'],group=r['group'],is_terminal=r['is_terminal'],native_complete=r['native_complete'],
                      p_complete=pred[r['id']]) for r in rows]
        pos=[r['p_complete'] for r in records if r['native_complete']];neg=[r['p_complete'] for r in records if not r['native_complete']]
        result[name]={'samples':len(records),'groups':len(groups),'auc':sum((p>n)+.5*(p==n) for p in pos for n in neg)/len(pos)/len(neg),
            'binary_at_half':{'false_positive':sum(r['p_complete']>=.5 and not r['native_complete'] for r in records),
                              'true_positive':sum(r['p_complete']>=.5 and r['native_complete'] for r in records)},
            'unchanged_gate_at_099':{'proposed_complete':sum(r['p_complete']>=.99 for r in records),
                                    'false_accepts':sum(r['p_complete']>=.99 and not r['native_complete'] for r in records)},
            'failed_terminals':[{**r,'false_accept_at_099':r['p_complete']>=.99} for r in records if r['is_terminal'] and not r['native_complete']],
            'rows':records}
    report=dict(status='COMPLETED',scope='Post-hoc development representation diagnostic with group exclusion, not a new final test or selected deployment threshold.',
        metrics=result,fit_seconds=fit_seconds,seconds=time.perf_counter()-start,new_backbone_calls=0,new_robot_episodes=0,
        representation_note='initial_delta needs an initial-image encoding and stored feature at deployment; it is not free just because already paid in this replay.',
        deployment_approved=False)
    (a.output/'RESULT.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='metrics'}))
    for k,v in result.items():print(k,json.dumps({n:x for n,x in v.items() if n!='rows'}))


if __name__=='__main__':main()
