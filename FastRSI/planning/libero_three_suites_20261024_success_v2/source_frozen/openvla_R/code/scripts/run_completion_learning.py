"""Frozen Clef visual features + an external supervised decision head.

Inference inputs exclude supervision. No native Agent or robot is controlled.
Outputs are saved before held-out scoring; no threshold/hyperparameter search.
"""
import argparse,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.completion_decision import encode,route


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,default=Path('/project'))
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(exist_ok=False);started=time.perf_counter()
    report=dict(status='LOADING',new_robot_episodes=0,controller_changed=False,calls=[])
    def save():
        report['seconds']=time.perf_counter()-started
        (args.output/'RESULT.json').write_text(json.dumps(report,indent=2))
    save()
    try:
        import numpy as np
        import torch
        from PIL import Image
        torch.set_num_threads(4);torch.manual_seed(17)
        manifest=json.loads((args.model/'DOWNLOAD.json').read_text())
        if manifest['status']!='VERIFIED':raise ValueError('Unverified model')
        for name,row in manifest['files'].items():
            p=args.model/name
            with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
            if p.stat().st_size!=row['bytes'] or h!=row['sha256']:raise ValueError('Model changed '+name)
        protocol=json.loads((args.inputs/'PROTOCOL.json').read_text())
        if manifest['revision']!=protocol['model_revision']:raise ValueError('Wrong revision')
        for path,sha in json.loads((args.inputs/'SOURCE_MANIFEST.json').read_text()).items():
            if hashlib.sha256((args.project/path).read_bytes()).hexdigest()!=sha:raise ValueError('Source changed '+path)
        spec=importlib.util.spec_from_file_location('completion_clef_source',args.model/'joint_schema_model.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        load_start=time.perf_counter()
        model,processor=module.load_release_model(args.model,device='cuda',attn_implementation='eager',local_files_only=True)
        torch.cuda.synchronize();report['load_seconds']=time.perf_counter()-load_start
        for p in model.parameters():p.requires_grad_(False)
        capture={}
        def hook(_,__ ,output):capture['field']=output.detach().float().cpu()
        handle=model.head.field_norm.register_forward_hook(hook)
        rows=json.loads((args.inputs/'INPUTS.json').read_text());features=[];probabilities=[]
        report['status']='EXTRACTING';save()
        for row in rows:
            begin=time.perf_counter();images=[]
            for ref in row['images']:
                p=args.project/ref['path'];raw=p.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('Image changed')
                with Image.open(p) as im:
                    if im.mode!='RGB' or hashlib.sha256(im.tobytes()).hexdigest()!=ref['pixel_sha256']:raise ValueError('Pixel mismatch')
                    images.append(im.copy())
            record={**row['request'],'images':images};encoded=encode(module.encode_record,processor,record)
            encoded_at=time.perf_counter()
            batch=module.collate_records([encoded],processor.tokenizer.pad_token_id,'cuda')
            capture.clear()
            with torch.inference_mode():logits=model(batch)[0][0]
            torch.cuda.synchronize();end=time.perf_counter()
            field=capture['field']
            if field.shape!=(1,1024) or not torch.isfinite(field).all():raise ValueError('Unexpected feature')
            values=dict(zip(encoded.questions[0].option_ids,logits.float().softmax(-1).cpu().tolist()))
            features.append(field[0].numpy());probabilities.append(values)
            report['calls'].append(dict(id=row['id'],seconds=end-begin,encode_seconds=encoded_at-begin,
                                       inference_seconds=end-encoded_at,input_tokens=len(encoded.input_ids),probabilities=values))
            save()
        handle.remove()
        X=np.stack(features);np.savez(args.output/'FEATURES.npz',features=X,ids=np.asarray([r['id'] for r in rows]))
        report['feature_sha256']=hashlib.sha256((args.output/'FEATURES.npz').read_bytes()).hexdigest()
        # CPU training updates only a 1024->1 external head; no backprop to Clef.
        labels=json.loads((args.inputs/'SUPERVISION.json').read_text());by_id={r['id']:r for r in labels}
        if len(by_id)!=len(rows):raise ValueError('Label mapping mismatch')
        labels=[by_id[r['id']] for r in rows]
        vector=torch.from_numpy(X).float();vector=torch.nn.functional.normalize(vector,dim=-1)
        y=torch.tensor([r['native_complete'] for r in labels],dtype=torch.float32)
        trained={};fit_times={};train_bindings={}
        for name,roles in [('frozen',{'seed'}),('updated',{'seed','update'})]:
            indices=[i for i,r in enumerate(labels) if r['role'] in roles]
            if any(labels[i]['role']=='test' for i in indices):raise ValueError('Test entered fitting')
            torch.manual_seed(protocol['fit']['seed']);head=torch.nn.Linear(1024,1)
            torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
            opt=torch.optim.AdamW(head.parameters(),lr=protocol['fit']['learning_rate'],weight_decay=protocol['fit']['weight_decay'])
            xx=vector[indices];yy=y[indices]
            counts={g:sum(labels[i]['group']==g for i in indices) for g in {labels[i]['group'] for i in indices}}
            w=torch.tensor([1./counts[labels[i]['group']] for i in indices]);
            for c in (0,1):w[yy==c]/=w[yy==c].sum()
            w/=w.sum();fit_start=time.perf_counter()
            for _ in range(protocol['fit']['iterations']):
                opt.zero_grad();loss=(torch.nn.functional.binary_cross_entropy_with_logits(head(xx)[:,0],yy,reduction='none')*w).sum()
                loss.backward();opt.step()
            fit_times[name]=time.perf_counter()-fit_start
            head.eval();torch.save(head.state_dict(),args.output/(name+'_head.pt'))
            with torch.no_grad():trained[name]=head(vector)[:,0].sigmoid().tolist()
            train_bindings[name]=dict(ids=[rows[i]['id'] for i in indices],groups=sorted(counts),
                roles=sorted(roles),parameters=sum(p.numel() for p in head.parameters()),
                checkpoint_sha256=hashlib.sha256((args.output/(name+'_head.pt')).read_bytes()).hexdigest())
        (args.output/'TRAINING_BINDINGS.json').write_text(json.dumps(train_bindings,indent=2))
        predictions=[]
        train_indices=[i for i,r in enumerate(labels) if r['role']!='test']
        for i,row in enumerate(rows):
            if labels[i]['role']!='test':continue
            candidates=[j for j in train_indices if labels[j]['goal']==labels[i]['goal']]
            # Same-feature, task-conditioned nearest-memory strong baseline.
            closest=max(candidates,key=lambda j:float(vector[i]@vector[j]))
            estimates={'upstream':probabilities[i]['complete'],**{k:v[i] for k,v in trained.items()},
                       'nearest_memory':float(y[closest]),'always_complete':1.}
            predictions.append(dict(id=row['id'],estimates=estimates,
                decisions={k:route(float(p),protocol['threshold']) for k,p in estimates.items()},
                nearest_source_id=rows[closest]['id']))
        (args.output/'PREDICTIONS.json').write_text(json.dumps(predictions,indent=2))
        # Score only after model versions/predictions have been persisted.
        metrics={}
        for name in predictions[0]['estimates']:
            scored=[]
            for pred in predictions:
                truth=by_id[pred['id']];accepted=pred['decisions'][name]=='propose_complete'
                scored.append(dict(id=pred['id'],group=truth['group'],native_complete=truth['native_complete'],
                    is_terminal=truth['is_terminal'],accepted=accepted,false_accept=accepted and not truth['native_complete']))
            metrics[name]=dict(samples=len(scored),proposed_complete=sum(r['accepted'] for r in scored),
                false_accepts=sum(r['false_accept'] for r in scored),
                missed_complete=sum(r['native_complete'] and not r['accepted'] for r in scored),
                terminal_samples=sum(r['is_terminal'] for r in scored),
                terminal_failures=sum(r['is_terminal'] and not r['native_complete'] for r in scored),
                terminal_proposed_complete=sum(r['is_terminal'] and r['accepted'] for r in scored),rows=scored)
        report.update(status='COMPLETED',metrics=metrics,fit_seconds=fit_times,training_bindings=train_bindings,
            test_initial_groups=len({r['group'] for r in labels if r['role']=='test'}),
            deployment_approved=False,learning_net_savings_established=False,
            scope='Real frozen-model image inference and external head fitting. Retrospective grouped development quality, not live OS cost or final evaluation.')
    except BaseException:
        report.update(status='FAILED',error=traceback.format_exc());raise
    finally:save()


if __name__=='__main__':main()
