"""One frozen visual encoder load; two real observed-outcome head versions.

Predeclared grouped development diagnostic, not a deployed controller. Labels
are read after features, and heldout labels only after predictions are saved.
"""
import argparse,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
 output=a.run/'learning';output.mkdir(exist_ok=False);start=time.perf_counter();report=dict(status='LOADING',calls=[],deployment=False)
 def save():
  report['seconds']=time.perf_counter()-start;(output/'RESULT.json').write_text(json.dumps(report,indent=2))
 save()
 try:
  import numpy as np
  import torch
  from PIL import Image
  torch.set_num_threads(4);torch.manual_seed(17)
  m=json.loads((a.model/'DOWNLOAD.json').read_text());assert m['status']=='VERIFIED'
  for name,row in m['files'].items():assert (a.model/name).stat().st_size==row['bytes'] and sha(a.model/name)==row['sha256'],name
  spec=importlib.util.spec_from_file_location('phase_clef',a.model/'joint_schema_model.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  tick=time.perf_counter();model,processor=module.load_release_model(a.model,device='cuda',attn_implementation='eager',local_files_only=True);torch.cuda.synchronize();report['load_seconds']=time.perf_counter()-tick
  for p in model.parameters():p.requires_grad_(False)
  capture={}
  hook=model.head.field_norm.register_forward_hook(lambda m,i,o:capture.update(value=o.detach().float().cpu()))
  features=[];zero={};bindings={};report['status']='WAITING_FOR_INPUTS';save()
  for role in ('seed','update','heldout'):
   while not (a.run/'inputs'/f'{role}.ready').exists():
    if time.perf_counter()-start>1300:raise TimeoutError('Input deadline')
    time.sleep(2)
   mark=json.loads((a.run/'inputs'/f'{role}.ready').read_text());path=a.run/'inputs'/f'{role}.json';assert sha(path)==mark['input_sha256'];row=json.loads(path.read_text());assert row['role']==role
   request=row['request'];assert set(request['state'])=={'goal','robot_state','observation_id','phase','existing_procedure','remaining_goal'}
   assert set(request['questions'])=={'phase'} and list(request['questions']['phase']['criteria'])==['handoff','continue']
   images=[]
   for item in row['images']:
    p=Path('/project')/item['path'];assert sha(p)==item['sha256']
    with Image.open(p) as im:
     assert im.mode=='RGB' and hashlib.sha256(im.tobytes()).hexdigest()==item['pixel_sha256'];images.append(im.copy())
   tick=time.perf_counter();request={**request,'images':images};encoded=module.encode_record(processor.tokenizer,request,max_length=1000000,processor=processor);assert len(encoded.input_ids)<=8192
   batch=module.collate_records([encoded],processor.tokenizer.pad_token_id,'cuda');capture.clear()
   with torch.inference_mode():logits=model(batch)[0][0]
   torch.cuda.synchronize();field=capture['value'];assert field.shape==(1,1024) and torch.isfinite(field).all()
   features.append(field[0].numpy());zero[role]=dict(zip(encoded.questions[0].option_ids,logits.float().softmax(-1).cpu().tolist()))
   bindings[role]=mark;report['calls'].append(dict(role=role,seconds=time.perf_counter()-tick,tokens=len(encoded.input_ids),zero_shot=zero[role]));save()
  hook.remove();np.savez(output/'FEATURES.npz',features=np.stack(features),roles=np.array(['seed','update','heldout']))
  vector=torch.nn.functional.normalize(torch.tensor(np.stack(features)),dim=-1)
  labels={}
  for role in ('seed','update'):
   p=a.run/'inputs'/f'{role}_labels.json';assert sha(p)==bindings[role]['label_sha256'];labels[role]=json.loads(p.read_text())
  fit=json.loads((a.run/'REGISTRATION.json').read_text())['fit'];versions={};train_seconds={}
  for version,roles in [('v1_seed',['seed']),('v2_updated',['seed','update'])]:
   xx=vector[:len(roles)];yy=torch.tensor([[float(labels[r]['outcomes'][action]['success']) for action in ('handoff','continue')] for r in roles])
   torch.manual_seed(fit['seed']);head=torch.nn.Linear(1024,2);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
   before={k:v.clone() for k,v in head.state_dict().items()};opt=torch.optim.AdamW(head.parameters(),lr=fit['learning_rate'],weight_decay=fit['weight_decay']);tick=time.perf_counter()
   for _ in range(fit['iterations']):
    opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(head(xx),yy);loss.backward();opt.step()
   train_seconds[version]=time.perf_counter()-tick;assert any(not torch.equal(before[k],v) for k,v in head.state_dict().items())
   checkpoint=output/(version+'.pt');torch.save(head.state_dict(),checkpoint)
   with torch.no_grad():prob=head(vector).sigmoid().tolist()
   versions[version]=dict(parameters=sum(p.numel() for p in head.parameters()),train_roles=roles,checkpoint_sha256=sha(checkpoint),final_train_loss=float(loss.detach()),predictions=prob,
    heldout_action=('continue' if prob[2][1]>prob[2][0] else 'handoff'))
  predictions=dict(versions=versions,zero_shot=zero,selection='Argmax estimated observed-success; exact tie handoff. Uncalibrated and not authorized for execution.',features_sha256=sha(output/'FEATURES.npz'))
  (output/'PREDICTIONS.json').write_text(json.dumps(predictions,indent=2))
  # No heldout supervision has been read before this persisted checkpoint/prediction.
  p=a.run/'inputs/heldout_labels.json';assert sha(p)==bindings['heldout']['label_sha256'];heldout=json.loads(p.read_text());outcomes=heldout['outcomes']
  selected={k:v['heldout_action'] for k,v in versions.items()};selected.update(fixed_continue='continue',fixed_handoff='handoff',upstream=max(zero['heldout'],key=zero['heldout'].get))
  scores={name:dict(action=action,observed_branch=outcomes[action]) for name,action in selected.items()}
  report.update(status='COMPLETED',versions=versions,fit_seconds=train_seconds,heldout_scores=scores,train_contexts=2,heldout_contexts=1,
   labels='Native development outcomes, conflicting semantic text never used as targets.',new_live_learner_calls=0,
   scope='Actual visual encoder and external 2050-parameter success-head training. Heldout scoring refers to executed branch data, not an online learned-policy rollout. No net gain established.')
 except BaseException:report.update(status='FAILED',error=traceback.format_exc());raise
 finally:save()
if __name__=='__main__':main()
