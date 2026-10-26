"""Bind a phase-130 observation; outcomes are a separate training sidecar."""
import argparse,hashlib,json,struct,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from fastrsi.decision_corpus import checked_image
NAME='phase_learning_v1_20261005'
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(stream):
 case=read(stream.parent/'ROSTER.json')[0];sid=case['id'];trace=stream/'stage_trace'/sid/'TRACE.jsonl'
 rows=lines(trace);phase=next(x for x in rows if x['step']==130)
 calls=lines(stream/'GOAL_SKILL_CALLS.jsonl');events=[e for e in lines(stream/'OS_COST_EVENTS.jsonl') if e.get('event')=='actual_policy_input']
 assert len(calls)==len(events)
 event=next(e for c,e in zip(calls,events) if c['step']==130 and c['runtime_session']==sid)
 assert hashlib.sha256(struct.pack('<8f',*event['robot_state'])).hexdigest()==event['array_bindings']['observation/state']['sha256']
 images=[]
 for key,cam in [('observation/image','agentview_image'),('observation/wrist_image','robot0_eye_in_hand_image')]:
  p=trace.parent/f'0130_{cam}.png';pixel=checked_image(p.read_bytes(),phase['images'][p.name],event['array_bindings'][key])
  images.append(dict(path=str(p.relative_to(R)),sha256=sha(p),pixel_sha256=pixel))
 result=read(stream/'RESULT.json');row=result['cases'][0];assert row['chain_status']=='PASSED'
 prefix=[dict(step=x['step'],actions=x.get('actions',[]),images=x['images']) for x in rows if x['step']<=130]
 outcome=dict(success=bool(row['environment_success']),vla_calls=sum(c['origin']=='vla' for c in calls),physical_steps=rows[-1]['step'],
  controller_seconds=read(stream.parent/'STATUS.json')['controller_seconds'],semantic=[o['verdict'] for o in row['outcomes']],
  result_sha256=sha(stream/'RESULT.json'),trace_sha256=sha(trace),stream=str(stream.relative_to(R)))
 return dict(goal=event['prompt'],robot_state=event['robot_state']),images,prefix,outcome
def main():
 ap=argparse.ArgumentParser();ap.add_argument('group',choices=['seed','update','heldout']);a=ap.parse_args()
 root=R/'runs'/NAME;inputs=root/'inputs';inputs.mkdir(parents=True,exist_ok=True)
 if a.group=='seed':paths=[R/'runs/goal_skill_handoff_v1_20261005/F/stream',R/'runs/source_completion_handoff_v1_20261005/F/stream']
 else:
  i=4 if a.group=='update' else 5;paths=[R/f'runs/phase_choice_v1_20261005/init{i}_h{n}/F/stream' for n in (130,262)]
 first,images,prefix,short=bind(paths[0]);second,images2,prefix2,long=bind(paths[1]);assert first==second and images==images2 or first==second and [x['sha256'] for x in images]==[x['sha256'] for x in images2]
 assert prefix==prefix2,'Mismatched observed prefix'
 state={**first,'observation_id':'current_phase_boundary','phase':'Completed first 130 source controller targets; now choose once for the remaining segment.',
  'existing_procedure':'Recorded source controller targets; current robot feedback tracks those targets. No object pose retargeting.',
  'remaining_goal':'The full original goal is unchanged. Do not assume the first bowl is complete or that source success transfers.'}
 request=dict(model='clef-flash',state=state,questions={'phase':dict(type='choice',instructions='Choose only from the two executable options using current images and state. Do not infer future success.',criteria={
  'handoff':'Resume original pi05 now in the same episode with the full goal.',
  'continue':'Continue the stored controller procedure through its source completion boundary, then resume pi05 in the same episode.'})})
 payload=dict(request=request,images=images,role=a.group)
 p=inputs/(a.group+'.json')
 with p.open('x') as f:json.dump(payload,f,indent=2)
 with (inputs/(a.group+'_labels.json')).open('x') as f:json.dump(dict(role=a.group,outcomes={'handoff':short,'continue':long},observed_prefix_equal=True,old_code_cost_excluded=a.group=='seed'),f,indent=2)
 # Atomic ready marker comes last; inference worker never sees partial inputs.
 tmp=inputs/(a.group+'.ready.tmp');tmp.write_text(json.dumps(dict(input_sha256=sha(p),label_sha256=sha(inputs/(a.group+'_labels.json')))));tmp.replace(inputs/(a.group+'.ready'))
 print(json.dumps(dict(group=a.group,observed_prefix_equal=True,success=[short['success'],long['success']])))
if __name__=='__main__':main()
