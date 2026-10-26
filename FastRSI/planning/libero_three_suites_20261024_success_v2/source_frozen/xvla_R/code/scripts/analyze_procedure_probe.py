"""Read-only consequences and complete runtime ledger of actual procedures."""
import argparse,hashlib,json
from collections import defaultdict,Counter
from pathlib import Path

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def analyze(run):
 status=read(run/'STATUS.json');assert status['status']=='COMPLETED'
 rows=[];pairs=[];sources={}
 for task in ('task2','task9'):
  branch={}
  for arm in ('N','F'):
   group=run/task/arm;stream=group/'stream';report=read(stream/'RESULT.json');assert report['status']=='COMPLETED'
   case=report['cases'][0];sid=case['case']['id'];events=lines(stream/'OS_COST_EVENTS.jsonl')
   assert case['chain_status']=='PASSED'
   inputs=[e for e in events if e['event']=='actual_policy_input']
   choices=[e for e in events if e['event']=='procedure_choice']
   teacher=[e for e in events if e['event']=='procedure_teacher_result']
   assert len(teacher)<=4 and (arm=='F' or not teacher)
   traces={s['session_id']:lines(stream/'stage_trace'/s['session_id']/'TRACE.jsonl') for s in case['sessions']}
   assert sum(len(t)-1 for t in traces.values())==len(inputs)
   spans=defaultdict(float)
   for e in events:
    if e['event']=='end':spans[e['name']]+=e['duration_seconds']
   changed=[e for e in choices if e['choice']!='full_goal']
   root=traces[sid];worker=read(group/'STATUS.json')
   r=dict(task=case['case']['task_id'],init=case['case']['init_state_id'],arm=arm,
     environment_success=case['environment_success'],semantic=[o['verdict'] for o in case['outcomes']],
     chain=case['chain_status'],admission=case['experience_admission']['status'],
     attempts=len(traces),steps=sum(t[-1]['step'] for t in traces.values()),vla_calls=len(inputs),
     case_seconds=case['seconds'],worker_seconds=worker['full_wall_seconds'],startup_seconds=worker['startup_seconds'],
     teacher_calls=len(teacher),teacher_facts=teacher,changed_policy_inputs=len(changed),
     changed_choices=dict(Counter(e['choice'] for e in changed)),first_changed_step=changed[0]['step'] if changed else None,
     qwen_service_calls=len(list((stream/'service_calls').glob('*/REQUEST.json'))),
     agent_calls=len(list((stream/sid).glob('agent_call_*/REQUEST.json'))),
     nested_stage_seconds=dict(spans))
   rows.append(r);branch[arm]=(root,inputs,r)
   for p in (group/'STATUS.json',stream/'RESULT.json',stream/'OS_COST_EVENTS.jsonl',stream/'stage_trace'/sid/'TRACE.jsonl'):
    sources[str(p.relative_to(run))]=sha(p)
  n,ni,nr=branch['N'];f,fi,fr=branch['F']
  root_equal=n[0]['images']==f[0]['images'] and ni[0]['array_bindings']==fi[0]['array_bindings']
  assert root_equal,'Initial observed state differs'
  # Check all physically executed prefix actions and images before the first actual intervention.
  first=fr['first_changed_step'];prefix=[(a,b) for a,b in zip(n,f) if first is None or a['step']<=first]
  matched=0
  for a,b in prefix:
   if a['step']==b['step'] and a['images']==b['images'] and a.get('actions')==b.get('actions'):matched+=1
   else:break
  pairs.append(dict(task=nr['task'],initial_images_and_policy_arrays_equal=True,
    matching_prefix_boundaries=matched,first_intervention_step=first,
    whole_trace_equal=n==f,case_seconds_delta_F_minus_N=fr['case_seconds']-nr['case_seconds'],
    worker_seconds_delta_F_minus_N=fr['worker_seconds']-nr['worker_seconds'],
    step_delta_F_minus_N=fr['steps']-nr['steps'],vla_delta_F_minus_N=fr['vla_calls']-nr['vla_calls'],
    pairing_scope='Same requested SDK initial state and seed plus exact observed images/proprio/action prefix; original trace has no full simulator snapshot.',
    training_label=None,training_label_status='Pending complete state/quality binding; teacher facts alone are not labels'))
 return dict(status='ANALYZED',batch_wall_seconds=status['seconds'],rows=rows,pairs=pairs,source_sha256=sources,
    scope='Paid fixed teacher procedure feasibility, not a trained decision model or unseen transfer evaluation.',
    total_cost_note='Stage spans are nested. Worker capacity times do not sum to parallel elapsed time. Teacher span includes input preparation.',
    trained=False,learning_gain_established=False)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 result=analyze(a.run)
 with a.output.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
