import sys,json,hashlib,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
from summarize_native_binding import summarize
runs=[]
for run in sorted((root/'runs').glob('instruction_trial_v*_20261002')):
 if not (run/'DELIVERY_MANIFEST.json').exists():continue
 row=summarize(run)
 calls=[json.loads(p.read_text()) for p in run.glob('*/stream/service_calls/*/RECEIPT.json')]
 row['model_calls']=len(calls);row['model_requests']=len(list(run.glob('*/stream/service_calls/*/REQUEST.json')))
 row['model_rejected']=sum(c.get('status')!='OK' for c in calls)
 row['incomplete_model_requests']=row['model_requests']-len(calls)
 for key in ['generation_seconds','prompt_tokens','completion_tokens']:row[key]=sum(c.get(key,0) for c in calls)
 row['arms']=[]
 for folder in sorted(run.glob('dev_*'))+sorted(run.glob('eval_*')):
  stream=folder/'stream';p=stream/'RESULT.json'
  if not p.exists():continue
  report=json.loads(p.read_text());case=report['cases'][0];a={'arm':folder.name,'controller_seconds':report['seconds'],'case':case}
  for rel in ['OS_COST_EVENTS.jsonl','TARGET_COST_EVENTS.jsonl']:
   f=stream/rel
   if f.exists():
    es=[json.loads(x) for x in f.read_text().splitlines()];tot=collections.defaultdict(float);n=collections.Counter()
    for e in es:
     if e.get('event')=='end':tot[e['name']]+=e['duration_seconds'];n[e['name']]+=1
    a[rel]={'spans_seconds':dict(tot),'counts':dict(n),'note':'Nested/overlapping spans; do not add them as wall time.'}
    if rel.startswith('OS'):
     prompts=collections.Counter(e.get('prompt') for e in es if e.get('event')=='actual_policy_input');a['actual_prompts']=dict(prompts)
  events=list(stream.glob('*/AGENT_EVENTS.jsonl'));a['queue_rejections']=[]
  for f in events:
   a['queue_rejections'] += [e for e in map(json.loads,f.read_text().splitlines()) if e.get('event') in ['queue_not_accepted','invalid_queue_edit_rolled_back']]
  a['initial_images']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (stream/case['case']['id']/'workspace/artifacts/runtime'/case['case']['id']/'rgb').glob('initial_*.png')}
  a['model_phases']={}
  for req in (stream/case['case']['id']).glob('*_call_*/REQUEST.json'):
   value=json.loads(req.read_text());phase='verifier' if req.parent.name.startswith('verifier') else 'planning'
   users=[m.get('content','') for m in value['messages'] if m['role']=='user']
   if phase=='planning' and users and 'Execution and its original verification have finished.' in str(users[-1]):phase='report'
   count=a['model_phases'].setdefault(phase,{'calls':0,'prompt_tokens':0,'completion_tokens':0});count['calls']+=1
   response=req.parent/'RESPONSE.json'
   if response.exists():
    usage=json.loads(response.read_text()).get('usage',{})
    for key in ['prompt_tokens','completion_tokens']:count[key]+=usage.get(key,0)
  row['arms'].append(a)
 by_arm={a['arm']:a for a in row['arms']}
 for phase in ['dev','eval']:
  if phase+'_A' in by_arm and phase+'_B' in by_arm:
   a,b=by_arm[phase+'_A'],by_arm[phase+'_B'];valid=all(x['case'].get('chain_status')=='PASSED' for x in [a,b])
   row[phase+'_comparison']={'both_chain_passed':valid,'initial_images_identical':bool(a['initial_images']) and a['initial_images']==b['initial_images'],
      'observed_controller_savings_seconds':a['controller_seconds']-b['controller_seconds'] if valid else None,
      'observed_controller_savings_fraction':1-b['controller_seconds']/a['controller_seconds'] if valid else None,
      'scope':'One paired development comparison only; incomplete chains are not speedup results.'}
 for key,file in [('policy_update','POLICY_UPDATE.json'),('learned','LEARNED.json'),('transfer','transfer/stream/RESULT.json')]:
  if (run/file).exists():row[key]=json.loads((run/file).read_text())
 runs.append(row)
keys=['full_wall_seconds','model_calls','model_requests','incomplete_model_requests','model_rejected','generation_seconds','prompt_tokens','completion_tokens','hcu_allocation_seconds','cpu_quota_seconds']
out={'scope':'This instruction-decision branch including failed attempts; not entire project or final A/B/C.', 'runs':runs,'totals':{k:sum(r.get(k,0) for r in runs) for k in keys},'accounting':'All campaign wall, failures and initialization included. HCU and CPU are allocated capacity-time. Local coding, transfer and human time are not timed. Overlapping spans not additive.'}
p=root/'planning/instruction_trial_v3/ANALYSIS.json';p.parent.mkdir(exist_ok=True)
with p.open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out['totals']))
