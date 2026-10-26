"""Summarize immutable native-binding attempts; not an efficacy comparison."""
import argparse
import datetime as dt
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def timestamp(value):
    value=re.sub(r'(\.\d{6})\d+',r'\1',value)
    return dt.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()

def summarize(run):
    status=json.loads((run/'STATUS.json').read_text())
    row={'run_id':run.name,**status,'model_calls':0,'model_rejected':0,'generation_seconds':0,
         'service_call_seconds':0,'prompt_tokens':0,'completion_tokens':0,
         'hcu_allocation_seconds':0,'cpu_quota_seconds':0,'containers':[]}
    for receipt in run.glob('stream/service_calls/*/RECEIPT.json'):
        r=json.loads(receipt.read_text());row['model_calls']+=1
        row['model_rejected']+=int(r.get('status')!='OK')
        for key in ('generation_seconds','prompt_tokens','completion_tokens'):row[key]+=r.get(key,0)
        row['service_call_seconds']+=r.get('seconds',0)
    requests=list(run.glob('stream/service_calls/*/REQUEST.json'))
    row['model_requests']=len(requests)
    row['incomplete_model_requests']=sum(not (p.parent/'RECEIPT.json').exists() for p in requests)
    row['incomplete_request_note']='Interrupted requests without final receipt have unknown generation/token totals; their wall and allocation costs remain included.'
    seen=set()
    for stop in run.rglob('STOP_RECEIPT.json'):
        for c in json.loads(stop.read_text()):
            if c['container'] in seen:continue
            seen.add(c['container'])
            launch=stop.parent/('LAUNCH_'+c['role']+'.json')
            if not launch.exists():
                # stop_all may place a test container's stop receipt in stream,
                # although its launch receipt belongs to tests. Bind by exact ID.
                candidates=[p for p in run.rglob('LAUNCH_'+c['role']+'.json')
                    if json.loads(p.read_text()).get('stdout','').strip()==c['container']]
                if len(candidates)!=1:raise ValueError('Missing/ambiguous launch receipt for '+c['container'])
                launch=candidates[0]
            argv=json.loads(launch.read_text())['argv']
            seconds=timestamp(c['finished'])-timestamp(c['started'])
            assert seconds>=0 and not c['running']
            cpus=float(argv[argv.index('--cpus')+1]) if '--cpus' in argv else 0
            devices=next((x.split('=',1)[1] for x in argv if x.startswith('HIP_VISIBLE_DEVICES=')),None)
            cards=len(devices.split(',')) if devices else 0
            row['cpu_quota_seconds']+=seconds*cpus
            row['hcu_allocation_seconds']+=seconds*cards
            row['containers'].append({'role':c['role'],'id':c['container'],'seconds':seconds,
                                     'cpu_quota':cpus,'visible_hcu_cards':cards,'exit_code':c['exit_code']})
    for key,rel in [('minimal_probe','stream/PROTOCOL_AGENT_RESULT.json'),('planning_probe','stream/planning_probe/RESULT.json'),('native_run','stream/RESULT.json')]:
        if (run/rel).exists():row[key]=json.loads((run/rel).read_text())
    row['delivery_verified']=(run/'DELIVERY_MANIFEST.json').exists()
    return row

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=ROOT/args.out;out.parent.mkdir(parents=True,exist_ok=True)
    runs=sorted([*ROOT.glob('runs/native_protocol_*/STATUS.json'),*ROOT.glob('runs/native_baseline_*/STATUS.json'),*ROOT.glob('runs/native_memory_*/STATUS.json')])
    rows=[summarize(p.parent) for p in runs]
    keys=['model_requests','incomplete_model_requests','full_wall_seconds','generation_seconds','service_call_seconds','model_calls','model_rejected',
          'prompt_tokens','completion_tokens','hcu_allocation_seconds','cpu_quota_seconds']
    report={'scope':'engineering binding and failed-attempt bill; not full-Agent steady-state or A/B/C cost',
      'accounting':'Wall totals sum independent campaign intervals. Container HCU allocation and CPU quota are capacity-time, not utilization. Service/model spans overlap residency and must not be added to wall. Local editing, network transfer and human time are not measured.',
      'runs':rows,'totals':{k:sum(r.get(k,0) for r in rows) for k in keys}}
    with out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report['totals'],indent=2))
if __name__=='__main__':main()
