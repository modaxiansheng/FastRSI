"""Read-only per-case cost and actual control audit for a native long-task run."""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter,defaultdict
from summarize_native_binding import summarize
ROOT=Path(__file__).resolve().parents[1]
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--out',required=True);a=ap.parse_args()
    run=ROOT/'runs'/a.run_id;stream=run/'stream';roster=read(run/'ROSTER.json');events=lines(stream/'OS_COST_EVENTS.jsonl')
    starts={e['span_id']:e for e in events if e.get('event')=='begin'}
    def case_of(e):
        seen=set()
        while e:
            if 'case' in e:return e['case']
            parent=e.get('parent_id')
            if not parent or parent in seen:return None
            seen.add(parent);e=starts.get(parent)
    stages=defaultdict(lambda:defaultdict(list))
    for e in events:
        if e.get('event')=='end':stages[case_of(e)][e['name']].append(e['duration_seconds'])
    target=lines(stream/'TARGET_COST_EVENTS.jsonl');cases=[]
    def message_key(d):return hashlib.sha256(json.dumps(d['messages'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    service_index=defaultdict(list)
    for p in stream.glob('service_calls/*/REQUEST.json'):
        service_index[message_key(read(p))].append(p.parent)

    for c in roster:
        folder=stream/c['id'];r=read(folder/'RESULT.json',{});tools=lines(folder/'AGENT_EVENTS.jsonl')
        responses=[read(p) for p in folder.glob('agent_call_*/RESPONSE.json')]
        requests=list(folder.glob('agent_call_*/REQUEST.json'))
        observations=[x for x in target if x.get('event')=='native_observation' and str(x.get('session_id','')).startswith(c['id'])]
        increments=[];previous={}
        for e in observations:
            sid=e['session_id']
            if e['method']=='action_chunk':increments.append(e['steps']-previous.get(sid,0))
            previous[sid]=e['steps']
        model_phases=defaultdict(lambda:{'requests':0,'receipts':0,'generation_seconds':0,'prompt_tokens':0,'completion_tokens':0,'rejected':0})
        for p in folder.glob('*_call_*/REQUEST.json'):
            request=read(p);role='verifier' if p.parent.name.startswith('verifier') else 'planning'
            users=[x.get('content','') for x in request['messages'] if x.get('role')=='user']
            if role=='planning' and users and 'Execution and its original verification have finished.' in str(users[-1]):role='terminal_report'
            stat=model_phases[role];stat['requests']+=1
            matched=service_index[message_key(request)]
            assert len(matched)==1, 'Ambiguous or missing raw request binding: '+str(p)
            receipt=read(matched[0]/'RECEIPT.json')
            if receipt:
                stat['receipts']+=1;stat['rejected']+=int(receipt.get('status')!='OK')
                for key in ['generation_seconds','prompt_tokens','completion_tokens']:stat[key]+=receipt.get(key,0)
        case_stages={k:{'count':len(v),'seconds_sum':sum(v)} for k,v in stages[c['id']].items()}
        cases.append({'id':c['id'],'chain':r.get('chain_status','NOT_STARTED_OR_NOT_FINISHED'),
            'environment_success':r.get('environment_success'),'seconds':r.get('seconds'),
            'verification_status':r.get('verification_status'),'attempts':len(r.get('outcomes',[])),
            'agent_requests':len(requests),'agent_responses':len(responses),
            'tool_counts':dict(Counter(x.get('name') for x in tools if x.get('event')=='tool')),
            'queue_accepted':sum(x.get('event')=='queue_accepted' for x in tools),
            'queue_rejected':sum(x.get('event')=='queue_not_accepted' for x in tools),
            'planning_reply':r.get('agent_response'),'terminal_report':r.get('terminal_report'),
            'goal_changed':bool(r.get('planned_session')) and r['planned_session']['task_description']!=c['task_description'],
            'stage_intervals':case_stages,'model_phases':dict(model_phases),'actual_environment_steps_per_chunk':dict(Counter(increments)),
            'last_environment_observation':observations[-1] if observations else None,
            'accepted_source':r.get('accepted_source'),'error':r.get('error')})
    report={'run_id':a.run_id,'scope':'Known development instances under native Agent; not a causal comparison or learning benefit',
        'cases':cases,'bill':summarize(run),
        'warnings':['Stage intervals overlap and nest; do not add them to wall time.','Model replacement and public decoder/queue fixes are controlled common infrastructure.','No forced memory consolidation. No new Laya call or learning selector.']}
    p=ROOT/a.out;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({'cases':[{k:v[k] for k in ['id','chain','environment_success','seconds','agent_requests','goal_changed','actual_environment_steps_per_chunk']} for v in cases],
        'bill':{k:report['bill'][k] for k in ['full_wall_seconds','model_requests','model_rejected','generation_seconds','hcu_allocation_seconds']}},indent=2))
if __name__=='__main__':main()
