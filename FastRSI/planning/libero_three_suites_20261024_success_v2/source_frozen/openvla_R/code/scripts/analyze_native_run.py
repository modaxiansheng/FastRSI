"""Read-only evidence map with distinct environment, verification and chain states."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default

def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []

def union_seconds(intervals):
    end=None;total=0
    for a,b in sorted(intervals):
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total/1e9

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--out',required=True);args=ap.parse_args()
    run=ROOT/'runs'/args.run_id;stream=run/'stream';native=lines(stream/'TARGET_COST_EVENTS.jsonl')
    events=lines(stream/'OS_COST_EVENTS.jsonl');cases=[]
    for case in read(run/'ROSTER.json',[]):
        folder=stream/case['id'];row=read(folder/'RESULT.json',{})
        state_events=[x for x in native if x.get('event')=='native_observation' and (x.get('session_id') or '').startswith(case['id'])]
        latest={x['session_id']:x for x in state_events}
        calls=sorted(folder.glob('agent_call_*/REQUEST.json'),key=lambda p:int(p.parent.name.rsplit('_',1)[-1]))
        requests=[read(p) for p in calls]
        accepted=row.get('accepted_source');source_text='';source_hash=None;memory_text='';memory_hash=None
        if accepted:
            source=ROOT/accepted.removeprefix('/project/')/'LESSONS.md'
            if source.exists():source_text=source.read_text().strip();source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
            memory_source=source.parent/'memory/MEMORY.md'
            if memory_source.exists():memory_text=memory_source.read_text().strip();memory_hash=hashlib.sha256(memory_source.read_bytes()).hexdigest()
        supplied=bool(source_text) and any(source_text in str(m.get('content','')) for r in requests for m in r.get('messages',[]))
        memory_supplied=bool(memory_text) and any(memory_text in str(m.get('content','')) for r in requests for m in r.get('messages',[]))
        planned=row.get('planned_session') or read(folder/'PLANNED_SESSION.json',{})
        prompts=sorted({str(e.get('prompt')) for e in events if e.get('event')=='actual_policy_input' and e.get('case')==case['id']})
        tools=lines(folder/'AGENT_EVENTS.jsonl')
        cases.append({'case':case,'chain_status':row.get('status','NOT_FINISHED_OR_NOT_STARTED'),
            'chain_error':row.get('error'),'environment_terminal_by_session':latest,
            'environment_success':state_events[-1]['native_success'] if state_events else None,
            'verification_outcomes':row.get('outcomes'),
            'accepted_source':accepted,'prior_lesson_sha256':source_hash,'prior_lesson_in_actual_model_request':supplied,
            'prior_memory_sha256':memory_hash,'prior_memory_in_actual_model_request':memory_supplied,
            'planned_task_description':planned.get('task_description'),
            'description_differs_from_fixed_goal':bool(planned) and planned.get('task_description')!=case['task_description'],
            'actual_vla_prompts':prompts,'agent_calls':len(calls),
            'model_updates_blocked':[e for e in tools if e.get('event')=='update_blocked'],
            'eval_fixed_hashes_equal':row.get('experience_before')==row.get('experience_after') if case['mode']=='evaluation' and row else None,
            'eval_closed_seal':(folder/'workspace/.FASTRSI_EVALUATION_CLOSED.json').exists(),
            'terminal_report':row.get('terminal_report'),
            'terminal_report_reference_rejections':[e for e in tools if e.get('event')=='report_artifact_rejected'],
            'queue_validation_failures':sum(e.get('event')=='queue_not_accepted' for e in tools)})
    stats={}
    for path in stream.glob('*COST_EVENTS.jsonl'):
        begins={};groups=defaultdict(list);errors=defaultdict(int)
        for e in lines(path):
            if e.get('event')=='begin':begins[e['span_id']]=e
            if e.get('event')=='end':
                start=begins.pop(e['span_id'],None)
                if start:
                    groups[e['name']].append((start['wall_ns'],start['wall_ns']+round(e['duration_seconds']*1e9)))
                    errors[e['name']]+=e.get('status')!='ok'
        stats[path.name]={'stages':{k:{'count':len(v),'duration_sum_seconds':sum(b-a for a,b in v)/1e9,
            'active_interval_union_seconds':union_seconds(v),'errors':errors[k]} for k,v in groups.items()},
            'unmatched_begins':len(begins)}
    result={'scope':'native binding evidence; changes are observational, not a causal ablation',
        'run_id':args.run_id,'status':read(run/'STATUS.json'),'cases':cases,'cost_spans':stats,
        'cost_warning':'Stage intervals can overlap; do not add stage durations to campaign wall time. Parent IDs are causal nesting and may outlive asynchronous parents.'}
    out=ROOT/args.out;out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'cases':[{k:r[k] for k in ['case','chain_status','environment_success','prior_lesson_in_actual_model_request','description_differs_from_fixed_goal','eval_fixed_hashes_equal']} for r in cases]},indent=2))
if __name__=='__main__':main()
