"""Trace admitted lessons into later actual requests without claiming causality."""
import argparse,hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def audit(run):
    rows=[]
    for case in load(run/'ROSTER.json'):
        folder=run/'stream'/case['id'];request=folder/'agent_call_1/REQUEST.json'
        if not request.exists():
            rows.append({'case':case['id'],'status':'NO_FIRST_REQUEST'});continue
        body=load(request)
        system='\n'.join(m['content'] for m in body['messages'] if m['role']=='system' and isinstance(m.get('content'),str))
        lessons=sorted(set(re.findall(r'lesson_[a-zA-Z0-9_]+',system)))
        events=[json.loads(s) for s in (folder/'AGENT_EVENTS.jsonl').read_text().splitlines() if s.strip()]
        queue=next((e['time'] for e in events if e['event']=='queue_accepted'),float('inf'))
        reads=[];decisions=[]
        for e in events:
            if e.get('time',float('inf'))>queue or e.get('event')!='tool':continue
            if e.get('name') in ('read_file','read_files'):
                ids=sorted(set(re.findall(r'(?:coverage_s[012]_t\d+_i\d+|native_long_t\d+_i\d+)',e.get('result','')))-{case['id']})
                if ids:reads.append({'path':e.get('params'),'prior_session_ids':ids,'time':e['time']})
            if e.get('name') in ('append_yaml_item','patch_yaml_item'):
                params=e.get('params',{});value=params.get('item',params.get('patch',{}))
                decisions.append({'time':e['time'],'tool':e['name'],'task_description':value.get('task_description'),'execution':value.get('execution'),'runtime_hints':value.get('runtime_hints')})
        terminal=folder/'RESULT.json';result=load(terminal) if terminal.exists() else {}
        lessons_path=folder/'workspace/LESSONS.md'
        final_ids=sorted(set(re.findall(r'lesson_[a-zA-Z0-9_]+',lessons_path.read_text()))) if lessons_path.exists() else []
        rows.append({'case':case['id'],'request':str(request.relative_to(ROOT)),'request_sha256':sha(request),
          'prior_lessons_in_initial_system_prompt':lessons,'explicit_history_reads_before_queue':reads,
          'actual_session_drafts':decisions,'new_lesson_ids_in_workspace':sorted(set(final_ids)-set(lessons)),
          'admission_status':result.get('experience_admission',{}).get('status','NO_ORIGINAL_RESULT'),
          'interpretation':'Actual context delivery and history reads confirmed where present; unchanged task/limits are required. No counterfactual or learned routing benefit established.'})
    return {'run':str(run.relative_to(ROOT)),'scope':'Read-only trace; no result mutation, extra model call or forced memory consolidation','cases':rows,
      'causal_decision_change':'NOT_ESTABLISHED_BY_THIS_AUDIT','warning':'LESSONS.md runtime feedback and save_memory consolidation are distinct paths; prompt inclusion does not prove utilization.'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--out',required=True);a=ap.parse_args()
    run=(ROOT/a.run).resolve();assert run.is_relative_to((ROOT/'runs').resolve())
    result=audit(run);out=ROOT/a.out
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'cases':[{'id':c['case'],'prior_lessons':c.get('prior_lessons_in_initial_system_prompt'),'admission':c.get('admission_status')} for c in result['cases']]},ensure_ascii=False))
