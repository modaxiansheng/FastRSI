"""Count observed recovery work from source-bound results, never assumed retries."""
import argparse, collections, hashlib, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    catalog=a.root/'planning/budget_history_refresh_v1_20261003/CATALOG.json'
    data=json.loads(catalog.read_text());rows=[];seen=set()
    for pair in data['records']:
        for arm in pair['arms']:
            if arm['path'] in seen:raise ValueError('Duplicate paid execution path')
            seen.add(arm['path'])
            matches=[(n,h) for n,h in arm['sources'].items() if n.endswith('/stream/RESULT.json')]
            if len(matches)!=1:raise ValueError('Unbound runtime result')
            name,digest=matches[0];raw=(a.root/name).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==digest,name
            result=json.loads(raw);assert result['status']=='COMPLETED' and len(result['cases'])==1
            case=result['cases'][0];outcomes=case['outcomes'];terminals=case.get('native_terminal_by_session',{})
            rows.append(dict(source=name,sha256=digest,suite=case['case']['suite'],task=case['case']['task_id'],
                init=case['case']['init_state_id'],window=arm['mode'],environment_success=case['environment_success'],
                observed_outcomes=len(outcomes),verdicts=[v.get('verdict') for v in outcomes],
                terminal_session_ids=list(terminals),terminal_session_count=len(terminals),
                admission=case.get('experience_admission',{}).get('status'),
                repeat_initial_condition=pair['repeat_of'] is not None))
    summary=dict(executions=len(rows),unique_task_initial_conditions=len({(r['suite'],r['task'],r['init']) for r in rows}),
        failed_executions=sum(not r['environment_success'] for r in rows),
        outcome_counts=dict(collections.Counter(r['observed_outcomes'] for r in rows)),
        terminal_session_counts=dict(collections.Counter(r['terminal_session_count'] for r in rows)),
        replan_verdicts=sum(r['verdicts'].count('replan') for r in rows))
    output=dict(status='AUDITED',catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),summary=summary,rows=rows,
        scope='Previously selected 32 paid arm executions only; repeated research runs are not automatic Agent recovery.',
        conclusion='Do not claim savings from unexecuted retry branches. Use prerequisite development task as a separate capability diagnosis, not an old-baseline speed comparison.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
