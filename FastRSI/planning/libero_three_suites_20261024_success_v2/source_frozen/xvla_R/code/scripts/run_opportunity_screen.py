"""Development-only full-OS screen using a roster fixed before any outcome."""
import asyncio
import hashlib
import json
import time
import traceback
from pathlib import Path
import run_round as runner
from fastrsi.versioning import initial_state, publish


async def main():
    root=runner.ROOT
    config=json.loads((root.parent/'PROTOCOL.json').read_text())
    roster=json.loads((root.parent/'ROSTER.json').read_text())
    expected=[(task,init) for task in config['task_ids'] for init in config['init_state_ids']]
    assert [(r['task_id'],r['init_state_id']) for r in roster]==expected
    assert len(roster)==config['max_cases'] and config['max_replans']==2
    assert all(r['task_description'] for r in roster)
    state=initial_state()
    publish(root/'INITIAL_VERSION.json',state,mode='development')
    report={'status':'RUNNING','purpose':config['purpose'],'cases':[],
            'protocol_sha256':hashlib.sha256((root.parent/'PROTOCOL.json').read_bytes()).hexdigest(),
            'roster_sha256':hashlib.sha256((root.parent/'ROSTER.json').read_bytes()).hexdigest()}
    start=time.perf_counter()
    try:
        for spec in roster:
            case_id=f't{spec["task_id"]}_i{spec["init_state_id"]}'
            report['current_case']=case_id
            (root/'RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            row,accepted,history=await runner.episode(spec['init_state_id'],'development',state,
                task_id=spec['task_id'],task_description=spec['task_description'],
                max_steps=config['max_steps_per_attempt'],case_id=case_id,suite=config['suite'])
            report['cases'].append(row)
            state['lessons'].extend(accepted)
            state['history_md']=history
            state['version']+=1
            state['update_receipts'].append({'case_id':case_id,'accepted_result_lessons':len(accepted),
                'scope':'Observed successful task result; no validated reusable action or routing policy'})
            publish(root/f'VERSION_{state["version"]}.json',state,mode='development')
            print(json.dumps({'case_id':case_id,'environment_success':row['environment_success'],
                  'attempts':len(row['outcomes']),'seconds':row['seconds']}),flush=True)
        report['status']='COMPLETED'
    except BaseException:
        report.update(status='FAILED',error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-start
        (root/'RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))


if __name__=='__main__':asyncio.run(main())
