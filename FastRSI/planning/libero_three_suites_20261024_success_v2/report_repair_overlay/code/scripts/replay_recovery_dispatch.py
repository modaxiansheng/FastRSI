"""Twelve actual Qwen calls with no robot, tools, or persistent updates."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
from fastrsi.recovery_dispatch_probe import prepare, summarize_verdict
from replay_resident_qwen import replay_one


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    run=args.run;protocol=json.loads((run/'PROTOCOL.json').read_text())
    prompts=json.loads((run/'inputs/PROMPTS.json').read_text())
    assert len(protocol['cases'])==6 and protocol['max_model_calls']==12
    started=time.time();deadline=time.monotonic()+max(1,protocol['deadline']-time.time()-20)
    rows=[];result=dict(status='RUNNING',rows=rows,robot_episodes=0,persistent_updates=0)
    try:
        for spec in protocol['cases']:
            data=(run/'inputs'/(spec['name']+'.json')).read_bytes()
            assert hashlib.sha256(data).hexdigest()==spec['sha256']
            body=json.loads(data)
            for variant in spec['order']:
                folder=run/'stream'/(spec['name']+'_'+variant)
                row=dict(name=spec['name'],variant=variant,environment_success=spec['environment_success'],archived_verdict=spec['archived_verdict'])
                try:
                    modified=prepare(body,variant,prompts['SESSION_PROMPT'],prompts['EPISODE_PROMPT'])
                    wire=data if variant=='session' else json.dumps(modified,ensure_ascii=False).encode()
                    receipt=replay_one(dict(url='http://127.0.0.1:49510/v1/chat/completions',log_dir=str(run/'stream/service_calls')),wire,folder,180,deadline)
                    response=json.loads((folder/'HTTP_RESPONSE.json').read_text())
                    row.update(status='PASSED',seconds=receipt['http_seconds'],verdict=summarize_verdict(response),usage=response['usage'])
                except Exception:
                    row.update(status='FAILED',error=traceback.format_exc())
                rows.append(row)
                (run/'stream/RESULT.json').write_text(json.dumps(result,indent=2))
        result['status']='COMPLETED' if all(r['status']=='PASSED' for r in rows) else 'COMPLETED_WITH_ERRORS'
    finally:
        result['seconds']=time.time()-started
        (run/'stream/RESULT.json').write_text(json.dumps(result,indent=2))
    if result['status']!='COMPLETED':raise SystemExit(1)


if __name__=='__main__':main()
