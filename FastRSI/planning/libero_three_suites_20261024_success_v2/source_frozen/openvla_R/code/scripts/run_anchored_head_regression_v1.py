"""CPU composition then bounded matched native Agent waves; no policy selection."""
import json
import os
from pathlib import Path
import subprocess
import time
import orchestrate_anchored_head_agent_v1 as mod
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
B=P/'planning/bowl_anchored_head_regression_v1_20261009'

def main():
    rows=json.loads((B/'FROZEN_ROSTER.json').read_text())
    assert len(rows)==4
    for row in rows:
        c=Path(row['campaign']);assert mod.native.sha(c/'MANIFEST.json')==row['manifest_sha256']
        assert not (c/'CPU_LAUNCH.json').exists() and not (c/'CONTROLLER_DISPATCH.json').exists()
    for row in rows:
        c=Path(row['campaign']);argv=mod.launch(P,c,'cpu');r=subprocess.run(argv,capture_output=True,text=True,timeout=45)
        mod.native.save(c/'CPU_LAUNCH.json',dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,time=time.time()))
        assert r.returncode==0,r.stderr;cid=r.stdout.strip()
        p=subprocess.run(['docker','wait',cid],capture_output=True,text=True,timeout=200)
        info=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
        mod.native.save(c/'CPU_CONTAINER.json',info)
        assert not info['State']['Running'] and info['State']['ExitCode']==0,(c,info['State'])
        check=json.loads((c/'preflight/CHECK_RESULT.json').read_text());assert check['status']=='PASSED' and check['manifest_sha256']==row['manifest_sha256']
        print('CPU_PASSED',row['seed'],row['arm'],flush=True)
    started=time.perf_counter();results=[]
    for wave in range(1):
        jobs=[]
        for row in [r for r in rows if r['wave']==wave]:
            c=Path(row['campaign']);argv=['python3',str(P/'scripts/orchestrate_anchored_head_agent_v1.py'),'--execute','--project',str(P),'--campaign',str(c),'--manifest-sha256',row['manifest_sha256'],'--allow-real-physical-calls']
            with (c/'CONTROLLER.log').open('xb') as out:r=subprocess.Popen(argv,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
            mod.native.save(c/'CONTROLLER_DISPATCH.json',dict(argv=argv,pid=r.pid,started_at=time.time(),manifest_sha256=row['manifest_sha256']))
            jobs.append((row,r));print('DISPATCH',wave,row['seed'],row['arm'],r.pid,flush=True)
        failures=[]
        for row,r in jobs:
            try:r.wait(timeout=1900)
            except subprocess.TimeoutExpired:
                # Ask only this owning controller to clean up its containers.
                r.terminate();r.wait(timeout=90)
            c=Path(row['campaign']);d=json.loads((c/'STATUS.json').read_text());results.append(dict(row,exit_code=r.returncode,status=d))
            if d['status']!='COMPLETED':failures.append(str(c))
            print('FINISH',wave,row['seed'],row['arm'],d['status'],d.get('environment_success'),flush=True)
        mod.native.save(B/'EXECUTION_STATUS.json',dict(results=results,experiment_elapsed_seconds=time.perf_counter()-started,failed_protocols=failures))
        if failures:raise RuntimeError('Protocol failure retained; remaining waves paused: '+str(failures))
if __name__=='__main__':main()
