"""Run the first two-case original-OS smoke, only after services and deps are ready."""
import json
import socket
import subprocess
from pathlib import Path

root=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
run=root/'runs/pilot_20261001'
core=root.parent/'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1/dependencies/PhyAgentOS-core'
image='sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81'
assert (run/'OS_REQUIREMENTS.txt').is_file(), 'Original OS import/setup has not passed'
for port in [49500,49510,49502]:
    with socket.create_connection(('127.0.0.1',port),timeout=3):pass

for role,script,environment in [
    ('qwen-recorder','qwen_recording_proxy.py',{'FASTRSI_TRACE':'/project/runs/pilot_20261001/QWEN_COST_EVENTS.jsonl'}),
    ('baseline-smoke','baseline_os_smoke.py',{})]:
    name='yuchen-fastrsi-'+role+'-20261001-v1'
    exists=subprocess.run(['docker','inspect','--format','{{.State.Status}}',name],capture_output=True,text=True)
    if exists.returncode==0:
        raise RuntimeError(f'{name} already exists: inspect before a repeat')
    argv=['docker','run','-d','--name',name,'--network','host','--cpus','8','--memory','16g',
          '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/core:/project/scripts',
          '-v',str(root)+':/project:rw','-v',str(core)+':/core:ro']
    for key,value in environment.items():argv+=['-e',key+'='+value]
    argv += [image,'timeout','1800','bash','-lc',
             'source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python /project/scripts/'+script]
    result=subprocess.run(argv,capture_output=True,text=True,timeout=60)
    with (run/('LAUNCH_'+role+'.json')).open('x') as f:
        json.dump({'argv':argv,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr},f,indent=2)
    print(role,result.returncode,result.stdout,result.stderr,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
    if role=='qwen-recorder':
        import time
        for _ in range(40):
            try:
                with socket.create_connection(('127.0.0.1',49511),timeout=0.5):break
            except OSError:time.sleep(0.25)
        else:raise RuntimeError('Recorder failed readiness')
