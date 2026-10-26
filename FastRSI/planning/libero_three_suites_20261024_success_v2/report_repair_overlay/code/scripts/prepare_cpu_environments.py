"""Prepare isolated OS dependencies and Laya on CPU; no shared runtime mutation."""
import json
import subprocess
from pathlib import Path

p=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
r=p/'runs/pilot_20261001'
core=str(p.parent/'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1/dependencies/PhyAgentOS-core')
image='sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81'
setup=('python -m venv --system-site-packages /project/envs/os-v1 && '
       '/project/envs/os-v1/bin/pip install loguru==0.7.3 litellm==1.82.1 '
       'pydantic-settings==2.12.0 json-repair==0.57.0 croniter==6.0.0 && '
       '/project/envs/os-v1/bin/python -c "from PhyAgentOS.runtime.watchdog.supervisor import WatchdogSupervisor; '
       'from PhyAgentOS.agent.session_verifier import SessionVerifier; '
       'from PhyAgentOS.providers.custom_provider import CustomProvider; print(\'OS_IMPORT_OK\')" && '
       '/project/envs/os-v1/bin/pip freeze > /project/runs/pilot_20261001/OS_REQUIREMENTS.txt')
jobs=[('ossetup',['-v',core+':/core:ro','-e','PYTHONPATH=/core'],['bash','-lc',setup]),
      ('laya-smoke',['-e','PYTHONPATH=/project/third_party/laya',
          '-e','HF_HOME=/project/models/huggingface','-e','HF_HUB_DISABLE_XET=1',
          '-e','HF_HUB_ETAG_TIMEOUT=15','-e','HF_HUB_DOWNLOAD_TIMEOUT=30'],
          ['python','/project/scripts/probe_laya.py'])]
for role,extra,command in jobs:
    name='yuchen-fastrsi-'+role+'-20261001-v1'
    existing=subprocess.run(['docker','inspect','--format','{{.State.Status}}',name],capture_output=True,text=True)
    if existing.returncode==0:
        print(name,'already exists:',existing.stdout.strip());continue
    argv=['docker','run','-d','--name',name,'--cpus','4','--memory','16g',
          '-e','PYTHONDONTWRITEBYTECODE=1','-v',str(p)+':/project:rw']+extra+[image,'timeout','900']+command
    result=subprocess.run(argv,capture_output=True,text=True,timeout=60)
    (r/('LAUNCH_'+role+'.json')).write_text(json.dumps({'argv':argv,'returncode':result.returncode,
        'stdout':result.stdout,'stderr':result.stderr},indent=2))
    print(role,result.returncode,result.stdout,result.stderr)
    if result.returncode:raise SystemExit(result.returncode)
