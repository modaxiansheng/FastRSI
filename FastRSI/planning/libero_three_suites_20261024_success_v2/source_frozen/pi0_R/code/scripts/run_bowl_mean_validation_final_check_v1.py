"""Check actual saved mean commands in a CPU container after all trials exit."""
from pathlib import Path
import fcntl,json,subprocess,time
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
A=P.parent/'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1'
lock=(P/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.loads((P/'planning/bowl_mean_validation_v1_20261010/AGGREGATE.json').read_text())['status']=='COMPLETED'
R=P/'planning/bowl_mean_validation_report_v1_20261010';R.mkdir(exist_ok=False)
name='yuchen-fastrsi-bowl-mean-validation-command-check-v1-20261010'
cmd=['docker','run','--name',name,'--network','none','--cpus','2','--memory','4g','--pids-limit','256','--entrypoint','python',
 '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/core',
 '-v',str(P)+':/project:ro','-v',str(A/'dependencies/PhyAgentOS-core')+':/core:ro','-v',str(R)+':/check',
 'sha256:3171fda2362d773433482f79c4fd53f6f428413efbf41c3fdfb43d8c799d6c95',
 '/project/scripts/check_bowl_mean_validation_commands_v1.py']
(R/'CHECK_LAUNCH.json').write_text(json.dumps(cmd,indent=2))
tick=time.perf_counter()
with (R/'CHECK.log').open('x') as log:
 r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=100)
s=json.loads(subprocess.check_output(['docker','inspect',name]))[0]['State']
(R/'CHECK_HOST.json').write_text(json.dumps(dict(returncode=r.returncode,seconds=time.perf_counter()-tick,container=name,state=s),indent=2))
assert r.returncode==0 and not s['Running'] and s['Pid']==0
subprocess.run(['python3',str(P/'scripts/finalize_bowl_mean_validation_v1.py')],check=True)
