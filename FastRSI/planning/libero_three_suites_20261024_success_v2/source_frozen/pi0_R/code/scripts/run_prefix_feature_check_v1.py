"""Owned saved-request feature audit after the candidate comparison exits."""
from pathlib import Path
import fcntl,hashlib,json,subprocess,time,shutil,re
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
A=P.parent/'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1'
lock=(P/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
R=P/'runs/prefix_feature_check_v1_20261010';R.mkdir(exist_ok=False)
def save(name,r):(R/name).write_text(json.dumps(r,indent=2)+'\n')
def cap(c):
 r=subprocess.run(c,capture_output=True,text=True,timeout=40);return dict(argv=c,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pre=dict(hcu=cap(['bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && hy-smi --showuse --showmemuse']),containers=cap(['docker','ps','--format','{{.Names}}']),holders=cap(['bash','-lc','fuser -v /dev/kfd /dev/mkfd /dev/dri/render*']))
save('PREFLIGHT.json',pre)
# Parse the actual management-tool spelling independently of the display padding.
mem=re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)',pre['hcu']['stdout'])
assert len(mem)==8 and all(float(v)==0 for i,v in mem)
assert pre['containers']['returncode']==0 and not pre['containers']['stdout'].strip()
identities=[]
for pid in {int(x) for x in pre['holders']['stdout'].split()}:
 comm=(Path('/proc')/str(pid)/'comm').read_text().strip();exe=str((Path('/proc')/str(pid)/'exe').resolve(strict=True))
 identities.append(dict(pid=pid,comm=comm,exe=exe));assert comm=='hymgr' and exe=='/usr/local/hyhal/bin/hymgr'
save('HOLDER_IDENTITIES.json',identities)
assert json.loads((P/'planning/bowl_mean_validation_v1_20261010/AGGREGATE.json').read_text())['status']=='COMPLETED'
before=sha(P/'CURRENT_NATIVE_ENTRY.json')
(R/'code').mkdir()
for n in ['prefix_feature_capture_v1.py','check_prefix_features_v1.py','run_prefix_feature_check_v1.py']:shutil.copy2(P/'scripts'/n,R/'code'/n)
shutil.copytree(P/'models/openpi_data',R/'tokenizer_cache')
(R/'frozen_sources').mkdir()
for n in ['pi0_pytorch.py','gemma_pytorch.py']:shutil.copy2(A/'dependencies/openpi/src/openpi/models_pytorch'/n,R/'frozen_sources'/n)
save('PROTOCOL.json',dict(robot_executions=0,training_steps=0,native_inferences=18,sampling_steps=10,default_sha256=before,device=0,code_sha256={f.name:sha(f) for f in (R/'code').iterdir()}))
name='yuchen-fastrsi-prefix-feature-check-v1-20261010'
cmd=['docker','run','--name',name,'--network','none','--cpus','8','--memory','64g','--pids-limit','1024','--shm-size','2g',
 '--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri','--group-add','video','--group-add','render',
 '-e','HIP_VISIBLE_DEVICES=0','-e','CUDA_VISIBLE_DEVICES=0','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',
 '-e','OPENPI_DATA_HOME=/workspace/openpi_data','-e','PYTHONPATH=/check/code:/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src:/core',
 '-v',str(P)+':/project:ro','-v',str(R)+':/check','-v',str(R/'tokenizer_cache')+':/workspace/openpi_data',
 '-v',str(A/'dependencies/openpi')+':/workspace/openpi:ro','-v',str(A/'dependencies/PhyAgentOS-core')+':/core:ro',
 '-v',str(A/'weights/pi05_libero_pytorch_fp32')+':/model:ro','-v','/opt/hyhal:/opt/hyhal:ro','-v','/etc/hfm:/etc/hfm:ro',
 'sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81','timeout','600','bash','-lc',
 'set -e; source /opt/dtk/env.sh; source /opt/hyhal/env.sh; exec python /check/code/check_prefix_features_v1.py']
save('LAUNCH.json',cmd);started=time.perf_counter();code=-1
try:
 with (R/'CHECK.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=650);code=r.returncode
finally:
 s=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
 if s['State']['Running']:
  cap(['docker','stop','-t','10',name]);s=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
 save('CONTAINER.json',s);save('HOST_RESULT.json',dict(exit_code=code,full_seconds=time.perf_counter()-started,default_unchanged=before==sha(P/'CURRENT_NATIVE_ENTRY.json')))
assert code==0 and not s['State']['Running'] and before==sha(P/'CURRENT_NATIVE_ENTRY.json')
rows=[]
for f in sorted(R.rglob('*')):
 rel=f.relative_to(R)
 if f.is_file() and not any(x in ('tokenizer_cache','__pycache__') for x in rel.parts):rows.append(dict(path=str(rel),bytes=f.stat().st_size,sha256=sha(f)))
save('COPY_MANIFEST.json',dict(files=rows,count=len(rows),bytes=sum(x['bytes'] for x in rows)))
print(json.dumps(json.loads((R/'RESULT.json').read_text())),flush=True)
