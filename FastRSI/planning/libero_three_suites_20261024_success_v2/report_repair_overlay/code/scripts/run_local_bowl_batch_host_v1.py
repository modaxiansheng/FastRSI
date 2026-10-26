"""Run the frozen20-root roster in paired host-local waves; no candidate update."""
import json,hashlib,os,subprocess,sys,time
from pathlib import Path
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
B=P/'planning/local_bowl_batch10_v1_20261009'
host=sys.argv[1];rows=[x for x in json.loads((B/'FROZEN_ROSTER.json').read_text()) if x['host']==host]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for row in rows:
 c=Path(row['campaign']);assert sha(c/'MANIFEST.json')==row['manifest_sha256']
 for f in ['sdk_local_transition/RESULT.json','preflight/CHECK_RESULT.json']:
  d=json.loads((c/f).read_text());assert d['manifest_sha256']==row['manifest_sha256']
  assert d.get('returncode',0)==0 and d.get('status','PASSED')=='PASSED'
 d=json.loads((c/'CPU_LAUNCH.json').read_text());assert d['host']==host
 assert subprocess.check_output(['docker','inspect','--format','{{.State.Status}} {{.State.ExitCode}}',d['stdout'].strip()],text=True).strip()=='exited 0'
 # Reject an old launch before any wave dispatch.
 assert not (c/'CONTROLLER_DISPATCH.json').exists()
for wave in range(3):
 jobs=[]
 for row in [r for r in rows if r['wave']==wave]:
  c=Path(row['campaign']);argv=['python3',str(P/'scripts/orchestrate_local_bowl_transition_v1.py'),'--execute','--project',str(P),'--campaign',str(c),'--manifest-sha256',row['manifest_sha256'],'--allow-real-physical-calls']
  log=(c/'CONTROLLER.log').open('xb');r=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy());log.close()
  (c/'CONTROLLER_DISPATCH.json').write_text(json.dumps(dict(argv=argv,pid=r.pid,start_stat=Path(f'/proc/{r.pid}/stat').read_text().split()[21],host=host,started_at=time.time(),manifest_sha256=row['manifest_sha256']),indent=2)+'\n');jobs.append((row,r))
  print('DISPATCH',wave,row['seed'],row['arm'],r.pid,flush=True)
 # Controller owns its bounded runner and cleanup. No external broad stops.
 for row,r in jobs:
  r.wait(timeout=1900);c=Path(row['campaign']);d=json.loads((c/'STATUS.json').read_text())
  if d['status']!='COMPLETED':raise RuntimeError('Protocol/resource failure; preserve and stop remaining waves: '+str(c))
  print('COMPLETE',wave,row['seed'],row['arm'],d['environment_success'],flush=True)
