"""Host-specific CPU checks for an immutable, predeclared A/B roster."""
import json,os,subprocess,sys,time
from pathlib import Path
import orchestrate_local_bowl_transition_v1 as mod
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
B=P/'planning/local_bowl_batch10_v1_20261009'
host=sys.argv[1]
rows=[x for x in json.loads((B/'FROZEN_ROSTER.json').read_text()) if x['host']==host]
for row in rows:
 c=Path(row['campaign'])
 if (c/'CPU_LAUNCH.json').exists():raise RuntimeError('Existing CPU launch must be inspected, not duplicated')
 argv=mod.launch(P,c,'cpu');r=subprocess.run(argv,capture_output=True,text=True,timeout=45)
 doc=dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,host=host,started_at=time.time())
 (c/'CPU_LAUNCH.json').write_text(json.dumps(doc,indent=2)+'\n')
 if r.returncode:raise RuntimeError(r.stderr)
 print(row['seed'],row['arm'],r.stdout.strip(),flush=True)
