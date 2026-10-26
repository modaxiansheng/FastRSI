"""Select exactly one new physical identity per runner, not one of two by task ID."""
import hashlib,json,os
from pathlib import Path
ROOT=Path('/eval')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def install():
 import numpy as np
 from libero.libero import benchmark
 suite=os.environ['LIBEROPRO_BASE_SUITE'];variant=os.environ['LIBEROPRO_VARIANT'];identity=os.environ['FASTRSI_DEV_ID']
 index=json.loads((ROOT/'ASSET_INDEX_FINAL.json').read_text());rows=sorted([r for r in index if r['suite']==suite and r['variant']==variant],key=lambda x:x['task_id']);assert len(rows)==10
 selected=[r for r in json.loads((ROOT/'generated/ROSTER_RAW.json').read_text())['rows'] if r['identity_id']==identity];assert len(selected)==1;s=selected[0];assert s['suite']==suite and s['variant']==variant
 for r in rows:assert sha(ROOT/'assets'/r['bddl_rel'])==r['bddl_sha256']
 class BoundSuite:
  name=suite
  def __init__(self,*a,**kw):
   self.n_tasks=10;self.tasks=[benchmark.Task(name=r['task_name'],language=r['language'],problem='Libero',problem_folder=r['official_suite'],bddl_file=Path(r['bddl_rel']).name,init_states_file=Path(r['init_rel']).name) for r in rows]
  def get_task(self,i):return self.tasks[i]
  def get_num_tasks(self):return 10
  def get_task_init_states(self,i):
   assert i==s['task_id'];p=ROOT/'generated'/s['state_path'];assert sha(p)==s['initial_state_sha256'];return np.load(str(p),allow_pickle=False)[None,:]
  def get_task_bddl_file_path(self,i):return str(ROOT/'assets'/rows[i]['bddl_rel'])
 benchmark.BENCHMARK_MAPPING[suite]=BoundSuite
 return s
