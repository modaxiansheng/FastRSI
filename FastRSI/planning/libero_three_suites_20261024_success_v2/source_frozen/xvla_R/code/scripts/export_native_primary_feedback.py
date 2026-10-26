"""New, explicit native-primary developer view; never rewrites original runs."""
import hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from fastrsi.benchmark_feedback import feedback,paired_feedback
NAME='goal_applicability_shadow_v1_20261005'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=R/'planning'/NAME
 protocol=dict(version='native_primary_v1',authority='User selected native benchmark success as primary on 2026-10-05; semantic and physical completion reported separately, conflicting prose excluded from memory.',scope='Prospective developer feedback and analyses; historical verdict/admission artifacts unchanged.',primary='native LIBERO success',secondary=['Qwen semantic verdict','physical release/stability evidence','complete controller and worker cost','VLA/Qwen/tool counts'],learning_permissions='Developer benchmark facts only; no test feedback, conflicting prose or invented action preference labels.',evaluation='Each evaluated version reads frozen experience; no persistent updates, including Session-end/background paths.')
 with (out/'FEEDBACK_PROTOCOL.json').open('x') as f:json.dump(protocol,f,indent=2)
 sources=[R/'planning/source_completion_handoff_v1_20261005/ANALYSIS.json']+[R/f'planning/m2_transfer_v1_20261005/INIT{i}_ANALYSIS.json' for i in (2,3)]
 pairs=[]
 for i,p in enumerate(sources,1):
  data=json.loads(p.read_text());rows=data['rows']
  # Existing analyzer verifies raw logs, images and actual action handoff.
  for r in rows:
   assert r['input_action_and_handoff_audit']=='PASSED'
   stream=R/'runs'/('source_completion_handoff_v1_20261005' if i==1 else f'm2_transfer_v1_20261005/init{i}')/r['arm']/'stream'
   parent=stream.parent
   for rel,digest in data['source_sha256'].items():
    if rel.startswith(r['arm']+'/'):
     target=parent.parent/rel
     assert sha(target)==digest,rel
  n,f=[feedback(r,mode='development',evidence_bound=True) for r in rows]
  pair=paired_feedback(n,f)
  pairs.append(dict(record_file=f'{i-1:03d}.json',source=str(p.relative_to(R)),source_sha256=sha(p),native=n,reuse=f,comparison=pair,training_target=None))
 with (out/'BENCHMARK_FEEDBACK.json').open('x') as f:json.dump(dict(protocol=protocol['version'],pairs=pairs,trained_model=False,notes='Observed outcomes/costs are usable developer feedback under new protocol. Preference/generalization supervision is not automatically inferred. Kept outside model requests.'),f,indent=2)
 print(json.dumps(dict(pairs=len(pairs),bound_native_outcomes=2*len(pairs),conflicts=sum(x['semantic_conflict'] for p in pairs for x in [p['native'],p['reuse']]),decision_labels=0)))
if __name__=='__main__':main()
