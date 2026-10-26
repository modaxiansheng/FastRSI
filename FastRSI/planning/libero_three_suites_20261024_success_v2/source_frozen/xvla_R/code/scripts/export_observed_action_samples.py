"""Export current three developer pairs as observed action regression samples."""
import hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from fastrsi.benchmark_feedback import observed_action_sample
NAME='learning_speed_review_20261005'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 origin=R/'planning/goal_applicability_shadow_v1_20261005';rows=json.loads((origin/'BENCHMARK_FEEDBACK.json').read_text())['pairs'];manifest=json.loads((origin/'requests/MANIFEST.json').read_text());digests={r['file']:r['sha256'] for r in manifest['records']};samples=[]
 actions={'native_pi05':'Use original pi05 for the complete task with the unchanged 10-step window.','source_then_pi05':'Reuse the fixed source 262-step controller goals, then resume pi05 in the same episode.'}
 for index,row in enumerate(rows,1):
  request_path=origin/'requests'/row['record_file'];assert sha(request_path)==digests[row['record_file']]
  record=json.loads(request_path.read_text());state=record['request']['state']
  context={k:state[k] for k in ('goal','observation_id','robot_state','candidate','image_order')}
  # Image identities/roles, not audit paths containing an initial-state ID.
  context['image_roles']=[im['role'] for im in record['images']]
  source=R/row['source'];assert sha(source)==row['source_sha256'];analysis=json.loads(source.read_text())
  for arm,action,kind in [('N','native_pi05','native'),('F','source_then_pi05','reuse')]:
   result=R/'runs'/('source_completion_handoff_v1_20261005' if index==1 else f'm2_transfer_v1_20261005/init{index}')/arm/'stream/RESULT.json'
   assert sha(result)==analysis['source_sha256'][arm+'/stream/RESULT.json']
   provenance=dict(input_sha256=sha(request_path),result_sha256=sha(result),group=f'task64_episode_{index}',task_family='stack_bowls',result_path=str(result.relative_to(R)),input_path=str(request_path.relative_to(R)),image_assets=record['images'],image_asset_scope='Resolve as pixels; paths and hashes are provenance, never encoded features',input_kind='Reconstructed from verified pre-execution inputs, not original emitted decision-model request')
   samples.append(observed_action_sample(context,action=action,available_actions=actions,observation=row[kind],provenance=provenance))
 out=R/'planning'/NAME;out.mkdir(exist_ok=True)
 with (out/'OBSERVED_ACTION_SAMPLES_V2.json').open('x') as f:json.dump(dict(status='EXPORTED_NOT_TRAINED',samples=samples,independent_context_groups=3,training_runs=0,warning='Six executed-arm samples from only three contexts; not six independent tasks. Same context paired actions must share data split. No guessed unexecuted branch, no preference or model deployment authorization.'),f,indent=2)
 print(json.dumps(dict(samples=len(samples),context_groups=3,trained=False)))
if __name__=='__main__':main()
