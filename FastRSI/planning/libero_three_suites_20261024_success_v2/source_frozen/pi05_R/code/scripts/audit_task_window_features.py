"""Offline inventory of decision-time legal features, not a trained selector."""
import hashlib,json,math
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(v):return math.sqrt(sum(float(x)**2 for x in v))
def main():
 rows=[]
 for name,pair,init in [('feedback_window_v1_20261003','pair1',49),('feedback_window_known_v1_20261003','pair0',31),('task_window_v1_20261003','pair0',5)]:
  group=R/'runs'/name/pair/'N';case=f'net_t4_i{init}';p=group/'stream/stage_trace'/case/'TRACE.jsonl'
  trace=[json.loads(l) for l in p.read_text().splitlines()];chunk=next(e for e in trace if e.get('method')=='action_chunk');a=chunk['actions'];assert len(a)==10
  # These were generated once before first action submission; no future observations are used.
  features={'translation_norm_mean':sum(norm(v[:3]) for v in a)/10,'rotation_norm_mean':sum(norm(v[3:6]) for v in a)/10,'translation_adjacent_delta_max':max(norm([x-y for x,y in zip(u[:3],v[:3])]) for u,v in zip(a,a[1:])),'gripper_sign_transitions':sum((u[6]>0)!=(v[6]>0) for u,v in zip(a,a[1:]))}
  rows.append({'development_init_for_pairing_only':init,'features_before_first_action':features,'source':str(p.relative_to(R)),'sha256':sha(p),'native_success_is_not_a_feature':True})
 report={'status':'FEATURE_AVAILABILITY_ONLY','rows':rows,'protocol':'Predicted first ten actions are already available at the first legal execution-window decision; not an extra policy call. Init labels are audit bookkeeping only.','limitations':['Three development instances cannot establish predictive generalization; no rule fit or labels optimized.','Trace saves already-visible image bytes and committed actions; actual_policy_input logs observation/state key, not its numeric values. Do not claim a retrospective proprioceptive classifier was evaluated.','Norms in existing action units, not a calibrated visual error or physical distance. A first-chunk signal cannot establish later-stage failure predictability.','Historical robot outcomes or hidden simulator object poses must not enter online features.'],'next':'If further state learning is justified, log a compact copy of the already supplied robot state at the actual decision boundary; run offline development diagnostics before purchasing new validation.'}
 (R/'planning/task_window_v1_20261003/FEATURE_AVAILABILITY.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
