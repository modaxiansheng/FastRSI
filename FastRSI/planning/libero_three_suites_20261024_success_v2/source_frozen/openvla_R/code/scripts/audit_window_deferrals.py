"""Replay only an observable feature state; never synthesize unexecuted robot outcomes."""
import json,hashlib,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from fastrsi.feedback_window import transition_diagnostics
rows=[]
for name,pair in [('feedback_window_known_v1_20261003','pair0'),('feedback_window_v1_20261003','pair1'),('task_window_v1_20261003','pair0')]:
 for arm in ('N','F','R'):
  p=R/'runs'/name/pair/arm/'stream/OS_COST_EVENTS.jsonl'
  if not p.exists():continue
  choices=[e for e in map(json.loads,p.read_text().splitlines()) if e.get('event')=='feedback_window_choice']
  direction=None;count=0;records=[]
  for e in choices:
   d,direction,count=transition_diagnostics(e['predicted_gripper_positive'],e['previous_committed_gripper_positive'],e['submitted_steps'],direction,count)
   records.append({'step':e['step'],**d})
  row=json.loads((p.parent/'RESULT.json').read_text())['cases'][0]
  rows.append({'run':name,'arm':arm,'init':row['case']['init_state_id'],'mode':choices[0]['mode'],'observed_native_success_for_analysis_only':row['environment_success'],'calls':len(choices),'steps':sum(v['steps'] for v in row['native_terminal_by_session'].values()),'deferral_count':sum(x['predicted_transition_outside_committed_prefix'] for x in records),'max_consecutive_deferrals':max(x['consecutive_deferrals_after'] for x in records),'decision_features':records,'source':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
report={'scope':'Chronological feature reconstruction from already saved predictions and actual committed prefixes; no offline reward or invented branch','rows':rows,'conclusions':['Existing broad gripper-transition rule can repeatedly predict a transition beyond the five actions it actually commits.','A repeated deferral can be beneficial stabilization or wasted replanning; these observations do not establish the counterfactual effect of forcing a longer window.','The counter is available before the next choice without Qwen, another VLA call, hidden state, task init ID, or future outcomes.','Log this diagnostic in every arm; do not turn one successful hard case into a learned threshold. No execution policy changed.']}
out=R/'planning/report_evidence_v1_20261003';out.mkdir(exist_ok=True);(out/'WINDOW_DEFERRALS.json').write_text(json.dumps(report,indent=2))
print(json.dumps([{k:r[k] for k in ('run','arm','init','mode','observed_native_success_for_analysis_only','calls','steps','deferral_count','max_consecutive_deferrals')} for r in rows],indent=2))
