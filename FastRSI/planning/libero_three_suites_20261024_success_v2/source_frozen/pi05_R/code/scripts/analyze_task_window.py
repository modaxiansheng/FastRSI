"""Prove the frozen table was actually read; record update and all paid lifecycle."""
import hashlib,json,time
from pathlib import Path
from task_window_study import ROOT,read,sha,pair_record
from fastrsi import task_window
NAME='task_window_v1_20261003'
def main():
 run=ROOT/'runs'/NAME;out=ROOT/'planning'/NAME;protocol=read(run/'PROTOCOL.json');assert read(run/'STATUS.json')['status']=='COMPLETED'
 summary=read(out/'SUMMARY.json');table=task_window.load(out/'TASK_TABLE.json',protocol['task_window_sha256']);initial=table
 new=[];proof=[]
 for pair in sorted(run.glob('pair*')):
  fact=pair_record(ROOT,pair);new.append(fact)
  p=pair/'F/stream/RESULT.json';row=read(p)['cases'][0];decision=row['task_window_decision'];expected=task_window.select(initial,row['case']['task_description'])
  assert decision['snapshot_sha256']==protocol['task_window_sha256']==sha(pair/'TASK_TABLE.json')
  assert all(decision[k]==v for k,v in expected.items())
  events=[json.loads(l) for l in (pair/'F/stream/OS_COST_EVENTS.jsonl').read_text().splitlines() if l.strip()]
  choices=[e for e in events if e.get('event')=='feedback_window_choice'];assert choices and all(e['mode']==expected['selected_mode'] for e in choices)
  agent_events=[json.loads(l) for l in (pair/'F/stream'/row['case']['id']/'AGENT_EVENTS.jsonl').read_text().splitlines() if l.strip()]
  assert any(e.get('event')=='task_window_read' for e in agent_events)
  proof.append({'pair':pair.name,'source':str(p.relative_to(ROOT)),'decision':decision,'actual_mode':choices[0]['mode'],'calls':len(choices),'snapshot_unchanged':True})
 t=time.perf_counter();updated=task_window.fit(initial['pairs']+new,phase='development');(out/'TASK_TABLE_NEXT.json').write_text(json.dumps(updated,indent=2));seconds=time.perf_counter()-t
 changes=[{'goal':v['goal'],'before':task_window.select(initial,v['goal'])['selected_mode'],'after':v['selected_mode']} for v in updated['table'].values()]
 follow=[read(p) for p in sorted(run.glob('CONTINUATION_*.json'))]
 assert all(r['status']=='NOT_DISPATCHED' for r in follow) or len(new)==2
 for item in follow:
  if item['status']=='NOT_DISPATCHED':assert not (run/'pair1').exists()
 frozen_cost=summary['totals']['F']['full_wall_seconds'];baseline_cost=summary['totals']['N']['full_wall_seconds']
 history=read(out/'HISTORY.json')
 report={'status':'COMPLETED','scope':'New development transfer check of a historical task table; no final evaluation, no adaptive within-episode routing or learned net-benefit claim','execution_proof':proof,'new_feedback':new,'update_seconds':seconds,'table_choices':changes,'policy_changed_after_new_feedback':any(x['before']!=x['after'] for x in changes),'continuation':follow,'ledger':{'unit':'two_gpu_reservation_seconds for lifecycle sums; CPU update time separate','historical_source_episodes':2*len(initial['pairs']),'historical_paid_lifecycle_seconds':history['paid_history_lifecycle_seconds'],'new_fixed10_lifecycle_seconds':baseline_cost,'new_frozen_table_lifecycle_seconds':frozen_cost,'new_acquisition_pair_lifecycle_seconds':baseline_cost+frozen_cost,'parallel_campaign_wall_seconds':summary['study_wall_seconds'],'cpu_update_seconds':seconds,'initial_table_preparation_seconds':None,'initial_preparation_limitation':'Original offline preparation not independently timed; no complete cold-start efficiency claim','no_double_count':'Acquisition pair includes both arms; do not add it again to those arms or to parallel campaign wall. History remains separately charged.'},'observed_frozen_table_minus_fixed10_seconds':frozen_cost-baseline_cost,'learning_increment_claim':None,'default_promoted':False,'sources':{str(p.relative_to(ROOT)):sha(p) for p in [out/'SUMMARY.json',out/'TASK_TABLE.json',out/'HISTORY.json',run/'PROTOCOL.json']}}
 (out/'LEARNING_AND_BUDGET.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ('policy_changed_after_new_feedback','table_choices','ledger','observed_frozen_table_minus_fixed10_seconds')},indent=2))
if __name__=='__main__':main()
