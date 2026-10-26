"""Observed report-only model costs; no conversion into robot/full-method gain."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];NAME='report_evidence_v2_20261003'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 run=ROOT/'runs'/NAME;out=ROOT/'planning'/NAME;assert read(run/'STATUS.json')['status']=='COMPLETED'
 result=read(run/'stream/RESULT.json');rows=[];sources={}
 for row in result['rows']:
  job=row['name'];folder=run/'stream'/job;receipt=row['report'];report=folder/'workspace'/receipt['report'];text=report.read_text()
  assert row['fixed_experience_unchanged'] and row['source_unchanged'] and row['canonical_unchanged']
  assert receipt['read_coverage']['complete'] and receipt['read_coverage']['fresh_report_write']
  if row['view']:assert receipt['evidence_view']['lossless_value_roundtrip']
  source=ROOT/row['source'].split('/project/',1)[1];saved=read(source/'RESULT.json');case=saved['case'];ep=read(source/'workspace/artifacts/runtime'/case['id']/'episode.json')
  calls=[]
  for p in (run/'stream/service_calls').glob('*/REQUEST.json'):
   request=read(p)
   if '/stream/'+job+'/workspace' not in json.dumps(request):continue
   r=read(p.parent/'RECEIPT.json');assert r['status']=='OK'
   calls.append({'id':p.parent.name,'generation_seconds':r['generation_seconds'],'service_seconds':r['seconds'],'prompt_tokens':r['prompt_tokens'],'completion_tokens':r['completion_tokens'],'finish':r['generation_finish_reason']})
   for path in (p,p.parent/'RECEIPT.json'):sources[str(path.relative_to(ROOT))]=sha(path)
  assert calls and all(c['finish']=='eos' for c in calls)
  # Do not allow a replay to silently read the old report and reuse an answer.
  reads=receipt['actual_reads'];allowed={'SESSIONS.md','artifacts/runtime/'+case['id']+'/episode.json'}
  assert all(p in allowed or (row['view'] and p.startswith('reports/evidence/')) for p in reads)
  rows.append({'job':job,'source_case':case['id'],'view':row['view'],'controller_seconds':row['seconds'],'report_calls':len(calls),'generation_seconds':sum(c['generation_seconds'] for c in calls),'prompt_tokens':sum(c['prompt_tokens'] for c in calls),'completion_tokens':sum(c['completion_tokens'] for c in calls),'prefix_sha256':row['prefix_sha256'],'prefix_messages':row['prefix_messages'],'report_path':str(report.relative_to(ROOT)),'report_text':text,'expected_native_success':ep['success'],'expected_steps':ep['num_steps'],'step_number_in_report':bool(re.search(r'(?<!\d)'+str(ep['num_steps'])+r'(?!\d)',text)),'read_coverage':receipt['read_coverage'],'evidence_view':receipt['evidence_view'],'sources_unchanged':True,'original_report_not_read':True})
  sources[str(report.relative_to(ROOT))]=sha(report)
 pairs=[]
 for prefix in ('N','F'):
  a=next(r for r in rows if r['job']==prefix+'_raw');b=next(r for r in rows if r['job']==prefix+'_view');assert a['prefix_sha256']==b['prefix_sha256']
  pairs.append({'source':prefix,'seconds_saved':a['controller_seconds']-b['controller_seconds'],'prompt_tokens_saved':a['prompt_tokens']-b['prompt_tokens'],'calls_difference':a['report_calls']-b['report_calls']})
 summary={'status':'COMPLETED','scope':'Same-saved-task native report replay. One observation per condition; no robot, verifier, learning or full OS speedup evidence.','rows':rows,'pairs':pairs,'project_wall_seconds':read(run/'STATUS.json')['full_wall_seconds'],'report_controller_seconds_total':sum(r['controller_seconds'] for r in rows),'sources':sources,'prior_failed_campaign_seconds':read(ROOT/'runs/report_evidence_v1_20261003/STATUS.json')['full_wall_seconds'],'default_changed':False,'limitations':['Shared service ordered N raw/view, F view/raw; not a repeated randomized latency estimate.','Both sides use equal copied fixed experience and original pre-report task prefix. Evaluation guards remain active through close.','Initial model loading, CPU tests and cleanup are in project wall. No model cold-start cost is charged repeatedly per report.','Historical raw-run bills are unchanged; improved view is a shared engineering candidate, not learned gain.']}
 out.mkdir(exist_ok=True);(out/'REPLAY_SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:summary[k] for k in ('pairs','project_wall_seconds','report_controller_seconds_total')},indent=2))
 for r in rows:print(r['job'],r['expected_native_success'],r['expected_steps'],r['report_text'])
if __name__=='__main__':main()
