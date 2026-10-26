"""Legacy development-only replay of a real continuation gate; no GPU dispatch."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.validation_budget import initial,observe,decide,dispatch,digest
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    start=time.perf_counter();out=args.output;out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'runs/instruction_trial_v3_20261002';manifest={}
    def read(p):
        manifest[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    proposal=read(source/'PROPOSED.json');rows=[]
    for arm in ['A','B']:
        group=source/('dev_'+arm);report=read(group/'stream/RESULT.json');status=read(group/'STATUS.json')
        assert status['status']=='COMPLETED' and report['status']=='COMPLETED' and len(report['cases'])==1
        row=report['cases'][0];assert row['case']['mode']=='development'
        case=group/'stream'/row['case']['id'];saved=read(case/'RESULT.json');assert saved['case']==row['case']
        expected=proposal['original_goal'] if arm=='A' else proposal['candidate_instruction']
        trace=group/'stream/OS_COST_EVENTS.jsonl';manifest[str(trace.relative_to(ROOT))]=hashlib.sha256(trace.read_bytes()).hexdigest()
        events=[json.loads(x) for x in trace.read_text().splitlines()]
        prompts=[x['prompt'] for x in events if x.get('event')=='actual_policy_input' and x.get('case')==row['case']['id']]
        assert prompts and set(prompts)=={expected}
        images={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (case/'workspace/artifacts/runtime'/row['case']['id']/'rgb').glob('initial_*.png')}
        assert images
        quality=row['chain_status']=='PASSED' and row['environment_success'] is True and bool(row['outcomes']) and all(x['verdict']=='success' for x in row['outcomes']) and row['experience_admission']['status']=='ADMITTED'
        rows.append({'seconds':report['seconds'],'quality':quality,'init':row['case']['init_state_id'],'images':images,'group_wall':status['full_wall_seconds']})
    a,b=rows;assert a['init']==b['init'] and a['images']==b['images']
    candidate_sha=digest({'goal':proposal['original_goal'],'candidate':proposal['candidate_instruction']})
    scope_sha=digest({'scope':proposal['scope'],'source_protocol':'instruction_trial_v3','measurement':'paired native controller seconds; identical cold service lifecycle in source; not new public fast baseline'})
    state=initial(candidate_sha,scope_sha)
    pair={'pair_id':'legacy_dev_t0_i8','mode':'development','candidate_sha256':candidate_sha,'scope_sha256':scope_sha,'init_id':a['init'],
          'native_task_seconds':a['seconds'],'candidate_task_seconds':b['seconds'],
          'project_source_run_seconds':a['group_wall']+b['group_wall'],'quality_resolved':a['quality'] and b['quality'],'sources':manifest.copy(),
          'scope_note':'Legacy admission, not retroactive certification under terminal_reads_v3. Group wall includes transfer diagnostic; project bill only, cold-start allocation remains unassigned.'}
    state=observe(state,pair)
    request={'mode':'development','candidate_sha256':candidate_sha,'scope_sha256':scope_sha,'init_id':21,
             'remaining_uses':2,'remaining_budget_seconds':900,'estimated_next_pair_seconds':a['seconds']+b['seconds'],
             'maximum_next_pair_seconds':900,'operation':'matched_development_pair',
             'horizon_note':'Explicit two-use mechanism fixture, not a claimed deployment horizon; no reservation or new experiment authorized by this replay.',
             'estimate_note':'Optimistic estimate from observed paired task controllers; excludes startup, extra discovery, and overhead. Not a guaranteed bound.'}
    decision=decide(state,request)
    def forbidden_executor(job):raise AssertionError('This regression must not launch or resurrect a parked candidate')
    receipt=dispatch(state,request,decision,forbidden_executor)
    baseline={'action':'validate_if_observed_saving_at_least_5_percent','observed_fraction':1-b['seconds']/a['seconds'],'would_continue':1-b['seconds']/a['seconds']>=.05,'scope':'Recorded legacy heuristic, not the strongest fixed-budget baseline'}
    result={'status':'COMPLETED','scope':'Historical development-only gate replay; not prospective learning or real cost savings',
      'old_gate':baseline,'new_gate':decision,'dispatch':receipt,'cold_start_accounting':'Not established: separate project group wall from mandatory deployment acquisition before any return claim.',
      'measured_saved_gpu_seconds':None,'model_calls':0,'new_robot_executions':0,'seconds':time.perf_counter()-start,
      'claims_excluded':['autonomous candidate discovery','prospective cross-round validation','learned efficiency versus a reasonable fixed rule','physical outcome of unexecuted validation','new strict baseline reproduction']}
    for name,value in [('STATE.json',state),('REQUEST.json',request),('RESULT.json',result),('INPUT_MANIFEST.json',manifest)]:
        (out/name).write_text(json.dumps(value,indent=2))
    assert all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h for rel,h in manifest.items())
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
