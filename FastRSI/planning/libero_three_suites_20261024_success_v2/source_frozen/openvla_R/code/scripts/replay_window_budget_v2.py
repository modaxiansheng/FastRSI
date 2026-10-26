"""Chronological replay using only executed development outcomes. Never simulate a missing branch."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from window_budget_evidence import build_ledger
from fastrsi import window_budget_v2 as budget
from fastrsi.validation_budget import digest

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    out=ROOT/'planning/window_budget_v2_final_20261003';out.mkdir(exist_ok=False);started=time.perf_counter()
    names=['feedback_window_v1_20261003/pair1','feedback_window_known_v1_20261003/pair0','task_window_v1_20261003/pair0']
    specs=[]
    for name in names:
        proto='runs/'+name.split('/')[0]+'/PROTOCOL.json'
        specs.append({'path':'runs/'+name,'arms':['N','F'],'protocol_group':'legacy_'+name.split('/')[0],
            'protocol_sources':{proto:sha(ROOT/proto)},'cost_compatible':False})
    config=ROOT/'runs/task_window_v1_20261003/pair0/N/MODEL_CONFIG.json'
    goal=read(ROOT/'runs/task_window_v1_20261003/pair0/N/stream/RESULT.json')['cases'][0]['case']['task_description']
    target='runs/structural_window_v1_20261003/PROTOCOL.json'
    candidate=digest({'operation':'fixed5_vs_fixed10','model_sha256':sha(config),'sampling_steps':10})
    trace=[]
    for n in range(1,len(specs)+1):
        ledger=build_ledger(ROOT,specs[:n],candidate_sha256=candidate,goal=goal,model_sha256=sha(config),
            protocol_group='resident_case_hot_terminal_view_v1',protocol_sources={target:sha(ROOT/target)})
        request={'phase':'development','ledger_sha256':digest(ledger),'candidate_sha256':candidate,
            'remaining_uses':1,'max_quality_followups':1,'allow_quality_confirmation':True,
            'remaining_budget_seconds':1700,'estimated_pair_seconds':750,'maximum_pair_seconds':1700,'next_init_id':2}
        decision=budget.decide(ledger,request)
        trace.append({'historical_prefix_records':n,'decision':decision,'request':request,
            'counterfactual_allocation_only':True,'historical_action_executed_by_this_policy':False})
        with (out/f'LEDGER_{n}.json').open('x') as f:json.dump(ledger,f,ensure_ascii=False,indent=2)
    def forbidden(_):raise AssertionError('Current paid evidence should not buy another identical confirmation')
    boundary=budget.dispatch(ledger,request,decision,forbidden)
    assert boundary['status']=='NOT_DISPATCHED'
    report={'status':'PASSED','scope':'Offline chronological decision replay plus real no-dispatch code boundary; no new robot data',
        'trace':trace,'current_dispatch':boundary,'observed_policy_sequence':[t['decision']['action'] for t in trace],
        'cost_comparable_pairs':sum(r['eligible_cost'] for r in ledger['records']),
        'historical_paid_lifecycle_seconds':sum(r['project_source_run_seconds'] for r in ledger['records']),
        'historical_bill_scope':'Heterogeneous old protocols, accounting only; excluded from resident cost fitting or method speed comparison',
        'new_model_calls':0,'new_robot_cases':0,'learning_gain_established':False,
        'measured_saved_seconds':None,'wall_seconds':time.perf_counter()-started}
    with (out/'REPLAY.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ['status','observed_policy_sequence','cost_comparable_pairs','historical_paid_lifecycle_seconds','wall_seconds']}))
if __name__=='__main__':main()
