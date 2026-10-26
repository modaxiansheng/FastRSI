"""Consume already-paid resident feedback; stop identical mature-policy reruns.

No GPU/robot/model calls, no new thresholds, and no counterfactual cost claims.
"""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fastrsi import task_window,validation_budget
from fastrsi.task_window_snapshot import advance_snapshot

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def extract(root, review, pair):
    byid={x['evidence_id']:x for x in review['executions']}
    arms=[byid[pair[k]] for k in ('fixed10_evidence_id','fixed5_evidence_id')]
    assert pair['native_fact_pair_eligible'] and all(pair['matching_checks'].values())
    assert all(x['case']['mode']=='development' and x['quality']['chain_status']=='PASSED' for x in arms)
    a,b=arms;case=a['case'];sources={}
    for row in arms:
        for name,digest in row['sources'].items():
            assert sha(root/name)==digest,name
            sources[name]=digest
    proto='runs/structural_window_v1_20261003/PROTOCOL.json'
    return {'phase':'development','evidence_id':'resident_pair/'+pair['pair_id'],
        'goal':case['task_description'],'task':case['task_id'],'init':case['init_state_id'],
        'cost_compatible':True,'protocol_group':'resident_case_hot_terminal_view_v1',
        'protocol_sources':{proto:sha(root/proto)},
        'independent_instance_key':f"libero_10/task{case['task_id']}/init{case['init_state_id']}",
        'fixed10':{'native_success':a['quality']['native_success'],'seconds':a['costs']['worker_lifecycle_seconds']},
        'fixed5':{'native_success':b['quality']['native_success'],'seconds':b['costs']['worker_lifecycle_seconds']},
        'sources':sources}, arms

def main():
    started=time.perf_counter();out=ROOT/'planning/mature_window_decisions_v1_20261003';out.mkdir(exist_ok=False)
    prior=ROOT/'runs/task_snapshot_native_v1_20261003/UPDATED.json';old=read(prior)
    review_path=ROOT/'planning/window_evidence_review_20261003/REVIEW.json';review=read(review_path)
    added=[];observations=[]
    for pair in review['fixed_policy_pairs']:
        if pair['run']=='structural_window_v1_20261003':
            fact,arms=extract(ROOT,review,pair);added.append(fact);observations.append((fact,arms))
    assert len(added)==4 and len(old['pairs'])==9
    # Previous snapshot already carries explicit provenance and verified facts.
    result=advance_snapshot(phase='development',previous_path=prior,previous_sha256=sha(prior),
        added_pairs=added,source_root=ROOT,output_path=out/'UPDATED.json',
        next_goal=added[0]['goal'],next_session_id='unlaunched-development-decision-check')
    new=read(out/'UPDATED.json');goals=sorted({p['goal'] for p in new['pairs']})
    changes=[{'goal':g,'before':task_window.select(old,g),'after':task_window.select(new,g)} for g in goals]
    changes=[{**x,'mode_changed':x['before']['selected_mode']!=x['after']['selected_mode']} for x in changes]
    gates=[]
    candidate=validation_budget.digest({'operation':'fixed5_vs_fixed10','model_config_sha':sha(ROOT/'runs/structural_window_v1_20261003/MODEL_CONFIG.json'),'cost_protocol':'resident_case_hot_terminal_view_v1'})
    for fact,arms in observations:
        scope=validation_budget.digest({'goal':fact['goal'],'candidate':candidate})
        state=validation_budget.initial(candidate,scope)
        a,b=fact['fixed10'],fact['fixed5']
        resolved=all(r['quality']['native_success'] and r['quality']['semantic_verdicts']==['success'] for r in arms)
        state=validation_budget.observe(state,{'candidate_sha256':candidate,'scope_sha256':scope,'mode':'development',
            'pair_id':fact['evidence_id'],'init_id':fact['init'],
            'native_task_seconds':a['seconds'],'candidate_task_seconds':b['seconds'],
            'project_source_run_seconds':a['seconds']+b['seconds'],'quality_resolved':resolved,'sources':fact['sources']})
        # Sensitivity probes are NOT real requested tasks and grant no dispatch authority.
        outcomes=[]
        for uses in (1,10,100):
            request={'candidate_sha256':candidate,'scope_sha256':scope,'operation':'matched_development_pair','mode':'development',
                'init_id':'NOT_SCHEDULED','remaining_uses':uses,'remaining_budget_seconds':1700,
                'estimated_next_pair_seconds':a['seconds']+b['seconds'],'maximum_next_pair_seconds':1700}
            decision=validation_budget.decide(state,request)
            outcomes.append({'remaining_uses_sensitivity_only':uses,'decision':decision})
        gates.append({'task':fact['task'],'native_success_10_5':[a['native_success'],b['native_success']],
            'observed_10_minus_5_seconds':a['seconds']-b['seconds'],
            'quality_resolved_under_existing_gate':resolved,'sensitivity':outcomes,
            'actual_executor_calls':0,'measured_saving_seconds':None})
    report={'status':'COMPLETED','scope':'Source-validated offline mature-policy update and budget sensitivity; no new execution or runtime gain',
        'previous_pairs':len(old['pairs']),'new_pairs':len(new['pairs']),'goals':len(goals),
        'changes':changes,'mode_change_count':sum(x['mode_changed'] for x in changes),
        'budget_gate_review':gates,'native_sessions_launched':0,'default_changed':False,'learning_gain_established':False,
        'policy_limitation':'A historical quality win with no harm holds fixed5 regardless of newer both-success costs; sparse new tasks remain fixed10.',
        'budget_limitation':'Existing gate stops both negative gain and unresolved quality; it neither acquires new-task evidence nor values capability gain.',
        'operational_decision':'No duplicate same-policy run. This does not reject all research acquisition or establish that no better policy exists.',
        'sources':{str(review_path.relative_to(ROOT)):sha(review_path),str(prior.relative_to(ROOT)):sha(prior)},
        'update_receipt':result['receipt'],'wall_seconds':time.perf_counter()-started}
    with (out/'ANALYSIS.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ['status','previous_pairs','new_pairs','goals','mode_change_count','wall_seconds']}))

if __name__=='__main__':main()
