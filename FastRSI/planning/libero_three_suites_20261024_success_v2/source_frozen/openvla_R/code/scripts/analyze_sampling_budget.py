"""Account real candidate probing without calling static gains learning gains."""
import argparse,hashlib,json,math
from pathlib import Path
from analyze_representative_batch import analyze
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--output',required=True);a=ap.parse_args()
    run=ROOT/'runs'/a.run_id;status=read(run/'STATUS.json');assert status['status'] in ('COMPLETED','FAILED')
    pairs=[];all_bills=[]
    for pair in sorted(run.glob('pair*')):
        if not pair.is_dir():continue
        arms={};comparisons={}
        for arm in ['N','F']:
            group=pair/arm
            if not (group/'STATUS.json').exists():continue
            report=analyze(group);bill=report['bill'];all_bills.append(bill)
            sample=lines(group/'stream/SAMPLING_EVENTS.jsonl');calls=[e for e in sample if e.get('event')=='sample_actions']
            arms[arm]={'status':report['status'],'cases':report['cases'],'full_wall_seconds':bill['full_wall_seconds'],
              'controller_seconds':read(group/'stream/RESULT.json')['seconds'] if (group/'stream/RESULT.json').exists() else None,
              'sampling':{'bindings':[e for e in sample if e.get('event')=='sampling_binding'],'calls':len(calls),'budgets':sorted({e['num_steps'] for e in calls}),'shapes':sorted({tuple(e['shape']) for e in calls}),
                  'summed_host_seconds':sum(e['seconds'] for e in calls),'host_seconds_per_call':sum(e['seconds'] for e in calls)/len(calls) if calls else None},
              'stage_seconds':report['stage_seconds'],'model_purposes':report['purpose_summary'],
              'bill':{k:bill.get(k) for k in ['hcu_allocation_seconds','cpu_quota_seconds','model_calls','model_requests','incomplete_model_requests','model_rejected','prompt_tokens','completion_tokens','generation_seconds']}}
        if len(arms)==2:
            n,f=arms['N'],arms['F'];diff=n['full_wall_seconds']-f['full_wall_seconds']
            comparisons={'raw_full_wall_N_minus_F_seconds':diff,'raw_fraction':diff/n['full_wall_seconds'],
                'single_parallel_layout_only':True,'learned_effect':None,'default_promotion':False}
            if n['controller_seconds'] is not None and f['controller_seconds'] is not None:
                gain=n['controller_seconds']-f['controller_seconds'];paid=n['full_wall_seconds']+f['full_wall_seconds']
                comparisons.update(raw_controller_N_minus_F_seconds=gain,
                    optimistic_reuses_to_cover_this_pair=math.ceil(paid/gain) if gain>0 else None,
                    payback_scope='Exploratory extrapolation from one pair, not a confidence bound; excludes unmeasured development labor, cannot establish deployed return.')
        pairs.append({'pair':pair.name,'status':read(pair/'STATUS.json'),'arms':arms,'comparison':comparisons})
    first=read(run/'PRIOR_ATTEMPT.json') if (run/'PRIOR_ATTEMPT.json').exists() else {}
    gate={k:read(run/k) for k in ['STATE_AFTER_P0.json','REQUEST.json','DECISION.json','DISPATCH_RECEIPT.json','STATE_AFTER_P1.json'] if (run/k).exists()}
    keys=['hcu_allocation_seconds','cpu_quota_seconds','model_calls','model_requests','incomplete_model_requests','prompt_tokens','completion_tokens']
    result={'status':status['status'],'scope':'Prospective paid sampling candidate and fixed economic continuation, not learned FastRSI efficacy',
       'pairs':pairs,'budget_gate':gate,'timing_validity':'Consult TIMING_DISPOSITION.json; historical uncached timings cannot establish return or seed post-repair cost learning.','ledger':{'project_current_wall_seconds':status['full_wall_seconds'],
           'prior_branch_status':first.get('status'),'prior_branch_wall_seconds':first.get('full_wall_seconds',0),'project_branch_wall_seconds':status['full_wall_seconds']+first.get('full_wall_seconds',0),
           'sum_arm_reservation_seconds':sum(b['full_wall_seconds'] for b in all_bills),
           'method_observed_probe_cost':{k:sum(b.get(k,0) for b in all_bills) for k in keys},
           'method_complete_cold_start_seconds':None,'future_incremental_saving_observed':None,
           'accounting':'Concurrent arm wall is not added to campaign wall. Full campaign contains learning, failures and shutdown; stage spans overlap. Capacity time is not GPU utilization. R&D labor/transfer unmeasured; source frozen-experience acquisition remains separately paid common history.'},
       'no_claim':['stable speedup','2pp noninferiority','30% total improvement-cost saving','learned selector benefit','broad task generalization'],
       'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output=ROOT/a.output;output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as file:json.dump(result,file,indent=2,ensure_ascii=False)
    print(json.dumps({'status':result['status'],'pairs':[{'pair':p['pair'],'comparison':p['comparison'],'arms':{k:{'wall':v['full_wall_seconds'],'sampling':v['sampling'],'cases':[{x:c.get(x) for x in ['id','environment_success','verification','chain_status']} for c in v['cases']]} for k,v in p['arms'].items()}} for p in pairs],'gate':gate.get('DECISION.json'),'ledger':result['ledger']},indent=2))
if __name__=='__main__':main()
