"""Aggregate complete native pairs, retaining failures and timing scopes."""
import json
import sys
from pathlib import Path

def main():
    project=Path('/project');batch=project/'planning/native_value_batch_v2_20261010'
    out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
    assert json.loads((batch/'STATUS.json').read_text())['status']=='COMPLETED'
    cases=[];sums={a:{} for a in ('N','F')}
    for version in range(8,12):
        name=f'native_value_phase_v{version}_20261010'
        audit=json.loads((project/'planning'/(name+'_audit')/'report_v2/AUDIT.json').read_text())
        assert audit['status']=='PASSED_NATIVE_BINDING_AUDIT'
        reg=json.loads((project/'runs'/name/'REGISTRATION.json').read_text())
        pair={r['arm']:r for r in audit['cases']}
        for a,r in pair.items():
            for k in ('native_success','actions','control_requests','candidate_inferences','selected_second',
                      'case_seconds','controller_seconds','startup_seconds','full_wall_seconds','sessions'):
                sums[a][k]=sums[a].get(k,0)+r[k]
        cases.append(dict(initial=reg['initial'],noise_base=reg['noise_base'],campaign=name,arms=pair,
            success_category=('both_success' if pair['N']['native_success'] and pair['F']['native_success'] else
                'rescued' if pair['F']['native_success'] else 'lost' if pair['N']['native_success'] else 'both_failure')))
    def reduction(k):return 1-sums['F'][k]/sums['N'][k]
    quality=sums['F']['native_success']>=sums['N']['native_success'] and sums['F']['native_success']>0
    benefit=dict(native_success_count_not_lower=quality,control_request_reduction=reduction('control_requests'),
        case_time_reduction=reduction('case_seconds'),full_host_time_reduction=reduction('full_wall_seconds'),
        candidate_inference_reduction=reduction('candidate_inferences'))
    benefit['stage_screening_thresholds_met']=quality and benefit['control_request_reduction']>=.10 and benefit['case_time_reduction']>=.05
    result=dict(status='AUDITED_NEW_COMBINATION_NATIVE_SCREEN',runs=8,cases=cases,arms=sums,benefit=benefit,
        parallel_execution=json.loads((batch/'EXECUTION.json').read_text()),
        default_unchanged=True,model_frozen_for_all_four_pairs=True,evaluation_updates=0,
        statistical_noninferiority_established=False,recursive_update_gain_established=False,
        complete_cold_start_net_gain_established=False,goal_complete=False,
        interpretation='Native Agent evaluation of an already learned frozen component; no new learned update or distillation in this batch.')
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(arms=sums,benefit=benefit,categories=[c['success_category'] for c in cases])),flush=True)

if __name__=='__main__':main()
