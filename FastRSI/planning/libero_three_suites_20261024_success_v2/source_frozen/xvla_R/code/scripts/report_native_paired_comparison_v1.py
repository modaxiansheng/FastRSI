"""All twelve native outcomes, separate stage targets from learning and repayment."""
import json,sys
from pathlib import Path

def main():
    p=Path('/project');b=p/'planning/native_paired_comparison_v1_20261010';o=Path(sys.argv[1]);o.mkdir(exist_ok=False)
    assert json.loads((b/'STATUS.json').read_text())['status']=='COMPLETED'
    sums={a:{} for a in ('N','F','R')};cases=[]
    for version in range(12,16):
        name=f'native_value_phase_v{version}_20261010'
        audit=json.loads((p/'planning'/(name+'_audit')/'report_v1/AUDIT.json').read_text())
        assert audit['status']=='PASSED_NATIVE_BINDING_AUDIT'
        reg=json.loads((p/'runs'/name/'REGISTRATION.json').read_text());pair={r['arm']:r for r in audit['cases']}
        assert set(pair)==set(sums)
        for a,r in pair.items():
            for k in ('native_success','actions','control_requests','candidate_inferences','selected_second',
                'case_seconds','controller_seconds','startup_seconds','full_wall_seconds','sessions'):
                sums[a][k]=sums[a].get(k,0)+r[k]
        cases.append(dict(initial=reg['initial'],noise_base=reg['noise_base'],campaign=name,arms=pair))
    def compare(other):
        result={k+'_reduction':1-sums['R'][k]/sums[other][k] for k in ('actions','control_requests','candidate_inferences','case_seconds','full_wall_seconds')}
        result['native_success_count_not_lower']=sums['R']['native_success']>=sums[other]['native_success'] and sums['R']['native_success']>0
        return result
    base=compare('N');frozen=compare('F')
    screening=base['native_success_count_not_lower'] and base['control_requests_reduction']>=.10 and base['case_seconds_reduction']>=.05
    incremental=frozen['native_success_count_not_lower'] and frozen['case_seconds_reduction']>0 and frozen['control_requests_reduction']>0
    training=p/'planning/native_paired_q_update_v1_20261010'
    branch_v1=json.loads((p/'planning/native_value_branch_batch_v1_20261010/RESULT.json').read_text())
    branch_v2=json.loads((p/'planning/native_value_branch_batch_v2_20261010/RESULT.json').read_text())
    update=json.loads((training/'training/RESULT.json').read_text())
    branch_episode_seconds=sum(c['seconds'] for r in (branch_v1,branch_v2) for c in r['cases'])
    # These costs are related scopes, never add overlapping scopes to each other.
    collection_hosts=[]
    for version in (1,2):
        for slot in range(4):
            root=p/'runs'/f'native_value_branch_gpu{slot}_v{version}_20261010'
            collection_hosts.append(json.loads((root/'HOST.json').read_text())['seconds'])
    learning_cost=dict(branch_episode_seconds=branch_episode_seconds,branch_full_host_seconds=sum(collection_hosts),
        q_training_seconds=update['seconds'],q_training_full_host_seconds=json.loads((training/'HOST_RESULT.json').read_text())['seconds'],
        reference_native_baseline_cost_not_free=True,prior_prefix_reproduction_cost_not_free=True,
        prior_source_data_and_feature_extraction_cost_not_free=True,
        full_cold_start_ledger_complete=False,do_not_add_overlapping_cost_scopes=True)
    result=dict(status='AUDITED_NATIVE_PAIRED_UPDATE_COMPARISON',runs=12,arms=sums,cases=cases,
        updated_vs_default=base,updated_vs_frozen=frozen,stage_screening_thresholds_met=screening,
        updated_component_has_observed_execution_increment=incremental,learning_cost=learning_cost,
        parallel_execution=json.loads((b/'EXECUTION.json').read_text()),
        model_and_roster_frozen_before_outcomes=True,evaluation_persistent_updates=0,default_unchanged=True,
        statistical_noninferiority_established=False,complete_cold_start_net_gain_established=False,goal_complete=False,
        interpretation='Updated Q versus unchanged Q on new combinations, all native stages retained; small development screen, not proof of generalization or repayment.')
    (o/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(arms=sums,updated_vs_default=base,updated_vs_frozen=frozen,screening_met=screening,increment=incremental)),flush=True)
if __name__=='__main__':main()
