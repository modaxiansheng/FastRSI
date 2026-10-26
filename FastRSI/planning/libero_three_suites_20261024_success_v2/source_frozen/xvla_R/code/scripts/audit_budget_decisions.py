"""Audit the real continuation and exercise all policies on source-bound logs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fastrsi.validation_policy import create, record, freeze, decide, UNIT
from fastrsi.budget_comparison import compare, MissingBranch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    def read(rel):
        raw = (ROOT/rel).read_bytes()
        manifest[rel] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    old = read('planning/sampling_budget_v1/ANALYSIS_V2.json')
    clean = read('planning/sampling_budget_v1/CACHE_ANALYSIS.json')
    disposition = read('planning/sampling_budget_v1/TIMING_DISPOSITION.json')
    original_decision = read('runs/sampling_budget_v2_20261002/DECISION.json')
    original_dispatch = read('runs/sampling_budget_v2_20261002/DISPATCH_RECEIPT.json')
    assert original_dispatch['decision'] == original_decision
    assert original_dispatch['executor_calls'] == 1
    # Verify the recorded raw input hashes; aggregates alone are not a binding.
    for analysis in (old, clean):
        key = 'STATE_AFTER_P1.json' if 'STATE_AFTER_P1.json' in analysis['budget_gate'] else 'STATE_AFTER_P0.json'
        for row in analysis['budget_gate'][key]['observations']:
            for rel, expected in row['sources'].items():
                actual = hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
                if actual != expected: raise ValueError('Source drift: '+rel)
                manifest[rel] = actual
    first, second = old['pairs']
    first_gain = first['arms']['N']['controller_seconds']-first['arms']['F']['controller_seconds']
    second_gain = second['arms']['N']['controller_seconds']-second['arms']['F']['controller_seconds']
    audit = {'scope':'As-recorded continuation plus later measurement audit',
             'before':{'observations':1, 'controller_gain':first_gain,
                       'assumed_future_uses':20, 'original_decision':original_decision},
             'action':{'automatic_dispatch':True, 'candidate_preselected':True,
                       'next_instance_preselected':True, 'receipt':original_dispatch},
             'after':{'controller_gain':second_gain, 'replicated_first_advantage':False,
                      'paid_pair_concurrent_wall':second['status']['full_wall_seconds'],
                      'paid_pair_two_gpu_reservation_seconds':sum(x['full_wall_seconds'] for x in second['status']['groups']),
                      'measured_information_value':None},
             'later_audit':disposition,
             'conclusion':'Dispatch was real; input timing was later invalidated. No evidence that the policy detected the cache issue or selected a cost-effective experiment.',
             'hindsight_warning':'Later timing invalidation was NOT available to the historical decision. Corrected replay cannot be credited as prospective prevention.'}
    # The sole eligible current-protocol pair uses full lifecycle costs, not a
    # controller saving compared against a full-service acquisition cost.
    raw_state = clean['budget_gate']['STATE_AFTER_P0.json']; pair = clean['pairs'][0]
    source_row = raw_state['observations'][0]
    assert source_row['quality_resolved'] is True
    for arm in ('N','F'):
        assert pair['arms'][arm]['status'] == 'COMPLETED'
        for case in pair['arms'][arm]['cases']:
            assert case['chain_status'] == 'PASSED' and case['environment_success'] is True
            assert case['verification'] and all(x['verdict']=='success' for x in case['verification'])
    wall = {k:pair['arms'][k]['full_wall_seconds'] for k in ('N','F')}
    row = {'pair_id':'cache_pair0', 'instance_id':'libero_10/t0/i44', 'mode':'development',
           'candidate_sha256':raw_state['candidate_sha256'], 'scope_sha256':raw_state['scope_sha256'],
           'cost_unit':UNIT, 'baseline_cost':wall['N'], 'candidate_cost':wall['F'],
           'paid_cost':sum(wall.values()), 'measurement':'valid', 'quality':'pass',
           'sources':source_row['sources'], 'quality_scope':'Observed instance only; not 2pp noninferiority'}
    # A sensitivity grid, not estimated priors, calibrated noise or a new study.
    sensitivity = []
    for uses in (2,20,100,200):
        for noise in (10.,30.,60.,120.):
            cfg = dict(cost_unit=UNIT, prior_mean=0., prior_sd=2*noise, observation_sd=noise,
                       total_budget=row['paid_cost']+3400, next_pair_estimate=row['paid_cost'],
                       next_pair_cap=1700., remaining_uses=uses, fixed_pairs=4, max_pairs=4,
                       minimum_quality_pairs=2)
            state = record(create(row['candidate_sha256'],row['scope_sha256'],cfg),row)
            request = dict(mode='development',operation='matched_development_pair',
                           candidate_sha256=row['candidate_sha256'],scope_sha256=row['scope_sha256'],
                           instance_id='UNEXECUTED_BRANCH',bootstrap=False)
            d = decide(state,request)
            sensitivity.append({'future_uses_assumption':uses,'observation_sd_assumption':noise,
                                'information_value':d['modeled_next_information_value'],
                                'net_value':d['modeled_next_net_value'],'actual_decision':d['reason']})
    cfg = dict(cost_unit=UNIT,prior_mean=0.,prior_sd=120.,observation_sd=60.,
               total_budget=row['paid_cost']+3400,next_pair_estimate=row['paid_cost'],next_pair_cap=1700.,
               remaining_uses=20,fixed_pairs=4,max_pairs=4,minimum_quality_pairs=2)
    def absent(_): raise MissingBranch('No eligible second pair in current protocol; parked candidate remains parked')
    replay = compare(row['candidate_sha256'],row['scope_sha256'],cfg,[row],
                     [{'instance_id':'UNEXECUTED_BRANCH','bootstrap':False}],absent)
    inventory = [
        {'candidate':'sampler 10 to 5','eligible_current_protocol_pairs':1,'status':'PARKED',
         'invalid_old_pairs':2,'source':'planning/sampling_budget_v1/TIMING_DISPOSITION.json'},
        {'candidate':'instruction wording reuse','status':'PARKED_DIFFERENT_PROTOCOLS',
         'source':'docs/INSTRUCTION_FOLLOWUP_PROGRESS_20261002.md'},
        {'candidate':'public Session constructor','status':'SHARED_OPTIMIZATION_FIXED_EVALUATION_NOT_POLICY_TRAINING',
         'source':'planning/representative_coverage_v1/SHARED_FLOW_NET_PARALLEL_V2_ANALYSIS.json'}]
    for item in inventory:
        rel = item['source']; raw = (ROOT/rel).read_bytes()
        manifest[rel] = hashlib.sha256(raw).hexdigest()
    result = {'status':'COMPLETED', 'scope':'Budget policy implementation and real-ledger diagnostic',
              'eligible_current_sampling_pairs':1,'new_gpu_runs':0,'new_model_calls':0,
              'actual_savings':None,'policy_quality_comparison':'NOT_ESTABLISHED_MISSING_EXECUTED_BRANCHES',
              'seconds':time.perf_counter()-started,
              'no_claim':['calibrated uncertainty','2pp noninferiority','learned speedup','counterfactual physical outcomes']}
    for name, obj in [('CONTINUATION_AUDIT.json',audit),('SENSITIVITY.json',sensitivity),
                      ('COMPARISON.json',replay),('CANDIDATES.json',inventory),('RESULT.json',result),
                      ('INPUT_MANIFEST.json',manifest)]:
        (args.output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    assert all(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h for rel,h in manifest.items())
    print(json.dumps({**result,'source_files_verified':len(manifest),
                      'comparison_status':{k:v['status'] for k,v in replay['policies'].items()}},ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
