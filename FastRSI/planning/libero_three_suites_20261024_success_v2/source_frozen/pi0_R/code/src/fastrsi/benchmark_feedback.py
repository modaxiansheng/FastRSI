"""Versioned developer feedback under a native-benchmark primary metric.

Does not replace session-verifier verdicts or promote quarantined prose. An
observed outcome is not a supervised label for the pre-execution route.
"""
import math

VERSION = 'native_primary_v1'

def feedback(row, *, mode, evidence_bound):
    if mode not in ('development','evaluation'):
        raise ValueError('Unknown data mode')
    if type(evidence_bound) is not bool:
        raise ValueError('Evidence binding must be explicit')
    if type(row.get('environment_success')) is not bool:
        raise ValueError('Missing native benchmark outcome')
    success=row['environment_success']
    semantic=row.get('semantic',[])
    if not isinstance(semantic,list) or any(v not in ('success','failure','replan','unknown') for v in semantic):
        raise ValueError('Invalid semantic observation')
    cost=row.get('controller_seconds')
    if type(cost) not in (int,float) or not math.isfinite(cost) or cost<0:
        raise ValueError('Invalid complete controller cost')
    usable=bool(evidence_bound and row.get('chain')=='PASSED' and mode=='development')
    conflict=any((v=='success')!=success for v in semantic if v in ('success','failure'))
    return dict(protocol=VERSION,mode=mode,benchmark_success=success,semantic_verdicts=list(semantic),
                semantic_conflict=conflict,controller_seconds=cost,benchmark_feedback_eligible=usable,
                narrative_feedback_eligible=False,physical_release_and_stability='UNKNOWN',
                decision_label=None,persistent_learning_allowed=usable,
                interpretation='Observed benchmark outcome only; not route correctness, stable release, or successful skill certification.')

def paired_feedback(native,reuse):
    if native['protocol']!=VERSION or reuse['protocol']!=VERSION:
        raise ValueError('Mixed outcome protocol')
    if not (native['benchmark_feedback_eligible'] and reuse['benchmark_feedback_eligible']):
        raise ValueError('Both branches require bound developer evidence')
    a,b=native['benchmark_success'],reuse['benchmark_success']
    relation='both_success' if a and b else 'both_failure' if not a and not b else 'reuse_success_only' if b else 'native_success_only'
    return dict(protocol=VERSION,observed_relation=relation,
        reuse_minus_native_controller_seconds=reuse['controller_seconds']-native['controller_seconds'],
        preference_label=None,label_status='OBSERVED_PAIR_NOT_VALIDATED_GENERALIZATION',
        reason='One paired realization does not identify expected route value or a reliable pre-execution separator.')


def observed_action_sample(context, *, action, available_actions, observation, provenance):
    """Supervise an executed action's outcome/cost without inventing its alternative.

    A training observation need not prove out-of-context generalization first.
    This function does not create a preference label or certify a new policy.
    """
    import copy
    import hashlib
    import json
    if observation.get('protocol') != VERSION or observation.get('mode') != 'development' or observation.get('benchmark_feedback_eligible') is not True:
        raise ValueError('Only bound developer observations may train')
    if type(observation.get('benchmark_success')) is not bool:
        raise ValueError('Missing observed benchmark result')
    cost=observation.get('controller_seconds')
    if type(cost) not in (int,float) or not math.isfinite(cost) or cost < 0:
        raise ValueError('Invalid observed cost')
    if not isinstance(available_actions,dict) or len(available_actions)<2 or action not in available_actions:
        raise ValueError('Executed action absent from available operations')
    if any(not isinstance(k,str) or not k or not isinstance(v,str) or not v for k,v in available_actions.items()):
        raise ValueError('Missing executable operation description')
    if not isinstance(context,dict) or not context.get('goal') or not context.get('observation_id'):
        raise ValueError('Missing pre-action context')
    forbidden={'native_success','final_success','benchmark_success','environment_success','semantic_verdict','semantic_verdicts','future','init_state_id','reward','controller_seconds','preference_label','training_target'}
    def validate(value):
        if isinstance(value,dict):
            if forbidden.intersection(value):raise ValueError('Outcome/index leaked into input')
            for v in value.values():validate(v)
        elif isinstance(value,(list,tuple)):
            for v in value:validate(v)
    validate(context)
    for key in ('input_sha256','result_sha256'):
        digest=provenance.get(key)
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Missing input/result provenance')
    if not isinstance(provenance.get('group'),str) or not provenance['group']:
        raise ValueError('Missing episode group for leakage-safe splitting')
    value=dict(version='observed_action_v1',model_input=dict(context=copy.deepcopy(context),available_actions=dict(available_actions),executed_action=action),
        targets=dict(benchmark_success=observation['benchmark_success'],controller_seconds=cost),
        provenance=copy.deepcopy(provenance),preference_label=None,
        cost_scope='Observed complete controller; excludes startup and acquisition/training. Other costs retained in ledger.',
        intended_use='Executed-action outcome/cost regression. No counterfactual label or causal policy-value guarantee.',
        release_authorized=False)
    value['sample_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return value
