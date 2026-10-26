"""Matched-episode updates; unmatched outcomes never become counterfactual costs."""
import copy
import math
import statistics
import time
from .feedback_window import MODES


def empty():
    return {'version':2, 'default':'gripper_transition', 'incumbent':'gripper_transition',
            'scope':'articulated_goal', 'records':[], 'updates':0, 'last_update':None}


def comparisons(state):
    from .window_learning import context_key
    blocks={}
    for r in state['records']:
        if context_key(r['goal']) != state['scope']:continue
        key=(r['block_id'],r['comparison_signature'])
        blocks.setdefault(key,{}).setdefault(r['policy'],[]).append(r)
    base=state['incumbent']; reports={}
    for candidate in MODES:
        if candidate==base:continue
        paired=[]
        for key,arms in blocks.items():
            if base not in arms or candidate not in arms:continue
            a,b=arms[base],arms[candidate]
            cost_a=statistics.mean(r['full_wall_seconds'] for r in a)
            cost_b=statistics.mean(r['full_wall_seconds'] for r in b)
            paired.append({'block_id':key[0], 'savings_fraction':(cost_a-cost_b)/cost_a,
                           'base_failure_rate':statistics.mean(not r['native_goal_success'] for r in a),
                           'candidate_failure_rate':statistics.mean(not r['native_goal_success'] for r in b),
                           'evidence_ids':[r['evidence_id'] for r in a+b]})
        values=[p['savings_fraction'] for p in paired];n=len(values)
        mean=statistics.mean(values) if n else None
        se=statistics.stdev(values)/math.sqrt(n) if n>=2 else None
        base_fail=sum(p['base_failure_rate'] for p in paired)
        candidate_fail=sum(p['candidate_failure_rate'] for p in paired)
        quality_improvement=n>=2 and candidate_fail<base_fail
        cost_improvement=(n>=2 and candidate_fail<=base_fail and mean-2*se>0)
        reports[candidate]={'paired_blocks':paired,'n_blocks':n,'mean_savings_fraction':mean,
                            'standard_error':se,'screen_lower':mean-2*se if n>=2 else None,
                            'base_failure_sum':base_fail,'candidate_failure_sum':candidate_fail,
                            'eligible':quality_improvement or cost_improvement,
                            'quality_improvement':quality_improvement,
                            'uncertainty_scope':'Two-standard-error development screen; NOT a calibrated confidence interval or noninferiority proof'}
    return reports


def update(state, records, *, phase):
    from .window_learning import update as validate_update
    # Reuse development-only, duplicate, source, outcome and finite-cost checks.
    for r in records:
        if not r.get('block_id') or not r.get('comparison_signature'):
            raise ValueError('Missing comparison binding')
    out=validate_update(state,records,phase=phase)
    reports=comparisons(out);eligible=[p for p,r in reports.items() if r['eligible']]
    selected=min(eligible,key=lambda p:(not reports[p]['quality_improvement'],
                 reports[p]['candidate_failure_sum']-reports[p]['base_failure_sum'],
                 -reports[p]['screen_lower'])) if eligible else out['incumbent']
    out['last_update']={'previous':out['incumbent'],'selected':selected,'comparisons':reports,
                        'reason':'matched_evidence_supports_change' if eligible else 'retain_incumbent_no_supported_change'}
    out['incumbent']=selected
    return out


def decide(state, goal):
    from .window_learning import context_key
    started=time.perf_counter();key=context_key(goal)
    return {'selected_mode':state['incumbent'] if key==state['scope'] else state['default'],
            'context':key,'updates':state['updates'],'reason':'matched_episode_incumbent' if key==state['scope'] else 'outside_scope_fallback',
            'evidence_ids':[r['evidence_id'] for r in state['records'] if context_key(r['goal'])==key],
            'last_update':copy.deepcopy(state['last_update']),
            'selection_seconds':time.perf_counter()-started,'scope':'Matched development prototype; no quality guarantee'}
