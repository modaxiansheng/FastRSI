"""Prefix-only comparison on saved development outcomes; no fabricated branches."""
import copy
import time

from .validation_policy import create, record, freeze, step, MODES


class MissingBranch(Exception):
    pass


def compare(candidate_sha, scope_sha, config, seed_records, jobs, read_outcome):
    """All policies start with identical paid evidence and the same job order.

read_outcome(job) is called ONLY after a policy requests that job. It must bind
the result to that precise candidate, scope and instance or raise MissingBranch.
Historical replay bills are reference costs, not newly incurred/saved GPU time.
"""
    started = time.perf_counter(); results = {}
    for mode in MODES:
        state = create(candidate_sha, scope_sha, config, mode)
        for row in seed_records:
            state = record(state, row)
        if mode == 'frozen': state = freeze(state)
        events = []; status = 'JOB_LIST_EXHAUSTED'; last = None
        for job in jobs:
            request = {**copy.deepcopy(job), 'mode':'development',
                       'operation':'matched_development_pair',
                       'candidate_sha256':candidate_sha, 'scope_sha256':scope_sha}
            try:
                state, receipt = step(state, request, read_outcome, events.append)
            except MissingBranch:
                status = 'MISSING_BRANCH'; break
            last = receipt
            if receipt['executor_calls'] == 0:
                status = 'POLICY_HELD'; break
        results[mode] = {'status':status, 'events':events, 'final_state':state,
                         'last_receipt':last,
                         'reference_selected_pair_cost':sum(r['paid_cost'] for r in state['records']),
                         'final_independent_quality':None, 'actual_gpu_savings':None}
    return {'scope':'Historical prefix replay, not live policy efficacy', 'policies':results,
            'new_robot_executions':0, 'seconds':time.perf_counter()-started,
            'policy_effectiveness_established':False}
