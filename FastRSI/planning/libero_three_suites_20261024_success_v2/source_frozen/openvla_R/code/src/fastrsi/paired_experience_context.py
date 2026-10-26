"""Keep factual paired outcomes bound to the continuation that produced them.

This controls a supervised data view, not robot execution. Q under a reference
policy can still propose policy improvement, but its labels must not silently be
presented as measured outcomes under the newly proposed policy.
"""
from collections import defaultdict
import hashlib
import json


def continuation_id(contract):
    if not isinstance(contract, dict) or not contract.get('kind'):
        raise ValueError('An explicit continuation contract is required')
    return hashlib.sha256(json.dumps(contract, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def training_view(records, target_continuation_id):
    if not target_continuation_id:
        raise ValueError('Do not infer a target continuation from mixed data')
    result = []
    for record in records:
        if record['continuation_id'] != continuation_id(record['continuation_contract']):
            raise ValueError('Continuation contract digest mismatch')
        if not record.get('both_outcomes_actually_executed'):
            raise ValueError('Unexecuted outcomes cannot form a factual pair')
        if record['continuation_id'] == target_continuation_id and record['preference'] is not None:
            result.append(record)
    return result


def context_dependent_preferences(records):
    by_input = defaultdict(list)
    for record in records:
        if record['preference'] is not None:
            by_input[(record['legal_input_sha256'], record['candidate_pair_sha256'])].append(record)
    return [rows for rows in by_input.values()
        if len({r['preference'] for r in rows}) > 1
        and len({r['continuation_id'] for r in rows}) > 1]
