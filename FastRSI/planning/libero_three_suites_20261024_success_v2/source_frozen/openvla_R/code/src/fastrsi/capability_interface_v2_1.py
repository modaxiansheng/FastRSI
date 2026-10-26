"""Compact opt-in v2 presentation; same original permissions, no policy update."""
import json

from .capability_interface_v2 import CASE_SWITCH, VERSION as BASE_VERSION, describe as base_describe

VERSION = 'capability_interface_v2_1'


def describe(workspace, case, maintenance):
    if case.get(CASE_SWITCH) != VERSION:
        raise ValueError('Explicit capability_interface_v2_1 opt-in required')
    value = base_describe(workspace, {**case, CASE_SWITCH: BASE_VERSION}, maintenance)
    value['version'] = VERSION
    return value


def terminal_hint(workspace, case, maintenance):
    value = describe(workspace, case, maintenance)
    if not value['enabled']:
        return ''
    request = value['request']
    budget = value['candidate_budget']
    budget_text = ('Lifetime candidate budget is full: new proposals cannot be applied. '
                   if budget['remaining'] == 0 else
                   'A proposal is conditional on native root success and the original compiler checks. ')
    file_text = ('Current request exists; inspect it only if needed. Existence is not approval. '
                 if request['exists'] else
                 'Current request does not exist: do not read it first. If making a valid '
                 'optional request, create it at the stated path with write_file. ')
    facts = dict(version=VERSION, candidate_slots_remaining=budget['remaining'],
        native_root_eligibility='UNKNOWN',
        request=dict(path=request['path'], exists=request['exists']),
        request_envelope=value['request_envelope'], catalog=value['original_catalog'])
    return ('\nShared capability interface v2.1, same optional maintenance step. '
            +budget_text+file_text+
            'Native eligibility is unknown to this hint; the original updater determines it. '
            'trial_candidate uses a catalog candidate with remaining probation budget; '
            'set_next_choice uses an accepted choice or a candidate with two actual '
            'probation successes. The updater observes current use first and may retire '
            'a candidate. Catalog operation names are schemas, not an approval. '
            'Add one to three chosen original operations to the exact envelope; '
            'catalog_sha256 is the raw catalog hash, not state_sha256. No request is forced. '
            'Bound facts: '+json.dumps(facts, separators=(',', ':'))+'. ')
