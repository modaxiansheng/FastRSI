"""Opt-in presentation of the original capability permissions and request state.

This is a public protocol correction, not a learned policy. It does not read
target/model outputs, infer benchmark success, submit requests, update skills,
change tool guards, or add an Agent turn. The original updater remains final.
"""
import hashlib
import json
from pathlib import Path

from .capability_interface import PREFIX, _bound_catalog
from .capability_updates import validate_request

VERSION = 'capability_interface_v2'
CASE_SWITCH = 'capability_interface_hint_version'


def _request_state(workspace, session_id):
    root = Path(workspace).resolve()
    relative = Path('skills/capability-requests') / (session_id + '.json')
    path = root / relative
    for current in (path, *path.parents):
        if current == root:
            break
        if current.is_symlink():
            raise ValueError('Current capability request cannot use a symlink')
    if path.exists() and not path.is_file():
        raise ValueError('Current capability request path is not a regular file')
    exists = path.is_file()
    return dict(path=relative.as_posix(), absolute_path=str(path), exists=exists,
                raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if exists else None,
                action=('Existing request may be inspected if its content is needed; '
                        'existence does not validate or approve it.' if exists else
                        'This request has not been created. Do not first read the missing file. '
                        'If you choose a permitted optional request, create it with write_file '
                        'at this exact path after the required evidence reads.'))


def describe(workspace, case, maintenance):
    """Return factual presentation only; UNKNOWN is never a success label.

    Catalog counters precede the current Session's completion processing. The
    updater observes actual use before applying requested operations, so choice
    availability below is conditional and never replaces that verification.
    """
    if type(maintenance) is not bool:
        raise ValueError('Explicit boolean native maintenance permission required')
    if not maintenance or case.get('mode') != 'development' or not case.get('capability_interface'):
        return dict(version=VERSION, enabled=False, operations=[], request=None,
                    native_root_eligibility='UNKNOWN', no_permissions_added=True)
    if case.get(CASE_SWITCH) != VERSION:
        raise ValueError('Explicit capability_interface_v2 opt-in required')
    raw, digest, catalog = _bound_catalog(case)
    local = Path(workspace) / PREFIX / 'catalogs' / (digest + '.json')
    if local.read_bytes() != raw:
        raise ValueError('Current capability catalog was modified')
    used = catalog['candidate_budget_used']
    limit = catalog['limits']['max_candidates']
    remaining = limit - used
    candidate_rows = []
    trials = []
    selections = list(catalog['accepted_choices'])
    for choice in catalog['candidate_choices']:
        status = catalog['candidate_statuses'][choice]
        candidate_rows.append(dict(choice=choice, **status))
        # Revalidation is reserved only by the original registered lifecycle.
        if status['status'] == 'revalidation_pending':
            continue
        if status['probation_uses'] < catalog['limits']['max_probation_uses']:
            trials.append(choice)
        # Do not approve a provisional choice here. The current completion can
        # supply the second success, or retire this choice, in the original
        # updater before it consumes set_next_choice.
        selections.append(choice)
    operations = []
    if remaining:
        operations.append(dict(op='propose_current_success', source='current',
            prerequisites=['Recorded native benchmark success of the actual root Session',
                           'Original compiler/controller evidence and duplicate-source checks'],
            candidate_slots_remaining=remaining))
    if trials or remaining:
        operations.append(dict(op='trial_candidate', catalog_choices=trials,
            symbolic_current_requires_successful_proposal=bool(remaining),
            prerequisites=['Original updater must confirm the candidate remains available',
                           'Probation uses must remain below the original limit after current-use observation'],
            extra_execution_now=False))
    operations.append(dict(op='set_next_choice', catalog_choices=selections,
        symbolic_current_requires_successful_proposal=bool(remaining),
        prerequisites=['Accepted choices retain their original permissions',
                       'A provisional candidate requires two actual ordinary probation successes '
                       'and must remain available after current-use observation']))
    # A valid JSON example is conditional syntax, not an instruction to propose.
    # Never display a new-proposal example after the lifetime budget is full.
    example = None
    if remaining:
        example = dict(version=1, session_id=case['id'], catalog_sha256=digest,
            operations=[dict(op='propose_current_success', source='current'),
                        dict(op='trial_candidate', choice='current')])
        validate_request(example, catalog, digest)
    return dict(version=VERSION, enabled=True, catalog_sha256=digest,
        session_id=case['id'], scope=catalog['scope'],
        candidate_budget=dict(used=used, limit=limit, remaining=remaining,
                              lifetime_budget_retired_candidates_still_count=True),
        native_root_eligibility='UNKNOWN',
        native_root_eligibility_reason='This presentation has no bound native terminal outcome. '
            'Runtime/semantic text is not substituted for native benchmark evidence. '
            'The original updater alone determines source qualification.',
        request=_request_state(workspace, case['id']),
        operations=operations, candidate_statuses_before_current_observation=candidate_rows,
        request_envelope=dict(version=1, session_id=case['id'], catalog_sha256=digest),
        conditional_proposal_example=example, original_catalog=catalog,
        no_permissions_added=True, no_request_created=True, no_forced_agent_turn=True)


def terminal_hint(workspace, case, maintenance):
    value = describe(workspace, case, maintenance)
    if not value['enabled']:
        return ''
    budget = value['candidate_budget']
    budget_text = ('The lifetime candidate budget is exhausted. A new proposal cannot '
                   'be applied; no new-proposal example is offered. Existing candidate '
                   'trial/selection permissions are still determined by the original '
                   'catalog and updater; they are not all disabled.' if not budget['remaining'] else
                   'A new proposal is conditional on actual native root eligibility '
                   'and the original compiler checks. The example is syntax only, '
                   'not evidence that the current source is qualified.')
    return ('\nShared capability interface v2, in this same optional experience-maintenance '
            'step. Read the complete current evidence and write the necessary benchmark '
            'report as originally required. This hint neither substitutes evidence nor '
            'adds reflection, verification, consolidation, or execution. '
            'Do not write a request solely to satisfy this prompt. '+budget_text+' '
            +value['request']['action']+' '
            'A request must use the exact Session and raw catalog hash below; state_sha256 '
            'is not catalog_sha256. The envelope needs a nonempty operations list using '
            'only the original schemas and prerequisites. At most one of each operation '
            'and three total. trial_candidate applies to a later ordinary task, not an '
            'extra task now. The original updater observes this Session first and may '
            'retire or otherwise change a provisional choice before deciding the request. '
            'Do not approve your own candidate, supply success labels, controller targets, '
            'or source paths. Semantic disagreement remains separate and is not a training '
            'label. The catalog and capability interface are read-only; requests are optional. '
            'The normal AgentLoop, required Verifier and native memory triggers are unchanged. '
            'Current bound operation/request facts: '+json.dumps(value, separators=(',', ':'))+'. ')
