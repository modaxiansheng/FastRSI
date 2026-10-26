"""Negative-only, SHA-bound real operation feedback for native withdrawal.

This adapter cannot promote a capability or infer unexecuted alternatives.
It rechecks complete raw records and an independently produced operation audit.
The existing global declaration has no conditional consumer, so an observed
failure withdraws that declaration; historical successful records remain.
"""
import copy
import hashlib
from pathlib import Path

from .empirical_scope_update_v1 import _read, digest, METRICS

VERSION = 'native_failed_operation_feedback_v1'


def load_failed_operation_observations(root, *, feedback, catalogue):
    root = Path(root)
    document = _read(root, feedback)
    candidates = _read(root, catalogue)['candidates']
    if document.get('version') != VERSION:
        raise ValueError('Explicit negative-only operation feedback required')
    audit = _read(root, document['audit'])
    registration = _read(root, document['registration'])
    inventory = _read(root, document['terminal_inventory'])
    if (audit.get('status') != 'PASSED_COMPLETE_FRESH_DEVELOPMENT_OPERATION_AUDIT'
            or inventory.get('status') != 'COMPLETE_REMOTE_FILES_AND_OWNED_STOP_VERIFIED'
            or registration.get('role') != 'DEVELOPMENT_OPERATION_PAIR'
            or audit['registration_sha256'] != document['registration']['sha256']):
        raise ValueError('Complete original audited development operation required')
    rawroot = document['raw_root']
    if (not isinstance(rawroot, str) or Path(rawroot).is_absolute()
            or '..' in Path(rawroot).parts or not Path(rawroot).parts):
        raise ValueError('Contained raw archive required')
    files = inventory['files']
    if len(files) != inventory['file_count'] or sum(x['size'] for x in files.values()) != inventory['total_bytes']:
        raise ValueError('Incomplete original terminal inventory')
    for key in ('audit', 'registration'):
        relative = Path(document[key]['path'])
        if Path(rawroot) not in relative.parents:
            raise ValueError('Audit and registration must come from the same raw inventory')
        identity = relative.relative_to(rawroot).as_posix()
        if files.get(identity, {}).get('sha256') != document[key]['sha256']:
            raise ValueError('Summary is not the original inventoried receipt')
    # Recheck all raw bytes, not just a summary or selected favourable images.
    for rel, value in files.items():
        _read_bytes(root, str(Path(rawroot)/rel), value)
    def raw(rel):
        value = files.get(rel)
        if value is None:
            raise ValueError('Actual branch file missing from complete inventory')
        return _read(root, dict(path=str(Path(rawroot)/rel), sha256=value['sha256']))
    expected = registration['draw_seeds']
    if [p['draw_seed'] for p in audit['pairs']] != expected or len(set(expected)) != len(expected):
        raise ValueError('Preregistered pairs omitted, duplicated or reordered')
    mapping = document['candidate_map']
    if set(mapping) != {'native', 'fixed60'} or mapping['native'] != 'native' or mapping['fixed60'] not in candidates:
        raise ValueError('Explicit executable catalogue mapping required')
    arms = {a['arm']: a for a in audit['arms']}
    if len(arms) != 2*len(expected) or set(arms) != set(registration['arms']):
        raise ValueError('Complete unique actual branch roster required')
    rows = []
    for pair in audit['pairs']:
        if (pair.get('initial_five_classes_exact') is not True
                or pair.get('cost_comparison_eligible') is not True):
            raise ValueError('Actual initial pairing and complete cost protocol required')
        for kind in ('native', 'fixed60'):
            name = kind+'_'+str(pair['draw_seed']); arm = arms[name]
            # Paths are copied under original project-relative identities.
            remote_path = Path(registration['arms'][name]['campaign'])
            marker = '/FastRSI/'
            if marker not in str(remote_path):
                raise ValueError('Original project campaign identity missing')
            prefix = str(remote_path).split(marker, 1)[1]
            result = raw(prefix+'/stream/RESULT.json')
            status = raw(prefix+'/STATUS.json')
            original = raw(prefix+'/REGISTRATION.json')
            cases = result['cases']
            if (len(cases) != 1 or cases[0]['case']['mode'] != 'development'
                    or cases[0]['environment_success'] is not False
                    or arm['native_success'] is not False
                    or cases[0]['chain_status'] != 'PASSED'
                    or arm['chain_status'] != 'PASSED'
                    or status['status'] != 'COMPLETED'
                    or status['environment_success'] is not False):
                raise ValueError('Adapter accepts complete real failures only, never positive admission')
            candidate = mapping[kind]
            if kind == 'fixed60':
                declaration = copy.deepcopy(candidates[candidate])
                actual = original['declaration']
                declared_sha = declaration['bindings']['model_config_sha256']
                if declared_sha not in ('$MODEL_CONFIG_SHA256', actual['bindings']['model_config_sha256']):
                    raise ValueError('Actual candidate model differs')
                declaration['bindings']['model_config_sha256'] = actual['bindings']['model_config_sha256']
                if declaration.get('execution_contract', {}).get('max_program_steps') != 60:
                    raise ValueError('Catalogue is not the actual 60-prefix candidate')
                declaration.pop('execution_contract', None)
                if declaration != actual or arm['work']['program_actions'] != 60 or arm['work']['program_requests'] != 6:
                    raise ValueError('Actual executed candidate differs from catalogue')
            elif arm['work']['program_actions'] or arm['work']['program_requests']:
                raise ValueError('Native arm has unexpected source execution')
            protocol = copy.deepcopy(arm['common_protocol'])
            if digest(protocol) != pair['common_protocol_sha256']:
                raise ValueError('Audited homogeneous protocol changed')
            protocol['model_config_sha256'] = protocol['models']['model_config_sha256']
            w = arm['work']
            work = {k: w[k] for k in METRICS if k not in ('vla_calls', 'qwen_calls')}
            work.update(vla_calls=w['vla_attempts'], qwen_calls=w['qwen_service_requests'])
            seconds = status['full_wall_seconds']
            audited_seconds = (arm['costs']['measurements']['full_wall_seconds']
                               if kind == 'native' else arm['costs']['full_wall_seconds'])
            if seconds != pair[kind+'_job_seconds'] or seconds != audited_seconds:
                raise ValueError('Complete inclusive cost receipt changed')
            rows.append(dict(candidate=candidate, role='development', native_success=False,
                context_sha256=pair['initial_pairing_only_sha256'], protocol_sha256=digest(protocol),
                protocol=protocol, result_sha256=files[prefix+'/stream/RESULT.json']['sha256'],
                chain_status='PASSED', verification_complete=True, work=work,
                full_wall_seconds=seconds, audited_costs=arm['costs'],
                acquisition_origin='Complete paid original operation failures; negative-only withdrawal',
                terminal_physical_release='UNKNOWN', administrative_ids_are_features=False,
                semantic_text_is_training_label=False))
    return rows


def _read_bytes(root, relative, value):
    path = Path(root)/relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts
            or path.is_symlink() or not path.is_file()
            or Path(root).resolve() not in path.resolve().parents):
        raise ValueError('Ordinary contained raw file required')
    blob = path.read_bytes()
    if len(blob) != value['size'] or hashlib.sha256(blob).hexdigest() != value['sha256']:
        raise ValueError('Actual raw evidence changed: '+relative)
