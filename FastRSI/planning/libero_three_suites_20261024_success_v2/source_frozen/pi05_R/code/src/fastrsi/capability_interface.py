"""Shared, optional capability requests in the existing native Results turn.

This module never compiles, approves or executes skills. Immutable catalogs are
installed by the caller; the Agent may only request an inter-Session operation.
"""
import hashlib
import json
from pathlib import Path
import re

PREFIX = Path('skills/capability-interface')
INSTRUCTIONS = '''# Executable capability requests

The current Results prompt supplies a versioned capability catalog. After reading
the current task evidence, you may request a capability operation in the existing
maintenance turn. A request is optional; no extra reflection turn is required.
Use only the exact Session, catalog hash and operations supplied there. The
trusted inter-Session updater binds source='current' to actual root execution;
you cannot supply success labels, controller targets or source file paths.
Candidates remain provisional until real later tasks provide validation.
The catalog is read-only. Evaluation never permits capability updates.
'''


def protected_path(path):
    return Path(path) == PREFIX or PREFIX in Path(path).parents


def _bound_catalog(case):
    from .capability_updates import validate_catalog
    spec = case['capability_interface']
    if set(spec) != {'artifact', 'sha256'}:
        raise ValueError('Capability interface requires an exact artifact binding')
    raw = Path(spec['artifact']).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != spec['sha256']:
        raise ValueError('Capability catalog digest mismatch')
    value = json.loads(raw)
    validate_catalog(value)
    if (not re.fullmatch(r'[A-Za-z0-9_-]+', case['id'])
            or value['session_id'] != case['id']
            or value['scope'] != dict(suite=case['suite'], task_id=case['task_id'],
                                      goal=case['task_description'])):
        raise ValueError('Capability catalog belongs to another Session or goal')
    return raw, digest, value


def install(workspace, case):
    root = Path(workspace)
    raw, digest, catalog = _bound_catalog(case)
    files = {PREFIX/'SKILL.md': INSTRUCTIONS.encode(),
             PREFIX/'catalogs'/f'{digest}.json': raw}
    for relative, content in files.items():
        target = root/relative
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError('Inherited capability interface changed')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(content)
    return dict(catalog_sha256=digest, session_id=catalog['session_id'],
                files={str(p): hashlib.sha256(v).hexdigest() for p, v in files.items()},
                request_path=f"skills/capability-requests/{case['id']}.json",
                scope='Shared operation permissions; no request or update is forced.')


def verify(workspace, receipt):
    for rel, digest in receipt['files'].items():
        if hashlib.sha256((Path(workspace)/rel).read_bytes()).hexdigest() != digest:
            raise ValueError('Read-only capability interface was modified')


def terminal_hint(workspace, case, maintenance):
    version = case.get('capability_interface_hint_version')
    if version is not None:
        if version == 'capability_interface_v2_1':
            from .capability_interface_v2_1 import terminal_hint as versioned_hint
        elif version == 'capability_interface_v2':
            from .capability_interface_v2 import terminal_hint as versioned_hint
        else:
            raise ValueError('Unknown opt-in capability hint version')
        return versioned_hint(workspace, case, maintenance)
    if not maintenance or case.get('mode') != 'development' or not case.get('capability_interface'):
        return ''
    _, digest, catalog = _bound_catalog(case)
    local = Path(workspace)/PREFIX/'catalogs'/f'{digest}.json'
    if hashlib.sha256(local.read_bytes()).hexdigest() != digest:
        raise ValueError('Current capability catalog was modified')
    example = dict(version=1, session_id=case['id'], catalog_sha256=digest,
                   operations=[dict(op='propose_current_success', source='current'),
                               dict(op='trial_candidate', choice='current')])
    return ('\nShared capability interface, available during this same optional experience-maintenance step. '
            'After reading current verified evidence, you may write one JSON request to '
            f"skills/capability-requests/{case['id']}.json. "
            'Do not write a request solely to satisfy this prompt. Do not alter the read-only catalog. '
            'A successful root trajectory can be proposed; trial_candidate schedules a provisional '
            'candidate on a later normal task, not an extra execution now. set_next_choice can select '
            'a catalog-listed accepted choice, native, or a provisional choice whose catalog '
            'permits bounded use in this registered flow. Provisional is not globally validated. '
            'Source eligibility uses the recorded native benchmark outcome of the root Session. '
            'Keep conflicting semantic prose separate; it is not a controller-training label. '
            'Do not approve your own candidate. '
            'At most one of each operation and three operations total. An optional proposal-and-trial '
            'request has this format: '+json.dumps(example, separators=(',', ':'))+
            '. Current trusted catalog: '+json.dumps(catalog, separators=(',', ':'))+'. ')
