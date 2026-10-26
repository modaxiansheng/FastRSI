"""CPU-only, immutable task-table preparation using the existing learner.

Pairs retain task_window.fit's schema and add protocol_group,
protocol_sources {relative_path: SHA256}, and independent_instance_key
(``libero_10/task<T>/init<I>``). Legacy pairs require explicit
previous_provenance keyed by evidence_id; their original facts are not changed.

The returned bindings work with run_native_baseline's existing artifact fields.
The caller persists the returned receipt. Preparing and reading a snapshot does
not establish that a native Session executed it or that learning saved costs.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import time

from . import task_window


PROVENANCE_KEYS = ('protocol_group', 'protocol_sources', 'independent_instance_key')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Missing ' + name)
    return value


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _binding(path, digest):
    return {'task_window_artifact': str(Path(path).resolve()),
            'task_window_sha256': digest}


def read_snapshot(*, binding, goal, session_id, phase):
    """Check an intended Session's immutable input; no Agent/robot is executed."""
    if phase not in ('development', 'evaluation'):
        raise ValueError('Explicit development/evaluation read phase required')
    _text(goal, 'goal'); _text(session_id, 'session_id')
    state = task_window.load(binding['task_window_artifact'], binding['task_window_sha256'])
    return {'session_id': session_id, 'phase': phase,
            'binding': dict(binding), 'decision': task_window.select(state, goal),
            'scope': 'CPU load/select check only; native Session execution not observed',
            'native_session_executed': False, 'persistent_update': False}


class _Sources:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.data = {}

    def verify(self, sources):
        if not isinstance(sources, dict) or not sources:
            raise ValueError('Real source bindings required')
        for name, expected in sources.items():
            _text(name, 'source path')
            path = (self.root / name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(self.root):
                raise ValueError('Source must remain within source_root')
            if not _digest(expected):
                raise ValueError('Invalid source SHA256')
            data = self.data.get(path)
            if data is None:
                data = path.read_bytes()
                self.data[path] = data
            if _sha(data) != expected:
                raise ValueError('Source hash mismatch: ' + name)

    def json(self, name):
        return json.loads(self.data[(self.root / name).resolve()])

    def json_lines(self, name):
        data = self.data[(self.root / name).resolve()]
        return [json.loads(line) for line in data.splitlines() if line.strip()]


def _prepare_pair(pair, provenance, sources):
    p = copy.deepcopy(pair)
    if provenance is not None:
        if set(provenance) != set(PROVENANCE_KEYS):
            raise ValueError('Previous provenance must contain only the three binding fields')
        for key in PROVENANCE_KEYS:
            if key in p and p[key] != provenance[key]:
                raise ValueError('Previous provenance changed')
            p[key] = copy.deepcopy(provenance[key])
    if p.get('phase') != 'development':
        raise ValueError('Evaluation feedback cannot update snapshots')
    for key in ('evidence_id', 'goal', 'protocol_group', 'independent_instance_key'):
        _text(p.get(key), key)
    if type(p.get('cost_compatible')) is not bool:
        raise ValueError('Explicit cost_compatible flag required')
    sources.verify(p.get('sources'))
    sources.verify(p.get('protocol_sources'))
    result_paths = [name for name in p['sources'] if name.endswith('/stream/RESULT.json')]
    if len(result_paths) != 2:
        raise ValueError('Exactly two original fixed-policy RESULT sources required')
    actual = {}; execution_ids = []; images = []; models = []
    for name in result_paths:
        result = sources.json(name)
        if result.get('status') != 'COMPLETED' or len(result.get('cases', [])) != 1:
            raise ValueError('Completed single-case source required')
        row = result['cases'][0]; case = row['case']
        if case.get('mode') != 'development' or case.get('suite') != 'libero_10':
            raise ValueError('Raw source is not LIBERO development')
        if row.get('chain_status') != 'PASSED' or case.get('max_steps') != 520:
            raise ValueError('Raw source chain/limit mismatch')
        if (case.get('task_description') != p['goal'] or case.get('task_id') != p.get('task')
                or case.get('init_state_id') != p.get('init')):
            raise ValueError('Raw task/init/goal mismatch')
        expected_instance = 'libero_10/task' + str(p['task']) + '/init' + str(p['init'])
        if p['independent_instance_key'] != expected_instance:
            raise ValueError('Independent instance key must match original task/init')
        mode = row.get('task_window_decision', row.get('window_learning_decision', {})).get(
            'selected_mode', case.get('feedback_window_mode'))
        if mode not in ('fixed10', 'fixed5') or mode in actual:
            raise ValueError('Need one actually executed fixed10 and one fixed5')
        group = str(Path(name).parent.parent)
        required = [group + '/STATUS.json', group + '/MODEL_CONFIG.json',
                    group + '/stream/OS_COST_EVENTS.jsonl',
                    group + '/stream/' + case['id'] + '/workspace/artifacts/runtime/'
                    + case['id'] + '/episode.json']
        if any(path not in p['sources'] for path in required):
            raise ValueError('Missing bound status/model/events/episode source')
        status, episode = sources.json(required[0]), sources.json(required[3])
        native = row.get('environment_success'); terms = list(row.get('native_terminal_by_session', {}).values())
        if (status.get('status') != 'COMPLETED' or type(native) is not bool
                or len(terms) != 1 or terms[0].get('native_success') != native
                or episode.get('success') != native or episode.get('num_steps') != terms[0].get('steps')):
            raise ValueError('Native facts mismatch or unsupported multi-Session recovery')
        fact = p.get(mode, {})
        seconds = fact.get('seconds')
        if (fact.get('native_success') is not native or type(seconds) not in (int, float)
                or not math.isfinite(seconds) or seconds <= 0 or seconds != status.get('full_wall_seconds')):
            raise ValueError('Pair cost/outcome does not match original source')
        events = sources.json_lines(required[2])
        inputs = [e for e in events if e.get('event') == 'actual_policy_input']
        choices = [e for e in events if e.get('event') == 'feedback_window_choice']
        if (not inputs or len(inputs) != len(choices)
                or any(e.get('prompt') != p['goal'] for e in inputs)
                or any(e.get('mode') != mode for e in choices)):
            raise ValueError('Real policy input/choice bindings mismatch')
        image_prefix = str(Path(required[3]).parent) + '/rgb/initial_'
        image_hashes = {Path(path).name: digest for path, digest in p['sources'].items()
                        if path.startswith(image_prefix) and path.endswith('.png')}
        if not image_hashes:
            raise ValueError('Original initial-image bindings required')
        images.append(image_hashes); models.append(p['sources'][required[1]])
        actual[mode] = name
        execution_ids.append((str((sources.root / name).resolve()), p['sources'][name]))
    if images[0] != images[1] or models[0] != models[1]:
        raise ValueError('Paired initial images/model configuration differ')
    return p, execution_ids


def advance_snapshot(*, phase, previous_path, previous_sha256, added_pairs,
                     source_root, output_path, next_goal, next_session_id,
                     previous_provenance=None):
    """Return updated/frozen bindings and receipt after exclusive snapshot creation.

    No existing file, default configuration, evaluation state or source is written.
    Cross-protocol cost pooling for one goal is rejected. Explicitly false
    cost_compatible records retain native facts and appear in the receipt.
    """
    started = time.perf_counter(); cpu_started = time.process_time()
    if phase != 'development':
        raise ValueError('Only explicit development can create an updated snapshot')
    _text(next_goal, 'next_goal'); _text(next_session_id, 'next_session_id')
    previous_path, output_path = Path(previous_path).resolve(), Path(output_path).absolute()
    if output_path.resolve() == previous_path or output_path.exists() or output_path.is_symlink():
        raise ValueError('Output must be a new unique path')
    if not isinstance(added_pairs, list) or not added_pairs:
        raise ValueError('At least one new real development pair required')
    previous = task_window.load(previous_path, previous_sha256)
    provenance = previous_provenance or {}
    old_ids = {p['evidence_id'] for p in previous['pairs']}
    if set(provenance) - old_ids:
        raise ValueError('Provenance references an unknown previous pair')
    sources = _Sources(source_root); prepared = []; seen_ids = set(); seen_instances = set()
    execution_paths = set(); execution_hashes = set(); cost_protocols = {}
    for pair, prior in [(p, True) for p in previous['pairs']] + [(p, False) for p in added_pairs]:
        p, executions = _prepare_pair(pair, provenance.get(pair['evidence_id']) if prior else None, sources)
        if p['evidence_id'] in seen_ids or p['independent_instance_key'] in seen_instances:
            raise ValueError('Duplicate evidence or repeated instance is not an independent pair')
        for path, digest in executions:
            if path in execution_paths or digest in execution_hashes:
                raise ValueError('Original execution source reused as another sample')
            execution_paths.add(path); execution_hashes.add(digest)
        seen_ids.add(p['evidence_id']); seen_instances.add(p['independent_instance_key'])
        if p['cost_compatible']:
            groups = cost_protocols.setdefault(task_window.key(p['goal']), set())
            groups.add(p['protocol_group'])
            if len(groups) > 1:
                raise ValueError('Same-goal cost protocols differ; explicitly exclude incompatible cost')
        prepared.append(p)
    new_state = task_window.fit(prepared, phase='development')
    data = (json.dumps(new_state, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    if _sha(previous_path.read_bytes()) != previous_sha256:
        raise ValueError('Previous artifact changed during preparation')
    new_sha = _sha(data)
    # x prevents overwrite even if a competing caller creates the path meanwhile.
    with output_path.open('xb') as stream:
        stream.write(data)
    updated = _binding(output_path, new_sha); frozen = _binding(previous_path, previous_sha256)
    check = read_snapshot(binding=updated, goal=next_goal, session_id=next_session_id, phase='development')
    before = task_window.select(previous, next_goal); after = check['decision']
    receipt = {
        'version': 'task_window_snapshot_v1', 'phase': phase,
        'scope': 'CPU preparation/read check only; no native Session or robot execution',
        'old_sha256': previous_sha256, 'new_sha256': new_sha,
        'frozen_binding': frozen, 'updated_binding': updated,
        'added_evidence_ids': [p['evidence_id'] for p in added_pairs],
        'previous_pair_count': len(previous['pairs']), 'new_pair_count': len(prepared),
        'verified_source_files': len(sources.data), 'next_session_read_check': check,
        'validation_scope': ('SHA-bound original RESULT/STATUS/episode facts, development phase, '
                             'actual input/window and paired initial-image consistency; '
                             'not a proof of physical truth, narrative correctness or skill quality'),
        'previous_decision': before, 'next_decision': after,
        'decision_changed': before != after,
        'selected_mode_changed': before['selected_mode'] != after['selected_mode'],
        'cost_excluded_evidence': [{'evidence_id': p['evidence_id'], 'protocol_group': p['protocol_group'],
                                    'reason': 'Caller explicitly declared cost_compatible false'}
                                   for p in prepared if not p['cost_compatible']],
        'cpu_process_seconds': time.process_time() - cpu_started,
        'preparation_wall_seconds': time.perf_counter() - started,
        'native_session_executed': False, 'default_changed': False,
        'measured_cost_saving_seconds': None, 'learning_gain_established': False,
    }
    return {'updated_binding': updated, 'frozen_binding': frozen, 'receipt': receipt}
