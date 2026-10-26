"""Reference for parent integration; metadata-only, plans are not execution evidence."""
import hashlib
import json
import os
from pathlib import Path

EXCLUDED_DIRS = {
    'code', 'tests', 'fixtures', 'testdata', '__pycache__', 'models',
    'tokenizer_cache', 'initial_experience', 'accepted_experience', 'revisions',
    'memory', 'sessions', 'skills', 'stage_trace', 'images', 'observations',
    'service_calls', 'replay_workspace',
}
PLANNING_KEYS = {
    'planned_session', 'plan', 'roster', 'protocol', 'configuration', 'config',
    'candidates', 'initial_experience', 'accepted_experience', 'experience_before',
    'source', 'source_case', 'source_request',
}


def scan_actual_results(runs_root, cases):
    """Recursively find requested identities in actual RESULT receipts.

    The filename RESULT is an attempted-run receipt, not proof of success.
    Search all depths and nested records; no dependency on a stream directory.
    ROSTER/PLAN/PROTOCOL are excluded. Copied source/experience and fixtures are
    excluded. Unknown suite identities are returned as ambiguous, never inferred
    from a numeric task ID. Caller must reject parse errors or ambiguity.
    """
    root = Path(runs_root)
    wanted = {(x['suite'], int(x['task_id']), int(x['init_state_id'])) for x in cases}
    pairs = {(t, i) for _, t, i in wanted}
    matches, ambiguous, errors, bindings = [], [], [], {}
    count = 0

    def integer(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdecimal():
            return int(value)
        return None

    def walk(value, pointer='', inherited_suite=None):
        if isinstance(value, list):
            for n, item in enumerate(value):
                yield from walk(item, pointer + '/' + str(n), inherited_suite)
        elif isinstance(value, dict):
            suite = next((value[k] for k in ('suite', 'suite_id', 'benchmark_name')
                          if isinstance(value.get(k), str)), inherited_suite)
            benchmark = value.get('benchmark')
            if isinstance(benchmark, dict) and isinstance(benchmark.get('suite_id'), str):
                suite = benchmark['suite_id']
            task = integer(value.get('task_id', value.get('task_index')))
            init = integer(value.get('init_state_id', value.get('instance_id')))
            if (task, init) in pairs:
                yield {'pointer': pointer or '/', 'suite': suite, 'task_id': task,
                       'init_state_id': init, 'status': value.get('status')}
            for key, item in value.items():
                if key not in PLANNING_KEYS and isinstance(item, (dict, list)):
                    escaped = key.replace('~', '~0').replace('/', '~1')
                    yield from walk(item, pointer + '/' + escaped, suite)

    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith('.')]
        if 'RESULT.json' not in files:
            continue
        path = Path(folder) / 'RESULT.json'
        relative = str(path.relative_to(root))
        try:
            raw = path.read_bytes()
            document = json.loads(raw)
            count += 1
            found = list(walk(document))
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            errors.append({'path': relative, 'error': str(exc)})
            continue
        if found:
            bindings[relative] = hashlib.sha256(raw).hexdigest()
        for row in found:
            record = {'path': relative, **row}
            if (row['suite'], row['task_id'], row['init_state_id']) in wanted:
                matches.append(record)
            elif row['suite'] is None:
                ambiguous.append(record)
    return {'result_files_read': count, 'matches': matches, 'ambiguous': ambiguous,
            'errors': errors, 'source_sha256': bindings,
            'scope': 'Own actual RESULT receipts including unsuccessful attempts; no plans; not an external/pretraining exposure claim.'}
