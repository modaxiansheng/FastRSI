"""Run the deployed allocation entrypoint on old/refreshed paid evidence.

CPU only: no worker/executor, no simulation of unexecuted branch outcomes.
Proposals diagnose allocation behavior, not held-out quality or saved runtime.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

from window_budget_continuation import continuation_scoped


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def run(root, output):
    root = root.resolve()
    output = (root / output).resolve()
    if not output.is_relative_to(root):
        raise ValueError('Output must remain inside project')
    output.mkdir(parents=True, exist_ok=False)
    started, cpu = time.time(), time.process_time()
    catalog_path = root / 'planning/budget_history_refresh_v1_20261003/CATALOG.json'
    catalog = json.loads(catalog_path.read_text())
    sources = dict(catalog['source_sha256'])
    sources[str(catalog_path.relative_to(root))] = sha(catalog_path)
    templates = {
        'old': 'planning/scoped_allocation_v1_20261003/PROTOCOL.json',
        'refreshed': 'planning/budget_history_refresh_v1_20261003/PROTOCOL.json',
    }
    for relative in templates.values():
        sources[relative] = sha(root / relative)
    for relative, expected in sources.items():
        if sha(root / relative) != expected:
            raise ValueError('Historical source changed: ' + relative)
    tasks = {}
    for record in catalog['records']:
        tasks.setdefault(record['task_id'], []).append(record)
    proposals = []
    for task, records in sorted(tasks.items()):
        seen = {r['init_state_id'] for r in records}
        initial = next(i for i in range(50) if i not in seen)
        case = {k: records[0]['arms'][0]['case'][k] for k in
                ('mode', 'suite', 'task_id', 'task_description', 'max_steps')}
        case.update(id=f'history_probe_t{task}_i{initial}', init_state_id=initial)
        proposals.append(case)
    write(output / 'PROPOSALS.json', {
        'cases': proposals,
        'scope': 'Predeclared CPU allocation probes; no claim that initial IDs are unseen project-wide.'})
    results = []
    for case in proposals:
        comparison = {'task_id': case['task_id'], 'goal': case['task_description']}
        for label, relative in templates.items():
            protocol = copy.deepcopy(json.loads((root / relative).read_text()))
            protocol['first_batch'] = [case]
            group = output / f'task{case["task_id"]}' / label
            group.mkdir(parents=True)
            write(group / 'PROTOCOL.json', protocol)
            model = root / tasks[case['task_id']][0]['arms'][0]['path'] / 'MODEL_CONFIG.json'
            (group / 'MODEL_CONFIG.json').write_bytes(model.read_bytes())
            write(group / 'STATUS.json', {'started_at': time.time(), 'status': 'ALLOCATION_CHECK_ONLY'})
            decision_started = time.perf_counter()
            receipt = continuation_scoped(root, group, protocol, 0)
            duration = time.perf_counter() - decision_started
            write(group / 'CONTINUATION_0.json', receipt)
            comparison[label] = {
                'action': receipt['decision']['action'],
                'reason': receipt['allocation_reason'],
                'allocation_status': receipt['status'],
                'evidence_count': len(receipt['ledger']['records']),
                'independent_count': len({r['init_id'] for r in receipt['ledger']['records']}),
                'paid_source_seconds': receipt['ledger']['paid_source_seconds'],
                'decision_wall_seconds': duration,
                'receipt': str((group / 'CONTINUATION_0.json').relative_to(root)),
            }
        comparison['reason_changed'] = comparison['old']['reason'] != comparison['refreshed']['reason']
        results.append(comparison)
    # These are narrow expected consequences of the imported historical facts.
    by_task = {row['task_id']: row for row in results}
    assert by_task[8]['old']['reason'] == 'no_evidence_bootstrap_requires_separate_research_plan'
    assert by_task[8]['refreshed']['reason'] == 'observed_candidate_quality_harm'
    assert by_task[0]['refreshed']['reason'] == 'no_comparable_both_success_cost_evidence'
    assert by_task[4]['refreshed']['reason'] == 'paid_quality_followup_limit_reached'
    assert all(row[label]['allocation_status'] == 'NOT_DISPATCHED'
               for row in results for label in templates)
    for relative, expected in sources.items():
        assert sha(root / relative) == expected, relative
    result = {
        'status': 'PASSED', 'version': 'budget_history_refresh_v1',
        'results': results, 'history_summary': catalog['summary'],
        'source_sha256': sources, 'started_at': started, 'finished_at': time.time(),
        'wall_seconds': time.time() - started, 'cpu_seconds': time.process_time() - cpu,
        'worker_dispatches': 0, 'model_calls': 0, 'robot_episodes': 0,
        'measured_saving_seconds': None, 'learning_gain_established': False,
        'scope': 'Actual allocation function with real paid history; CPU diagnostic only. Both old and refreshed gates decline all proposals. Changed reasons are not saved experiments.',
    }
    write(output / 'RESULT.json', result)
    return {k: result[k] for k in ('status', 'wall_seconds', 'cpu_seconds', 'worker_dispatches')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output)))
