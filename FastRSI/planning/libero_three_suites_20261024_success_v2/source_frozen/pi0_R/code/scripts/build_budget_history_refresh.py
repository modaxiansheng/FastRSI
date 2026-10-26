"""Build a new SHA-bound historical fixed10/fixed5 catalog without GPU work.

Original run paths remain authoritative. Repeated initial conditions retain
their paid execution costs but are not additional independent observations.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

VERSION = 'budget_history_refresh_v1'
OLD_TEMPLATE = 'planning/scoped_allocation_v1_20261003/PROTOCOL.json'
OUTPUT = 'planning/budget_history_refresh_v1_20261003'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


class Sources:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = {}

    def path(self, relative):
        path = Path(relative)
        require(not path.is_absolute(), 'Source must be project relative')
        result = (self.root / path).resolve()
        require(result.is_relative_to(self.root), 'Source escaped project')
        return result

    def read(self, relative):
        path = self.path(relative)
        data = path.read_bytes()
        key = path.relative_to(self.root).as_posix()
        digest = hashlib.sha256(data).hexdigest()
        require(key not in self.manifest or self.manifest[key] == digest, 'Source changed: ' + key)
        self.manifest[key] = digest
        return data

    def json(self, relative):
        return json.loads(self.read(relative))

    def binding(self, relative):
        self.read(relative)
        return self.manifest[self.path(relative).relative_to(self.root).as_posix()]

    def verify(self):
        for relative, expected in self.manifest.items():
            require(hashlib.sha256(self.path(relative).read_bytes()).hexdigest() == expected,
                    'Source changed after audit: ' + relative)


def finite_positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0, 'Invalid ' + name)
    return value


def proposal_specs(sources, template):
    specs = copy.deepcopy(template['budget_history_specs'])
    require(len(specs) == 9, 'Expected immutable nine-spec source template')
    for spec in specs:
        require(spec['cost_compatible'] is False, 'Unexpected old cost eligibility')
    additions = [
        {'path': 'runs/action_window_v1_20261003/pair0', 'arms': ['N', 'F'],
         'record_schema': 'action_window_v1',
         'protocol_group': 'legacy_quality_only_action_window_v1_20261003',
         'protocol': 'runs/action_window_v1_20261003/PROTOCOL.json'},
        *[{'path': 'runs/short_exit_v1_20261003/pair' + str(i), 'arms': ['N', 'F'],
           'protocol_group': 'fresh_case_cold_terminal_view_bound_destination',
           'protocol': 'runs/short_exit_v1_20261003/PROTOCOL.json'} for i in (0, 1)],
    ]
    for host in ('187', '188'):
        for index in (0, 1):
            base = 'runs/structural_window_v1_20261003/host' + host
            path = base + '/N/pair' + str(index)
            additions.append({'path': path, 'arms': ['N', 'F'],
                'arm_paths': {arm: base + '/' + arm + '/pair' + str(index) + '/' + arm for arm in ('N', 'F')},
                'protocol_group': 'resident_case_hot_terminal_view_bound_destination',
                'protocol': 'runs/structural_window_v1_20261003/PROTOCOL.json'})
    for spec in additions:
        protocol = spec.pop('protocol')
        spec.update(cost_compatible=False, protocol_sources={protocol: sources.binding(protocol)})
        specs.append(spec)
    require(len(specs) == 16, 'Expected sixteen development pairs')
    return specs


def inspect_arm(sources, relative, expected_mode, legacy):
    before = set(sources.manifest)
    status_path = relative + '/STATUS.json'
    result_path = relative + '/stream/RESULT.json'
    status, result = sources.json(status_path), sources.json(result_path)
    require(status['status'] == result['status'] == 'COMPLETED', 'Incomplete original execution: ' + relative)
    require(len(result['cases']) == 1, 'Cost must bind one original case: ' + relative)
    row = result['cases'][0]; case = row['case']
    require(row['chain_status'] == 'PASSED', 'Unresolved chain: ' + relative)
    require(case['mode'] == 'development' and case['suite'] == 'libero_10' and case['max_steps'] == 520,
            'Not an eligible development scope: ' + relative)
    for key in ('task_id', 'init_state_id'):
        require(type(case[key]) is int and case[key] >= 0, 'Invalid instance identity')
    require(len(row['native_terminal_by_session']) == 1, 'Multi-Session pair needs separate audit')
    fact = next(iter(row['native_terminal_by_session'].values()))
    require(type(row['environment_success']) is bool and row['environment_success'] == fact['native_success'],
            'Native outcome disagreement')
    model_path = relative + '/MODEL_CONFIG.json'
    model_sha = sources.binding(model_path)
    events_path = relative + '/stream/OS_COST_EVENTS.jsonl'
    events = [json.loads(line) for line in sources.read(events_path).splitlines() if line.strip()]
    inputs = [event for event in events if event.get('event') == 'actual_policy_input']
    require(inputs and all(event['prompt'] == case['task_description'] for event in inputs), 'Actual prompt changed')
    runtime = relative + '/stream/' + case['id'] + '/workspace/artifacts/runtime/' + case['id']
    episode_path = runtime + '/episode.json'; episode = sources.json(episode_path)
    require(episode['success'] == fact['native_success'] and episode['num_steps'] == fact['steps'],
            'Episode/native fact mismatch')
    chunks = episode['metadata']['final_status']['episode_summary']['chunks']
    width = 10 if expected_mode == 'fixed10' else 5
    require(len(chunks) == len(inputs) and sum(c['executed_steps'] for c in chunks) == fact['steps'],
            'Actual chunk counts/steps disagree')
    require(all(type(c['requested_steps']) is int and 0 < c['requested_steps'] <= width
                and c['action_shape'] == [c['requested_steps'], 7]
                and 0 <= c['executed_steps'] <= c['requested_steps'] for c in chunks),
            'Submitted action shape/window mismatch')
    choices = [event for event in events if event.get('event') ==
               ('action_window_choice' if legacy else 'feedback_window_choice')]
    if legacy and expected_mode == 'fixed10':
        require(not choices and not row.get('action_window_decision'), 'Unexpected legacy native override')
        require(all(c['requested_steps'] == 10 for c in chunks), 'Legacy native chunk is not fixed10')
        mode_proof = 'original_episode_chunks_and_actual_policy_inputs'
    else:
        require(len(choices) == len(inputs), 'Missing window choices')
        require(all(e['original_steps'] == 10 and e['submitted_steps'] == c['requested_steps']
                    for e, c in zip(choices, chunks)), 'Choice/episode prefix mismatch')
        if legacy:
            require(row['action_window_decision']['action'] == 'short_window', 'Legacy candidate differs')
            mode_proof = 'original_action_window_choices_and_episode_chunks'
        else:
            actual = row.get('task_window_decision', row.get('window_learning_decision', {})).get(
                'selected_mode', case.get('feedback_window_mode'))
            require(actual == expected_mode and all(e['mode'] == expected_mode for e in choices),
                    'Actual fixed-window mode differs')
            mode_proof = 'original_feedback_window_choices_and_episode_chunks'
    images = sorted(sources.path(runtime + '/rgb').glob('initial_*.png'))
    require(images, 'Missing initial images')
    image_hashes = {p.name: sources.binding(p.relative_to(sources.root).as_posix()) for p in images}
    verdicts = [o.get('verdict') for o in row.get('outcomes', [])]
    alignment = ('aligned' if len(verdicts) == 1 and verdicts[0] in ('success', 'failure')
                 and (verdicts[0] == 'success') == fact['native_success'] else 'conflict_or_unresolved')
    return {
        'path': relative, 'mode': expected_mode, 'mode_proof': mode_proof,
        'case': {key: case[key] for key in ('id', 'mode', 'suite', 'task_id', 'init_state_id', 'task_description', 'max_steps')},
        'native_success': fact['native_success'], 'steps': fact['steps'], 'vla_calls': len(inputs),
        'semantic_verdicts': verdicts, 'semantic_alignment': alignment,
        'admission': row.get('experience_admission'), 'chain_status': row['chain_status'],
        'full_wall_seconds': finite_positive(status['full_wall_seconds'], 'arm lifecycle'),
        'started_at': finite_positive(status['started_at'], 'arm start'),
        'finished_at': finite_positive(status['finished_at'], 'arm finish'),
        'original_status': status, 'model_config_sha256': model_sha, 'initial_image_sha256': image_hashes,
        'sources': {key: sources.manifest[key] for key in set(sources.manifest)-before},
    }


def inspect_pair(sources, spec):
    require(sources.path(spec['path']).is_dir(), 'Pair identity must be an existing original directory')
    for path, digest in spec['protocol_sources'].items():
        require(sources.binding(path) == digest, 'Protocol SHA changed: ' + path)
    arms = spec.get('arms', ['N', 'F'])
    require(len(arms) == 2 and len(set(arms)) == 2, 'Exactly two actual arms required')
    arm_paths = spec.get('arm_paths', {a: spec['path'] + '/' + a for a in arms})
    require(set(arm_paths) == set(arms), 'Explicit source arm mapping differs')
    legacy = spec.get('record_schema') == 'action_window_v1'
    records = [inspect_arm(sources, arm_paths[a], mode, legacy) for a, mode in zip(arms, ('fixed10', 'fixed5'))]
    a, b = records
    identity_keys = ('suite', 'task_id', 'init_state_id', 'task_description', 'mode', 'max_steps')
    require(all(a['case'][k] == b['case'][k] for k in identity_keys), 'Pair identities differ')
    require(a['model_config_sha256'] == b['model_config_sha256'], 'Pair model config differs')
    require(a['initial_image_sha256'] == b['initial_image_sha256'], 'Pair initial images differ')
    if legacy:
        artifact_path = spec['path'] + '/STAGE_FLOW.json'; artifact = sources.json(artifact_path)
        require(artifact['operation'] == 'execute_prefix' and artifact['steps'] == 5
                and artifact['goal'] == a['case']['task_description']
                and artifact['scope'] == {'suite': 'libero_10', 'task_id': a['case']['task_id'],
                    'max_steps': 520, 'model_sha': a['model_config_sha256']}, 'Legacy artifact scope differs')
        original = sources.json(b['path'] + '/stream/RESULT.json')['cases'][0]
        require(original['action_window_decision']['sha256'] == sources.binding(artifact_path), 'Legacy artifact SHA differs')
        spec['protocol_sources'][artifact_path] = sources.binding(artifact_path)
    hot = 'arm_paths' in spec
    view = 'terminal_view' if hot or 'short_exit' in spec['path'] else 'historical_report'
    relation = ('win' if b['native_success'] and not a['native_success'] else
                'harm' if a['native_success'] and not b['native_success'] else 'equal')
    observed = max(a['finished_at'], b['finished_at'])
    return {'evidence_id': spec['path'], 'spec': spec, 'arms': records,
        'independent_instance_key': 'libero_10/task%d/init%d' % (a['case']['task_id'], a['case']['init_state_id']),
        'task_id': a['case']['task_id'], 'init_state_id': a['case']['init_state_id'],
        'goal': a['case']['task_description'], 'observed_at': observed, 'observed_at_utc': utc(observed),
        'native_quality_relation': relation, 'pair_arm_lifecycle_seconds': a['full_wall_seconds'] + b['full_wall_seconds'],
        'same_initial_images': True, 'same_model_config': True, 'actual_window_trace_checked': True,
        'cost_scope': 'resident_case_hot_lifecycle' if hot else 'cold_per_arm_lifecycle',
        'cost_compatible_with_current_continuous_F': False,
        'cost_incompatibility_reasons': [
            'isolated_initial_experience_not_continuous_admitted_workspace',
            'per_case_model_startup_included' if not hot else 'pool_startup_excluded_from_case_lifecycle',
            'report_protocol_' + view + '_not_identical_whole_continuous_stream',
        ],
        'quality_scope': 'Historical exact-task numeric evidence; not proof of current continuous-F quality. '
                         'Quarantined narrative remains excluded; no cross-task transfer inferred.'}


def shared_service_costs(sources, records):
    rows = []
    for host in ('187', '188'):
        for arm in ('N', 'F'):
            branch = 'runs/structural_window_v1_20261003/host' + host + '/' + arm
            status = sources.json(branch + '/STATUS.json')
            selected = [r for pair in records for r in pair['arms'] if r['path'].startswith(branch + '/pair')]
            require(len(selected) == 2, 'Expected two selected hot cases per structural branch')
            require({r['path'].split('structural_window_v1_20261003/')[1] for r in selected} ==
                    {r['path'] for r in status['groups']}, 'Pool bill covers unexpected cases')
            hot = math.fsum(r['full_wall_seconds'] for r in selected)
            full = finite_positive(status['full_wall_seconds'], 'structural branch cost')
            startup = finite_positive(status['pool_startup_seconds'], 'pool startup')
            require(full >= hot + startup, 'Branch accounting inconsistent')
            rows.append({'branch': branch, 'status': status['status'],
                'full_branch_wall_seconds': full, 'included_hot_arm_lifecycle_seconds': hot,
                'included_pool_startup_seconds': startup, 'other_branch_overhead_seconds': full-hot-startup,
                'increment_over_already_counted_hot_cases_seconds': full-hot,
                'cost_rule': 'Full branch replaces its two hot case costs; do not add all three.',
                'source_sha256': sources.binding(branch + '/STATUS.json')})
    return rows


def build(project, output):
    sources = Sources(project)
    template = sources.json(OLD_TEMPLATE)
    records = [inspect_pair(sources, spec) for spec in proposal_specs(sources, template)]
    records.sort(key=lambda r: (r['observed_at'], r['evidence_id']))
    first = {}; executions = set()
    for record in records:
        key = record['independent_instance_key']
        original = first.setdefault(key, record)
        repeated = original is not record
        record.update(is_independent_initial_condition=not repeated,
                      first_observed_at=original['observed_at'], first_observed_at_utc=original['observed_at_utc'],
                      first_evidence_id=original['evidence_id'], repeat_of=original['evidence_id'] if repeated else None)
        if repeated:
            record['spec']['repeat_of'] = original['evidence_id']
        for arm in record['arms']:
            require(arm['path'] not in executions, 'Same paid arm included twice')
            executions.add(arm['path'])
    require(len(records) == 16 and len(first) == 13, 'Historical scope differs from sixteen/ thirteen audit')
    require(sum(r['repeat_of'] is not None for r in records) == 3, 'Expected three repeated-instance pairs')
    branches = shared_service_costs(sources, records)
    pairs_paid = math.fsum(r['pair_arm_lifecycle_seconds'] for r in records)
    extra = math.fsum(b['increment_over_already_counted_hot_cases_seconds'] for b in branches)
    new_protocol = copy.deepcopy(template)
    new_protocol['budget_history_specs'] = [r['spec'] for r in records]
    new_protocol['budget_repeat_policy'] = 'retain'
    catalog = {'version': VERSION, 'status': 'AUDITED_CPU_ONLY',
        'old_template': {'path': OLD_TEMPLATE, 'sha256': sources.manifest[OLD_TEMPLATE]},
        'summary': {'paid_development_pairs': len(records), 'unique_initial_conditions': len(first),
            'unique_arm_executions': len(executions), 'repeated_initial_condition_pairs': 3,
            'all_selected_pair_arm_lifecycle_seconds': pairs_paid,
            'structural_hot_case_seconds_already_in_pair_bill': math.fsum(b['included_hot_arm_lifecycle_seconds'] for b in branches),
            'structural_pool_startup_seconds_once': math.fsum(b['included_pool_startup_seconds'] for b in branches),
            'structural_other_branch_overhead_seconds': math.fsum(b['other_branch_overhead_seconds'] for b in branches),
            'structural_full_branch_seconds': math.fsum(b['full_branch_wall_seconds'] for b in branches),
            'selected_history_lifecycle_seconds_including_structural_shared_services_once': pairs_paid + extra,
            'native_failed_arm_paths': [a['path'] for r in records for a in r['arms'] if not a['native_success']],
            'quarantined_arm_paths': [a['path'] for r in records for a in r['arms'] if a['admission']['status'] == 'QUARANTINED'],
            'task0_pairs': [r['evidence_id'] for r in records if r['task_id'] == 0],
            'current_cost_compatible_pairs': 0},
        'records': records, 'shared_service_costs': branches,
        'proposed_specs': new_protocol['budget_history_specs'],
        'source_sha256': sources.manifest,
        'scope': 'Historical fixed10/fixed5 development evidence only. Original evaluation action_window/pair1 excluded. '
                 'No copy or rename of run evidence; no new P/F pairing. Repeat observations are paid executions, '
                 'not independent initial conditions and cannot reset earliest-win/followup chronology.',
        'accounting': 'Sum of selected arm lifecycles plus structural pool/branch overhead counted once. '
                      'Not parallel wall clock, GPU kernel time, current cold-start cost or complete project bill. '
                      'Other R/S study arms and unrelated diagnostics are outside this selected candidate catalog. '
                      'Different cost protocols are not pooled for fitting.',
        'quality': 'Native outcomes, semantic verdicts and admission states remain separate. No narrative relabeling, '
                   'candidate promotion, learned benefit or contextual generalization is established.',
        'source_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    sources.verify()
    output = Path(output).resolve()
    require(output.is_relative_to(sources.root), 'Output must remain in project')
    require(not (output/'CATALOG.json').exists() and not (output/'PROTOCOL.json').exists(), 'Output version already exists')
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('CATALOG.json', catalog), ('PROTOCOL.json', new_protocol)]:
        with (output/name).open('x') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.project.resolve(); output = args.output or root/OUTPUT
    catalog = build(root, output)
    print(json.dumps({'output': str(output), 'summary': catalog['summary'],
                      'bound_source_files': len(catalog['source_sha256'])}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
