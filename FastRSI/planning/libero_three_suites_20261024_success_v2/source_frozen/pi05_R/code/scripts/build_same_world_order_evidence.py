"""Bind completed order observations; never generate a label or run a model.

Collection verifies pinned transfer manifests, actual action/input ledgers, and
the admitted workspace seal. Deployment may explicitly validate only the copied
receipt's trusted SHA and strict schema; original source paths need not exist.
"""
import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path

VERSION = 'same_world_order_evidence_v1'
JOINT_GOAL = 'put the cream cheese in the bowl and put the bowl on the plate'
STAGES = {'goal6': 'put the cream cheese in the bowl',
          'goal8': 'put the bowl on the plate'}
ORDERS = {'order68': ['goal6', 'goal8'], 'order86': ['goal8', 'goal6']}
GOALS = [['on', 'cream_cheese_1', 'akita_black_bowl_1'],
         ['on', 'akita_black_bowl_1', 'plate_1']]
DIAGNOSTIC_SCOPE = 'native_oracle_stage_feedback_operation_value_upper_bound_not_learning_or_full_agent'
TARGET_SCOPE = 'same_world_development_stream_not_official_libero_episode'
DIAGNOSTIC_MANIFEST_SHA256 = '6e9f6b6ffc64ae6e82a286642a06641cbc38f20ffb1ee5c2f7be8421e32abc98'
NATIVE_MANIFEST_SHA256 = 'c602c8146a3e6f1090bbf125545591e7447e2d202938bd2c8fe42a4b126ae68f'
LIMITATIONS = [
    'Observed completed development runs only; no unexecuted branch labels.',
    'Simulator stage feedback is shared oracle feedback, not learned recognition.',
    'Policy diagnostics and full native admission are separate evidence layers.',
    'No current pre-reset visual observation or initial-state feature is supplied.',
    'No learning efficacy, transferable skill, official score or net speedup claim.',
    'Cost scopes overlap; components must not be added to inclusive wall times.',
]
COST_SCOPES = {
    'policy_operation_diagnostic': {
        'worker_wall': 'Inclusive worker: reset, policy, environment, oracle and trace; excludes model startup and parent scheduling',
        'script_wall': 'Inclusive diagnostic script wall; contains worker; excludes model startup and parent scheduling',
        'policy_inference': 'Component within worker wall; nonadditive',
        'environment_action': 'Component within worker wall; nonadditive',
        'environment_reset': 'Component within worker wall, including settling actions; nonadditive'},
    'full_native_admitted': {
        'case_wall': 'Inclusive full native case: common construction, execution, verification, original reporting and shutdown; excludes admission, model startup and parent scheduling',
        'worker_wall': 'Inclusive worker with model/service startup; contains case wall',
        'outer_controller_wall': 'Inclusive controller wall; contains worker; not additive',
        'policy_inference': 'Component within case wall; nonadditive',
        'environment_action': 'Component within case wall; nonadditive',
        'environment_reset': 'Component within case wall, including settling actions; nonadditive',
        'experience_admission': 'After recorded case wall, within worker and outer wall; not additive to those inclusive walls'}
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def digest(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')
    return value


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'Cost must be finite and nonnegative')
    return value


def equal_number(a, b, name, tolerance=1e-6):
    require(abs(number(a) - number(b)) <= tolerance, 'Cost mismatch: ' + name)


class Sources:
    """Only explicitly requested source files are read, not archive trees."""
    def __init__(self, root, manifest_name, expected_sha256):
        self.root = Path(root).resolve()
        self.files = {}
        self.manifest_name = manifest_name
        self.expected_sha256 = digest(expected_sha256)
        manifest_path = self.root / manifest_name
        require(manifest_path.is_file(), 'Missing transfer manifest')
        require(sha256(manifest_path) == expected_sha256, 'Transfer manifest SHA mismatch')
        self._remember(manifest_name, manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if isinstance(manifest['files'], list):
            self.index = {row['path']: row['sha256'] for row in manifest['files']}
            require(len(self.index) == len(manifest['files']), 'Duplicate manifest paths')
        else:
            self.index = {key: row['sha256'] for key, row in manifest['files'].items()}

    def _path(self, relative):
        require(isinstance(relative, str) and not Path(relative).is_absolute(), 'Relative source path required')
        path = (self.root / relative).resolve()
        require(self.root in path.parents, 'Source path escapes evidence root')
        require(path.is_file(), 'Missing source: ' + relative)
        return path

    def _remember(self, relative, path):
        self.files[relative] = {'sha256': sha256(path), 'bytes': path.stat().st_size}

    def file(self, relative, manifested=True):
        path = self._path(relative)
        actual = sha256(path)
        if manifested:
            require(relative in self.index, 'Source absent from verified manifest: ' + relative)
            require(actual == self.index[relative], 'Source SHA mismatch: ' + relative)
        self._remember(relative, path)
        return path

    def json(self, relative, manifested=True):
        return json.loads(self.file(relative, manifested).read_text())

    def lines(self, relative):
        return [json.loads(line) for line in self.file(relative).read_text().splitlines() if line.strip()]

    def provenance(self):
        return {'root': str(self.root), 'manifest': self.manifest_name,
                'manifest_sha256': self.expected_sha256, 'files': copy.deepcopy(self.files)}


def registration(reg):
    require(reg['version'] == 2 and reg['suite'] == 'libero_goal', 'Expected v2 LIBERO_goal')
    require(reg['anchor_task_id'] == 6 and reg['init_state_id'] == 0 and reg['seed'] == 7,
            'Source development protocol changed')
    require(reg['total_steps'] == 600 and reg['replan_steps'] == 10, 'Source budget changed')
    require(reg['arm_orders'] == ORDERS, 'Registered orders changed')
    require({row['key']: row['language'] for row in reg['tasks']} == STAGES, 'Task semantics changed')
    require(reg['final_goal_state_expected'] == GOALS, 'Native goals changed')
    require(len({row['world_signature'] for row in reg['tasks']}) == 1, 'Different world definitions')
    return {'joint_goal': JOINT_GOAL, 'stages': copy.deepcopy(STAGES),
            'native_goal_predicates': copy.deepcopy(GOALS)}


def native_definitions(reg, rows):
    require(len(rows) == 2, 'Missing native definition bindings')
    by_key = {row['key']: row for row in rows}
    require(set(by_key) == set(STAGES), 'Native definition keys changed')
    for task, goal in zip(reg['tasks'], GOALS):
        row = by_key[task['key']]
        require(row['bddl_sha256'] == task['bddl_sha256'] and
                row['world_signature'] == task['world_signature'] and
                row['benchmark_language'] == task['language'], 'Native definition mismatch')
        require(row['native_goal_state'] == [goal], 'Parsed native goal mismatch')


def final_state(value, step):
    require(value['joint_success'] is True and
            value['goals'] == {'goal6': True, 'goal8': True}, 'Final joint native goals not satisfied')
    before, after = value['same_state_before'], value['same_state_after']
    require(before == after, 'Final goal evaluations changed physical state')
    require(before.get('physical_steps', before.get('step')) == step,
            'Final predicate evaluation step mismatch')
    digest(before['sim_state_sha256'])


def actual_actions(actions, expected):
    require(isinstance(actions, list) and len(actions) == expected, 'Executed actions/count mismatch')
    require(all(isinstance(row, list) and len(row) == 7 and
                all(type(v) in (int, float) and math.isfinite(v) for v in row)
                for row in actions), 'Invalid physical action rows')


def cost(seconds, scope):
    return {'seconds': number(seconds), 'scope': scope}


def diagnostic_record(sources, reg, arm):
    base = arm + '/result/'
    result = sources.json(base + 'RESULT.json')
    events = sources.lines(base + 'EVENTS.jsonl')
    require(result['status'] == 'COMPLETED' and result['success'] is True and
            result['scope'] == DIAGNOSTIC_SCOPE and result['target_scope'] == TARGET_SCOPE,
            'Diagnostic is not a completed operation-value observation')
    require(result['arm'] == arm and result['registered_sequence'] == ORDERS[arm], 'Wrong diagnostic order')
    require(result['registration_sha256'] == sources.files['REGISTRATION.json']['sha256'],
            'Diagnostic registration SHA mismatch')
    require(result['qwen_calls'] == 0 and result['learning_updates'] == 0 and
            result['environment_resets'] == 1 and not result['exceptions'], 'Diagnostic scope changed')
    native_definitions(reg, result['native_definition_bindings'])
    require({k: result['policy_metadata'][k] for k in reg['policy_metadata_expected']} ==
            reg['policy_metadata_expected'], 'Diagnostic policy metadata mismatch')
    activations = [row for row in events if row['event'] == 'stage_activation']
    calls = [row for row in events if row['event'] == 'policy_call']
    executed = [row for row in events if row['event'] == 'execution']
    require([row['stage'] for row in activations] == ORDERS[arm] and
            [{k: v for k, v in row.items() if k != 'event'} for row in activations] == result['transitions'],
            'Actual diagnostic sequence differs')
    require(len(calls) == len(executed) == result['policy_calls'] == result['policy_attempts'],
            'Diagnostic actual policy count mismatch')
    step = 0
    for ordinal, (call, action) in enumerate(zip(calls, executed), 1):
        require(call['ordinal'] == action['ordinal'] == ordinal and
                call['stage'] == action['stage'] and call['prompt'] == STAGES[call['stage']],
                'Policy input/action binding mismatch')
        require(call['step_before'] == action['step_before'] == step, 'Noncontiguous diagnostic actions')
        require(call['predicted_horizon'] == 10 and len(call['submitted_actions']) <= 10,
                'Diagnostic H10/W10 changed')
        actual_actions(call['predicted_actions'], 10)
        actual_actions(call['submitted_actions'], len(call['submitted_actions']))
        actual_actions(action['actual_actions'], action['executed_steps'])
        require(action['actual_actions'] == call['submitted_actions'][:action['executed_steps']],
                'Actual actions differ from submitted prefix')
        step += action['executed_steps']
        require(step == action['step_after'], 'Diagnostic step delta mismatch')
    require(step == result['physical_steps'] and result['counted_target_action_steps'] ==
            step + result['reset_settling_steps'], 'Diagnostic total steps mismatch')
    require(result['reset_settling_steps'] == 10, 'Settling actions missing')
    final_state(result['final_native_evaluation'], step)
    for name, rows, key in [('policy_seconds', calls, 'policy_seconds'),
                            ('action_seconds', executed, 'action_seconds')]:
        equal_number(sum(number(row[key]) for row in rows), result['costs'][name], name)
    require('startup' in result['cost_scope'] and 'do not add' in result['cost_scope'], 'Diagnostic cost scope missing')
    require(result['script_wall_seconds'] >= result['worker_wall_seconds'] >=
            sum(number(v) for v in result['costs'].values()), 'Diagnostic inclusive wall cost inconsistency')
    for filename, image_sha in result['initial_images'].items():
        matches = [relative for relative in sources.index
                   if relative.startswith(base + 'stage_trace/') and Path(relative).name == filename]
        require(len(matches) == 1, 'Missing/ambiguous diagnostic image source')
        require(sha256(sources.file(matches[0])) == image_sha, 'Initial image mismatch')
    record = {'evidence_layer': 'policy_operation_diagnostic', 'task': registration(reg),
              'actual_sequence': ORDERS[arm], 'result': {
                  'joint_native_success': True, 'native_goal_results': {'goal6': True, 'goal8': True},
                  'policy_action_steps': step, 'reset_settling_steps': 10,
                  'policy_calls': len(calls), 'environment_resets': 1},
              'costs': {
                  'worker_wall': cost(result['worker_wall_seconds'], 'Inclusive worker: reset, policy, environment, oracle and trace; excludes model startup and parent scheduling'),
                  'script_wall': cost(result['script_wall_seconds'], 'Inclusive diagnostic script wall; contains worker; excludes model startup and parent scheduling'),
                  'policy_inference': cost(result['costs']['policy_seconds'], 'Component within worker wall; nonadditive'),
                  'environment_action': cost(result['costs']['action_seconds'], 'Component within worker wall; nonadditive'),
                  'environment_reset': cost(result['costs']['reset_seconds'], 'Component within worker wall, including settling actions; nonadditive')}}
    pairing = {'initial_state': copy.deepcopy(result['initial_state']),
               'initial_images': copy.deepcopy(result['initial_images'])}
    return record, pairing


def span_total(events, name):
    starts = {row['span_id']: row for row in events if row.get('event') == 'begin' and row.get('name') == name}
    ends = [row for row in events if row.get('event') == 'end' and row.get('name') == name]
    require(starts and len(starts) == len(ends), 'Missing/unclosed cost spans: ' + name)
    require(len({row['span_id'] for row in ends}) == len(ends), 'Repeated cost span')
    for row in ends:
        require(row['span_id'] in starts and row['status'] == 'ok', 'Failed/unbound cost span')
        equal_number((row['mono_ns'] - starts[row['span_id']]['mono_ns']) / 1e9,
                     row['duration_seconds'], name)
    return len(ends), sum(number(row['duration_seconds']) for row in ends)


def native_record(sources, reg):
    stream = 'F/stream/'
    campaign = sources.json(stream + 'RESULT.json')
    require(campaign['status'] == 'COMPLETED' and campaign['experience_admission_mode'] == 'guarded_v1'
            and len(campaign['cases']) == 1, 'Expected one completed guarded native case')
    row = campaign['cases'][0]
    case_id = row['case']['id']
    require(re.fullmatch('[A-Za-z0-9_-]+', case_id), 'Unsafe Session identifier')
    base = stream + case_id + '/'
    actual_row = sources.json(base + 'RESULT.json')
    require(actual_row == row, 'Campaign/case result differs')
    entry = sources.json(stream + 'stage_trace/' + case_id + '/ENTRY_REPORT.json')
    binding = sources.json(base + 'workspace/SAME_WORLD_BINDING.json')
    planned = sources.json(base + 'PLANNED_SESSION.json')
    audit = sources.json('NATIVE_EVIDENCE_AUDIT_V1.json', manifested=False)
    admission = sources.json(base + 'admission/ADMISSION.json')
    sealed_base = base + 'admission/accepted_experience/'
    seal = sources.json(sealed_base + '.FASTRSI_ADMITTED.json')
    require(row['status'] == row['chain_status'] == 'PASSED' and row['environment_success'] is True,
            'Native chain did not pass')
    require(row['case']['same_world_arm'] == binding['arm'] == 'order86' and
            binding['sequence'] == ORDERS['order86'] and binding['joint_goal'] == JOINT_GOAL,
            'Native fixed order binding differs')
    require(binding['registration_sha256'] == sources.files['SAME_WORLD_REGISTRATION.json']['sha256'],
            'Native registration SHA differs')
    require(planned == row['planned_session'] and planned['task_description'] == JOINT_GOAL and
            planned['execution']['max_steps'] == 600 and planned['execution']['replan_every_steps'] == 10,
            'Native planned inputs differ')
    require(entry['status'] == 'CLOSED' and entry['scope'] == TARGET_SCOPE and not entry['exceptions'],
            'Native Target entry did not close cleanly')
    require(entry['binding']['registered_sequence'] == binding['sequence'] and
            entry['binding']['registration_sha256'] == binding['registration_sha256'], 'Actual target binding differs')
    native_definitions(reg, entry['binding']['native_definition_bindings'])
    require([item['stage'] for item in entry['stream']['transitions']] == binding['sequence'],
            'Actual native transition order differs')
    target = sources.lines(stream + 'TARGET_COST_EVENTS.jsonl')
    os_events = sources.lines(stream + 'OS_COST_EVENTS.jsonl')
    admitted_target = [item for item in target if item.get('event') == 'same_world_target_admitted']
    require(len(admitted_target) == 1 and admitted_target[0]['binding'] == entry['binding'],
            'Target ledger/entry binding differs')
    observations = [item for item in target if item.get('event') == 'native_observation']
    resets = [item for item in observations if item.get('method') == 'reset']
    actions = [item for item in observations if item.get('method') == 'action_chunk']
    inputs = [item for item in os_events if item.get('event') == 'actual_policy_input']
    require(len(resets) == entry['environment_resets'] == entry['metrics']['reset_calls'] == 1,
            'Native reset count differs')
    require(len(actions) == len(inputs) == entry['metrics']['action_calls'], 'Actual native model/action count differs')
    metadata = [item for item in os_events if item.get('event') == 'same_world_policy_metadata']
    require(len(metadata) == 1 and metadata[0]['actual'] == audit['policy_metadata']['actual'],
            'Native model metadata audit differs')
    require({k: metadata[0]['actual'][k] for k in reg['policy_metadata_expected']} ==
            reg['policy_metadata_expected'], 'Actual native policy metadata differs')
    step = 0
    for input_row, action in zip(inputs, actions):
        require(input_row['case'] == action['session_id'] == case_id and
                input_row['prompt'] == STAGES[action['stage_before']], 'Native prompt/action stage binding differs')
        require(action['step_before'] == step and action['action_input_binding']['source_observation_id'] ==
                action['action_input_binding']['actual_observation_id'], 'Native stale observation/action binding')
        count = action['executed_steps']
        require(type(count) is int and 0 < count <= 10, 'Native execution exceeds W10')
        actual_actions(action['native_executed_actions'], count)
        actual_actions(action['requested_actions'], len(action['requested_actions']))
        require(len(action['requested_actions']) <= 10 and action['native_executed_actions'] ==
                action['requested_actions'][:count], 'Native action prefix mismatch')
        step += count
        require(action['steps'] == action['physical_binding']['physical_steps'] == step,
                'Native actual step delta differs')
        require(action['native_success'] == action['stream_native_success'] and
                action['native_success_scope'] == 'Current joint native predicates', 'Stage success confused with joint success')
    require(actions and actions[-1]['native_success'] is True and entry['stream_native_success'] is True,
            'Native final current-state success missing')
    require(step == entry['physical_steps'] == audit['physical_steps'] and
            entry['reset_settling_steps'] == 10, 'Native physical counts differ')
    final_state(entry['stream']['final_evaluation'], step)
    require(entry['stream']['final_evaluation'] == audit['final_evaluation'], 'Final audit differs from entry')
    require(admission['eligible'] is True and admission['status'] == seal['status'] == 'ADMITTED' and
            not admission['reasons'] and admission['source_case'] == seal['source_case'] == case_id,
            'Native experience not admitted')
    require(admission['source_files'] == seal['files'], 'Admission seal/source set differs')
    for relative, expected in seal['files'].items():
        require(sha256(sources.file(sealed_base + relative)) == expected, 'Sealed experience changed')
    require(len(admission['attempts']) == len(row['outcomes']) == 1, 'Native verification attempt ambiguity')
    attempt = admission['attempts'][0]
    require(attempt['session_id'] == case_id and attempt['runtime_success'] is True and
            attempt['protocol_code'] == 'verified' and attempt['verdict'] == 'success', 'Admitted verification differs')
    bundle = sources.json(sealed_base + attempt['bundle'])
    require(sources.files[sealed_base + attempt['bundle']]['sha256'] == attempt['bundle_sha256'] and
            bundle['runtime_result']['success'] is True and bundle['runtime_result']['num_steps'] == step,
            'Admitted runtime bundle differs')
    require(row['outcomes'][0]['code'] == 'verified' and row['outcomes'][0]['verdict'] == 'success' and
            row['experience_admission']['status'] == 'ADMITTED', 'Native verdict/admission row differs')
    status = sources.json('STATUS.json')
    require(status['status'] == status['arm']['status'] == 'COMPLETED', 'Native controller not completed')
    equal_number(audit['costs']['case_seconds'], row['seconds'], 'native case wall')
    equal_number(audit['costs']['outer_controller_wall_seconds'], status['wall_seconds'], 'outer wall')
    equal_number(audit['costs']['worker_wall_seconds'], status['arm']['full_wall_seconds'], 'worker wall')
    require(status['wall_seconds'] >= status['arm']['full_wall_seconds'] >= row['seconds'], 'Native wall cost scope inconsistency')
    policy_count, policy_seconds = span_total(os_events, 'policy_client')
    require(policy_count == len(inputs) == audit['vla_inputs'], 'Native policy span count differs')
    for name in ['os_execution', 'semantic_verification', 'native_terminal_report', 'agent_model_call', 'experience_admission']:
        count, seconds = span_total(os_events, name)
        require(count == audit['inclusive_spans']['os'][name]['calls'], 'Audit cost span count differs')
        equal_number(seconds, audit['inclusive_spans']['os'][name]['seconds'], name)
    equal_number(policy_seconds, audit['inclusive_spans']['os']['policy_client']['seconds'], 'native policy inference')
    admission_count, admission_seconds = span_total(os_events, 'experience_admission')
    require(admission_count == 1, 'Unexpected native admission count')
    require(entry['metrics'] == audit['entry_metrics'] and
            sha256(sources.file(sealed_base + '.FASTRSI_ADMITTED.json')) == audit['admission']['seal_sha256'],
            'Entry/seal audit mismatch')
    result = {'joint_native_success': True, 'native_goal_results': {'goal6': True, 'goal8': True},
              'policy_action_steps': step, 'reset_settling_steps': 10,
              'policy_calls': len(inputs), 'environment_resets': 1,
              'chain_status': 'PASSED', 'semantic_verdict': 'success', 'experience_admission': 'ADMITTED'}
    record = {'evidence_layer': 'full_native_admitted', 'task': registration(reg),
              'actual_sequence': copy.deepcopy(binding['sequence']), 'result': result,
              'costs': {
                  'case_wall': cost(row['seconds'], COST_SCOPES['full_native_admitted']['case_wall']),
                  'worker_wall': cost(status['arm']['full_wall_seconds'], 'Inclusive worker with model/service startup; contains case wall'),
                  'outer_controller_wall': cost(status['wall_seconds'], 'Inclusive controller wall; contains worker; not additive'),
                  'policy_inference': cost(policy_seconds, 'Component within case wall; nonadditive'),
                  'environment_action': cost(entry['metrics']['action_seconds'], 'Component within case wall; nonadditive'),
                  'environment_reset': cost(entry['metrics']['reset_seconds'], 'Component within case wall, including settling actions; nonadditive'),
                  'experience_admission': cost(admission_seconds, COST_SCOPES['full_native_admitted']['experience_admission'])}}
    pairing = {'initial_state': copy.deepcopy(resets[0]['physical_binding']),
               'initial_images': copy.deepcopy(resets[0]['raw_images'])}
    return record, pairing


def decision_facts(records):
    return {'task': copy.deepcopy(records['policy_operation_diagnostic'][0]['task']),
            'observations': copy.deepcopy(records), 'limitations': list(LIMITATIONS)}


def build_receipt(diagnostic_root, native_root,
                  diagnostic_manifest_sha256=DIAGNOSTIC_MANIFEST_SHA256,
                  native_manifest_sha256=NATIVE_MANIFEST_SHA256):
    diagnostic = Sources(diagnostic_root, 'LOCAL_OUTPUT_MANIFEST.json', diagnostic_manifest_sha256)
    native = Sources(native_root, 'EVIDENCE_TRANSFER_MANIFEST_V1.json', native_manifest_sha256)
    check = native.json('LOCAL_EVIDENCE_VERIFY_V1.json', manifested=False)
    require(check['status'] == 'PASSED' and check['manifest_sha256'] == native_manifest_sha256,
            'Native independent transfer verification missing')
    dreg = diagnostic.json('REGISTRATION.json')
    nreg = native.json('SAME_WORLD_REGISTRATION.json')
    require(registration(dreg) == registration(nreg), 'Diagnostic/native task meanings differ')
    diag_rows, pairings = [], []
    for arm in ('order68', 'order86'):
        record, pairing = diagnostic_record(diagnostic, dreg, arm)
        diag_rows.append(record)
        pairings.append(pairing)
    native_row, native_pairing = native_record(native, nreg)
    pairings.append(native_pairing)
    require(len({p['initial_state']['sim_state_sha256'] for p in pairings}) == 1 and
            all(p['initial_images'] == pairings[0]['initial_images'] for p in pairings),
            'Evidence is not from the same observed initial state/images')
    records = {'policy_operation_diagnostic': diag_rows, 'full_native_admitted': [native_row]}
    receipt = {'version': VERSION, 'records': records, 'decision_facts': decision_facts(records),
               'provenance': {'diagnostic': diagnostic.provenance(), 'native': native.provenance(),
                              'pairing_only_not_decision_features': pairings}}
    validate_schema(receipt)
    return receipt


def exact_keys(value, keys, name):
    require(isinstance(value, dict) and set(value) == set(keys), 'Unexpected schema fields: ' + name)


def validate_schema(receipt):
    exact_keys(receipt, ['version', 'records', 'decision_facts', 'provenance'], 'receipt')
    require(receipt['version'] == VERSION, 'Wrong receipt version')
    records = receipt['records']
    exact_keys(records, ['policy_operation_diagnostic', 'full_native_admitted'], 'records')
    require(len(records['policy_operation_diagnostic']) == 2 and len(records['full_native_admitted']) == 1,
            'Missing/extra real observations')
    for layer, rows in records.items():
        for row in rows:
            exact_keys(row, ['evidence_layer', 'task', 'actual_sequence', 'result', 'costs'], 'record')
            require(row['evidence_layer'] == layer, 'Mixed evidence layers')
            require(row['task'] == {'joint_goal': JOINT_GOAL, 'stages': STAGES, 'native_goal_predicates': GOALS},
                    'Unexpected task features')
            require(row['actual_sequence'] in list(ORDERS.values()), 'Unknown actual sequence')
            result = row['result']
            result_keys = ['joint_native_success', 'native_goal_results', 'policy_action_steps',
                           'reset_settling_steps', 'policy_calls', 'environment_resets']
            if layer == 'full_native_admitted':
                result_keys += ['chain_status', 'semantic_verdict', 'experience_admission']
                require(result.get('chain_status') == 'PASSED' and result.get('semantic_verdict') == 'success'
                        and result.get('experience_admission') == 'ADMITTED', 'Unadmitted native fact')
            exact_keys(result, result_keys, 'result')
            require(result['joint_native_success'] is True and result['native_goal_results'] == {'goal6': True, 'goal8': True},
                    'Incomplete final goals')
            require(type(result['policy_action_steps']) is int and 0 < result['policy_action_steps'] <= 600 and
                    type(result['policy_calls']) is int and 0 < result['policy_calls'] <= 600 and
                    result['reset_settling_steps'] == 10 and result['environment_resets'] == 1, 'Invalid actual counts')
            cost_keys = ['policy_inference', 'environment_action', 'environment_reset']
            cost_keys += (['worker_wall', 'script_wall'] if layer == 'policy_operation_diagnostic'
                          else ['case_wall', 'worker_wall', 'outer_controller_wall', 'experience_admission'])
            exact_keys(row['costs'], cost_keys, 'costs')
            for name, metric in row['costs'].items():
                exact_keys(metric, ['seconds', 'scope'], 'cost')
                number(metric['seconds'])
                require(metric['scope'] == COST_SCOPES[layer][name], 'Cost scope changed or mixed across layers')
    require([r['actual_sequence'] for r in records['policy_operation_diagnostic']] == list(ORDERS.values())
            and records['full_native_admitted'][0]['actual_sequence'] == ORDERS['order86'], 'Record sequence order differs')
    require(receipt['decision_facts'] == decision_facts(records), 'Decision facts contain additional features or altered evidence')
    provenance = receipt['provenance']
    exact_keys(provenance, ['diagnostic', 'native', 'pairing_only_not_decision_features'], 'provenance')
    for layer in ('diagnostic', 'native'):
        source = provenance[layer]
        exact_keys(source, ['root', 'manifest', 'manifest_sha256', 'files'], 'source provenance')
        require(Path(source['root']).is_absolute(), 'Absolute collection root required')
        digest(source['manifest_sha256'])
        require(source['files'], 'No actual source bindings')
        for relative, info in source['files'].items():
            require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Unsafe source provenance')
            exact_keys(info, ['sha256', 'bytes'], 'source file')
            digest(info['sha256'])
            require(type(info['bytes']) is int and info['bytes'] >= 0, 'Bad source size')
    return receipt


def relocate(root, source_base_mapping):
    root = Path(root)
    mappings = source_base_mapping or {}
    matches = []
    for old, new in mappings.items():
        old, new = Path(old), Path(new)
        require(old.is_absolute() and new.is_absolute(), 'Explicit absolute base mapping required')
        if root == old or old in root.parents:
            matches.append((len(old.parts), new / root.relative_to(old)))
    return max(matches, key=lambda pair: pair[0])[1] if matches else root


def validate_evidence(path, expected_sha256=None, *, verify_sources=True, source_base_mapping=None):
    """Default: source rebuild. Deployment: verify_sources=False + trusted SHA.

    Explicit mappings relocate collection roots only. Source verification reads
    bounded referenced files and reconstructs every fact, not just file hashes.
    """
    path = Path(path)
    require(path.is_file(), 'Missing evidence receipt')
    if expected_sha256 is not None:
        require(sha256(path) == digest(expected_sha256), 'Evidence receipt SHA mismatch')
    require(verify_sources or expected_sha256 is not None, 'Schema-only deployment requires a trusted receipt SHA')
    receipt = validate_schema(json.loads(path.read_text()))
    if verify_sources:
        roots = {layer: relocate(receipt['provenance'][layer]['root'], source_base_mapping)
                 for layer in ('diagnostic', 'native')}
        for layer, root in roots.items():
            for relative, info in receipt['provenance'][layer]['files'].items():
                source = (root / relative).resolve()
                require(root.resolve() in source.parents and source.is_file(), 'Missing/escaping receipt source: ' + relative)
                require(sha256(source) == info['sha256'] and source.stat().st_size == info['bytes'],
                        'Receipt source changed: ' + relative)
        rebuilt = build_receipt(roots['diagnostic'], roots['native'],
                                receipt['provenance']['diagnostic']['manifest_sha256'],
                                receipt['provenance']['native']['manifest_sha256'])
        require(rebuilt['records'] == receipt['records'] and rebuilt['decision_facts'] == receipt['decision_facts'] and
                rebuilt['provenance']['pairing_only_not_decision_features'] == receipt['provenance']['pairing_only_not_decision_features'],
                'Receipt facts differ from actual sources')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-root', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory, never overwritten')
    parser.add_argument('--diagnostic-manifest-sha256', default=DIAGNOSTIC_MANIFEST_SHA256)
    parser.add_argument('--native-manifest-sha256', default=NATIVE_MANIFEST_SHA256)
    args = parser.parse_args()
    receipt = build_receipt(args.diagnostic_root, args.native_root,
                            args.diagnostic_manifest_sha256, args.native_manifest_sha256)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'RECEIPT.json'
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    receipt_sha = sha256(path)
    validate_evidence(path, receipt_sha)
    verification = {'status': 'PASSED', 'version': VERSION, 'receipt_sha256': receipt_sha,
                    'mode': 'COLLECTION_PINNED_MANIFEST_AND_ACTUAL_SOURCE_REBUILD',
                    'builder_sha256': sha256(Path(__file__)),
                    'source_files_verified': sum(len(v['files']) for k, v in receipt['provenance'].items()
                                                 if k in ('diagnostic', 'native')),
                    'scope': 'No robot/model execution, no learning label, no sealed workspace change',
                    'deployment_validation': 'validate_evidence(path, trusted_sha, verify_sources=False)',
                    'explicit_source_relocation': 'source_base_mapping={old_absolute_root:new_absolute_root}'}
    (args.output / 'VERIFICATION.json').write_text(json.dumps(verification, indent=2) + '\n')
    path.chmod(0o444)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
