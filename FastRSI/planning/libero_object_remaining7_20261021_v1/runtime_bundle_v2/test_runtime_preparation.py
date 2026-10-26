"""Pure fake integration; no Docker, neural inference, Env, SSH, or real states.

FAKE scope/authorization here tests lifecycle wiring only. These fixtures never
prove human authorization, actual sampling, hardware availability or a rescue.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import contextlib
import importlib
import json
import shutil
import tempfile
import unittest

import row_factory as factory
import coordinator_contract as contract
import isolated_coordinator as coordinator
import runtime_admission as admission
import launch_specs
import audit_results

HERE = Path(__file__).resolve().parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def states():
    _, design = factory.proposal()
    return [dict(row, state_path='states/' + row['identity_id'] + '.npy', nq=58,
                 initial_state_sha256=('%064x' % (i + 1)),
                 physical_state_sha256=('%064x' % (i + 101)),
                 asset_sha256='a' * 64, source_receipt_sha256='b' * 64)
            for i, row in enumerate(design)]


class FakeExecution:
    def __init__(self, directory, task=0, baseline_first_success=False, fault_branch=None,
                 cleanup_fault=False, resource_wait=False, create_timeout=False, start_timeout=False):
        self.root = Path(directory)
        self.task, self.baseline_first_success = task, baseline_first_success
        self.fault_branch, self.cleanup_fault = fault_branch, cleanup_fault
        self.resource_wait = resource_wait
        self.create_timeout, self.start_timeout = create_timeout, start_timeout
        self.launches, self.created_commands, self.containers = [], [], {}
        self.rng = {'python': 'fake-python', 'numpy': 'fake-numpy', 'torch': 'fake-torch'}
        self.all_rows = [factory.condition_row(state, model, '187', 0)
                         for model in factory.MODELS for state in states()]
        groups = []
        for group in contract.groups():
            selected = [row for row in self.all_rows if row['model'] == group['model']
                        and row['source']['task_id'] == group['task_id']]
            groups.append(dict(group, node='187', gpu=0,
                               condition_ids=[row['identity_id'] for row in selected]))
        write(self.root / 'BUILD_INPUT_MANIFEST.json', {})
        write(self.root / 'STATE_OUTPUT_BINDING.json', {})
        write(self.root / 'AUTHORIZATION_SCOPE.json', {'fixture': 'FAKE_NOT_HUMAN_AUTHORIZATION'})
        write(self.root / 'FROZEN_RUNTIME_SCOPE.json', dict(
            status='ALL56_OBJECT7_CONDITIONS_FROZEN_BEFORE_TRIAL', groups=groups,
            conditions=self.all_rows, normal_physical_attempts_max=336,
            input_manifest_sha256=contract.sha(self.root / 'BUILD_INPUT_MANIFEST.json'),
            state_output_binding_sha256=contract.sha(self.root / 'STATE_OUTPUT_BINDING.json')))
        execution = dict(status='EXECUTION_BYTES_AND56_OBJECT7_CONDITIONS_FROZEN')
        for field, name in [('input_manifest_sha256', 'BUILD_INPUT_MANIFEST.json'),
                            ('frozen_runtime_scope_sha256', 'FROZEN_RUNTIME_SCOPE.json'),
                            ('state_output_binding_sha256', 'STATE_OUTPUT_BINDING.json'),
                            ('authorization_scope_sha256', 'AUTHORIZATION_SCOPE.json')]:
            execution[field] = contract.sha(self.root / name)
        write(self.root / 'EXECUTION_BINDINGS.json', execution)
        self.args = SimpleNamespace(model='pi05', suite='libero_object', task=task,
                                    node='187', gpu=0,
                                    authorization=self.root / 'AUTHORIZATION_SCOPE.json',
                                    resource_binding=self.root / 'RESOURCE_BINDING.json')
        write(self.args.resource_binding, {'fixture': 'FAKE'})
        self.lane = self.root / 'campaigns/libero_object__pi05' / ('task' + str(task)) / 'node187/gpu0'

    def process(self, argv, **kwargs):
        if argv[:2] == ['docker', 'inspect']:
            matches = [v for v in self.containers.values() if v['Name'].lstrip('/') == argv[2]]
            return SimpleNamespace(returncode=0 if matches else 1, stdout=json.dumps(matches),
                                   stderr='' if matches else 'Error: No such object: ' + argv[2])
        if argv[:2] == ['docker', 'logs']:
            return SimpleNamespace(returncode=0, stdout='fake logs', stderr='')
        if argv[:2] == ['docker', 'stop']:
            cid = argv[-1]
            if self.cleanup_fault:
                return SimpleNamespace(returncode=1, stdout='', stderr='FAKE cleanup failure')
            self.containers[cid]['State']['Running'] = False
            return SimpleNamespace(returncode=0, stdout=cid, stderr='')
        if argv[:2] == ['docker', 'start']:
            cid = argv[2]
            role = self.containers[cid]['FAKE_role']
            self.launches.append(role)
            self.containers[cid]['State']['Running'] = role == 'policy'
            if self.start_timeout and role == 'policy':
                raise coordinator.subprocess.TimeoutExpired(argv, timeout=90)
            return SimpleNamespace(returncode=0, stdout=cid, stderr='')
        assert argv[:2] == ['docker', 'create']
        role = argv[argv.index('--name') + 1]
        cid = '%064x' % (len(self.containers) + 1)
        policy = role == 'policy'
        label = argv[argv.index('--label') + 1].split('=', 1)
        self.containers[cid] = dict(
            Id=cid, Name='/' + role, FAKE_role=role,
            Image=coordinator.POLICY if policy else coordinator.ENV,
            HostConfig=dict(Privileged=False, DeviceRequests=[],
                            Devices=[{'PathOnHost': name} for name in
                                     ['/dev/kfd', '/dev/mkfd', '/dev/dri']] if policy else []),
            Config=dict(Env=['HIP_VISIBLE_DEVICES=0', 'CUDA_VISIBLE_DEVICES=0'], Labels={label[0]: label[1]}),
            State=dict(Running=False, ExitCode=0))
        Path(argv[argv.index('--cidfile') + 1]).write_text(cid)
        if self.create_timeout and role == 'policy':
            raise coordinator.subprocess.TimeoutExpired(argv, timeout=90)
        return SimpleNamespace(returncode=0, stdout=cid, stderr='')

    def argv(self, role):
        return ['docker', 'run', '-d', '--name', role,
                coordinator.POLICY if role == 'policy' else coordinator.ENV, 'FAKE_NEVER_EXECUTED']

    def policy_command(self, lane, *args):
        write(lane / 'DEVICE_BINDING.json', dict(device_count=1, hip='0', cuda='0'))
        return self.argv('policy')

    def wire_command(self, folder, *args):
        write(folder / 'RESULT.json', dict(accepted=True, wire_horizon=10,
                                         environment_constructors=0, physical_actions=0))
        return self.argv('wire')

    def environment_command(self, folder, lane, node, gpu, row, kind, branch=None):
        canonical = row.get('canonical_condition', row['identity_id'])
        self.created_commands.append((canonical, kind, branch))
        # The actual coordinator must persist intent BEFORE calling the launcher.
        assert (folder / 'ATTEMPT_INTENT.json').is_file()
        if kind == 'base':
            path = folder / row['physical_identity_id'] / 'stream' / (canonical + '--base') / 'PROTOCOL_RESULT.json'
            success = self.baseline_first_success and row['source']['variant'] == 'position'
            count = 1 if success else 3
            write(path, dict(software_ok=True, first_success=success, final_success=success,
                             attempts=[dict(index=i + 1) for i in range(count)]))
            for i in range(count):
                write(path.parent / ('attempt_' + str(i + 1)) / 'RAW.json',
                      dict(error=None, episode_completed=True, physical_success=success,
                           reset_state_sha256='r' * 64, first_input_sha256='i' * 64,
                           actions=10, seconds=0.01,
                           initial_state_sha256=row['identity']['initial_state_sha256']))
            checkpoint = lane / 'resident_receipts' / (canonical + '.json')
            write(checkpoint, {'fixture': 'FAKE_CHECKPOINT_NOT_UNPICKLED'})
        else:
            success = branch == 'frozen'
            raw = dict(physical_success=success, error=None, seconds=0.02, episode_completed=True)
            write(folder / 'RAW.json', raw)
            result = dict(software_ok=True, error=None, episode_completed=True,
                          physical_success=success, native_accounting_complete=True,
                          GT_control_input=False, official_actions=10,
                          visual_requests=0 if row['skill'] is None else 1,
                          stop_reason='SOURCE_SKILL_UNAVAILABLE' if row['skill'] is None else None)
            if branch == self.fault_branch:
                # Preserve the physical observation while rejecting software/compare closure.
                result.update(physical_success=True, software_ok=False, error='FAKE accounting fault')
                raw.update(physical_success=True, error='FAKE accounting fault')
                write(folder / 'RAW.json', raw)
            write(folder / 'RESULT.json', result)
            write(folder / 'RESIDENT_ENTRY.json', {'receipt': {'paired_start_rng_hashes': self.rng}})
            write(folder / 'INITIAL_INPUT_ACCEPTANCE.json',
                  dict(reset_state_sha256='r' * 64, initial_policy_input_sha256='i' * 64))
        return self.argv(canonical + '_' + (branch or 'base'))

    def vision_command(self, folder, *args):
        write(folder / 'vision_rpc/READY.json', {'fixture': 'FAKE'})
        write(folder / 'vision_rpc/SERVICE_RESULT.json', dict(error=None, cleanup_error=None))
        return self.argv(folder.parent.name + '_' + folder.name + '_vision')

    def baseline_evidence(self, path, row, lane):
        protocol = contract.read(path)
        raws = [contract.read(path.parent / ('attempt_' + str(a['index'])) / 'RAW.json')
                for a in protocol['attempts']]
        return protocol, raws, self.rng, lane / 'resident_receipts' / (row['identity_id'] + '.json')

    def run(self):
        def audit(*args):
            if self.resource_wait:
                raise admission.ResourceUnavailable('FAKE unrelated other task remains untouched')
            return {}
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): return False
        with contextlib.ExitStack() as stack:
            overrides = dict(ROOT=self.root, frozen_files=lambda *args: {},
                             validate_authorization=lambda *args: {},
                             audit=audit, ident=lambda pid: {'fixture': 'FAKE_PID'},
                             inspect=lambda cid: self.containers[cid],
                             policy_command=self.policy_command, wire_command=self.wire_command,
                             environment_command=self.environment_command,
                             vision_command=self.vision_command,
                             baseline_evidence=self.baseline_evidence)
            for name, value in overrides.items():
                stack.enter_context(mock.patch.object(coordinator, name, value))
            stack.enter_context(mock.patch.object(coordinator.subprocess, 'run', self.process))
            stack.enter_context(mock.patch.object(coordinator.urllib.request, 'urlopen', lambda *args, **kwargs: Response()))
            coordinator.run_group(self.args)


class Contracts(unittest.TestCase):
    def test_partial_summary_does_not_score_unsupported_or_missing_compute_as_zero(self):
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_') as tmp:
            fake = FakeExecution(tmp, task=2)
            fake.run()
            result = audit_results.audit(fake.root)
            measured = result['models']['pi05']
            self.assertEqual(measured['verified_closed_conditions'], 2)
            self.assertEqual(measured['final_by_arm']['candidate'], dict(successes=0, known=0, unknown=14))
            self.assertIsNone(measured['paired_subset_only']['ordinary']['complete_14_identity_paired_mean'])
            self.assertIsNone(measured['closed_attempt_actor_costs_only']['ordinary']['raw_declared_policy_counts']['neural_forward_end'])
            self.assertIsNone(result['money'])

    def test_actual_proven_launch_descriptors_and_original_backend_routing(self):
        manifest = contract.read(HERE / 'BUILD_INPUT_MANIFEST.json')
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_DESCRIPTOR_') as tmp:
            root = Path(tmp)
            # Only text bytes needed to build real argv; none of this code is executed.
            for name in ('launch_parent_frozen.py', 'sources/multi_vla_server.py',
                         'runtime/backend_server_frozen.py'):
                declared = next(item for item in manifest['files'] if item['destination'] == name)
                source = Path(declared['local_path'])
                self.assertEqual(contract.sha(source), declared['sha256'])
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, root / name)
            write(root / 'BUILD_INPUT_MANIFEST.json', manifest)
            with mock.patch.object(launch_specs, 'M', root), \
                    mock.patch.object(launch_specs, 'generated_root', lambda: root / 'FAKE_GENERATED'), \
                    mock.patch.object(Path, 'exists', lambda self: True):
                # Existence is FAKE here; actual host paths are a separate root admission.
                for model in factory.MODELS:
                    row = factory.condition_row(states()[0], model, '187', 0)
                    lane, folder = root / ('FAKE_LANE_' + model), root / ('FAKE_ATTEMPT_' + model)
                    argv = launch_specs.policy_command(lane, '187', 0, model,
                                                       [row['identity_id']], 'libero_object')
                    self.assertIn(launch_specs.POLICY, argv)
                    self.assertIn('LIBEROPRO_BASE_SUITE=libero_object', argv)
                    self.assertIn('HIP_VISIBLE_DEVICES=0', argv)
                    self.assertIn('CUDA_VISIBLE_DEVICES=0', argv)
                    expected_weights = dict(pi05='pi05_libero_pytorch_fp32', pi0='pi0_libero_pytorch_fp32',
                                            openvla='openvla_object', xvla='xvla_libero')[model]
                    self.assertTrue(any('/weights/' + expected_weights + ':/model:ro' in arg for arg in argv))
                    for branch in factory.BRANCHES:
                        paired = factory.branch_row(row, branch, {'fixture': 'FAKE_RNG'}, 'r' * 64, 'i' * 64)
                        actor = launch_specs.environment_command(folder, lane, '187', 0, paired,
                            'ordinary_fourth' if branch == 'ordinary' else 'paired_fourth', branch)
                        self.assertIn(launch_specs.ENV, actor)
                        self.assertIn('FASTRSI_RESIDENT_INFERENCE_HORIZON=' + str(factory.HORIZONS[model][0]), actor)
                        self.assertIn(str(root / 'FAKE_GENERATED') + ':/eval/generated:ro', actor)
                        self.assertNotIn('--device', actor)
                        if branch == 'candidate':
                            self.assertIn(str(root / 'candidate_runtime') + ':/eval/runtime:ro', actor)
                        if branch == 'frozen':
                            self.assertIn(str(root / 'previous_runtime') + ':/eval/runtime:ro', actor)
                    visual = launch_specs.vision_command(folder, '187', 'candidate', row)
                    self.assertNotIn('--device', visual)
                    self.assertNotIn('--gpus', visual)
                    self.assertIn(launch_specs.ENV, visual)

    def test_fixed_shape_and_no_alphabet_in_old_skill_namespace(self):
        value, rows = factory.proposal()
        self.assertEqual(len(rows), 14)
        self.assertEqual(len(contract.groups()), 28)
        self.assertEqual(sum(g['normal_attempts_max'] for g in contract.groups()), 336)
        self.assertTrue(all(factory.capability('libero_object', task, 'frozen') is None for task in factory.TASKS))
        self.assertEqual(factory.capability('libero_object', 0, 'candidate'), 'alphabet_transfer_v1')
        factory.check_capabilities(HERE)

    def test_true_native_retained_and_salad_unknown_never_copied(self):
        self.assertEqual(factory.final_outcomes(True, {}), dict(ordinary=True, frozen=True, candidate=True))
        value = factory.final_outcomes(False, dict(
            ordinary={'result': {'physical_success': True}},
            frozen={'result': {'physical_success': False}},
            candidate={'status': 'SKIPPED_UNSUPPORTED', 'physical_attempts': 0}))
        self.assertEqual(value, dict(ordinary=True, frozen=False, candidate=None))
        self.assertIsNone(factory.paired_difference(value['candidate'], value['ordinary']))

    def test_all_backend_wire_horizons_unchanged(self):
        self.assertEqual(factory.HORIZONS, dict(pi05=(10, 10), pi0=(50, 10), openvla=(1, 1), xvla=(1, 1)))

    def test_object_nq_and_frozen_roster_are_not_substitutable(self):
        row = states()[0]
        with self.assertRaises(AssertionError): factory.physical_row(dict(row, nq=65))
        with self.assertRaises(AssertionError): factory.physical_row(dict(row, seed=7))
        with self.assertRaises(AssertionError): factory.physical_row(dict(row, state_path='../other.npy'))


class Lifecycle(unittest.TestCase):
    def test_create_timeout_never_sends_start_and_stopped_cid_is_known(self):
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_') as tmp:
            fake = FakeExecution(tmp, create_timeout=True)
            with self.assertRaises(RuntimeError): fake.run()
            self.assertEqual(fake.launches, [])
            stop = contract.read(fake.lane / 'OWNED_STOP_RECEIPT.json')
            self.assertIn('policy', stop['states'])
            self.assertFalse(stop['states']['policy']['state']['Running'])
            self.assertIsNotNone(contract.read(fake.lane / 'LAUNCH_policy.json')['container_id'])

    def test_start_timeout_uses_already_known_cid_and_cleans_without_retry(self):
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_') as tmp:
            fake = FakeExecution(tmp, start_timeout=True)
            with self.assertRaises(RuntimeError): fake.run()
            self.assertEqual(fake.launches, ['policy'])
            self.assertTrue(all(not c['State']['Running'] for c in fake.containers.values()))
            receipt = contract.read(fake.lane / 'LAUNCH_policy.json')
            self.assertEqual(receipt['start_argv'], ['docker', 'start', receipt['container_id']])
            self.assertIsNotNone(receipt['error'])

    def test_native_success_skips_all_extra_other_failure_runs_every_arm(self):
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_') as tmp:
            fake = FakeExecution(tmp, baseline_first_success=True)
            fake.run()
            position, environment = [r['identity_id'] for r in fake.all_rows
                                     if r['model'] == 'pi05' and r['source']['task_id'] == 0]
            self.assertFalse(any(role.startswith(position) and not role.endswith('_base') for role in fake.launches))
            self.assertTrue(any(role == environment + '_candidate' for role in fake.launches))
            # Old success does not prevent running the independent candidate arm.
            result = contract.read(fake.lane / environment / 'CONDITION_RESULT.json')
            self.assertEqual(result['final'], dict(ordinary=False, frozen=True, candidate=False))
            self.assertEqual(result['net_vs_frozen'], -1)
            self.assertTrue(all(not c['State']['Running'] for c in fake.containers.values()))

    def test_salad_candidate_has_no_launch_raw_or_fake_outcome(self):
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_') as tmp:
            fake = FakeExecution(tmp, task=2)
            fake.run()
            for physical in ['object7_dev_t2_position', 'object7_dev_t2_environment']:
                case = fake.lane / ('pi05-' + physical)
                self.assertFalse((case / 'candidate/RAW.json').exists())
                self.assertFalse((case / 'candidate/ATTEMPT_INTENT.json').exists())
                result = contract.read(case / 'CONDITION_RESULT.json')
                self.assertIsNone(result['final']['candidate'])
                self.assertEqual(result['branches']['candidate']['physical_attempts'], 0)
            self.assertFalse(any('candidate' in role for role in fake.launches))

    def test_first_technical_fault_preserves_physical_true_and_stops(self):
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_') as tmp:
            fake = FakeExecution(tmp, fault_branch='ordinary')
            with self.assertRaises(RuntimeError): fake.run()
            case = fake.lane / 'pi05-object7_dev_t0_position'
            self.assertTrue(contract.read(case / 'ordinary/RAW.json')['physical_success'])
            self.assertFalse((case / 'CONDITION_RESULT.json').exists())
            self.assertFalse(any('frozen' in role or 'candidate' in role for role in fake.launches))
            self.assertTrue((fake.root / 'HALT_OBJECT7.json').exists())
            self.assertTrue(all(not c['State']['Running'] for c in fake.containers.values()))


class PhaseCloseout(unittest.TestCase):
    def fixture(self, root):
        groups, pins = [], []
        for task in factory.TASKS:
            ids = ['FAKE_t%d_%s' % (task, variant) for variant in ('position', 'environment')]
            group = dict(model='pi05', task_id=task, node='187', gpu=0, condition_ids=ids)
            groups.append(group)
            lane = root / 'campaigns/libero_object__pi05' / ('task' + str(task)) / 'node187/gpu0'
            write(lane / 'GROUP_RESULT.json', dict(status='FROZEN_GROUP_COMPLETED', completed=ids,
                                                 normal_physical_attempts=10 if task == 2 else 12))
            write(lane / 'OWNED_STOP_RECEIPT.json', dict(cleanup_errors=[],
                  states={'FAKE': {'state': {'Running': False}}}))
            write(lane / 'LIFECYCLE.json', dict(error=None, cleanup_errors=[]))
            pin = dict(task_id=task, files={}, conditions={})
            for name in ('GROUP_RESULT.json', 'OWNED_STOP_RECEIPT.json', 'LIFECYCLE.json'):
                pin['files'][name] = contract.sha(lane / name)
            for identity in ids:
                write(lane / identity / 'CONDITION_RESULT.json', dict(software_ok=True,
                      native_up_to3=False, final=dict(ordinary=False, frozen=False,
                                                      candidate=None if task == 2 else True),
                      branches={'candidate': {'status': 'SKIPPED_UNSUPPORTED' if task == 2 else 'FAKE'}}))
                pin['conditions'][identity] = contract.sha(lane / identity / 'CONDITION_RESULT.json')
            pins.append(pin)
        write(root / 'FROZEN_RUNTIME_SCOPE.json', {'groups': groups})
        return dict(schema='fastrsi_object7_pi05_stage_complete_v1', groups=pins,
                    normal_physical_attempts_completed=82,
                    frozen_runtime_scope_sha256=contract.sha(root / 'FROZEN_RUNTIME_SCOPE.json'))

    def test_seven_groups_close_despite_explicit_salad_coverage_gap(self):
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_') as tmp:
            root = Path(tmp)
            marker = self.fixture(root)
            contract.verify_pi05_stage(root, marker)
            self.assertFalse((root / 'PI05_STAGE_COMPLETE.json').exists())

    def test_missing_group_or_cleanup_error_cannot_open_later_models(self):
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_') as tmp:
            root = Path(tmp)
            marker = self.fixture(root)
            with self.assertRaises(AssertionError):
                contract.verify_pi05_stage(root, dict(marker, groups=marker['groups'][:-1]))
            lane = root / 'campaigns/libero_object__pi05/task0/node187/gpu0'
            write(lane / 'LIFECYCLE.json', dict(error='FAKE technical fault', cleanup_errors=[]))
            marker['groups'][0]['files']['LIFECYCLE.json'] = contract.sha(lane / 'LIFECYCLE.json')
            with self.assertRaises(AssertionError): contract.verify_pi05_stage(root, marker)

    def test_duplicate_group_does_not_launch_again(self):
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_') as tmp:
            fake = FakeExecution(tmp)
            fake.run()
            count = len(fake.launches)
            with self.assertRaises(FileExistsError): fake.run()
            self.assertEqual(len(fake.launches), count)

    def test_cleanup_failure_never_accepted_as_normal_close(self):
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_') as tmp:
            fake = FakeExecution(tmp, cleanup_fault=True)
            with self.assertRaises(RuntimeError): fake.run()
            self.assertTrue(contract.read(fake.lane / 'OWNED_STOP_RECEIPT.json')['cleanup_errors'])
            self.assertTrue((fake.root / 'HALT_OBJECT7.json').exists())


class ResourceAdmission(unittest.TestCase):
    def test_resource_wait_has_no_new_model_and_no_global_failure_halt(self):
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_') as tmp:
            fake = FakeExecution(tmp, resource_wait=True)
            with self.assertRaises(RuntimeError): fake.run()
            self.assertEqual(fake.launches, [])
            self.assertTrue((fake.lane / 'GROUP_RESOURCE_WAIT.json').exists())
            self.assertFalse((fake.root / 'HALT_OBJECT7.json').exists())
            self.assertFalse((fake.lane / 'GROUP_FAILURE.json').exists())
            self.assertEqual(contract.read(fake.lane / 'LIFECYCLE.json')['normal_physical_attempts_completed'], 0)

    def report(self, holder='', containers=None, locks=None):
        return dict(holders={'stdout': holder}, containers=containers or [],
                    held_locks={'stdout': json.dumps({'locks': locks or []})})

    def approval(self, infra=None):
        return dict(exact_infrastructure_holders={'187': infra or []})

    def test_hymgr_name_does_not_bypass_exact_current_identity(self):
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_') as tmp:
            report = self.report('/dev/kfd: root 313 F.... hymgr')
            with self.assertRaises(admission.ResourceUnavailable):
                admission.validate_occupancy(Path(tmp), report, self.approval(), '187', 0)
            expected = dict(pid=313, boot='FAKE', start='12', argv=['FAKE_hymgr'])
            with mock.patch.object(admission, 'ident', lambda pid: expected):
                admission.validate_occupancy(Path(tmp), report, self.approval([expected]), '187', 0)
            with mock.patch.object(admission, 'ident', lambda pid: dict(expected, start='13')):
                with self.assertRaises(AssertionError):
                    admission.validate_occupancy(Path(tmp), report, self.approval([expected]), '187', 0)

    def own_container(self, root, running):
        lane = root / 'campaigns/libero_object__pi05/task0/node187/gpu1'
        cid = 'a' * 64
        owner = dict(pid=101, boot='FAKE', start='1', argv=['FAKE_OWNER'])
        write(lane / 'LAUNCH_policy.json', dict(returncode=0, container_id=cid))
        write(lane / 'IDENTITY.json', dict(node='187', gpu=1, process=owner))
        actual = dict(Id=cid, State={'Running': running},
                      HostConfig=dict(Devices=[{'PathOnHost': '/dev/kfd'}],
                                      DeviceRequests=[], Privileged=False),
                      Config={'Env': ['HIP_VISIBLE_DEVICES=1', 'CUDA_VISIBLE_DEVICES=1']})
        item = dict(id=cid[:12], name='FAKE_own_other_card',
                    top={'stdout': 'PID COMMAND\n999 FAKE_POLICY'},
                    devices=['/dev/kfd'], device_requests=[], privileged=False,
                    visible=['HIP_VISIBLE_DEVICES=1', 'CUDA_VISIBLE_DEVICES=1'])
        return cid, owner, actual, item

    def test_live_own_other_card_allowed_but_stopped_history_cannot_explain_holder(self):
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_') as tmp:
            root = Path(tmp)
            cid, owner, actual, item = self.own_container(root, True)
            report = self.report('/dev/kfd: root 999 F.... FAKE_POLICY', [item])
            with mock.patch.object(admission.subprocess, 'run', lambda *args, **kwargs:
                                   SimpleNamespace(returncode=0, stdout=json.dumps([actual]), stderr='')), \
                    mock.patch.object(admission, 'ident', lambda pid: owner):
                admission.validate_occupancy(root, report, self.approval(), '187', 0)
                with self.assertRaises(admission.ResourceUnavailable):
                    admission.validate_occupancy(root, report, self.approval(), '187', 1)
                actual['State']['Running'] = False
                with self.assertRaises(admission.ResourceUnavailable):
                    admission.validate_occupancy(root, report, self.approval(), '187', 0)
                admission.validate_occupancy(root, self.report('', [item]), self.approval(), '187', 0)

    def test_parallel_lease_requires_real_receipt_identity_and_other_card(self):
        with tempfile.TemporaryDirectory(prefix='fastrsi_object7_FAKE_') as tmp:
            root = Path(tmp)
            path = root / 'leases/FAKE_lease.json'
            write(path, {'fixture': 'FAKE_NOT_REAL_LEASE'})
            owner = dict(pid=101, boot='FAKE', start='1', argv=['FAKE_OWNER'])
            receipt = root / 'campaigns/FAKE/IDENTITY.json'
            write(receipt, dict(process=owner, node='187', gpu=1))
            named = '/mnt/data/task/yuchen/FastPhyagentos/FastRSI/planning/liberopro_eval_20261012_v1/leases/FAKE'
            report = self.report(locks=[{'path': named}])
            with self.assertRaises(admission.ResourceUnavailable):
                admission.validate_occupancy(root, report, self.approval(), '187', 0)
            item = dict(path=str(path), sha256=contract.sha(path), project_root=str(root.resolve()),
                        node='187', gpu=1, owner=owner,
                        owner_receipt={'path': str(receipt), 'sha256': contract.sha(receipt)})
            approve = dict(self.approval(), exact_parallel_reservations=[item])
            with mock.patch.object(admission, 'ident', lambda pid: owner), \
                    mock.patch.object(admission, 'LEASE_ROOT', str(root / 'leases')):
                admission.validate_occupancy(root, self.report(locks=[{'path': str(path)}]), approve, '187', 0)
                with self.assertRaises(AssertionError):
                    admission.validate_occupancy(root, self.report(locks=[{'path': str(path)}]), approve, '187', 1)
                # Historical lease metadata with no current held lock is not a live owner.
                with mock.patch.object(admission, 'ident', side_effect=ProcessLookupError('FAKE exited')):
                    admission.validate_occupancy(root, self.report(), approve, '187', 0)
            item['sha256'] = '0' * 64
            with mock.patch.object(admission, 'ident', lambda pid: owner):
                with self.assertRaises(AssertionError):
                    with mock.patch.object(admission, 'LEASE_ROOT', str(root / 'leases')):
                        admission.validate_occupancy(root, self.report(locks=[{'path': str(path)}]), approve, '187', 0)

    def test_target_card_zero_does_not_match_fractional_nonzero(self):
        for use, memory, accepted in [('0', '0', True), ('0.00', '0.0', True),
                                      ('0.5', '0', False), ('0', '0.5', False),
                                      ('10', '0', False)]:
            report = {'hcu': {'stdout': 'HCU[0] HCU use (%%): %s\nHCU[0] HCU memory use (%%): %s\n' % (use, memory)}}
            if accepted:
                admission.validate_target_card(report, 0)
            else:
                with self.assertRaises(admission.ResourceUnavailable): admission.validate_target_card(report, 0)


if __name__ == '__main__':
    unittest.main()
