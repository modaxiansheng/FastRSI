"""Local fake children and metadata only: no subprocess, SSH, Env or model."""
from pathlib import Path
from unittest import mock
import json
import tempfile
import unittest

import later_models_driver_v1 as driver


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class Fake:
    def __init__(self, root, pi05_complete=True, failure=None, resource_wait=None, never_finish=False):
        self.root, self.now = root, 0.0
        self.children, self.calls, self.maximum = {}, [], 0
        self.failure, self.resource_wait, self.never_finish = failure, resource_wait, never_finish
        groups = []
        for model in ('pi05',) + driver.MODELS:
            for i, task in enumerate((0, 1, 2, 4, 6, 7, 9)):
                groups.append(dict(model=model, suite='libero_object', task_id=task,
                     node='187' if task in (7, 0, 1, 2) else '188', gpu=i,
                     normal_attempts_max=12, condition_ids=['FAKE_%s_%da' % (model, task),
                                                           'FAKE_%s_%db' % (model, task)]))
        write(root / 'FROZEN_RUNTIME_SCOPE.json', dict(status='ALL56_OBJECT7_CONDITIONS_FROZEN_BEFORE_TRIAL',
              groups=groups, conditions=[{'fixture': 'FAKE'}] * 56,
              normal_physical_attempts_max=336, pi05_first_stage_normal_attempts_max=84))
        for name in ('isolated_coordinator.py', 'close_pi05_stage.py'):
            (root / name).write_text('# FAKE SOURCE NEVER EXECUTED\n')
        write(root / 'AUTHORIZATION_SCOPE.json', {'fixture': 'FAKE_NOT_AUTHORITY'})
        write(root / 'RESOURCE_BINDING.json', {'fixture': 'FAKE_NOT_RESOURCE_PROOF'})
        if pi05_complete:
            for group in groups:
                if group['model'] == 'pi05':
                    self.close(group)
            for node in driver.NODES:
                producer = root / 'controllers' / ('PI05_BATCH_NODE' + node)
                write(producer / 'LAUNCH.json', dict(
                    frozen_runtime_scope_sha256=driver.sha(root / 'FROZEN_RUNTIME_SCOPE.json'),
                    coordinator_sha256=driver.sha(root / 'isolated_coordinator.py')))
                write(producer / 'STATUS.json', dict(node=node, finished=[dict(
                    task_id=g['task_id'], returncode=0, status='VERIFIED_GROUP_CLOSED')
                    for g in groups if g['model'] == 'pi05' and g['node'] == node]))

    def close(self, group):
        lane = driver.lane_path(self.root, group)
        write(lane / 'GROUP_RESULT.json', dict(status='FROZEN_GROUP_COMPLETED',
              completed=group['condition_ids'], normal_physical_attempts=10))
        write(lane / 'OWNED_STOP_RECEIPT.json', dict(cleanup_errors=[],
              states={'FAKE_OWNED': {'state': {'Running': False}}}))
        write(lane / 'LIFECYCLE.json', dict(error=None, cleanup_errors=[]))
        for condition in group['condition_ids']:
            write(lane / condition / 'CONDITION_RESULT.json', dict(software_ok=True,
                native_up_to3=False, final=dict(ordinary=False, frozen=False,
                                                candidate=None if group['task_id'] == 2 else True),
                branches={'candidate': {'status': 'SKIPPED_UNSUPPORTED' if group['task_id'] == 2 else 'FAKE'}}))

    def marker(self):
        rows, count = [], 0
        for group in driver.groups(self.root):
            if group['model'] != 'pi05':
                continue
            lane = driver.lane_path(self.root, group)
            rows.append(dict(task_id=group['task_id'],
                files={n: driver.sha(lane / n) for n in
                       ('GROUP_RESULT.json', 'OWNED_STOP_RECEIPT.json', 'LIFECYCLE.json')},
                conditions={c: driver.sha(lane / c / 'CONDITION_RESULT.json') for c in group['condition_ids']}))
            count += driver.read(lane / 'GROUP_RESULT.json')['normal_physical_attempts']
        value = dict(schema='fastrsi_object7_pi05_stage_complete_v1', groups=rows,
                normal_physical_attempts_completed=count,
                frozen_runtime_scope_sha256=driver.sha(self.root / 'FROZEN_RUNTIME_SCOPE.json'))
        driver.verify_pi05_stage(self.root, value)
        write(self.root / 'PI05_STAGE_COMPLETE.json', value)

    def clock(self): return self.now
    def sleep(self, value): self.now += value
    def identity(self, pid):
        return dict(pid=pid, boot='FAKE', start=str(pid),
                    argv=self.children[pid].argv if pid in self.children else ['FAKE_OWNER'])

    def popen(self, argv, **kwargs):
        assert kwargs['start_new_session'] is True
        closer = Path(argv[2]).name == 'close_pi05_stage.py'
        if closer:
            key = ('closer', -1)
        else:
            key = (argv[argv.index('--model') + 1], int(argv[argv.index('--task') + 1]))
            driver.verify_pi05_stage(self.root)
        self.calls.append(key)
        fake = self

        class Process:
            def __init__(self):
                self.pid, self.argv = 50000 + len(fake.calls), argv
                self.code, self.polls = None, 0

            def poll(self):
                if self.code is not None:
                    return self.code
                self.polls += 1
                if self.polls < 2 or (fake.never_finish and not closer):
                    return None
                if closer:
                    fake.marker()
                    self.code = 0
                    return self.code
                group = next(g for g in driver.groups(fake.root) if (g['model'], g['task_id']) == key)
                if key == fake.failure:
                    write(driver.lane_path(fake.root, group) / 'GROUP_FAILURE.json', {'fixture': 'FAKE_UNKNOWN'})
                    self.code = 1
                elif key == fake.resource_wait:
                    write(driver.lane_path(fake.root, group) / 'GROUP_RESOURCE_WAIT.json', {'fixture': 'FAKE_WAIT'})
                    self.code = 1
                else:
                    fake.close(group)
                    self.code = 0
                return self.code

        process = Process()
        self.children[process.pid] = process
        self.maximum = max(self.maximum, sum(p.code is None for p in self.children.values()))
        return process

    def run(self, node='187', wait=20, wall=100):
        with mock.patch.object(driver, 'frozen_files', lambda root: None), \
                mock.patch.object(driver, 'helper_pins', lambda root: {'fixture': 'FAKE_PIN'}):
            return driver.run(self.root, node, self.root / 'AUTHORIZATION_SCOPE.json',
                self.root / 'RESOURCE_BINDING.json', wait_seconds=wait, wall_seconds=wall,
                poll_seconds=1, popen=self.popen, identify=self.identity,
                clock=self.clock, sleep=self.sleep)


class LaterModels(unittest.TestCase):
    def test_prior_global_closer_claim_without_marker_never_spawns_again(self):
        with tempfile.TemporaryDirectory(prefix='OBJECT7_LATER_FAKE_') as tmp:
            fake = Fake(Path(tmp))
            write(fake.root / 'controllers/OBJECT7_PI05_STAGE_CLOSE_INTENT.json',
                  {'fixture': 'FAKE_PRIOR_UNKNOWN_CLOSER_NO_MARKER'})
            with self.assertRaises(FileExistsError): fake.run()
            self.assertEqual(fake.calls, [])
            self.assertTrue((fake.root / 'HALT_OBJECT7.json').exists())

    def test_unpublished_exclusive_file_is_waited_not_read_or_faked(self):
        with tempfile.TemporaryDirectory(prefix='OBJECT7_LATER_FAKE_') as tmp:
            fake = Fake(Path(tmp))
            producer = fake.root / 'controllers/PI05_BATCH_NODE188/STATUS.json'
            producer.unlink()
            target = driver.lane_path(fake.root, next(g for g in driver.groups(fake.root)
                                     if g['model'] == 'pi05' and g['node'] == '188'))
            (target / 'LIFECYCLE.json').write_text('{')  # FAKE in-progress exclusive write.
            self.assertFalse(driver.pi05_ready(fake.root, driver.groups(fake.root)))
            self.assertEqual(fake.run(wait=2, wall=5), 2)
            self.assertFalse((fake.root / 'HALT_OBJECT7.json').exists())
            self.assertEqual(fake.calls, [])

    def test_real_marker_contract_once_then_local_models_in_order_max_two(self):
        with tempfile.TemporaryDirectory(prefix='OBJECT7_LATER_FAKE_') as tmp:
            fake = Fake(Path(tmp))
            self.assertEqual(fake.run(), 0)
            self.assertEqual(fake.calls[0], ('closer', -1))
            self.assertEqual(fake.maximum, 2)
            expected = [('closer', -1)] + [(m, t) for m in driver.MODELS for t in (7, 0, 1, 2)]
            self.assertEqual(fake.calls, expected)
            driver.verify_pi05_stage(fake.root)
            # Second node verifies the same original marker, without a second closer.
            self.assertEqual(fake.run(node='188'), 0)
            self.assertEqual(sum(model == 'closer' for model, _ in fake.calls), 1)
            self.assertEqual(len(fake.calls), 22)

    def test_incomplete_pi05_ends_bounded_cpu_wait_zero_children(self):
        with tempfile.TemporaryDirectory(prefix='OBJECT7_LATER_FAKE_') as tmp:
            fake = Fake(Path(tmp), pi05_complete=False)
            self.assertEqual(fake.run(wait=2, wall=10), 2)
            self.assertEqual(fake.calls, [])
            self.assertFalse((fake.root / 'PI05_STAGE_COMPLETE.json').exists())
            self.assertFalse((fake.root / 'HALT_OBJECT7.json').exists())

    def test_corrupt_existing_marker_never_starts_a_model(self):
        with tempfile.TemporaryDirectory(prefix='OBJECT7_LATER_FAKE_') as tmp:
            fake = Fake(Path(tmp))
            fake.marker()
            marker = driver.read(fake.root / 'PI05_STAGE_COMPLETE.json')
            marker['normal_physical_attempts_completed'] = 0
            write(fake.root / 'PI05_STAGE_COMPLETE.json', marker)
            with self.assertRaises(AssertionError): fake.run()
            self.assertEqual(fake.calls, [])
            self.assertTrue((fake.root / 'HALT_OBJECT7.json').exists())

    def test_technical_failure_drains_current_and_never_advances_model(self):
        with tempfile.TemporaryDirectory(prefix='OBJECT7_LATER_FAKE_') as tmp:
            fake = Fake(Path(tmp), failure=('pi0', 7))
            self.assertEqual(fake.run(), 2)
            self.assertEqual(fake.calls, [('closer', -1), ('pi0', 7), ('pi0', 0)])
            self.assertTrue((fake.root / 'HALT_OBJECT7.json').exists())
            self.assertTrue(all(p.code is not None for p in fake.children.values()))

    def test_resource_wait_does_not_retry_or_start_next_model(self):
        with tempfile.TemporaryDirectory(prefix='OBJECT7_LATER_FAKE_') as tmp:
            fake = Fake(Path(tmp), resource_wait=('pi0', 7))
            self.assertEqual(fake.run(), 2)
            self.assertEqual(fake.calls, [('closer', -1), ('pi0', 7), ('pi0', 0)])
            self.assertFalse((fake.root / 'HALT_OBJECT7.json').exists())

    def test_observer_timeout_preserves_exact_live_coordinators_without_restart(self):
        with tempfile.TemporaryDirectory(prefix='OBJECT7_LATER_FAKE_') as tmp:
            fake = Fake(Path(tmp), never_finish=True)
            self.assertEqual(fake.run(wait=2, wall=4), 2)
            self.assertEqual(fake.calls, [('closer', -1), ('pi0', 7), ('pi0', 0)])
            status = driver.read(fake.root / 'controllers/LATER_MODELS_V1_NODE187/STATUS.json')
            self.assertEqual(len(status['active']), 2)
            self.assertTrue(all(row['process']['boot'] == 'FAKE' for row in status['active']))
            self.assertTrue(all(p.code is None for p in list(fake.children.values())[1:]))

    def test_once_only_batch_directory_cannot_repeat_conditions(self):
        with tempfile.TemporaryDirectory(prefix='OBJECT7_LATER_FAKE_') as tmp:
            fake = Fake(Path(tmp))
            fake.run()
            before = len(fake.calls)
            with self.assertRaises(FileExistsError): fake.run()
            self.assertEqual(len(fake.calls), before)


if __name__ == '__main__':
    unittest.main()
