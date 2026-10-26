"""Pure fake host scheduling tests; no real children, SSH, Docker, Env or model."""
from pathlib import Path
from unittest import mock
import json
import os
import tempfile
import unittest

import pi05_batch_driver as batch


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class Fake:
    def __init__(self, root, failure=None, wait=None, never_finish=False):
        self.root, self.failure, self.wait = root, failure, wait
        self.never_finish = never_finish
        self.now, self.calls, self.children = 0.0, [], {}
        self.maximum = 0
        groups = []
        for model in ('pi05', 'pi0', 'openvla', 'xvla'):
            for i, task in enumerate((0, 1, 2, 4, 6, 7, 9)):
                node = '187' if task in (7, 0, 1, 2) else '188'
                groups.append(dict(model=model, suite='libero_object', task_id=task,
                                   node=node, gpu=i, normal_attempts_max=12,
                                   condition_ids=['FAKE_%s_t%d_a' % (model, task),
                                                  'FAKE_%s_t%d_b' % (model, task)]))
        write(root / 'FROZEN_RUNTIME_SCOPE.json', dict(
            status='ALL56_OBJECT7_CONDITIONS_FROZEN_BEFORE_TRIAL', groups=groups,
            conditions=[{'fixture': 'FAKE'}] * 56, normal_physical_attempts_max=336,
            pi05_first_stage_normal_attempts_max=84))
        (root / 'isolated_coordinator.py').write_text('# FAKE ENTRY NEVER EXECUTED\n')
        write(root / 'AUTHORIZATION_SCOPE.json', {'fixture': 'FAKE_NOT_AUTHORITY'})
        write(root / 'RESOURCE_BINDING.json', {'fixture': 'FAKE_NOT_RESOURCE_PROOF'})

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds

    def ident(self, pid):
        return dict(pid=pid, boot='FAKE_BOOT', start=str(pid),
                    argv=self.children[pid].argv if pid in self.children else ['FAKE_OWNER'])

    def popen(self, argv, **kwargs):
        assert kwargs['start_new_session'] is True and kwargs['close_fds'] is True
        task = int(argv[argv.index('--task') + 1])
        self.calls.append(task)
        fixture = self

        class Process:
            def __init__(self):
                self.pid = 10000 + len(fixture.calls)
                self.argv, self.code, self.polls = argv, None, 0

            def poll(self):
                if self.code is not None:
                    return self.code
                if fixture.never_finish:
                    return None
                self.polls += 1
                if self.polls < 2:
                    return None
                group = next(g for g in batch.selected_groups(fixture.root, '187')
                             if g['task_id'] == task)
                lane = batch.lane_path(fixture.root, group)
                if task == fixture.wait:
                    write(lane / 'GROUP_RESOURCE_WAIT.json', {'fixture': 'FAKE_WAIT_NO_RETRY'})
                    self.code = 1
                elif task == fixture.failure:
                    write(lane / 'GROUP_FAILURE.json', {'fixture': 'FAKE_UNKNOWN_NOT_PHYSICAL'})
                    self.code = 1
                else:
                    write(lane / 'GROUP_RESULT.json', dict(status='FROZEN_GROUP_COMPLETED',
                         completed=group['condition_ids'], normal_physical_attempts=12))
                    write(lane / 'OWNED_STOP_RECEIPT.json', dict(cleanup_errors=[],
                         states={'FAKE': {'state': {'Running': False}}}))
                    write(lane / 'LIFECYCLE.json', dict(error=None, cleanup_errors=[]))
                    self.code = 0
                return self.code

        result = Process()
        self.children[result.pid] = result
        self.maximum = max(self.maximum, sum(child.code is None for child in self.children.values()))
        return result

    def run(self, wall=30):
        with mock.patch.object(batch, 'frozen_files', lambda root: None), \
                mock.patch.object(batch, 'helper_bindings', lambda root: {'fixture': 'FAKE_HELPER_PIN'}):
            return batch.run(self.root, '187', self.root / 'AUTHORIZATION_SCOPE.json',
                self.root / 'RESOURCE_BINDING.json', wall_seconds=wall, poll_seconds=1,
                popen=self.popen, identify=self.ident, clock=self.clock, sleep=self.sleep)


class Scheduling(unittest.TestCase):
    def test_helper_source_must_equal_actual_sealed_bundle(self):
        import shutil
        with tempfile.TemporaryDirectory(prefix='object7_BATCH_FAKE_') as tmp:
            root = Path(tmp)
            local = Path(batch.__file__).resolve().parent
            for name in ('coordinator_contract.py', 'row_factory.py', 'runtime_admission.py'):
                shutil.copyfile(local / name, root / name)
            self.assertEqual(len(batch.helper_bindings(root)), 3)
            (root / 'row_factory.py').write_text('# FAKE_DIFFERENT_HELPER\n')
            with self.assertRaises(AssertionError):
                batch.helper_bindings(root)

    def test_fixed_milk_alphabet_first_max_two_and_exact_launch_receipts(self):
        with tempfile.TemporaryDirectory(prefix='object7_BATCH_FAKE_') as tmp:
            fake = Fake(Path(tmp))
            self.assertEqual(fake.run(), 0)
            self.assertEqual(fake.calls, [7, 0, 1, 2])
            self.assertEqual(fake.maximum, 2)
            directory = fake.root / 'controllers/PI05_BATCH_NODE187'
            receipt = batch.read(directory / 'CHILD_LAUNCH_task7.json')
            self.assertEqual(receipt['process']['argv'], receipt['argv'])
            self.assertEqual(receipt['process']['pid'], 10001)
            self.assertTrue(receipt['detached'])
            self.assertFalse((fake.root / 'PI05_STAGE_COMPLETE.json').exists())
            state = batch.read(directory / 'STATUS.json')
            self.assertEqual(state['status'], 'LOCAL_PI05_GROUPS_CLOSED_ROOT_MUST_VERIFY_GLOBAL_STAGE')
            self.assertEqual(len(state['finished']), 4)

    def test_same_batch_never_restarts_groups(self):
        with tempfile.TemporaryDirectory(prefix='object7_BATCH_FAKE_') as tmp:
            fake = Fake(Path(tmp))
            fake.run()
            before = len(fake.calls)
            with self.assertRaises(FileExistsError):
                fake.run()
            self.assertEqual(len(fake.calls), before)

    def test_first_technical_fault_halts_new_dispatch_without_killing_active(self):
        with tempfile.TemporaryDirectory(prefix='object7_BATCH_FAKE_') as tmp:
            fake = Fake(Path(tmp), failure=7)
            self.assertEqual(fake.run(), 2)
            self.assertEqual(fake.calls, [7, 0])
            self.assertTrue((fake.root / 'HALT_OBJECT7.json').exists())
            state = batch.read(fake.root / 'controllers/PI05_BATCH_NODE187/STATUS.json')
            self.assertEqual(len(state['finished']), 2)
            self.assertEqual(state['active'], [])

    def test_resource_wait_stops_local_batch_preserves_no_global_failure_claim(self):
        with tempfile.TemporaryDirectory(prefix='object7_BATCH_FAKE_') as tmp:
            fake = Fake(Path(tmp), wait=7)
            self.assertEqual(fake.run(), 2)
            self.assertEqual(fake.calls, [7, 0])
            self.assertFalse((fake.root / 'HALT_OBJECT7.json').exists())
            state = batch.read(fake.root / 'controllers/PI05_BATCH_NODE187/STATUS.json')
            self.assertEqual(state['stop_reason'], 'RESOURCE_WAIT_NO_AUTOMATIC_RETRY')

    def test_observer_expiry_retains_live_identity_and_does_not_restart_or_stop(self):
        with tempfile.TemporaryDirectory(prefix='object7_BATCH_FAKE_') as tmp:
            fake = Fake(Path(tmp), never_finish=True)
            self.assertEqual(fake.run(wall=2), 2)
            self.assertEqual(fake.calls, [7, 0])
            self.assertTrue(all(child.code is None for child in fake.children.values()))
            state = batch.read(fake.root / 'controllers/PI05_BATCH_NODE187/STATUS.json')
            self.assertEqual(state['status'], 'OBSERVATION_CLOSED_WORKERS_PRESERVED')
            self.assertEqual(len(state['active']), 2)

    def test_preexisting_group_is_blocked_without_spawning_it(self):
        with tempfile.TemporaryDirectory(prefix='object7_BATCH_FAKE_') as tmp:
            fake = Fake(Path(tmp))
            group = batch.selected_groups(fake.root, '187')[0]
            batch.lane_path(fake.root, group).mkdir(parents=True)
            with self.assertRaises(AssertionError):
                fake.run()
            self.assertEqual(fake.calls, [])
            self.assertTrue((fake.root / 'HALT_OBJECT7.json').exists())

    def test_preexisting_global_halt_has_zero_new_children(self):
        with tempfile.TemporaryDirectory(prefix='object7_BATCH_FAKE_') as tmp:
            fake = Fake(Path(tmp))
            write(fake.root / 'HALT_OBJECT7.json', {'fixture': 'FAKE_EXISTING_HALT'})
            with self.assertRaises(AssertionError):
                fake.run()
            self.assertEqual(fake.calls, [])


if __name__ == '__main__':
    unittest.main()
