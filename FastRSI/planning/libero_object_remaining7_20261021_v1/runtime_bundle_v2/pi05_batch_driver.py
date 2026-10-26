"""Host-only fixed-placement scheduler for the seven pi05 Object groups.

No SSH, Docker, Env, inference or scientific implementation is present here.
Only the frozen isolated_coordinator entry is spawned. Run once per node; an
existing batch/group directory is never resumed or turned into another attempt.
An interrupted observer is not proof that its detached coordinator stopped.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback

from coordinator_contract import read, sha, frozen_files
from runtime_admission import ident

ROOT = Path(__file__).resolve().parent
ORDER = (7, 0, 1, 4, 6, 9, 2)  # Milk, alphabet, then fixed remaining tasks.
NODES = ('187', '188')


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def replace_status(path, value):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(str(temporary), str(path))


def selected_groups(root, node):
    scope = read(root / 'FROZEN_RUNTIME_SCOPE.json')
    assert scope['status'] == 'ALL56_OBJECT7_CONDITIONS_FROZEN_BEFORE_TRIAL'
    assert len(scope['groups']) == 28 and len(scope['conditions']) == 56
    assert scope['normal_physical_attempts_max'] == 336
    assert scope['pi05_first_stage_normal_attempts_max'] == 84
    assert len({(g['model'], g['suite'], g['task_id']) for g in scope['groups']}) == 28
    pi05 = [g for g in scope['groups'] if g['model'] == 'pi05']
    assert len(pi05) == 7 and {g['task_id'] for g in pi05} == set(ORDER)
    for group in scope['groups']:
        assert group['node'] in NODES and type(group['gpu']) is int and group['gpu'] in range(8)
        assert group['suite'] == 'libero_object'
        assert len(group['condition_ids']) == 2 and group['normal_attempts_max'] == 12
    return sorted((g for g in pi05 if g['node'] == node),
                  key=lambda group: ORDER.index(group['task_id']))


def lane_path(root, group):
    return (root / 'campaigns/libero_object__pi05' / ('task' + str(group['task_id'])) /
            ('node' + group['node']) / ('gpu' + str(group['gpu'])))


def command(root, group, authorization, resource_binding):
    return [sys.executable, '-B', str(root / 'isolated_coordinator.py'), '--run',
            '--model', 'pi05', '--suite', 'libero_object', '--task', str(group['task_id']),
            '--node', group['node'], '--gpu', str(group['gpu']),
            '--authorization', str(authorization), '--resource-binding', str(resource_binding)]


def helper_bindings(root):
    """A separate host driver may import only the exact sealed helper bytes."""
    local = Path(__file__).resolve().parent
    result = {}
    for name in ('coordinator_contract.py', 'row_factory.py', 'runtime_admission.py'):
        digest = sha(local / name)
        assert sha(root / name) == digest, 'Host-driver helper differs from sealed bundle: ' + name
        result[name] = dict(executed_source=str(local / name), sha256=digest,
                            sealed_source=str(root / name))
    return result


def closed_group(root, group):
    lane = lane_path(root, group)
    if (lane / 'GROUP_RESOURCE_WAIT.json').exists():
        return dict(status='RESOURCE_WAIT_NO_AUTOMATIC_RETRY',
                    receipt=str(lane / 'GROUP_RESOURCE_WAIT.json'))
    result = read(lane / 'GROUP_RESULT.json')
    stops = read(lane / 'OWNED_STOP_RECEIPT.json')
    lifecycle = read(lane / 'LIFECYCLE.json')
    assert result['status'] == 'FROZEN_GROUP_COMPLETED'
    assert result['completed'] == group['condition_ids']
    assert type(result['normal_physical_attempts']) is int
    assert 0 <= result['normal_physical_attempts'] <= 12
    assert not lifecycle['error'] and not lifecycle['cleanup_errors']
    assert not stops['cleanup_errors'] and stops['states']
    assert all(value['state']['Running'] is False for value in stops['states'].values())
    return dict(status='VERIFIED_GROUP_CLOSED',
                normal_physical_attempts_completed=result['normal_physical_attempts'],
                evidence={name: {'path': str(lane / name), 'sha256': sha(lane / name)}
                          for name in ('GROUP_RESULT.json', 'OWNED_STOP_RECEIPT.json', 'LIFECYCLE.json')})


def halt(root, batch, reason):
    path = root / 'HALT_OBJECT7.json'
    try:
        save(path, dict(time=time.time(), source='pi05_batch_driver',
                        batch=str(batch), reason=reason, workers_not_restarted=True,
                        unknown_physical_outcomes_not_scored=True))
    except FileExistsError:
        pass


def run(root, node, authorization, resource_binding, wall_seconds=43200,
        poll_seconds=2.0, popen=None, identify=None, clock=None, sleep=None):
    """Observe retained Popen handles; never infer task failure from SSH state.

    Expiry stops this observer/new dispatch without signalling detached workers.
    Each coordinator retains its own exact CID cleanup responsibility.
    """
    popen = popen or subprocess.Popen
    identify = identify or ident
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    root, authorization, resource_binding = map(lambda p: Path(p).resolve(),
                                               (root, authorization, resource_binding))
    assert node in NODES and 1 <= wall_seconds <= 86400 and poll_seconds > 0
    frozen_files(root)
    helpers = helper_bindings(root)
    groups = selected_groups(root, node)
    assert not (root / 'HALT_OBJECT7.json').exists()
    assert authorization.is_file() and resource_binding.is_file()
    # The placement is precommitted for all28 groups, not allocated after results.
    batch = root / 'controllers' / ('PI05_BATCH_NODE' + node)
    batch.parent.mkdir(exist_ok=True)
    batch.mkdir(exist_ok=False)
    lock = (batch / 'batch.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    began, deadline = clock(), clock() + wall_seconds
    owner = identify(os.getpid())
    save(batch / 'LAUNCH.json', dict(time=time.time(), process=owner,
         source_path=str(Path(__file__).resolve()), source_sha256=sha(Path(__file__)),
         coordinator_sha256=sha(root / 'isolated_coordinator.py'),
         frozen_runtime_scope_sha256=sha(root / 'FROZEN_RUNTIME_SCOPE.json'),
         authorization={'path': str(authorization), 'sha256': sha(authorization)},
         resource_binding={'path': str(resource_binding), 'sha256': sha(resource_binding)},
         imported_helper_bindings=helpers,
         node=node, task_order=list(ORDER), local_tasks=[g['task_id'] for g in groups],
         maximum_active_groups_per_node=2, maximum_active_groups_two_nodes=4,
         observer_wall_seconds=wall_seconds, full_evaluation='PAUSED',
         model_environment_launches_by_driver=0,
         spawn_target='EXISTING_FROZEN_ISOLATED_COORDINATOR',
         actual_environment_and_physical_counts_require_child_receipts=True))
    queue, active, finished = list(groups), {}, []
    stopped_reason, fault = None, None

    def status(state):
        replace_status(batch / 'STATUS.json', dict(
            time=time.time(), status=state, node=node, process=owner,
            elapsed_seconds=clock() - began, queued_tasks=[g['task_id'] for g in queue],
            active=[dict(task_id=item['group']['task_id'], pid=item['process'].pid,
                         process_identity=item['identity'], receipt=item['launch_receipt'])
                    for item in active.values()], finished=finished,
            stop_reason=stopped_reason, technical_fault=fault,
            no_automatic_retry=True, detached_children_preserved=True,
            scientific_counts_require_original_raw=True, full_evaluation='PAUSED'))

    try:
        while queue or active:
            if (root / 'HALT_OBJECT7.json').exists():
                stopped_reason = stopped_reason or 'GLOBAL_TECHNICAL_HALT_NO_FURTHER_DISPATCH'
            for task, item in list(active.items()):
                code = item['process'].poll()
                if code is None:
                    try:
                        current = identify(item['process'].pid)
                    except (FileNotFoundError, ProcessLookupError):
                        # A normal exit can race the previous poll. Its Popen
                        # status is collected next iteration, without a restart.
                        if item['process'].poll() is not None:
                            continue
                        raise
                    assert current == item['identity'], 'Current child identity changed'
                    continue
                outcome = dict(task_id=task, returncode=code,
                               process_identity=item['identity'], host_end_unix=time.time())
                try:
                    observed = closed_group(root, item['group'])
                    if observed['status'] == 'RESOURCE_WAIT_NO_AUTOMATIC_RETRY':
                        stopped_reason = stopped_reason or observed['status']
                    else:
                        assert code == 0, 'Nonzero exit cannot prove normal group completion'
                    outcome.update(observed)
                except BaseException:
                    outcome.update(status='TECHNICAL_OR_UNKNOWN_STOP_NOT_PHYSICAL_FAILURE',
                                   error=traceback.format_exc())
                    fault = outcome['error']
                    stopped_reason = 'TECHNICAL_OR_UNKNOWN_STOP_NO_FURTHER_DISPATCH'
                    halt(root, batch, outcome)
                save(batch / ('EXIT_task' + str(task) + '.json'), outcome)
                finished.append(outcome)
                del active[task]
            if clock() >= deadline:
                stopped_reason = 'OBSERVER_DEADLINE_NO_RESTART_ACTIVE_WORKERS_PRESERVED'
                status('OBSERVATION_CLOSED_WORKERS_PRESERVED')
                return 2
            if stopped_reason and not active:
                status('STOPPED_NO_AUTOMATIC_RETRY')
                return 2
            # Two local workers total, and never two groups assigned the same card.
            while queue and not stopped_reason and len(active) < 2:
                assert not (root / 'HALT_OBJECT7.json').exists()
                used = {item['group']['gpu'] for item in active.values()}
                index = next((i for i, group in enumerate(queue) if group['gpu'] not in used), None)
                if index is None:
                    break
                group = queue.pop(index)
                task = group['task_id']
                assert not lane_path(root, group).exists(), 'Group already dispatched; inspect original owner, never restart'
                argv = command(root, group, authorization, resource_binding)
                save(batch / ('INTENT_task' + str(task) + '.json'), dict(
                    time=time.time(), argv=argv, group=group,
                    coordinator_sha256=sha(root / 'isolated_coordinator.py'),
                    maximum_normal_physical_attempts=12, actual_starts_unknown_until_child_evidence=True))
                logfile = batch / ('task' + str(task) + '.log')
                with logfile.open('x') as stream:
                    process = popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True, close_fds=True)
                # Child may close before /proc observation; retain exact Popen exit
                # evidence and stop instead of guessing a birth identity or retrying.
                try:
                    child_identity = identify(process.pid)
                    assert child_identity['argv'] == argv
                except BaseException:
                    code = process.poll()
                    save(batch / ('BIRTH_UNOBSERVED_task' + str(task) + '.json'), dict(
                        time=time.time(), pid=process.pid, intended_argv=argv,
                        observed_returncode=code, error=traceback.format_exc(),
                        live_child_may_remain=True, no_restart=True))
                    halt(root, batch, 'CHILD_BIRTH_UNOBSERVED_OBSERVE_EXACT_PID_NO_RESTART')
                    raise
                receipt = batch / ('CHILD_LAUNCH_task' + str(task) + '.json')
                # Retain the exact live handle before any subsequent receipt IO.
                active[task] = dict(group=group, process=process,
                                    identity=child_identity, launch_receipt=str(receipt))
                save(receipt, dict(time=time.time(), process=child_identity,
                     argv=argv, log_path=str(logfile), group=group,
                     coordinator_source_sha256=sha(root / 'isolated_coordinator.py'),
                     detached=True, automatic_retry=False))
            status('DRAINING_EXISTING_GROUPS' if stopped_reason else 'OBSERVING_FIXED_GROUPS')
            if queue or active:
                sleep(poll_seconds)
        status('LOCAL_PI05_GROUPS_CLOSED_ROOT_MUST_VERIFY_GLOBAL_STAGE')
        return 0
    except BaseException:
        fault = traceback.format_exc()
        halt(root, batch, fault)
        status('OBSERVER_FAILED_ACTIVE_WORKERS_PRESERVED_NO_RESTART')
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--node', choices=NODES, required=True)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--resource-binding', type=Path)
    parser.add_argument('--observer-wall-seconds', type=int, default=43200)
    args = parser.parse_args()
    if not args.run:
        print(json.dumps(dict(status='METADATA_PLAN_ONLY_NO_SPAWN',
             groups=selected_groups(args.root, args.node), task_order=list(ORDER),
             max_active_per_node=2, max_active_two_nodes=4,
             global_pi05_normal_attempts_max=84, later_model_gate='PI05_STAGE_COMPLETE.json',
             actual_model_environment_and_physical_attempts=0), indent=2))
        return
    assert args.authorization and args.resource_binding
    sys.exit(run(args.root, args.node, args.authorization, args.resource_binding,
                 wall_seconds=args.observer_wall_seconds))


if __name__ == '__main__':
    main()
