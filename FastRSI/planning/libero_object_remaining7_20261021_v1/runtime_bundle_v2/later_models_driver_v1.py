"""Independent host observer/dispatch version; all scientific actors stay frozen.

Wait for seven real pi05 group closeouts, serialize the original stage closer,
then run pi0 -> openvla -> xvla locally with at most two frozen groups active.
No SSH, Docker, Env, model imports, new identities or automatic retries.
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

from coordinator_contract import read, sha, frozen_files, verify_pi05_stage
from runtime_admission import ident

ROOT = Path(__file__).resolve().parent
MODELS = ('pi0', 'openvla', 'xvla')
ORDER = (7, 0, 1, 4, 6, 9, 2)
NODES = ('187', '188')


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def status_write(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(str(temporary), str(path))


def groups(root):
    value = read(root / 'FROZEN_RUNTIME_SCOPE.json')
    assert value['status'] == 'ALL56_OBJECT7_CONDITIONS_FROZEN_BEFORE_TRIAL'
    assert len(value['groups']) == 28 and len(value['conditions']) == 56
    assert value['normal_physical_attempts_max'] == 336
    assert value['pi05_first_stage_normal_attempts_max'] == 84
    actual = value['groups']
    assert {(g['model'], g['task_id']) for g in actual} == {
        (model, task) for model in ('pi05',) + MODELS for task in ORDER}
    for group in actual:
        assert group['suite'] == 'libero_object' and group['node'] in NODES
        assert type(group['gpu']) is int and group['gpu'] in range(8)
        assert len(group['condition_ids']) == 2 and group['normal_attempts_max'] == 12
    return actual


def lane_path(root, group):
    return (root / 'campaigns' / ('libero_object__' + group['model']) /
            ('task' + str(group['task_id'])) / ('node' + group['node']) /
            ('gpu' + str(group['gpu'])))


def closure(root, group):
    lane = lane_path(root, group)
    if (lane / 'GROUP_RESOURCE_WAIT.json').exists():
        return dict(status='RESOURCE_WAIT_NO_RETRY', path=str(lane / 'GROUP_RESOURCE_WAIT.json'))
    result = read(lane / 'GROUP_RESULT.json')
    stopped, lifecycle = read(lane / 'OWNED_STOP_RECEIPT.json'), read(lane / 'LIFECYCLE.json')
    assert result['status'] == 'FROZEN_GROUP_COMPLETED'
    assert result['completed'] == group['condition_ids']
    assert type(result['normal_physical_attempts']) is int and 0 <= result['normal_physical_attempts'] <= 12
    assert not lifecycle['error'] and not lifecycle['cleanup_errors']
    assert not stopped['cleanup_errors'] and stopped['states']
    assert all(row['state']['Running'] is False for row in stopped['states'].values())
    return dict(status='VERIFIED_GROUP_CLOSED', normal_physical_attempts_completed=result['normal_physical_attempts'],
                evidence={name: dict(path=str(lane / name), sha256=sha(lane / name))
                          for name in ('GROUP_RESULT.json', 'OWNED_STOP_RECEIPT.json', 'LIFECYCLE.json')})


def pi05_ready(root, all_groups):
    """Use the producer's atomic publication before reading immutable closeouts.

    The original coordinator writes exclusive files, not atomic replacements.
    Its host driver's atomic STATUS is published only after Popen exit and
    closure validation. Without that publication, wait or let root invoke the
    original closer; never treat a partially written result as a failed task.
    """
    for node in NODES:
        local = [g for g in all_groups if g['model'] == 'pi05' and g['node'] == node]
        if not local:
            continue
        producer = root / 'controllers' / ('PI05_BATCH_NODE' + node)
        if not (producer / 'STATUS.json').is_file():
            return False
        published = read(producer / 'STATUS.json')
        assert published['node'] == node
        launch = read(producer / 'LAUNCH.json')
        assert launch['frozen_runtime_scope_sha256'] == sha(root / 'FROZEN_RUNTIME_SCOPE.json')
        assert launch['coordinator_sha256'] == sha(root / 'isolated_coordinator.py')
        completed = {row['task_id'] for row in published['finished']
                     if row.get('returncode') == 0 and row.get('status') == 'VERIFIED_GROUP_CLOSED'}
        if not {g['task_id'] for g in local} <= completed:
            return False
    for group in all_groups:
        if group['model'] != 'pi05':
            continue
        lane = lane_path(root, group)
        if not all((lane / name).is_file() for name in
                   ('GROUP_RESULT.json', 'OWNED_STOP_RECEIPT.json', 'LIFECYCLE.json')):
            return False
        if closure(root, group)['status'] != 'VERIFIED_GROUP_CLOSED':
            return False
        if not all((lane / condition / 'CONDITION_RESULT.json').is_file()
                   for condition in group['condition_ids']):
            return False
    return True


def helper_pins(root):
    local = Path(__file__).resolve().parent
    result = {}
    for name in ('coordinator_contract.py', 'row_factory.py', 'runtime_admission.py'):
        digest = sha(local / name)
        assert sha(root / name) == digest, 'Host helper differs from actual sealed bundle: ' + name
        result[name] = dict(executed_source=str(local / name), sealed_source=str(root / name), sha256=digest)
    return result


def halt(root, batch, reason):
    try:
        save(root / 'HALT_OBJECT7.json', dict(time=time.time(), source='later_models_driver_v1',
             batch=str(batch), reason=reason, no_restart=True, unknown_results_not_physical_failures=True))
    except FileExistsError:
        pass


def run(root, node, authorization, resource_binding, wait_seconds=14400, wall_seconds=86400,
        poll_seconds=2, popen=None, identify=None, clock=None, sleep=None):
    popen, identify = popen or subprocess.Popen, identify or ident
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    began = clock()
    deadline, wait_deadline = began + wall_seconds, began + wait_seconds
    assert node in NODES and 1 <= wait_seconds <= 14400 and wait_seconds <= wall_seconds <= 86400
    root, authorization, resource_binding = [Path(p).resolve() for p in (root, authorization, resource_binding)]
    frozen_files(root)
    helpers, all_groups = helper_pins(root), groups(root)
    assert authorization.is_file() and resource_binding.is_file()
    assert not (root / 'HALT_OBJECT7.json').exists()
    batch = root / 'controllers' / ('LATER_MODELS_V1_NODE' + node)
    batch.parent.mkdir(exist_ok=True)
    batch.mkdir(exist_ok=False)
    lock = (batch / 'observer.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    owner, active, finished = identify(os.getpid()), {}, []
    stop_reason, fault, model_now = None, None, None
    source_pins = {name: sha(root / name) for name in ('isolated_coordinator.py', 'close_pi05_stage.py')}
    save(batch / 'LAUNCH.json', dict(time=time.time(), process=owner,
         observer_source=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__))),
         frozen_runtime_scope_sha256=sha(root / 'FROZEN_RUNTIME_SCOPE.json'),
         imported_helpers=helpers, frozen_entrypoints=source_pins,
         authorization=dict(path=str(authorization), sha256=sha(authorization)),
         resource_binding=dict(path=str(resource_binding), sha256=sha(resource_binding)),
         model_order=list(MODELS), task_order=list(ORDER), node=node,
         max_active_groups_per_node=2, max_active_two_nodes=4,
         maximum_pi05_gate_cpu_wait_seconds=wait_seconds,
         maximum_observer_wall_seconds=wall_seconds, includes_wait_time=True,
         direct_env_or_model_launches_by_host_observer=0,
         actual_child_model_env_physical_counts_require_original_receipts=True,
         full_evaluation='PAUSED', new_scope=False, training=False))

    def state(label, queued=None):
        status_write(batch / 'STATUS.json', dict(time=time.time(), status=label,
            process=owner, node=node, model=model_now, elapsed_seconds=clock() - began,
            queued_tasks=[g['task_id'] for g in (queued or [])], finished=finished,
            active=[dict(model=item['group']['model'], task_id=item['group']['task_id'],
                         process=item['identity'], launch_receipt=item['receipt']) for item in active.values()],
            stop_reason=stop_reason, fault=fault, no_automatic_retry=True,
            detached_children_preserved=True, monetary_cost=None, assistant_tokens=None,
            lifecycle_wait_and_observer_wall_preserved=True, full_evaluation='PAUSED'))

    def launch_owned(argv, key, log_name):
        assert not (root / 'HALT_OBJECT7.json').exists()
        save(batch / ('INTENT_' + key + '.json'), dict(time=time.time(), argv=argv,
             actual_environment_or_physical_starts_unknown_until_child_receipts=True))
        with (batch / log_name).open('x') as stream:
            process = popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                            start_new_session=True, close_fds=True)
        try:
            birth = identify(process.pid)
            assert birth['argv'] == argv
        except BaseException:
            save(batch / ('BIRTH_UNOBSERVED_' + key + '.json'), dict(
                time=time.time(), pid=process.pid, intended_argv=argv,
                observed_returncode=process.poll(), error=traceback.format_exc(),
                child_may_remain=True, automatic_retry=False))
            raise
        receipt = batch / ('CHILD_LAUNCH_' + key + '.json')
        # Register before receipt IO so outer error status retains the live PID.
        active[key] = dict(process=process, identity=birth, receipt=str(receipt),
                           group=dict(model='metadata', task_id=-1))
        save(receipt, dict(time=time.time(), process=birth, argv=argv,
             executable_source_sha256=sha(Path(argv[2])), log_path=str(batch / log_name),
             detached=True, automatic_retry=False))
        return process, birth, receipt

    def observe_live(item):
        try:
            current = identify(item['process'].pid)
        except (FileNotFoundError, ProcessLookupError):
            if item['process'].poll() is not None:
                return
            raise
        assert current == item['identity'], 'Exact child birth/argv changed; do not restart'

    try:
        # Shared filesystem lock: at most one host invokes the original closer.
        gate_lock = (batch.parent / 'OBJECT7_PI05_STAGE_CLOSE.lock').open('a')
        gate_locked = False
        try:
            while True:
                assert not (root / 'HALT_OBJECT7.json').exists(), 'Global technical HALT while CPU waiting'
                if clock() >= wait_deadline or clock() >= deadline:
                    stop_reason = 'BOUNDED_CPU_WAIT_ENDED_NO_MODEL_DISPATCH_NO_RETRY'
                    state('WAIT_TIMEOUT_NO_MODEL_DISPATCH')
                    return 2
                marker = root / 'PI05_STAGE_COMPLETE.json'
                if marker.exists() or pi05_ready(root, all_groups):
                    try:
                        fcntl.flock(gate_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        gate_locked = True
                    except BlockingIOError:
                        state('WAIT_GLOBAL_PI05_CLOSER_LOCK')
                        sleep(poll_seconds)
                        continue
                    if marker.exists():
                        verify_pi05_stage(root)
                    else:
                        assert pi05_ready(root, all_groups)
                        assert sha(root / 'close_pi05_stage.py') == source_pins['close_pi05_stage.py']
                        argv = [sys.executable, '-B', str(root / 'close_pi05_stage.py')]
                        # Flock alone is not enough after observer death/timeout:
                        # a detached closer may still exist after its lock owner
                        # exits. The durable exclusive claim prevents a second
                        # host from silently launching that metadata work again.
                        save(batch.parent / 'OBJECT7_PI05_STAGE_CLOSE_INTENT.json', dict(
                            time=time.time(), observer=owner, node=node, batch=str(batch),
                            argv=argv, closer_source_sha256=source_pins['close_pi05_stage.py'],
                            frozen_scope_sha256=sha(root / 'FROZEN_RUNTIME_SCOPE.json'),
                            missing_marker_after_this_claim_requires_diagnosis=True,
                            automatic_retry=False))
                        process, birth, receipt = launch_owned(argv, 'pi05_closer', 'pi05_closer.log')
                        closer_deadline = min(deadline, wait_deadline, clock() + 60)
                        while process.poll() is None:
                            observe_live(active['pi05_closer'])
                            assert clock() < closer_deadline, 'Metadata closer observation deadline; never restart'
                            sleep(poll_seconds)
                        code = process.poll()
                        save(batch / 'EXIT_pi05_closer.json', dict(time=time.time(),
                             process=birth, returncode=code, launch_receipt=str(receipt)))
                        del active['pi05_closer']
                        assert code == 0, 'Original stage closer failed; keep all original records'
                        verify_pi05_stage(root)
                    save(batch / 'PI05_GATE_VERIFIED.json', dict(time=time.time(),
                         marker_path=str(marker), marker_sha256=sha(marker),
                         global_close_lock=str(gate_lock.name), original_closer_source_sha256=source_pins['close_pi05_stage.py']))
                    break
                state('CPU_WAIT_FOR_SEVEN_ORIGINAL_PI05_GROUP_CLOSEOUTS')
                sleep(poll_seconds)
        finally:
            if gate_locked:
                fcntl.flock(gate_lock, fcntl.LOCK_UN)
            gate_lock.close()

        for model in MODELS:
            model_now = model
            verify_pi05_stage(root)
            queue = sorted((g for g in all_groups if g['model'] == model and g['node'] == node),
                           key=lambda group: ORDER.index(group['task_id']))
            while queue or active:
                if (root / 'HALT_OBJECT7.json').exists():
                    stop_reason = stop_reason or 'GLOBAL_TECHNICAL_HALT_NO_FURTHER_DISPATCH'
                for key, item in list(active.items()):
                    code = item['process'].poll()
                    if code is None:
                        observe_live(item)
                        continue
                    outcome = dict(model=model, task_id=item['group']['task_id'],
                                   returncode=code, process=item['identity'], host_end_unix=time.time())
                    try:
                        value = closure(root, item['group'])
                        if value['status'] == 'RESOURCE_WAIT_NO_RETRY':
                            stop_reason = stop_reason or value['status']
                        else:
                            assert code == 0
                        outcome.update(value)
                    except BaseException:
                        fault = traceback.format_exc()
                        outcome.update(status='TECHNICAL_OR_UNKNOWN_NOT_PHYSICAL_FAILURE', error=fault)
                        stop_reason = 'TECHNICAL_OR_UNKNOWN_NO_FURTHER_DISPATCH'
                        halt(root, batch, outcome)
                    save(batch / ('EXIT_' + key + '.json'), outcome)
                    finished.append(outcome)
                    del active[key]
                if clock() >= deadline:
                    stop_reason = 'OBSERVER_TIMEOUT_DETACHED_WORKERS_PRESERVED_NO_RESTART'
                    state('OBSERVATION_CLOSED_WORKERS_PRESERVED', queue)
                    return 2
                if stop_reason and not active:
                    state('STOPPED_NO_AUTOMATIC_RETRY', queue)
                    return 2
                while queue and not stop_reason and len(active) < 2:
                    assert not (root / 'HALT_OBJECT7.json').exists()
                    used = {item['group']['gpu'] for item in active.values()}
                    index = next((i for i, g in enumerate(queue) if g['gpu'] not in used), None)
                    if index is None:
                        break
                    group = queue.pop(index)
                    assert not lane_path(root, group).exists(), 'Existing group is not a restart opportunity'
                    assert sha(root / 'isolated_coordinator.py') == source_pins['isolated_coordinator.py']
                    key = model + '_task' + str(group['task_id'])
                    argv = [sys.executable, '-B', str(root / 'isolated_coordinator.py'), '--run',
                            '--model', model, '--suite', 'libero_object', '--task', str(group['task_id']),
                            '--node', node, '--gpu', str(group['gpu']), '--authorization', str(authorization),
                            '--resource-binding', str(resource_binding)]
                    launch_owned(argv, key, key + '.log')
                    active[key]['group'] = group
                state('DRAINING_OWNED_GROUPS' if stop_reason else 'OBSERVING_FIXED_MODEL_GROUPS', queue)
                if queue or active:
                    sleep(poll_seconds)
            save(batch / ('LOCAL_MODEL_CLOSED_' + model + '.json'), dict(time=time.time(),
                 model=model, local_groups=len([g for g in all_groups if g['model'] == model and g['node'] == node]),
                 no_result_pooling=True, marker_is_local_not_benchmark_completion=True))
        state('LOCAL_LATER_MODEL_GROUPS_CLOSED_ROOT_MUST_AUDIT_GLOBAL_RESULTS')
        return 0
    except BaseException:
        fault = traceback.format_exc()
        halt(root, batch, fault)
        state('OBSERVER_FAILED_DETACHED_WORKERS_PRESERVED_NO_RESTART')
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
    parser.add_argument('--pi05-wait-seconds', type=int, default=14400)
    parser.add_argument('--observer-wall-seconds', type=int, default=86400)
    args = parser.parse_args()
    if not args.run:
        print(json.dumps(dict(status='METADATA_PLAN_ONLY_NO_SPAWN', model_order=MODELS,
             task_order=ORDER, local_groups=[g for g in groups(args.root) if g['node'] == args.node
                    and g['model'] in MODELS], maximum_active_per_node=2,
             maximum_active_two_nodes=4, pi05_gate='ORIGINAL_GLOBAL_SEVEN_GROUP_CLOSEOUT',
             actual_model_env_or_physical_attempts=0), indent=2))
        return
    assert args.authorization and args.resource_binding
    sys.exit(run(args.root, args.node, args.authorization, args.resource_binding,
                 wait_seconds=args.pi05_wait_seconds, wall_seconds=args.observer_wall_seconds))


if __name__ == '__main__':
    main()
