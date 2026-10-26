"""Same56 host continuation only. No mutable STATUS is a progression input.

The fixed plan is authoritative for reuse/partial/new routing. Only the original
coordinator or the separately pinned continuation coordinator can execute work.
Observer IO faults stop this CPU observer, not unrelated/in-flight physics.
"""
from pathlib import Path
import argparse
import errno
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

from coordinator_contract import read, sha, frozen_files, verify_pi05_stage
from runtime_admission import ident

ROOT = Path(__file__).resolve().parent
MODELS = ('pi05', 'pi0', 'openvla', 'xvla')
TASKS = (7, 0, 1, 4, 6, 9, 2)
NODES = ('187', '188')
CORE_NAMES = ('GROUP_RESULT.json', 'OWNED_STOP_RECEIPT.json', 'LIFECYCLE.json')


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def status(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(str(temporary), str(path))


def lane(root, group):
    return (root / 'campaigns' / ('libero_object__' + group['model']) /
            ('task' + str(group['task_id'])) / ('node' + group['node']) /
            ('gpu' + str(group['gpu'])))


def token_path(root, group):
    return root / 'closed_publications_v3' / ('CLOSED_' + group['model'] + '_task' + str(group['task_id']) + '.json')


def token_once(root, group):
    """One optional publication read. Only absent publication is pending."""
    try:
        return read(token_path(root, group))
    except FileNotFoundError:
        return None


def publish_exclusive(path, value):
    """Publish fully written bytes by exclusive hard link, never os.replace."""
    path = Path(path)
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    temporary = path.parent / ('.' + path.name + '.owner' + str(os.getpid()) + '.' + str(time.time_ns()))
    try:
        with temporary.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(path))  # EEXIST is blocking; no rewrite or retry.
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(raw).hexdigest()


def load_plan(root, plan_path):
    scope = read(root / 'FROZEN_RUNTIME_SCOPE.json')
    plan = read(plan_path)
    assert scope['status'] == 'ALL56_OBJECT7_CONDITIONS_FROZEN_BEFORE_TRIAL'
    assert len(scope['groups']) == 28 and len(scope['conditions']) == 56
    assert scope['normal_physical_attempts_max'] == 336
    assert scope['pi05_first_stage_normal_attempts_max'] == 84
    assert plan['scope'] == scope and len(plan['groups']) == 28
    assert plan['original_scope']['sha256'] == sha(root / 'FROZEN_RUNTIME_SCOPE.json')
    key = lambda g: (g['model'], g['task_id'])
    assert len({key(g) for g in plan['groups']}) == 28
    original = {key(g): g for g in scope['groups']}
    assert set(original) == {(m, t) for m in MODELS for t in TASKS}
    result = []
    for row in plan['groups']:
        source = original[key(row)]
        for name in ('model', 'suite', 'task_id', 'node', 'gpu', 'condition_ids'):
            assert row[name] == source[name]
        assert row['action'] in ('REUSE_COMPLETE_GROUP', 'RESUME_PARTIAL', 'NEW_GROUP')
        assert source['node'] in NODES and type(source['gpu']) is int and source['gpu'] in range(8)
        assert len(source['condition_ids']) == 2 and source['normal_attempts_max'] == 12
        assert source['suite'] == 'libero_object'
        if row['action'] != 'NEW_GROUP':
            assert row['model'] == 'pi05'
        result.append(row)
    return plan, result


def closed_evidence(root, group):
    folder = lane(root, group)
    result, stopped, life = [read(folder / name) for name in CORE_NAMES]
    assert result['status'] == 'FROZEN_GROUP_COMPLETED'
    assert result['completed'] == group['condition_ids']
    assert type(result['normal_physical_attempts']) is int and 0 <= result['normal_physical_attempts'] <= 12
    assert not stopped['cleanup_errors'] and stopped['states']
    assert all(item['state']['Running'] is False for item in stopped['states'].values())
    assert life['error'] is None and life['cleanup_errors'] == []
    condition_pins = {}
    for condition in group['condition_ids']:
        path = folder / condition / 'CONDITION_RESULT.json'
        document = read(path)
        assert document['software_ok'] is True and document['condition'] == condition
        assert type(document['native_up_to3']) is bool
        assert all(type(document['final'][branch]) is bool for branch in ('ordinary', 'frozen'))
        if group['task_id'] == 2 and document['native_up_to3'] is False:
            assert document['final']['candidate'] is None
            assert document['branches']['candidate']['status'] == 'SKIPPED_UNSUPPORTED'
        else:
            assert type(document['final']['candidate']) is bool
        condition_pins[condition] = dict(path=str(path), sha256=sha(path))
    return dict(normal_physical_attempts=result['normal_physical_attempts'],
                files={n: dict(path=str(folder / n), sha256=sha(folder / n)) for n in CORE_NAMES},
                conditions=condition_pins)


def validate_token(root, group, token, plan_sha):
    assert token['schema'] == 'fastrsi_object7_v3_immutable_closed_group_v1'
    assert token['continuation_plan_sha256'] == plan_sha
    assert token['frozen_scope_sha256'] == sha(root / 'FROZEN_RUNTIME_SCOPE.json')
    assert token['group'] == group and token['action'] == group['action']
    current = closed_evidence(root, group)  # Original result files: one strict pass, no retries/cache.
    assert token['closed_evidence'] == current
    return token


def global_halt(root, batch, reason):
    try:
        save(root / 'HALT_OBJECT7.json', dict(time=time.time(), source='node_host_controller_v3',
            batch=str(batch), reason=reason, physical_failure_not_inferred=True,
            no_automatic_restart=True))
    except FileExistsError:
        pass


def run(root, node, authorization, resource_binding, plan_path, wait_seconds=14400,
        wall_seconds=86400, poll_seconds=2, popen=None, identify=None, clock=None, sleep=None):
    popen, identify = popen or subprocess.Popen, identify or ident
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    began = clock()
    assert node in NODES and 1 <= wait_seconds <= 14400 and wait_seconds <= wall_seconds <= 86400
    root, authorization, resource_binding, plan_path = [Path(p).resolve() for p in
        (root, authorization, resource_binding, plan_path)]
    frozen_files(root)
    bindings = read(root / 'RECOVERY_SOURCE_BINDINGS.json')
    pins = {'host_controller_sha256': sha(Path(__file__)),
            'resume_coordinator_sha256': sha(root / 'resume_coordinator_v3.py'),
            'continuation_plan_sha256': sha(plan_path),
            'original_coordinator_sha256': sha(root / 'isolated_coordinator.py')}
    assert all(bindings[key] == digest for key, digest in pins.items())
    for name in ('coordinator_contract.py', 'row_factory.py', 'runtime_admission.py'):
        assert sha(Path(__file__).parent / name) == sha(root / name), name
    plan, all_groups = load_plan(root, plan_path)
    assert authorization.is_file() and resource_binding.is_file()
    assert not (root / 'HALT_OBJECT7.json').exists()
    controls = root / 'controllers'
    controls.mkdir(exist_ok=True)
    batch = controls / ('NODE_HOST_V3_' + node)
    batch.mkdir(exist_ok=False)
    lock = (batch / 'owner.lock').open('x')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    publications = root / 'closed_publications_v3'
    publications.mkdir(exist_ok=True)
    owner, active, finished = identify(os.getpid()), {}, []
    stop_reason, model_now, observer_fault = None, None, None
    deadline = began + wall_seconds
    save(batch / 'LAUNCH.json', dict(time=time.time(), process=owner, source_pins=pins,
         node=node, continuation_plan_path=str(plan_path), model_order=list(MODELS),
         task_order=list(TASKS), max_active_per_node=2, max_active_two_nodes=4,
         physical_identities=14, model_conditions=56, physical_attempt_limit=336,
         same56_continuation=True, retained_prior_costs=True, no_status_progression_reads=True,
         max_barrier_cpu_wait_seconds=wait_seconds, observer_wall_seconds=wall_seconds,
         direct_model_or_environment_calls=0, full_evaluation='PAUSED'))

    def update(label, queued=None):
        status(batch / 'STATUS.json', dict(time=time.time(), status=label, process=owner,
             node=node, model=model_now, elapsed_seconds=clock()-began,
             queued=[g['task_id'] for g in (queued or [])], finished=finished,
             active=[dict(group=v['group'], process=v['identity'], launch_receipt=v['receipt'])
                     for v in active.values()], stop_reason=stop_reason, observer_fault=observer_fault,
             no_auto_restart=True, status_is_ui_only=True, active_workers_preserved=True,
             model_and_physical_counts_require_child_evidence=True, money=None, assistant_tokens=None))

    def spawn(argv, key, group):
        assert not (root / 'HALT_OBJECT7.json').exists()
        save(batch / ('INTENT_' + key + '.json'), dict(time=time.time(), argv=argv, group=group,
             source_sha256=sha(Path(argv[2])), no_auto_restart=True))
        with (batch / (key + '.log')).open('x') as stream:
            process = popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                            start_new_session=True, close_fds=True)
        try:
            birth = identify(process.pid)
            assert birth['argv'] == argv
        except BaseException:
            save(batch / ('BIRTH_UNOBSERVED_' + key + '.json'), dict(time=time.time(),
                 pid=process.pid, intended_argv=argv, returncode=process.poll(),
                 error=traceback.format_exc(), live_worker_may_remain=True, restart=False))
            raise
        receipt = batch / ('CHILD_LAUNCH_' + key + '.json')
        active[key] = dict(process=process, identity=birth, group=group, receipt=str(receipt))
        save(receipt, dict(time=time.time(), process=birth, argv=argv, group=group,
             executable_sha256=sha(Path(argv[2])), detached=True, restart=False))
        return process

    def verify_live(item):
        try:
            current = identify(item['process'].pid)
        except (FileNotFoundError, ProcessLookupError):
            if item['process'].poll() is not None:
                return
            raise
        assert current == item['identity']

    def publish(group, source):
        evidence = closed_evidence(root, group)
        if group['action'] == 'REUSE_COMPLETE_GROUP':
            # Root has byte-materialized original closed metadata, not new runs.
            for name, pin in group['reused_group_files'].items():
                assert evidence['files'][name]['sha256'] == pin['sha256']
            for condition, pin in group['reused_condition_files'].items():
                assert evidence['conditions'][condition]['sha256'] == pin['sha256']
        value = dict(schema='fastrsi_object7_v3_immutable_closed_group_v1', time=time.time(),
             group=group, action=group['action'], publisher=owner, source=source,
             continuation_plan_sha256=pins['continuation_plan_sha256'],
             frozen_scope_sha256=sha(root / 'FROZEN_RUNTIME_SCOPE.json'), closed_evidence=evidence,
             no_new_trial_inferred_from_reuse=group['action'] == 'REUSE_COMPLETE_GROUP')
        digest = publish_exclusive(token_path(root, group), value)
        return dict(group=group, status='IMMUTABLE_GROUP_PUBLISHED', token=str(token_path(root, group)),
                    token_sha256=digest, normal_physical_attempts=evidence['normal_physical_attempts'])

    try:
        for model in MODELS:
            model_now = model
            if model != 'pi05':
                verify_pi05_stage(root)
            queue = sorted([g for g in all_groups if g['model'] == model and g['node'] == node],
                           key=lambda g: TASKS.index(g['task_id']))
            while queue or active:
                if (root / 'HALT_OBJECT7.json').exists():
                    stop_reason = stop_reason or 'GLOBAL_TECHNICAL_HALT_NO_NEW_DISPATCH'
                for key, item in list(active.items()):
                    code = item['process'].poll()
                    if code is None:
                        verify_live(item)
                        continue
                    group = item['group']
                    record = dict(time=time.time(), process=item['identity'], group=group, returncode=code)
                    resource_wait = lane(root, group) / 'GROUP_RESOURCE_WAIT.json'
                    if resource_wait.is_file():
                        record.update(status='RESOURCE_WAIT_NO_RETRY', evidence=read(resource_wait))
                        stop_reason = 'RESOURCE_WAIT_NO_RETRY_ROOT_INTERVENTION'
                    elif code != 0:
                        failure_path = lane(root, group) / 'GROUP_FAILURE.json'
                        # A bare process exit or an observation failure does not
                        # establish a coordinator failure. Require its immutable
                        # failure receipt; absence/IO stays CPU-only diagnosis.
                        failure = read(failure_path)
                        assert failure.get('error'), 'Missing actual coordinator failure evidence'
                        record['coordinator_failure_evidence'] = dict(
                            path=str(failure_path), sha256=sha(failure_path), failure=failure)
                        record.update(status='COORDINATOR_TECHNICAL_FAILURE_NOT_PHYSICAL_FAILURE')
                        global_halt(root, batch, record)
                        stop_reason = 'COORDINATOR_FAILURE_NO_NEW_DISPATCH'
                    else:
                        record.update(publish(group, dict(child_launch=item['receipt'], process=item['identity'], returncode=code)))
                    save(batch / ('EXIT_' + key + '.json'), record)
                    finished.append(record)
                    del active[key]
                if clock() >= deadline:
                    stop_reason = 'OBSERVER_DEADLINE_WORKERS_PRESERVED_NO_RESTART'
                    update('OBSERVATION_STOPPED_CPU_ONLY', queue)
                    return 2
                if stop_reason and not active:
                    update('STOPPED_NO_RETRY', queue)
                    return 2
                while queue and not stop_reason and len(active) < 2:
                    used = {v['group']['gpu'] for v in active.values()}
                    index = next((i for i, g in enumerate(queue)
                                  if g['action'] == 'REUSE_COMPLETE_GROUP' or g['gpu'] not in used), None)
                    if index is None:
                        break
                    group = queue.pop(index)
                    assert token_once(root, group) is None, 'Existing publication: diagnose original owner, never rerun'
                    if group['action'] == 'REUSE_COMPLETE_GROUP':
                        finished.append(publish(group, dict(kind='PINNED_PRIOR_GROUP_REUSE_ZERO_GPU', original_lane=group['original_lane'])))
                        continue
                    assert not lane(root, group).exists(), 'Group already reserved; no automatic restart'
                    entry = 'resume_coordinator_v3.py' if group['action'] == 'RESUME_PARTIAL' else 'isolated_coordinator.py'
                    field = 'resume_coordinator_sha256' if group['action'] == 'RESUME_PARTIAL' else 'original_coordinator_sha256'
                    assert sha(root / entry) == pins[field]
                    argv = [sys.executable, '-B', str(root / entry), '--run', '--model', model,
                            '--suite', 'libero_object', '--task', str(group['task_id']), '--node', node,
                            '--gpu', str(group['gpu']), '--authorization', str(authorization),
                            '--resource-binding', str(resource_binding)]
                    if group['action'] == 'RESUME_PARTIAL':
                        argv += ['--continuation-plan', str(plan_path)]
                    spawn(argv, model + '_task' + str(group['task_id']), group)
                update('DRAINING_OWNED_COORDINATORS' if stop_reason else 'OBSERVING_FIXED_MODEL_GROUPS', queue)
                if queue or active:
                    sleep(poll_seconds)

            if model == 'pi05':
                wait_deadline = min(deadline, clock() + wait_seconds)
                gate_lock = (controls / 'V3_PI05_STAGE_CLOSE.lock').open('a')
                gate_locked = False
                try:
                    while True:
                        if (root / 'HALT_OBJECT7.json').exists():
                            stop_reason = 'GLOBAL_HALT_WHILE_CPU_BARRIER_WAIT'
                            update('CPU_BARRIER_STOPPED')
                            return 2
                        if clock() >= wait_deadline:
                            stop_reason = 'PUBLICATION_WAIT_BOUNDED_TIMEOUT_OWNER_REVIEW_REQUIRED'
                            update('CPU_BARRIER_WAIT_TIMEOUT_NO_RETRY')
                            return 2
                        expected = [g for g in all_groups if g['model'] == 'pi05']
                        tokens = [token_once(root, group) for group in expected]
                        if any(token is None for token in tokens):
                            update('CPU_WAIT_IMMUTABLE_PI05_PUBLICATIONS')
                            sleep(poll_seconds)
                            continue
                        for group, token in zip(expected, tokens):
                            validate_token(root, group, token, pins['continuation_plan_sha256'])
                        try:
                            fcntl.flock(gate_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            gate_locked = True
                        except BlockingIOError:
                            sleep(poll_seconds)
                            continue
                        marker = root / 'PI05_STAGE_COMPLETE.json'
                        if marker.is_file():
                            verify_pi05_stage(root)
                        else:
                            save(controls / 'V3_PI05_STAGE_CLOSE_INTENT.json', dict(time=time.time(),
                                publisher=owner, plan_sha256=pins['continuation_plan_sha256'],
                                no_repeat_after_observer_loss=True))
                            argv = [sys.executable, '-B', str(root / 'close_pi05_stage.py')]
                            process = spawn(argv, 'pi05_stage_closer', dict(model='metadata', task_id=-1))
                            close_deadline = min(deadline, clock() + 60)
                            while process.poll() is None:
                                verify_live(active['pi05_stage_closer'])
                                assert clock() < close_deadline, 'Metadata closer unknown; never restart'
                                sleep(poll_seconds)
                            code = process.poll()
                            save(batch / 'EXIT_pi05_stage_closer.json', dict(time=time.time(), returncode=code,
                                 process=active['pi05_stage_closer']['identity']))
                            del active['pi05_stage_closer']
                            assert code == 0
                            verify_pi05_stage(root)
                        save(batch / 'PI05_STAGE_VERIFIED.json', dict(time=time.time(),
                             marker_path=str(marker), marker_sha256=sha(marker), original_gate=True))
                        break
                finally:
                    if gate_locked:
                        fcntl.flock(gate_lock, fcntl.LOCK_UN)
                    gate_lock.close()
        update('LOCAL_ALL_FOUR_MODELS_CLOSED_GLOBAL_AUDIT_REQUIRED')
        return 0
    except BaseException:
        observer_fault = traceback.format_exc()
        stop_reason = 'CPU_OBSERVER_NEEDS_DIAGNOSIS_ACTIVE_WORKERS_UNTOUCHED'
        # Deliberately no global_halt here: an observer/read/publication fault is
        # not an actual coordinator/physics failure and cannot drain other jobs.
        update('CPU_OBSERVER_STOPPED_NO_GLOBAL_HALT')
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
    parser.add_argument('--continuation-plan', type=Path)
    parser.add_argument('--publication-wait-seconds', type=int, default=14400)
    parser.add_argument('--observer-wall-seconds', type=int, default=86400)
    args = parser.parse_args()
    plan_path = args.continuation_plan or args.root / 'CONTINUATION_PLAN.json'
    if not args.run:
        plan, planned = load_plan(args.root, plan_path)
        print(json.dumps(dict(status='METADATA_PLAN_ONLY_NO_CHILDREN', groups=planned,
             same56=True, physical_attempts=0, full_evaluation='PAUSED'), indent=2))
        return
    assert args.authorization and args.resource_binding
    sys.exit(run(args.root, args.node, args.authorization, args.resource_binding, plan_path,
                 args.publication_wait_seconds, args.observer_wall_seconds))


if __name__ == '__main__':
    main()
