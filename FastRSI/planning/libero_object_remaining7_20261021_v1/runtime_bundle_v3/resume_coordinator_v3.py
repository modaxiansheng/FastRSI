"""One fresh homogeneous model/suite group from the frozen56-condition development trial.

Default --plan is metadata-only. Explicit --run checks the existing user scope,
exact assembled bytes, independent state binding and frozen placement.
No historical controller, recovery-state reuse or technical rerun is supported.
"""
from pathlib import Path
from same_cohort_resume import (load_plan, prepare_lane, reuse_closed_condition,
                               reuse_baseline, reuse_arm, policy_restore_command)
import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
import traceback
import urllib.request
from coordinator_contract import (groups, read, sha, frozen_files, validate_authorization,
                                  validate_arm, branch_final)
from row_factory import (branch_row, baseline_roster_row, BRANCHES,
                         branch_disposition, final_outcomes, paired_difference)
from runtime_admission import audit, ident, ResourceUnavailable
from launch_specs import POLICY, ENV, policy_command, environment_command, vision_command, wire_command

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def inspect(cid):
    assert re.fullmatch('[0-9a-f]{64}', cid)
    result = subprocess.run(['docker', 'inspect', cid], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)[0]
    assert value['Id'] == cid
    return value


def baseline_evidence(protocol_path, row, lane):
    protocol = read(protocol_path)
    assert protocol['software_ok'] is True and type(protocol['final_success']) is bool
    attempts = protocol['attempts']
    assert 1 <= len(attempts) <= 3
    raw = [read(protocol_path.parent / ('attempt_' + str(a['index'])) / 'RAW.json')
           for a in attempts]
    assert all(r['error'] is None and r['episode_completed'] is True
               and type(r['physical_success']) is bool for r in raw)
    assert all(r['initial_state_sha256'] == row['identity']['initial_state_sha256'] for r in raw)
    assert all(r['actions'] <= row['source']['max_steps'] for r in raw)
    assert len({r['reset_state_sha256'] for r in raw}) == 1
    assert len({r['first_input_sha256'] for r in raw}) == 1
    lifecycle = read(protocol_path.parent / 'CASE_LIFECYCLE.json')
    rng = lifecycle['resident_end']['receipt']['end_rng_hashes']
    key = hashlib.sha256(row['identity_id'].encode()).hexdigest()
    checkpoint_path = lane / 'resident_receipts' / ('CHECKPOINT_' + key + '.json')
    checkpoint = read(checkpoint_path)
    assert checkpoint['rng_hashes'] == rng and checkpoint['roundtrip_verified'] is True
    serialized = lane / 'resident_receipts' / ('CHECKPOINT_' + key + '.pickle')
    assert sha(serialized) == checkpoint['sha256']
    return protocol, raw, rng, checkpoint_path


def run_group(args):
    started = time.time()
    frozen_files(ROOT)
    scope = read(ROOT / 'FROZEN_RUNTIME_SCOPE.json')
    assert scope['status'] == 'ALL56_OBJECT7_CONDITIONS_FROZEN_BEFORE_TRIAL'
    assert scope['input_manifest_sha256'] == sha(ROOT / 'BUILD_INPUT_MANIFEST.json')
    assert scope['state_output_binding_sha256'] == sha(ROOT / 'STATE_OUTPUT_BINDING.json')
    assert len(scope['conditions']) == 56 and scope['normal_physical_attempts_max'] == 336
    selected = next(g for g in scope['groups']
                    if g['model'] == args.model and g['suite'] == args.suite and g['task_id'] == args.task)
    assert (selected['node'], selected['gpu']) == (args.node, args.gpu)
    rows = [r for r in scope['conditions'] if r['identity_id'] in selected['condition_ids']]
    assert [r['identity_id'] for r in rows] == selected['condition_ids']
    assert all(r['model'] == args.model and r['source']['suite'] == args.suite for r in rows)
    execution = read(ROOT / 'EXECUTION_BINDINGS.json')
    assert execution['status'] == 'EXECUTION_BYTES_AND56_OBJECT7_CONDITIONS_FROZEN'
    for field, name in [('input_manifest_sha256', 'BUILD_INPUT_MANIFEST.json'),
                        ('frozen_runtime_scope_sha256', 'FROZEN_RUNTIME_SCOPE.json'),
                        ('state_output_binding_sha256', 'STATE_OUTPUT_BINDING.json'),
                        ('authorization_scope_sha256', 'AUTHORIZATION_SCOPE.json')]:
        assert execution[field] == sha(ROOT / name)
    resources = validate_authorization(ROOT, read(args.authorization),
                   read(args.resource_binding), args.node, args.gpu, args.model, args.suite)
    resume_plan = load_plan(ROOT, args.model, args.task, args.node, args.gpu, args.continuation_plan)
    assert not (ROOT / 'HALT_OBJECT7.json').exists()
    controls = ROOT / 'controllers'
    controls.mkdir(exist_ok=True)
    lock = (controls / ('isolated_node' + args.node + '_gpu' + str(args.gpu) + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lane = ROOT / 'campaigns' / (args.suite + '__' + args.model) / ('task' + str(args.task)) / ('node' + args.node) / ('gpu' + str(args.gpu))
    try:
        lane.mkdir(parents=True, exist_ok=False)
    except BaseException:
        lock.close()
        raise
    save(lane / 'IDENTITY.json', dict(time=time.time(), process=ident(os.getpid()),
         model=args.model, suite=args.suite, task_id=args.task, node=args.node, gpu=args.gpu,
         authorization_path=str(args.authorization.resolve()), authorization_sha256=sha(args.authorization),
         resource_binding_path=str(args.resource_binding.resolve()), resource_binding_sha256=sha(args.resource_binding),
         frozen_scope_sha256=sha(ROOT / 'FROZEN_RUNTIME_SCOPE.json'),
         input_manifest_sha256=sha(ROOT / 'BUILD_INPUT_MANIFEST.json')))
    prepare_lane(resume_plan, lane, rows)
    (lane / 'POLICY_COST_EVENTS.jsonl').touch(exist_ok=False)
    owned, stops, completed = {}, {}, []
    pending_create_uncertainties = []
    policy, error, normal_attempts = None, None, 0
    reused_normal_attempts = 0
    dispatcher = (controls / ('isolated_node' + args.node + '_dispatch.lock')).open('a')
    cold = None

    def protect():
        assert not (ROOT / 'HALT_OBJECT7.json').exists(), 'Protective drain before next action'

    def launch(role, argv, folder, dispatch_locked=False):
        protect()
        if not dispatch_locked:
            fcntl.flock(dispatcher, fcntl.LOCK_EX)
        begin = time.time()
        receipt = dict(host_start_unix=begin, argv=argv, create_argv=None,
                       start_argv=None, returncode=None, container_id=None,
                       stdout='', stderr='', error=None)
        attempted_create = False
        name = None
        cidfile = lane / ('CID_' + hashlib.sha256(role.encode()).hexdigest() + '.txt')
        label_key = 'fastrsi.object7.launch_owner'
        owner = hashlib.sha256(json.dumps(dict(process=ident(os.getpid()),
            root=str(ROOT.resolve()), role=role, source_sha256=sha(Path(__file__))),
            sort_keys=True).encode()).hexdigest()
        expected_image = POLICY if role == 'policy' else ENV

        def exact_name():
            query = subprocess.run(['docker', 'inspect', name], capture_output=True,
                                   text=True, timeout=30)
            if query.returncode != 0:
                assert 'No such object' in query.stderr or 'No such container' in query.stderr, query.stderr
                return None
            values = json.loads(query.stdout)
            assert len(values) == 1 and values[0]['Name'].lstrip('/') == name
            return values[0]

        def register(bound):
            cid = bound['Id']
            assert re.fullmatch('[0-9a-f]{64}', cid)
            assert bound['Name'].lstrip('/') == name
            assert bound['Config']['Labels'].get(label_key) == owner
            assert bound['Image'] == expected_image
            owned[role] = cid  # Before docker start: every potentially live CID is cleanup-owned.
            receipt['container_id'] = cid
            return cid

        try:
            assert argv[:3] == ['docker', 'run', '-d']
            assert argv.count('--name') == 1 and not cidfile.exists()
            name = argv[argv.index('--name') + 1]
            assert exact_name() is None, 'Existing exact name is not a new execution opportunity'
            create = ['docker', 'create'] + argv[3:]
            index = create.index(expected_image)
            create[index:index] = ['--cidfile', str(cidfile), '--label', label_key + '=' + owner]
            receipt['create_argv'] = create
            save(lane / ('LAUNCH_INTENT_' + role + '.json'), dict(
                time=begin, role=role, exact_name=name, expected_image=expected_image,
                owner_label={label_key: owner}, cidfile=str(cidfile),
                argv=create, start_sent=False, scientific_attempts_require_raw=True))
            attempted_create = True
            result = subprocess.run(create, capture_output=True, text=True, timeout=90)
            receipt.update(create_returncode=result.returncode,
                           stdout=result.stdout, stderr=result.stderr)
            assert result.returncode == 0, (role, result.stderr)
            cid = result.stdout.strip()
            assert re.fullmatch('[0-9a-f]{64}', cid)
            assert cidfile.read_text().strip() == cid
            bound = inspect(cid)
            register(bound)
            assert bound['State']['Running'] is False, 'docker create must not execute the image'
            assert bound['HostConfig']['Privileged'] is False
            if role == 'policy':
                paths = {v['PathOnHost'] for v in bound['HostConfig']['Devices']}
                assert paths == {'/dev/kfd', '/dev/mkfd', '/dev/dri'}
                assert 'HIP_VISIBLE_DEVICES=' + str(args.gpu) in bound['Config']['Env']
                assert 'CUDA_VISIBLE_DEVICES=' + str(args.gpu) in bound['Config']['Env']
            else:
                assert not bound['HostConfig']['Devices'] and not bound['HostConfig']['DeviceRequests']
            save(folder / (role + '_BINDING.json'), bound)
            start = ['docker', 'start', cid]
            receipt['start_argv'] = start
            save(lane / ('START_INTENT_' + role + '.json'), dict(
                time=time.time(), role=role, container_id=cid, argv=start,
                known_current_owner_label={label_key: owner}))
            started_result = subprocess.run(start, capture_output=True, text=True, timeout=90)
            receipt.update(returncode=started_result.returncode,
                           start_stdout=started_result.stdout, start_stderr=started_result.stderr)
            assert started_result.returncode == 0, (role, started_result.stderr)
            assert started_result.stdout.strip() == cid
            return cid
        except BaseException:
            receipt['error'] = traceback.format_exc()
            # A CLI disconnect is not a retry. Observe only this pre-registered name/CID.
            if attempted_create and role not in owned:
                try:
                    bound = inspect(cidfile.read_text().strip()) if cidfile.exists() else exact_name()
                    if bound is not None:
                        register(bound)
                    else:
                        pending_create_uncertainties.append(dict(role=role, exact_name=name,
                            reason='CREATE_OUTCOME_UNKNOWN_NO_START_SENT', owner_label={label_key: owner}))
                except BaseException:
                    receipt['ownership_recovery_error'] = traceback.format_exc()
                    pending_create_uncertainties.append(dict(role=role, exact_name=name,
                        reason='EXACT_CREATE_IDENTITY_NOT_PROVED_NO_RESTART', owner_label={label_key: owner}))
            raise
        finally:
            receipt['host_end_unix'] = time.time()
            save(lane / ('LAUNCH_' + role + '.json'), receipt)
            if not dispatch_locked:
                fcntl.flock(dispatcher, fcntl.LOCK_UN)

    def wait(cid, deadline_seconds):
        deadline = time.monotonic() + deadline_seconds
        while inspect(cid)['State']['Running']:
            assert policy is None or inspect(policy)['State']['Running'], 'Owned policy exited'
            assert time.monotonic() < deadline, 'Bounded work deadline exceeded'
            time.sleep(2)
        assert inspect(cid)['State']['ExitCode'] == 0, inspect(cid)['State']

    def capture(role, folder):
        value = inspect(owned[role])
        assert value['State']['Running'] is False
        save(folder / ('STOP_' + role + '.json'), dict(time=time.time(),
             container_id=owned[role], image=value['Image'], state=value['State']))
        process = subprocess.run(['docker', 'logs', owned[role]], capture_output=True,
                                 text=True, timeout=30)
        (folder / ('LOG_' + role + '.txt')).write_text(process.stdout + process.stderr)

    try:
        # Exactly2 actual cold policy loads per node; hold until health is ready.
        cold_wait_started = time.time()
        deadline = time.monotonic() + 1800
        while cold is None:
            protect()
            assert time.monotonic() < deadline, 'Bounded cold-slot wait exceeded'
            for index in range(2):
                candidate = (controls / ('isolated_node' + args.node + '_cold' + str(index) + '.lock')).open('a')
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    cold = candidate
                    break
                except BlockingIOError:
                    candidate.close()
            if cold is None:
                time.sleep(1)
        save(lane / 'COLD_SLOT_ACQUIRED.json', dict(time=time.time(),
             wait_started=cold_wait_started, wait_seconds=time.time() - cold_wait_started,
             lock_path=str(cold.name), process=ident(os.getpid()), max_cold_per_node=2))
        fcntl.flock(dispatcher, fcntl.LOCK_EX)
        try:
            audit(ROOT, lane, resources, args.node, args.gpu)
            policy_argv = policy_command(lane, args.node, args.gpu,
                             args.model, selected['condition_ids'], args.suite)
            policy_argv = policy_restore_command(resume_plan, lane, policy_argv)
            policy = launch('policy', policy_argv, lane, dispatch_locked=True)
        finally:
            fcntl.flock(dispatcher, fcntl.LOCK_UN)
        deadline = time.monotonic() + 600
        while True:
            assert inspect(policy)['State']['Running'] and time.monotonic() < deadline
            try:
                with urllib.request.urlopen('http://127.0.0.1:' + str(31000 + 100 * args.gpu)
                                            + '/healthz', timeout=2) as response:
                    assert response.status == 200
                break
            except (OSError, TimeoutError):
                time.sleep(2)
        device = read(lane / 'DEVICE_BINDING.json')
        assert device['device_count'] == 1
        assert device['hip'] == device['cuda'] == str(args.gpu)
        save(lane / 'COLD_SLOT_RELEASED_AFTER_HEALTH.json', dict(time=time.time(),
             policy_container_id=policy, lock_path=str(cold.name), health_ready=True))
        cold.close()
        cold = None
        wire = lane / 'wire'
        wire.mkdir()
        cid = launch('wire', wire_command(wire, args.node, args.gpu, args.model), wire)
        wait(cid, 330)
        capture('wire', wire)
        checked = read(wire / 'RESULT.json')
        assert checked['accepted'] is True and checked['wire_horizon'] == selected['wire_horizon']
        assert checked['environment_constructors'] == checked['physical_actions'] == 0
        for row in rows:
            protect()
            retained = reuse_closed_condition(resume_plan, row, lane)
            if retained is not None:
                prior_count = retained['baseline_attempts'] + sum(
                    1 for arm in retained['branches'].values() if 'result' in arm)
                normal_attempts += prior_count
                reused_normal_attempts += prior_count
                completed.append(row['identity_id'])
                assert normal_attempts <= 6 * len(completed) <= selected['normal_attempts_max']
                continue
            case = lane / row['identity_id']
            case.mkdir()
            save(case / 'ROW.json', row)
            protocol_path = reuse_baseline(resume_plan, row, lane, case)
            if protocol_path is None:
                base = case / 'base'
                stream = base / row['physical_identity_id'] / 'stream'
                stream.mkdir(parents=True)
                save(stream.parent / 'ROSTER.json', [baseline_roster_row(row)])
                role = row['identity_id'] + '_base'
                save(base / 'ATTEMPT_INTENT.json', dict(time=time.time(), role=role,
                     condition=row['identity_id'], arm='shared_native_up_to3',
                     maximum_physical_attempts=3, actual_attempts_require_raw=True))
                cid = launch(role, environment_command(base, lane, args.node, args.gpu,
                                                        row, 'base'), base)
                wait(cid, 3660)
                capture(role, base)
                protocol_path = stream / (row['identity_id'] + '--base') / 'PROTOCOL_RESULT.json'
            else:
                reused_normal_attempts += len(read(protocol_path)['attempts'])
            protocol, raw, rng, checkpoint_path = baseline_evidence(protocol_path, row, lane)
            normal_attempts += len(protocol['attempts'])
            arms = {}
            if protocol['final_success'] is False:
                for branch in BRANCHES:
                    protect()
                    retained_arm = reuse_arm(resume_plan, row, branch, case)
                    if retained_arm is not None:
                        arms[branch] = retained_arm
                        old_count = 1 if 'result' in retained_arm else 0
                        normal_attempts += old_count
                        reused_normal_attempts += old_count
                        continue
                    if branch_disposition(row, branch) == 'SKIPPED_UNSUPPORTED':
                        folder = case / branch
                        folder.mkdir()
                        skipped = dict(status='SKIPPED_UNSUPPORTED', physical_attempts=0,
                            task_id=row['source']['task_id'], condition=row['identity_id'],
                            physical_success=None, raw_path=None, raw_sha256=None,
                            reason='No proved salad recovery in this frozen candidate version')
                        save(folder / 'ARM_UNAVAILABLE.json', skipped)
                        arms[branch] = skipped
                        continue
                    paired = branch_row(row, branch, rng, raw[0]['reset_state_sha256'],
                                        raw[0]['first_input_sha256'])
                    assert paired['skill'] == row['skill_by_branch'][branch]
                    folder = case / branch
                    folder.mkdir()
                    (folder / 'vision_rpc').mkdir()
                    mode = 'ordinary_fourth' if branch == 'ordinary' else 'paired_fourth'
                    save(folder / 'JOB.json', dict(row=paired, mode=mode))
                    vision = None
                    if branch != 'ordinary' and paired['skill'] is not None:
                        vr = row['identity_id'] + '_' + branch + '_vision'
                        vision = launch(vr, vision_command(folder, args.node, branch, paired), folder)
                        deadline = time.monotonic() + 180
                        while not (folder / 'vision_rpc/READY.json').exists():
                            assert inspect(vision)['State']['Running'] and time.monotonic() < deadline
                            time.sleep(1)
                    role = row['identity_id'] + '_' + branch + '_actor'
                    save(folder / 'ATTEMPT_INTENT.json', dict(time=time.time(), role=role,
                         condition=row['identity_id'], arm=branch, maximum_physical_attempts=1,
                         actual_attempts_require_raw=True))
                    cid = launch(role, environment_command(folder, lane, args.node, args.gpu,
                                                           paired, mode, branch), folder)
                    wait(cid, 1860)
                    capture(role, folder)
                    result = validate_arm(read(folder / 'RESULT.json'), paired)
                    entry = read(folder / 'RESIDENT_ENTRY.json')['receipt']
                    assert entry['paired_start_rng_hashes'] == rng
                    assert read(folder / 'INITIAL_INPUT_ACCEPTANCE.json')['reset_state_sha256'] == raw[0]['reset_state_sha256']
                    assert read(folder / 'INITIAL_INPUT_ACCEPTANCE.json')['initial_policy_input_sha256'] == raw[0]['first_input_sha256']
                    if vision is not None:
                        wait(vision, 180)
                        capture(vr, folder)
                        service = read(folder / 'vision_rpc/SERVICE_RESULT.json')
                        assert service['error'] is None and service['cleanup_error'] is None
                    normal_attempts += 1
                    arms[branch] = dict(result=result, raw_path=str(folder / 'RAW.json'),
                                        raw_sha256=sha(folder / 'RAW.json'))
            finals = final_outcomes(protocol['final_success'], arms)
            outcome = dict(time=time.time(), condition=row['identity_id'],
                 physical_identity=row['physical_identity_id'], model=args.model, suite=args.suite,
                 source=row['source'], skill_by_branch=row['skill_by_branch'],
                 first_success=protocol['first_success'], native_up_to3=protocol['final_success'],
                 baseline_protocol_path=str(protocol_path), baseline_protocol_sha256=sha(protocol_path),
                 baseline_attempts=len(protocol['attempts']), shared_baseline_end_rng=rng,
                 baseline_checkpoint_path=str(checkpoint_path), baseline_checkpoint_sha256=sha(checkpoint_path),
                 branches=arms, final=finals,
                 net_vs_ordinary=paired_difference(finals['candidate'], finals['ordinary']),
                 net_vs_frozen=paired_difference(finals['candidate'], finals['frozen']),
                 success_pooling=False, software_ok=True, compute_budget_matched=False,
                 fixed_physical_denominator=14, candidate_missing_not_scored=True, source_success_counted_as_trial=False,
                 same_cohort_continuation_plan_sha256=sha(args.continuation_plan),
                 previous_complete_physical_results_retained=True)
            save(case / 'CONDITION_RESULT.json', outcome)
            completed.append(row['identity_id'])
            assert normal_attempts <= 6 * len(completed) <= selected['normal_attempts_max']
        assert completed == selected['condition_ids']
        save(lane / 'GROUP_RESULT.json', dict(status='FROZEN_GROUP_COMPLETED',
             model=args.model, suite=args.suite, completed=completed,
             normal_physical_attempts=normal_attempts,
             reused_normal_physical_attempts=reused_normal_attempts,
             new_normal_physical_attempts=normal_attempts - reused_normal_attempts,
             same_cohort_continuation_plan_sha256=sha(args.continuation_plan),
             normal_physical_attempts_max=selected['normal_attempts_max'],
             whole_trial_conditions=56, whole_trial_normal_attempts_max=336))
    except ResourceUnavailable:
        error = traceback.format_exc()
        save(lane / 'GROUP_RESOURCE_WAIT.json', dict(time=time.time(), reason=error,
             completed=completed, last_normal_attempt_count=normal_attempts,
             classification='RESOURCE_OCCUPIED_OR_UNEXPLAINED_NOT_MODEL_OR_PHYSICAL_FAILURE',
             automatic_retry=False, unrelated_tasks_touched=False))
    except BaseException:
        error = traceback.format_exc()
        save(lane / 'GROUP_FAILURE.json', dict(time=time.time(), error=error,
             completed=completed, last_normal_attempt_count=normal_attempts,
             note='Error/startup/unfinished outcomes remain unknown, not physical failures; no automatic retry.'))
        try:
            save(ROOT / 'HALT_OBJECT7.json', dict(time=time.time(),
                 node=args.node, gpu=args.gpu, model=args.model, suite=args.suite,
                 error=error, old_controllers_affected=False))
        except FileExistsError:
            pass
    finally:
        cleanup_errors = list(pending_create_uncertainties)
        for role, cid in reversed(list(owned.items())):
            try:
                value = inspect(cid)
                if value['State']['Running']:
                    stopped = subprocess.run(['docker', 'stop', '--time', '30', cid],
                                             capture_output=True, text=True, timeout=60)
                    assert stopped.returncode == 0, stopped.stderr
                    value = inspect(cid)
                assert value['State']['Running'] is False
                stops[role] = dict(container_id=cid, image=value['Image'], state=value['State'])
            except BaseException:
                cleanup_errors.append(dict(role=role, container_id=cid, error=traceback.format_exc()))
        if cleanup_errors:
            try:
                save(ROOT / 'HALT_OBJECT7.json', dict(time=time.time(),
                     reason='OWNED_CLEANUP_INCOMPLETE_NO_NEW_LOADS', errors=cleanup_errors))
            except FileExistsError:
                pass
        # A failed health/load keeps its cold token until owned services stop.
        if cold is not None:
            cold.close()
        save(lane / 'OWNED_STOP_RECEIPT.json', dict(time=time.time(), states=stops,
             cleanup_errors=cleanup_errors, other_containers_touched=False))
        save(lane / 'LIFECYCLE.json', dict(host_start_unix=started, host_end_unix=time.time(),
             host_elapsed_seconds=time.time() - started, completed=completed, error=error,
             cleanup_errors=cleanup_errors, normal_physical_attempts_completed=normal_attempts,
             reused_normal_physical_attempts=reused_normal_attempts,
             new_normal_physical_attempts_completed=normal_attempts - reused_normal_attempts,
             prior_failed_runtime_preserved=True, same_cohort_continuation_plan_sha256=sha(args.continuation_plan),
             policy_and_cpu_load_hold_costs_retained=True, overlapping_intervals_not_summed=True,
             assistant_token_cost=None, money=None, full_evaluation='PAUSED'))
        lock.close()
        dispatcher.close()
        if cleanup_errors:
            raise RuntimeError('Owned container cleanup incomplete; inspect exact stop receipt')
    if error:
        raise RuntimeError(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--model', choices=['pi05', 'pi0', 'openvla', 'xvla'])
    parser.add_argument('--suite', choices=['libero_object'], default='libero_object')
    parser.add_argument('--task', type=int, choices=[0, 1, 2, 4, 6, 7, 9])
    parser.add_argument('--node', choices=['187', '188'])
    parser.add_argument('--gpu', type=int, choices=range(8))
    parser.add_argument('--authorization', type=Path, default=ROOT / 'AUTHORIZATION_SCOPE.json')
    parser.add_argument('--resource-binding', type=Path)
    parser.add_argument('--continuation-plan', type=Path, default=ROOT / 'CONTINUATION_PLAN.json')
    args = parser.parse_args()
    assert not (args.plan and args.run)
    if not args.run:
        print(json.dumps(dict(status='METADATA_PLAN_ONLY_NO_SUBPROCESS', groups=groups(),
             conditions=56, normal_physical_attempts_max=336, full_evaluation='PAUSED'), indent=2))
        return
    assert args.model and args.suite and args.task is not None and args.node and args.gpu is not None and args.resource_binding
    run_group(args)


if __name__ == '__main__':
    main()
