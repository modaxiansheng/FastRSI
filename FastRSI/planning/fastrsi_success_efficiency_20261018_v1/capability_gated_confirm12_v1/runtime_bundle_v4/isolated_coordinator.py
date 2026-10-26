"""One fresh homogeneous model/suite group from the frozen48-condition trial.

Default --plan is metadata-only. Explicit --run checks the existing user scope,
exact assembled bytes, independent state binding and frozen placement.
No historical controller, recovery-state reuse or technical rerun is supported.
"""
from pathlib import Path
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
from row_factory import branch_row, baseline_roster_row, BRANCHES
from runtime_admission import audit, ident
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
    assert scope['status'] == 'ALL48_RUNTIME_CONDITIONS_FROZEN_BEFORE_TRIAL'
    assert scope['input_manifest_sha256'] == sha(ROOT / 'BUILD_INPUT_MANIFEST.json')
    assert scope['state_output_binding_sha256'] == sha(ROOT / 'STATE_OUTPUT_BINDING.json')
    assert len(scope['conditions']) == 48 and scope['normal_physical_attempts_max'] == 288
    selected = next(g for g in scope['groups']
                    if g['model'] == args.model and g['suite'] == args.suite)
    assert (selected['node'], selected['gpu']) == (args.node, args.gpu)
    rows = [r for r in scope['conditions'] if r['identity_id'] in selected['condition_ids']]
    assert [r['identity_id'] for r in rows] == selected['condition_ids']
    assert all(r['model'] == args.model and r['source']['suite'] == args.suite for r in rows)
    execution = read(ROOT / 'EXECUTION_BINDINGS.json')
    assert execution['status'] == 'EXECUTION_BYTES_AND48_CONDITIONS_FROZEN'
    for field, name in [('input_manifest_sha256', 'BUILD_INPUT_MANIFEST.json'),
                        ('frozen_runtime_scope_sha256', 'FROZEN_RUNTIME_SCOPE.json'),
                        ('state_output_binding_sha256', 'STATE_OUTPUT_BINDING.json'),
                        ('authorization_scope_sha256', 'AUTHORIZATION_SCOPE.json')]:
        assert execution[field] == sha(ROOT / name)
    resources = validate_authorization(ROOT, read(args.authorization),
                   read(args.resource_binding), args.node, args.gpu, args.model, args.suite)
    assert not (ROOT / 'HALT_ISOLATED_CONFIRM12.json').exists()
    controls = ROOT / 'controllers'
    controls.mkdir(exist_ok=True)
    lock = (controls / ('isolated_node' + args.node + '_gpu' + str(args.gpu) + '.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lane = ROOT / 'campaigns' / (args.suite + '__' + args.model) / ('node' + args.node) / ('gpu' + str(args.gpu))
    lane.mkdir(parents=True, exist_ok=False)
    save(lane / 'IDENTITY.json', dict(time=time.time(), process=ident(os.getpid()),
         model=args.model, suite=args.suite, node=args.node, gpu=args.gpu,
         authorization_path=str(args.authorization.resolve()), authorization_sha256=sha(args.authorization),
         resource_binding_path=str(args.resource_binding.resolve()), resource_binding_sha256=sha(args.resource_binding),
         frozen_scope_sha256=sha(ROOT / 'FROZEN_RUNTIME_SCOPE.json'),
         input_manifest_sha256=sha(ROOT / 'BUILD_INPUT_MANIFEST.json')))
    (lane / 'POLICY_COST_EVENTS.jsonl').touch(exist_ok=False)
    owned, stops, completed = {}, {}, []
    policy, error, normal_attempts = None, None, 0
    dispatcher = (controls / ('isolated_node' + args.node + '_dispatch.lock')).open('a')
    cold = None

    def protect():
        assert not (ROOT / 'HALT_ISOLATED_CONFIRM12.json').exists(), 'Protective drain before next action'

    def launch(role, argv, folder, dispatch_locked=False):
        protect()
        if not dispatch_locked:
            fcntl.flock(dispatcher, fcntl.LOCK_EX)
        try:
            before = time.time()
            process = subprocess.run(argv, capture_output=True, text=True, timeout=90)
            cid = process.stdout.strip() if process.returncode == 0 else None
            if cid is not None:
                assert re.fullmatch('[0-9a-f]{64}', cid)
                owned[role] = cid
            save(lane / ('LAUNCH_' + role + '.json'), dict(
                 host_start_unix=before, host_end_unix=time.time(), argv=argv,
                 returncode=process.returncode, container_id=cid,
                 stdout=process.stdout, stderr=process.stderr))
            assert process.returncode == 0, (role, process.stderr)
            bound = inspect(cid)
            assert bound['HostConfig']['Privileged'] is False
            assert bound['Image'] == (POLICY if role == 'policy' else ENV)
            if role == 'policy':
                paths = {v['PathOnHost'] for v in bound['HostConfig']['Devices']}
                assert paths == {'/dev/kfd', '/dev/mkfd', '/dev/dri'}
                assert 'HIP_VISIBLE_DEVICES=' + str(args.gpu) in bound['Config']['Env']
                assert 'CUDA_VISIBLE_DEVICES=' + str(args.gpu) in bound['Config']['Env']
            else:
                assert not bound['HostConfig']['Devices'] and not bound['HostConfig']['DeviceRequests']
            save(folder / (role + '_BINDING.json'), bound)
            return cid
        finally:
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
            policy = launch('policy', policy_command(lane, args.node, args.gpu,
                             args.model, selected['condition_ids'], args.suite), lane,
                             dispatch_locked=True)
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
            case = lane / row['identity_id']
            case.mkdir()
            save(case / 'ROW.json', row)
            base = case / 'base'
            stream = base / row['physical_identity_id'] / 'stream'
            stream.mkdir(parents=True)
            save(stream.parent / 'ROSTER.json', [baseline_roster_row(row)])
            role = row['identity_id'] + '_base'
            cid = launch(role, environment_command(base, lane, args.node, args.gpu,
                                                    row, 'base'), base)
            wait(cid, 3660)
            capture(role, base)
            protocol_path = stream / (row['identity_id'] + '--base') / 'PROTOCOL_RESULT.json'
            protocol, raw, rng, checkpoint_path = baseline_evidence(protocol_path, row, lane)
            normal_attempts += len(protocol['attempts'])
            arms = {}
            if protocol['final_success'] is False:
                for branch in BRANCHES:
                    protect()
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
            finals = {branch: (True if protocol['final_success'] else
                               branch_final(False, arms[branch]['result']['physical_success']))
                      for branch in BRANCHES}
            outcome = dict(time=time.time(), condition=row['identity_id'],
                 physical_identity=row['physical_identity_id'], model=args.model, suite=args.suite,
                 source=row['source'], skill_by_branch=row['skill_by_branch'],
                 first_success=protocol['first_success'], native_up_to3=protocol['final_success'],
                 baseline_protocol_path=str(protocol_path), baseline_protocol_sha256=sha(protocol_path),
                 baseline_attempts=len(protocol['attempts']), shared_baseline_end_rng=rng,
                 baseline_checkpoint_path=str(checkpoint_path), baseline_checkpoint_sha256=sha(checkpoint_path),
                 branches=arms, final=finals,
                 net_vs_ordinary=int(finals['candidate']) - int(finals['ordinary']),
                 net_vs_frozen=int(finals['candidate']) - int(finals['frozen']),
                 success_pooling=False, software_ok=True, compute_budget_matched=False,
                 primary_denominator_remains48=True, source_success_counted_as_trial=False)
            save(case / 'CONDITION_RESULT.json', outcome)
            completed.append(row['identity_id'])
            assert normal_attempts <= 6 * len(completed) <= selected['normal_attempts_max']
        assert completed == selected['condition_ids']
        save(lane / 'GROUP_RESULT.json', dict(status='FROZEN_GROUP_COMPLETED',
             model=args.model, suite=args.suite, completed=completed,
             normal_physical_attempts=normal_attempts,
             normal_physical_attempts_max=selected['normal_attempts_max'],
             whole_trial_conditions=48, whole_trial_normal_attempts_max=288))
    except BaseException:
        error = traceback.format_exc()
        save(lane / 'GROUP_FAILURE.json', dict(time=time.time(), error=error,
             completed=completed, last_normal_attempt_count=normal_attempts,
             note='Error/startup/unfinished outcomes remain unknown, not physical failures; no automatic retry.'))
        try:
            save(ROOT / 'HALT_ISOLATED_CONFIRM12.json', dict(time=time.time(),
                 node=args.node, gpu=args.gpu, model=args.model, suite=args.suite,
                 error=error, old_controllers_affected=False))
        except FileExistsError:
            pass
    finally:
        cleanup_errors = []
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
                save(ROOT / 'HALT_ISOLATED_CONFIRM12.json', dict(time=time.time(),
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
    parser.add_argument('--suite', choices=['libero_object', 'libero_spatial', 'libero_goal', 'libero_10'])
    parser.add_argument('--node', choices=['187', '188'])
    parser.add_argument('--gpu', type=int, choices=range(8))
    parser.add_argument('--authorization', type=Path, default=ROOT / 'AUTHORIZATION_SCOPE.json')
    parser.add_argument('--resource-binding', type=Path)
    args = parser.parse_args()
    assert not (args.plan and args.run)
    if not args.run:
        print(json.dumps(dict(status='METADATA_PLAN_ONLY_NO_SUBPROCESS', groups=groups(),
             conditions=48, normal_physical_attempts_max=288, full_evaluation='PAUSED'), indent=2))
        return
    assert args.model and args.suite and args.node and args.gpu is not None and args.resource_binding
    run_group(args)


if __name__ == '__main__':
    main()
