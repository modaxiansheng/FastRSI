"""Own audited107 HCU2/3/6/7; four lanes, memory-gated expansion to eight whole-device lanes; same evaluation identity."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
SOURCE = ROOT
IMAGE = 'sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81'
HOST = 'j01r2n07'
BOOT = 'a9998682-7741-4ea7-bac1-51b6f2280f66'
TOKEN = 'fastrsi-original-three-suites-20261024-resource05'
CONTROL = ROOT / 'control05'
stop_requested = False


def write(name, value):
    p = CONTROL / name
    tmp = p.with_name(p.name + '.pending.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    tmp.replace(p)


def request_stop(*_):
    global stop_requested
    stop_requested = True


def command(argv, timeout=20):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(dict(argv=argv, code=p.returncode, stderr=p.stderr[-3000:]))
    return p.stdout


def inspect(cid):
    return json.loads(command(['docker', 'inspect', cid]))[0]


def memory(gpu):
    return int(Path('/sys/class/drm/card%d/device/mem_info_vram_used' % (gpu + 1)).read_text())


def initial_gate():
    assert os.uname().nodename == HOST
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == BOOT
    # These exact banks and the adapted lane were pinned during preparation.
    bindings = json.loads((ROOT / 'RESOURCE05_BINDINGS.json').read_text())
    assert hashlib.sha256((ROOT / 'run_lane_resource05.py').read_bytes()).hexdigest() == bindings['lane_sha256']
    for model, files in bindings['models'].items():
        for name, digest in files.items():
            assert hashlib.sha256((SOURCE / 'source_frozen' / (model + '_R') / name).read_bytes()).hexdigest() == digest
    assert all(memory(g) < 256 * 1024 ** 2 for g in [2, 3, 6, 7]), 'Authorized pair is not idle'
    smi = command(['bash', '-lc', 'source /opt/dtk/env.sh && hy-smi --showpids'])
    claimed_indices = {int(x) for match in re.findall(r"HCU Index:\s*\[([^]]*)\]", smi)
                       for x in re.findall(r'\d+', match)}
    assert not claimed_indices & {2, 3, 6, 7}, 'A KFD process already holds this pair'
    for row in bindings['resource05_overlays'].values():
        assert hashlib.sha256(Path(row['overlay']).read_bytes()).hexdigest() == row['overlay_sha256']
    inventory = []
    for cid in command(['docker', 'ps', '-q']).split():
        c = inspect(cid)
        env = dict(x.split('=', 1) for x in c['Config'].get('Env', []) if '=' in x)
        masks = [env[k] for k in ['HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'] if k in env]
        claimed = {int(x) for mask in masks for x in mask.split(',') if x.isdigit()}
        if claimed & {2, 3, 6, 7}:
            raise RuntimeError('Another container claims the authorized pair: ' + c['Name'])
        if c['HostConfig'].get('Devices') and not masks:
            # A parked user workspace is retained. No active code, owner lease,
            # memory, or device holder may be treated as this exception.
            assert c['Name'] == '/bevheight-plus-repair-eval-20261022'
            top = command(['docker', 'top', cid, '-eo', 'pid,ppid,comm,args'])
            assert len(top.strip().splitlines()) == 2 and top.strip().endswith('sleep infinity')
        inventory.append(dict(id=c['Id'], name=c['Name'], visible=masks, paused=c['State']['Paused']))
    write('PRELAUNCH_AUDIT.json', dict(time=time.time(), host=HOST, boot=BOOT,
          memory={g: memory(g) for g in [2, 3, 6, 7]}, containers=inventory,
          preserved_workspace_exception='Exact parked BEVHeight sleep-only container, no model or device holder'))


def cleanup(prefix, guard):
    receipts = []
    # Only exact new namespace containers can ever be stopped by this owner.
    for cid in command(['docker', 'ps', '-aq', '--filter', 'name=' + prefix]).split():
        c = inspect(cid)
        assert c['Name'].startswith('/' + prefix)
        if c['State']['Running']:
            command(['docker', 'stop', '-t', '15', c['Id']], 40)
        receipts.append(dict(id=c['Id'], name=c['Name'], state=inspect(c['Id'])['State']))
    if guard:
        c = inspect(guard)
        assert c['Name'] == '/' + TOKEN + '-guard'
        if c['State']['Running']:
            command(['docker', 'stop', '-t', '10', c['Id']], 30)
        receipts.append(dict(id=c['Id'], name=c['Name'], state=inspect(c['Id'])['State']))
    write('CLEANUP.json', dict(time=time.time(), containers=receipts, unrelated_work_touched=False))


def summarize():
    groups = []
    for p in sorted((ROOT / 'group_status').glob('*.json')):
        groups.append(dict(group=p.stem, **json.loads(p.read_text())))
    rows = []
    for campaign in (ROOT / 'campaigns').glob('*') if (ROOT / 'campaigns').exists() else []:
        records = [json.loads(p.read_text()) for p in campaign.glob('node*/lane*/case*/TERMINAL_CASE.json')]
        identities = {(x['task_id'], x['init_state_id']) for x in records}
        assert len(identities) == len(records), 'Duplicate closed identity'
        rows.append(dict(campaign=campaign.name, closed=len(records),
            physical_success=sum(x.get('physical_final_success') is True for x in records),
            physical_unknown=sum(type(x.get('physical_final_success')) is not bool for x in records),
            end_to_end_success=sum(x.get('end_to_end_success') is True for x in records),
            tokens_known=sum(x.get('qwen_total_tokens') or 0 for x in records),
            token_unknown_cases=sum(x.get('qwen_token_usage_complete') is not True for x in records)))
    write('PROGRESS.json', dict(time=time.time(), expected=6000,
          completed=sum(x['closed'] for x in rows), rows=rows, groups=groups,
          scope='All closed successes and failures; service/loading/technical costs retained separately'))


def main():
    assert ROOT.name == 'libero_three_suites_20261024_success_v2'
    parent = ROOT
    deadline = time.time() + 3800
    while not (parent / 'control03/OWNER_RESULT.json').exists():
        if time.time() > deadline: raise TimeoutError('Previous version did not drain')
        time.sleep(5)
    CONTROL.mkdir(exist_ok=True)
    with (CONTROL / 'owner.lock').open('a+') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (CONTROL / 'OWNER_RESULT.json').exists():
            raise RuntimeError('Closed owner must not restart automatically')
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        start = time.time()
        guard = None
        leases = []
        process = None
        result = dict(status='STARTING', pid=os.getpid(), started=start,
                      boot=BOOT, host=HOST, physical_gpus=[2, 3, 6, 7], node='107')
        write('OWNER_STATUS.json', result)
        prefix = 'yuchen-fastrsi-libero_original_best_20261024_'
        try:
            allocation = Path('/data/FastRSI/revisions/harness-rsi-v1/deployments/recovery-plan-20261007/gpu-allocation.lock')
            with allocation.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if (ROOT / 'HALT.json').exists():
                    prior = json.loads((ROOT / 'HALT.json').read_text())
                    assert prior.get('reason') == 'Resource03 drain after stop, error, or deadline'
                    (ROOT / 'HALT.json').rename(CONTROL / 'PREVIOUS_DISPATCH_HALT.json')
                initial_gate()
                for gpu in [2, 3, 6, 7]:
                    p = Path('/data/HCU-coordination/leases/gpu-' + str(gpu))
                    p.mkdir(exist_ok=False)
                    data = dict(owner=TOKEN, token=TOKEN, pid=os.getpid(), host=HOST,
                                boot=BOOT, gpu=gpu, root=str(ROOT), created=time.time())
                    (p / 'OWNER.json').write_text(json.dumps(data, indent=2))
                    leases.append(p)
                argv = ['docker', 'run', '-d', '--name', TOKEN + '-guard',
                        '--label', 'fastrsi.owner=' + TOKEN, '--network', 'none',
                        '--cpus', '1', '--memory', '8g', '--device', '/dev/kfd',
                        '--device', '/dev/mkfd', '--device', '/dev/dri',
                        '-e', 'HIP_VISIBLE_DEVICES=2,3,6,7', '-e', 'CUDA_VISIBLE_DEVICES=2,3,6,7',
                        '-e', 'PYTHONDONTWRITEBYTECODE=1',
                        '-v', '/opt/hyhal:/opt/hyhal:ro', '-v', '/etc/hfm:/etc/hfm:ro',
                        '-v', str(ROOT) + ':/eval:ro', '-v', str(CONTROL) + ':/control:rw',
                        IMAGE, 'bash', '-lc',
                        'source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec python /eval/gpu_guard_resource03.py']
                guard = command(argv, 45).strip()
                write('GUARD_LAUNCH.json', dict(argv=argv, id=guard, time=time.time()))
                until = time.time() + 120
                while not (CONTROL / 'GPU_GUARD_READY.json').exists():
                    assert inspect(guard)['State']['Running'], 'Allocation guard exited'
                    if time.time() >= until or stop_requested:
                        raise TimeoutError('Visible allocation guard readiness')
                    time.sleep(2)
                assert all(memory(g) > 256 * 1024 ** 2 for g in [2, 3, 6, 7])
                # Existing dynamic allocators now observe this pair as occupied.
            plan = json.loads((ROOT / 'RUN_PLAN.json').read_text())
            def run_slot(slot):
                for index in range(slot, len(plan['groups']), 8):
                    if stop_requested or (ROOT / 'HALT.json').exists():
                        return 'DRAINED'
                    group = plan['groups'][index]
                    py = PROJECT / 'envs/host-controller-v1/bin/python'
                    argv = [str(py), str(ROOT / 'run_lane_resource05.py'), '--model', group['model'],
                            '--arm', 'R', '--suite', group['suite'], '--variant', 'original',
                            '--node', '107', '--gpu-base', str({0:2,1:3,2:2,3:3,4:6,5:7,6:6,7:7}[slot]), '--slot-index', str(slot),
                            '--resource-profile', 'whole_gpu_eight_lanes_resource05', '--stage', 'full']
                    log = CONTROL / ('GROUP_%02d_%s_%s.log' % (index, group['model'], group['suite']))
                    with log.open('ab') as stream:
                        child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        write('GROUP_LAUNCH_%02d.json' % index, dict(time=time.time(), pid=child.pid, argv=argv,
                              start_ticks=Path('/proc/%d/stat' % child.pid).read_text().split()[21], logical_slot=slot, physical_gpus=[{0:2,1:3,2:2,3:3,4:6,5:7,6:6,7:7}[slot]], resource_profile='whole_gpu_eight_lanes_resource05'))
                        code = child.wait()
                    if code:
                        return dict(status='FAILED', group=group, code=code)
                return 'COMPLETE'
            def extra_gate(slot):
                initial = {2:0,3:1,6:4,7:5}[slot]
                group = plan['groups'][initial]
                campaign = ROOT / 'campaigns' / ('libero_original_best_20261024_' + group['model'] + '_R_' + group['suite'] + '_original')
                fresh = [p for p in campaign.glob('node*/lane*/case*/TERMINAL_CASE.json') if p.stat().st_mtime > start]
                if not fresh: return None
                gpu = {2:2,3:3,6:6,7:7}[slot]
                used = memory(gpu)
                total = int(Path('/sys/class/drm/card%d/device/mem_info_vram_total' % (gpu+1)).read_text())
                reserve_gib = 30 if plan['groups'][slot]['model']=='pi0' else 33
                if used + reserve_gib*1024**3 > total*.90: return None
                available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
                if available < 100*1024**3: return None
                return dict(time=time.time(), slot=slot, physical_gpu=gpu, used_bytes=used,
                            total_bytes=total, predicted_bytes=used+reserve_gib*1024**3,
                            memory_admission_ceiling_fraction=.90, native_case_evidence=str(fresh[0]),
                            placement='Two independent policy/verifier/environment lanes per GPU; no shared scenes or resident RNG')
            result.update(status='RUNNING', parallel_lanes=4, maximum_parallel_lanes=8,
                          allocation_guard=guard, maximum_native_replans=5, updated=time.time())
            write('OWNER_STATUS.json', result)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                jobs = {slot: pool.submit(run_slot, slot) for slot in [0,1,4,5]}
                pending = [2,3,6,7]
                while pending or not all(j.done() for j in jobs.values()):
                    summarize()
                    errors = [j.result() for j in jobs.values() if j.done() and j.result() != 'COMPLETE']
                    halted = errors or stop_requested or time.time()-start > 96*3600
                    if halted:
                        pending.clear()
                        if not (ROOT / 'HALT.json').exists():
                            (ROOT / 'HALT.json').write_text(json.dumps(dict(time=time.time(), reason='Resource05 drain after stop, error, or deadline', errors=errors)))
                    else:
                        for slot in list(pending):
                            gate = extra_gate(slot)
                            # If no current lane remains, the new slot has exclusive pair capacity.
                            if gate is None and all(j.done() for j in jobs.values()):
                                gate = dict(time=time.time(),slot=slot,reason='Earlier slots all finished; no extra concurrency')
                            if gate is not None:
                                write('EXPAND_SLOT_%d.json' % slot, gate)
                                jobs[slot] = pool.submit(run_slot, slot)
                                pending.remove(slot)
                                result.update(parallel_lanes=len(jobs), updated=time.time())
                                write('OWNER_STATUS.json', result)
                    time.sleep(20)
                result['slots'] = {slot: j.result() for slot,j in jobs.items()}
            result['status'] = 'COMPLETE' if len(jobs)==8 and all(x == 'COMPLETE' for x in result['slots'].values()) else 'STOPPED_FOR_DIAGNOSIS'
            result['finished'] = time.time()
        except BaseException:
            result.update(status='STOPPED_FOR_DIAGNOSIS', error=traceback.format_exc(), finished=time.time())
        finally:
            try:
                cleanup(prefix, guard)
            finally:
                for p in leases:
                    d = json.loads((p / 'OWNER.json').read_text())
                    assert d['token'] == TOKEN and d['pid'] == os.getpid()
                    write('RELEASED_GPU_' + str(d['gpu']) + '.json', dict(**d, released=time.time()))
                    (p / 'OWNER.json').unlink()
                    p.rmdir()
                summarize()
                result['elapsed_seconds'] = time.time() - start
                write('OWNER_RESULT.json', result)
                write('OWNER_STATUS.json', result)
                print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
