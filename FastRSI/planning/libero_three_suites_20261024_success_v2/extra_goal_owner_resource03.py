"""Admit up to two extra Goal groups on spare memory in the owned policy GPUs."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / 'control03'
PARENT_PID = 604189
PARENT_START = '910896065'
BOOT = 'a9998682-7741-4ea7-bac1-51b6f2280f66'
START = time.time()
JOBS = [(6, 2, 'pi05'), (7, 6, 'xvla')]


def write(name, data):
    p = CONTROL / name
    tmp = p.with_name(p.name + '.pending.' + str(os.getpid()))
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(p)


def parent_live():
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == BOOT
    try:
        assert Path('/proc/%d/stat' % PARENT_PID).read_text().split()[21] == PARENT_START
        assert str(ROOT / 'owner_resource03.py') in Path('/proc/%d/cmdline' % PARENT_PID).read_bytes().replace(b'\0', b' ').decode()
    except (FileNotFoundError, AssertionError):
        return False
    return not (ROOT / 'HALT.json').exists() and not (CONTROL / 'OWNER_RESULT.json').exists()


def memory(gpu, suffix):
    return int(Path('/sys/class/drm/card%d/device/mem_info_vram_%s' % (gpu + 1, suffix)).read_text())


def gate(slot, gpu, model):
    if not parent_live():
        raise RuntimeError('Registered parent stopped or halted')
    if not all((CONTROL / ('EXPAND_SLOT_%d.json' % s)).exists() for s in [4, 5]):
        return None
    for m in ['pi05', 'xvla']:
        campaign = ROOT / 'campaigns' / ('libero_original_best_20261024_' + m + '_R_libero_object_original')
        if not any(campaign.glob('node*/lane*/case*/TERMINAL_CASE.json')):
            return None
    lease = json.loads(Path('/data/HCU-coordination/leases/gpu-%d/OWNER.json' % gpu).read_text())
    assert lease['pid'] == PARENT_PID and lease['boot'] == BOOT
    used, total = memory(gpu, 'used'), memory(gpu, 'total')
    reserve = (32 if model == 'pi05' else 28) * 1024 ** 3
    if used + reserve > total * .90:
        return None
    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))) * 1024
    if available < 100 * 1024 ** 3:
        return None
    return dict(time=time.time(), logical_slot=slot, policy_gpu=gpu, qwen_gpu=gpu,
                used_bytes=used, total_bytes=total, reserved_additional_bytes=reserve,
                predicted_fraction=(used + reserve) / total, ceiling_fraction=.90,
                host_available_bytes=available, scope='Existing registered Goal group; same group lock and complete500 denominator; extra independent resident services')


def run_group(slot, gpu, model):
    argv = [str(ROOT.parent.parent / 'envs/host-controller-v1/bin/python'),
            str(ROOT / 'run_lane_whole_device_resource03.py'), '--model', model, '--arm', 'R',
            '--suite', 'libero_goal', '--variant', 'original', '--node', '107',
            '--gpu-base', str(gpu), '--slot-index', str(slot),
            '--resource-profile', 'success_v2_whole_device_extra', '--stage', 'full']
    with (CONTROL / ('EXTRA_SLOT_%d.log' % slot)).open('xb') as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write('EXTRA_LAUNCH_%d.json' % slot, dict(time=time.time(), pid=child.pid,
              start_ticks=Path('/proc/%d/stat' % child.pid).read_text().split()[21], argv=argv,
              parent_pid=PARENT_PID, same_scientific_configuration=True))
        code = child.wait()
    return dict(slot=slot, model=model, gpu=gpu, code=code, finished=time.time())


def cleanup():
    receipts = []
    ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=yuchen-fastrsi-libero_original_best_20261024_'], text=True).split()
    for cid in ids:
        c = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
        if not any(('-n107-lane%d_full_' % s) in c['Name'] for s in [6, 7]):
            continue
        if c['State']['Running']:
            subprocess.run(['docker', 'stop', '-t', '15', cid], check=True, capture_output=True, timeout=40)
        receipts.append(dict(id=cid, name=c['Name']))
    write('EXTRA_CLEANUP.json', dict(time=time.time(), own_slots_only=[6,7], containers=receipts))


def main():
    with (CONTROL / 'extra_goal.lock').open('a+') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (CONTROL / 'EXTRA_RESULT.json').exists()
        result = dict(status='WAITING_FOR_SIX_NATIVE_LANES_AND_MEMORY', pid=os.getpid(), started=START,
                      parent_pid=PARENT_PID, potential_total_lanes=8)
        write('EXTRA_STATUS.json', result)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                pending = list(JOBS)
                jobs = {}
                while pending or not all(j.done() for j in jobs.values()):
                    if not parent_live() or time.time() - START > 96 * 3600:
                        raise RuntimeError('Registered parent halted or deadline reached')
                    for item in list(pending):
                        admission = gate(*item)
                        if admission is not None:
                            write('EXTRA_ADMISSION_%d.json' % item[0], admission)
                            jobs[item[0]] = pool.submit(run_group, *item)
                            pending.remove(item)
                            result.update(status='RUNNING', admitted_slots=list(jobs), updated=time.time())
                            write('EXTRA_STATUS.json', result)
                    failures = [j.result() for j in jobs.values() if j.done() and j.result()['code']]
                    if failures:
                        if not (ROOT / 'HALT.json').exists():
                            (ROOT / 'HALT.json').write_text(json.dumps(dict(time=time.time(), reason='Extra resource lane requires diagnosis; retain all paid cases', failures=failures)))
                        pending.clear()
                    time.sleep(20)
                result.update(status='COMPLETE' if len(jobs)==2 and all(j.result()['code']==0 for j in jobs.values()) else 'STOPPED_FOR_DIAGNOSIS', slots={s:j.result() for s,j in jobs.items()})
        except BaseException:
            result.update(status='STOPPED_FOR_DIAGNOSIS', error=traceback.format_exc())
        finally:
            cleanup()
            result.update(finished=time.time(), elapsed_seconds=time.time()-START)
            write('EXTRA_RESULT.json', result)
            write('EXTRA_STATUS.json', result)


if __name__ == '__main__':
    main()
