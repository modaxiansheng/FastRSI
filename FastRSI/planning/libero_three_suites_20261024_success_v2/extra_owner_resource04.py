"""Use spare policy-card VRAM when the verifier card cannot admit another lane."""
import concurrent.futures
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import extra_goal_owner_resource03 as common

ROOT, CONTROL = common.ROOT, common.CONTROL
START = time.time()
SCHEDULE = {6: (2, [('pi05', 'libero_goal'), ('xvla', 'libero_goal')]),
            7: (6, [('xvla', 'libero_object'), ('openvla', 'libero_goal')])}


def closed(model, suite):
    campaign = ROOT / 'campaigns' / ('libero_original_best_20261024_' + model + '_R_' + suite + '_original')
    return list(campaign.glob('node*/lane*/case*/TERMINAL_CASE.json'))


def gate(slot, gpu, model, suite):
    if not common.parent_live():
        raise RuntimeError('Parent halted or exited')
    if not (CONTROL / 'EXPAND_SLOT_4.json').exists() or not closed('pi05', 'libero_object'):
        return None
    lease = json.loads(Path('/data/HCU-coordination/leases/gpu-%d/OWNER.json' % gpu).read_text())
    assert lease['pid'] == common.PARENT_PID and lease['boot'] == common.BOOT
    used, total = common.memory(gpu, 'used'), common.memory(gpu, 'total')
    reserve = {'pi05': 32, 'xvla': 28, 'openvla': 36}[model] * 1024 ** 3
    if used + reserve > total * .90:
        return None
    available = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))) * 1024
    if available < 100 * 1024 ** 3:
        return None
    return dict(time=time.time(), slot=slot, model=model, suite=suite, policy_gpu=gpu,
                qwen_gpu=gpu, used_bytes=used, total_bytes=total,
                reserved_additional_bytes=reserve, predicted_fraction=(used+reserve)/total,
                ceiling_fraction=.90, same_registered_group_lock=True,
                reason='Verifier-only card cannot admit another full verifier with headroom; place independent policy and verifier together on spare policy-card VRAM')


def run_slot(slot):
    gpu, groups = SCHEDULE[slot]
    results = []
    for model, suite in groups:
        while len(closed(model, suite)) < 500:
            if time.time()-START > 96*3600:
                raise TimeoutError('Bounded resource owner deadline')
            admission = gate(slot, gpu, model, suite)
            if admission:
                break
            common.write('EXTRA04_SLOT_%d.json' % slot, dict(status='WAITING_FOR_MEMORY', time=time.time(), model=model, suite=suite, gpu=gpu))
            time.sleep(20)
        else:
            results.append(dict(model=model, suite=suite, code=0, state='ALREADY_COMPLETED_NO_RERUN'))
            continue
        common.write('EXTRA04_ADMISSION_%d_%s_%s.json' % (slot,model,suite), admission)
        argv = [str(ROOT.parent.parent / 'envs/host-controller-v1/bin/python'),
                str(ROOT / 'run_lane_whole_device_resource03.py'), '--model', model, '--arm', 'R',
                '--suite', suite, '--variant', 'original', '--node', '107', '--gpu-base', str(gpu),
                '--slot-index', str(slot), '--resource-profile', 'success_v2_whole_device_extra', '--stage', 'full']
        logpath = CONTROL / ('EXTRA04_SLOT_%d_%s_%s.log' % (slot,model,suite))
        with logpath.open('xb') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            launch = dict(time=time.time(), pid=child.pid, start_ticks=Path('/proc/%d/stat'%child.pid).read_text().split()[21],
                          argv=argv, parent_pid=common.PARENT_PID, same_scientific_configuration=True)
            common.write('EXTRA04_LAUNCH_%d_%s_%s.json' % (slot,model,suite), launch)
            common.write('EXTRA04_SLOT_%d.json' % slot, dict(status='RUNNING', model=model,suite=suite,gpu=gpu,**launch))
            code = child.wait()
        results.append(dict(model=model,suite=suite,code=code,finished=time.time()))
        if code:
            if not (ROOT/'HALT.json').exists():
                (ROOT/'HALT.json').write_text(json.dumps(dict(time=time.time(),reason='Resource04 lane requires diagnosis',results=results)))
            break
    return results


def main():
    with (CONTROL/'extra_resource04.lock').open('a+') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (CONTROL/'EXTRA04_RESULT.json').exists()
        result=dict(status='RUNNING_WITH_MEMORY_ADMISSION',pid=os.getpid(),started=START,parent_pid=common.PARENT_PID,schedule=SCHEDULE)
        common.write('EXTRA04_STATUS.json', result)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                jobs={s:pool.submit(run_slot,s) for s in SCHEDULE}
                while not all(j.done() for j in jobs.values()):
                    time.sleep(20)
                result['groups']={s:j.result() for s,j in jobs.items()}
            result['status']='COMPLETE' if all(v['code']==0 for rows in result['groups'].values() for v in rows) else 'STOPPED_FOR_DIAGNOSIS'
        except BaseException:
            result.update(status='STOPPED_FOR_DIAGNOSIS',error=traceback.format_exc())
        finally:
            common.cleanup()
            result.update(finished=time.time(),elapsed_seconds=time.time()-START)
            common.write('EXTRA04_RESULT.json',result)
            common.write('EXTRA04_STATUS.json',result)


if __name__=='__main__':
    main()
