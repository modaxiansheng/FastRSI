"""Move the queued OpenVLA Goal group into a card that finishes shorter groups."""
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT/'control05'
PARENT_PID, PARENT_START = 687870, '910990350'
START = time.time()


def write(name, data):
    p=CONTROL/name
    temp=p.with_name(p.name+'.pending.'+str(os.getpid()))
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(p)


def closed():
    return list((ROOT/'campaigns/libero_original_best_20261024_openvla_R_libero_goal_original').glob('node*/lane*/case*/TERMINAL_CASE.json'))


def live():
    try:
        return (Path('/proc/%d/stat'%PARENT_PID).read_text().split()[21]==PARENT_START
                and str(ROOT/'owner_resource05.py') in Path('/proc/%d/cmdline'%PARENT_PID).read_bytes().replace(b'\0',b' ').decode()
                and not (ROOT/'HALT.json').exists() and not (CONTROL/'OWNER_RESULT.json').exists())
    except FileNotFoundError:
        return False


def ready():
    if not all((CONTROL/('EXPAND_SLOT_%d.json'%s)).exists() for s in [2,3,6,7]):
        return False
    started=json.loads((CONTROL/'OWNER_STATUS.json').read_text())['started']
    for s,g in enumerate(json.loads((ROOT/'RUN_PLAN.json').read_text())['groups'][:8]):
        campaign=ROOT/'campaigns'/('libero_original_best_20261024_'+g['model']+'_R_'+g['suite']+'_original')
        if not any(p.stat().st_mtime>started for p in campaign.glob('node*/lane*/case*/TERMINAL_CASE.json')):
            return False
    return True


def main():
    with (CONTROL/'long_group_backfill.lock').open('a+') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (CONTROL/'BACKFILL_RESULT.json').exists()
        result=dict(status='WAITING_FOR_INITIAL_EIGHT_LANES_AND_FREE_MEMORY',pid=os.getpid(),started=START,
                    group='openvla_R_libero_goal_original',same500_cases=True,parent_pid=PARENT_PID)
        write('BACKFILL_STATUS.json',result)
        try:
            while len(closed())<500:
                if not live() or time.time()-START>96*3600:
                    raise RuntimeError('Parent halted or bounded deadline reached')
                if not ready():
                    time.sleep(20);continue
                group_lock=ROOT/'leases/openvla_R_libero_goal_original.lock'
                with group_lock.open('a+') as probe:
                    try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        result['status']='ALREADY_DISPATCHED_BY_MAIN_OWNER';break
                for gpu in [6,7,2,3]:
                    lease=json.loads(Path('/data/HCU-coordination/leases/gpu-%d/OWNER.json'%gpu).read_text())
                    assert lease['pid']==PARENT_PID
                    base=Path('/sys/class/drm/card%d/device'%(gpu+1))
                    used=int((base/'mem_info_vram_used').read_text());total=int((base/'mem_info_vram_total').read_text())
                    if used+36*1024**3>total*.90:continue
                    busy=False
                    for port in [21900,21902,21910,21920]:
                        with socket.socket() as sock:busy |= sock.connect_ex(('127.0.0.1',port))==0
                    if busy:continue
                    admission=dict(time=time.time(),gpu=gpu,used_bytes=used,total_bytes=total,reserved_additional_bytes=36*1024**3,ceiling_fraction=.90)
                    write('BACKFILL_ADMISSION.json',admission)
                    argv=[str(ROOT.parent.parent/'envs/host-controller-v1/bin/python'),str(ROOT/'run_lane_backfill_resource06.py'),
                          '--model','openvla','--arm','R','--suite','libero_goal','--variant','original','--node','107',
                          '--gpu-base',str(gpu),'--slot-index','8','--resource-profile','whole_gpu_eight_lanes_resource05','--skip-busy-group','--stage','full']
                    env=dict(os.environ,RESOURCE06_BACKFILL_GPU=str(gpu))
                    with (CONTROL/'BACKFILL_GROUP.log').open('xb') as log:
                        child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        write('BACKFILL_LAUNCH.json',dict(time=time.time(),pid=child.pid,start_ticks=Path('/proc/%d/stat'%child.pid).read_text().split()[21],argv=argv,environment_gpu=gpu,adapter_sha256=hashlib.sha256((ROOT/'run_lane_backfill_resource06.py').read_bytes()).hexdigest()))
                        result.update(status='RUNNING',gpu=gpu,child_pid=child.pid);write('BACKFILL_STATUS.json',result)
                        code=child.wait()
                    if code:
                        if not (ROOT/'HALT.json').exists():(ROOT/'HALT.json').write_text(json.dumps(dict(time=time.time(),reason='Long-group backfill requires diagnosis',code=code)))
                        raise RuntimeError('Backfill lane failed; costs retained')
                    result.update(status='COMPLETE' if len(closed())==500 else 'GROUP_LOCK_RACE_DEFERRED',closed=len(closed()))
                    break
                else:
                    time.sleep(20);continue
                break
            else:result['status']='COMPLETE'
        except BaseException:
            result.update(status='STOPPED_FOR_DIAGNOSIS',error=traceback.format_exc())
        finally:
            result.update(finished=time.time(),elapsed_seconds=time.time()-START)
            write('BACKFILL_RESULT.json',result);write('BACKFILL_STATUS.json',result)


if __name__=='__main__':main()
