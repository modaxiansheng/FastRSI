"""Start original Goal on independently audited node108 GPU3/6; keep existing500 identities."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import signal
import time
import traceback
from pathlib import Path

import owner_resource05 as base

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT/'control07'
HOST, BOOT = 'j01r2n08', 'c5b72a98-90a4-460b-87e5-75716165b991'
TOKEN = 'fastrsi-original-goal-node108-resource07'
base.CONTROL, base.TOKEN = CONTROL, TOKEN
GPUS = [3,6]
SLOTS = {'pi05':9,'xvla':10,'pi0':11,'openvla':12}
stop_requested = False


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def stop(*_):
    global stop_requested
    stop_requested=True


def closed(model):
    p=ROOT/'campaigns'/('libero_original_best_20261024_'+model+'_R_libero_goal_original')
    return list(p.glob('node*/lane*/case*/TERMINAL_CASE.json'))


def parent_live():
    return (not (ROOT/'HALT.json').exists() and not (ROOT/'control05/OWNER_RESULT.json').exists()
            and time.time()-(ROOT/'control05/PROGRESS.json').stat().st_mtime<180)


def fresh_gate():
    assert os.uname().nodename==HOST and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
    assert parent_live()
    bindings=json.loads((ROOT/'RESOURCE07_BINDINGS.json').read_text())
    assert sha(ROOT/'run_lane_goal_resource07.py')==bindings['lane_sha256']
    for model,files in bindings['models'].items():
        for name,digest in files.items():assert sha(ROOT/'source_frozen'/(model+'_R')/name)==digest
    for row in bindings['resource05_overlays'].values():assert sha(Path(row['overlay']))==row['overlay_sha256']
    assert json.loads((CONTROL/'CPU_PREFLIGHT_108.json').read_text())['code']==0
    assert json.loads((CONTROL/'CROSS_HOST_FLOCK_RECEIPT.json').read_text())['held_on108_rejected_on107']
    proof=json.loads((ROOT/'LEASE_OWNER162_CLOSED_PROOF_07.json').read_text())
    assert time.time()-proof['time']<900 and not proof['controller_processes'] and not proof['running_fba_services']
    assert proof['files']['PID']['proc_exists'] is False
    assert not any(proof['lease_owner_pids'].values())
    for gpu in GPUS:assert proof['files']['node108-g%d.json'%gpu]['summary']['status']=='dispatch_window_ended'
    secondary=json.loads((ROOT/'LEASE_OWNER162_ADMISSION_PID_CHECK_07.json').read_text())
    assert not any(secondary['secondary_admission_pids'].values())
    assert all(base.memory(g)<256*1024**2 for g in GPUS)
    smi=base.command(['bash','-lc','source /opt/dtk/env.sh && hy-smi --showpids'])
    holders={int(n) for line in smi.splitlines() if 'HCU Index:' in line for n in re.findall(r'\d+',line)}
    assert not holders.intersection(GPUS)
    containers=[]
    for cid in base.command(['docker','ps','-q']).split():
        c=base.inspect(cid);env=dict(x.split('=',1) for x in c['Config']['Env'] if '=' in x)
        masks=[env[k] for k in ['HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'] if k in env]
        claimed={int(x) for mask in masks for x in mask.split(',') if x.isdigit()}
        assert not claimed.intersection(GPUS),c['Name']
        assert not(c['HostConfig'].get('Devices') and not masks),('Unmasked device owner',c['Name'])
        containers.append(dict(id=c['Id'],name=c['Name'],masks=masks,paused=c['State']['Paused']))
    for gpu in GPUS:assert not Path('/data/fba-closedloop-pool/HOLD.g%d'%gpu).exists()
    base.write('FRESH_ADMISSION.json',dict(time=time.time(),host=HOST,boot=BOOT,gpus=GPUS,
               memory_bytes={g:base.memory(g) for g in GPUS},holders=sorted(holders),containers=containers,
               preserved_reservations=[0,1,2,4,5,7],closed_owner_proof_sha256=sha(ROOT/'LEASE_OWNER162_CLOSED_PROOF_07.json')))


def claim(leases):
    archive=Path('/data/HCU-coordination')/('archived-leases-libero-goal07-'+str(int(time.time())))
    archive.mkdir(exist_ok=False)
    expected={3:'242703a9-530a-4082-8501-22bc1cce18c7',6:'43d2fcca-ed58-4b32-b573-bb76d4029ef1'}
    previous=[]
    for gpu in GPUS:
        lease=Path('/data/HCU-coordination/leases/gpu-%d'%gpu)
        old=json.loads((lease/'OWNER.json').read_text());digest=sha(lease/'OWNER.json')
        assert old['token']==expected[gpu] and old['controller_host']=='10.232.1.162' and old['controller']=='/data/fba-closedloop-pool-v1'
        lease.rename(archive/lease.name)
        assert sha(archive/lease.name/'OWNER.json')==digest
        lease.mkdir(exist_ok=False)
        owner=dict(owner=TOKEN,token=TOKEN,pid=os.getpid(),host=HOST,boot=BOOT,gpu=gpu,root=str(ROOT),created=time.time(),scope='Original LIBERO Goal only; user requested107/108 free resources')
        (lease/'OWNER.json').write_text(json.dumps(owner,indent=2));leases.append(lease)
        previous.append(dict(gpu=gpu,archive=str(archive/lease.name),sha256=digest,old=old))
    base.write('RETIRED_CLOSED_LEASES.json',dict(time=time.time(),reason='Owner PIDs absent, controller absent, no running FBA services, per-card dispatch_window_ended, no live holder or reservation; original directories atomically archived without replacing their files',rows=previous))
    return leases


def run_group(model,gpu):
    slot=SLOTS[model]
    argv=[str(ROOT.parent.parent/'envs/host-controller-v1/bin/python'),str(ROOT/'run_lane_goal_resource07.py'),
          '--model',model,'--arm','R','--suite','libero_goal','--variant','original','--node','108',
          '--gpu-base',str(gpu),'--slot-index',str(slot),'--resource-profile','whole_gpu_goal_resource07','--skip-busy-group','--stage','full']
    with (CONTROL/('GROUP_'+model+'.log')).open('xb') as log:
        import subprocess
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        base.write('GROUP_LAUNCH_'+model+'.json',dict(time=time.time(),pid=child.pid,start_ticks=Path('/proc/%d/stat'%child.pid).read_text().split()[21],argv=argv,node108=True,physical_gpu=gpu))
        code=child.wait()
    if code:return dict(status='FAILED',model=model,code=code)
    return dict(status='COMPLETE' if len(closed(model))==500 else 'DEFERRED_TO_EXISTING_OWNER',model=model,closed=len(closed(model)))


def memory_gate(model,gpu):
    reserve={'pi0':28,'xvla':30}[model]*1024**3
    total=int(Path('/sys/class/drm/card%d/device/mem_info_vram_total'%(gpu+1)).read_text())
    used=base.memory(gpu)
    if used+reserve>total*.90:return None
    return dict(time=time.time(),model=model,gpu=gpu,used_bytes=used,reserved_additional_bytes=reserve,total_bytes=total,ceiling_fraction=.90)


def group_busy(model):
    path=ROOT/'leases'/(model+'_R_libero_goal_original.lock')
    with path.open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
    return False


def cleanup(guard):
    receipts=[]
    for model,slot in SLOTS.items():
        prefix='yuchen-fastrsi-libero_original_best_20261024_'+model+'_R_libero_goal_original-n108-lane%d_'%slot
        for cid in base.command(['docker','ps','-aq','--filter','name='+prefix]).split():
            c=base.inspect(cid);assert c['Name'].startswith('/'+prefix)
            if c['State']['Running']:base.command(['docker','stop','-t','15',cid],40)
            receipts.append(dict(id=cid,name=c['Name']))
    if guard:
        c=base.inspect(guard);assert c['Name']=='/'+TOKEN+'-guard'
        if c['State']['Running']:base.command(['docker','stop','-t','10',guard],30)
    base.write('CLEANUP.json',dict(time=time.time(),own_node108_goal_only=True,containers=receipts,guard=guard))


def main():
    with (CONTROL/'owner.lock').open('a+') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (CONTROL/'OWNER_RESULT.json').exists()
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        start=time.time();guard=None;leases=[]
        result=dict(status='STARTING',pid=os.getpid(),started=start,host=HOST,boot=BOOT,node='108',physical_gpus=GPUS)
        base.write('OWNER_STATUS.json',result)
        try:
            allocation=Path('/data/FastRSI/revisions/harness-rsi-v1/deployments/recovery-plan-20261007/gpu-allocation.lock')
            if not allocation.parent.exists():allocation=Path('/data/HCU-coordination/libero-goal07-allocation.lock')
            with allocation.open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                fresh_gate();claim(leases)
                argv=['docker','run','-d','--name',TOKEN+'-guard','--network','none','--cpus','1','--memory','8g',
                      '--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri',
                      '-e','HIP_VISIBLE_DEVICES=3,6','-e','CUDA_VISIBLE_DEVICES=3,6','-e','PYTHONDONTWRITEBYTECODE=1',
                      '-v','/opt/hyhal:/opt/hyhal:ro','-v','/etc/hfm:/etc/hfm:ro','-v',str(ROOT)+':/eval:ro','-v',str(CONTROL)+':/control:rw',
                      base.IMAGE,'bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec python /eval/gpu_guard_resource07.py']
                guard=base.command(argv,45).strip();base.write('GUARD_LAUNCH.json',dict(time=time.time(),argv=argv,id=guard))
                deadline=time.time()+120
                while not (CONTROL/'GPU_GUARD_READY.json').exists():
                    assert base.inspect(guard)['State']['Running']
                    if time.time()>deadline:raise TimeoutError('Visible guard readiness')
                    time.sleep(2)
            result.update(status='RUNNING',active_initial_models=['pi05','openvla'],guard=guard,updated=time.time())
            base.write('OWNER_STATUS.json',result)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                jobs={'pi05':pool.submit(run_group,'pi05',3),'openvla':pool.submit(run_group,'openvla',6)}
                pending=['pi0','xvla'];deferred=[]
                while pending or not all(j.done() for j in jobs.values()):
                    base.summarize()
                    errors=[j.result() for j in jobs.values() if j.done() and j.result()['status']=='FAILED']
                    if errors or stop_requested or not parent_live() or time.time()-start>96*3600:
                        pending.clear()
                        if not (ROOT/'HALT.json').exists():
                            (ROOT/'HALT.json').write_text(json.dumps(dict(time=time.time(),reason='Goal07 drain for diagnosis or stop',errors=errors)))
                    else:
                        for model in list(pending):
                            if len(closed(model))==500 or group_busy(model):
                                deferred.append(model);pending.remove(model);continue
                            if not closed('pi05'):continue
                            if model=='xvla' and ('pi0' not in jobs or not closed('pi0') or not closed('openvla')):continue
                            for gpu in ([3] if model=='pi0' else [3,6]):
                                admission=memory_gate(model,gpu)
                                if admission:
                                    base.write('ADMIT_'+model+'.json',admission)
                                    jobs[model]=pool.submit(run_group,model,gpu);pending.remove(model);break
                    result.update(models_dispatched=list(jobs),models_pending=pending[:],models_in_existing_queue=deferred[:],updated=time.time())
                    base.write('OWNER_STATUS.json',result)
                    time.sleep(20)
                result['groups']={m:j.result() for m,j in jobs.items()}
                result['status']='COMPLETE_OWN_GROUPS' if all(v['status']!='FAILED' for v in result['groups'].values()) and not (ROOT/'HALT.json').exists() else 'STOPPED_FOR_DIAGNOSIS'
        except BaseException:result.update(status='STOPPED_FOR_DIAGNOSIS',error=traceback.format_exc())
        finally:
            cleanup(guard)
            for lease in leases:
                owner=json.loads((lease/'OWNER.json').read_text());assert owner['pid']==os.getpid() and owner['token']==TOKEN
                base.write('RELEASED_GPU_%d.json'%owner['gpu'],dict(**owner,released=time.time()))
                (lease/'OWNER.json').unlink();lease.rmdir()
            base.summarize();result.update(finished=time.time(),elapsed_seconds=time.time()-start)
            base.write('OWNER_RESULT.json',result);base.write('OWNER_STATUS.json',result)


if __name__=='__main__':main()
