"""Six sequential CPU-only replays; no CPU contention across experimental arms."""
import fcntl
import hashlib
import json
import time
import traceback
import orchestrate_full_os as life

NAME='renderer_cost_v1_20261004'


def main():
    root=life.PROJECT;run=root/'runs'/NAME
    lock=open(root/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/'STATUS.json').exists()
    start=time.time();life.deadline=start+900;status=dict(status='RUNNING',started_at=start,jobs=[])
    life.save(run/'STATUS.json',status)
    try:
        life.gate(run)
        for p,h in json.loads((run/'MANIFEST.json').read_text()).items():
            assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
        protocol=json.loads((run/'PROTOCOL.json').read_text())
        for job in protocol['jobs']:
            folder=run/job['id'];folder.mkdir();source=root/job['source']
            old=json.loads((source/'LAUNCH_target.json').read_text())['argv']
            image=next(x for x in old if x.startswith('sha256:'));argv=old[:old.index(image)]
            argv[argv.index('--name')+1]='yuchen-fastrsi-'+NAME+'-'+job['id']
            clean=[];i=0
            while i<len(argv):
                if argv[i]=='-e' and (argv[i+1].startswith('FASTRSI_') or argv[i+1].startswith('LP_NUM_THREADS=')
                        or ('blas_threads' in job and argv[i+1].startswith('OPENBLAS_NUM_THREADS='))):
                    i+=2;continue
                clean.append(argv[i]);i+=1
            # Preserve original dependency search path and all CPU/memory quotas.
            if job['threads'] is not None:clean+=['-e','LP_NUM_THREADS='+str(job['threads'])]
            if job.get('blas_threads') is not None:clean+=['-e','OPENBLAS_NUM_THREADS='+str(job['blas_threads'])]
            base='/project/runs/'+NAME
            cid=life.launch('replay',clean+[image,'timeout','220','python',base+'/code/scripts/replay_renderer_cost.py',
                '--stream','/project/'+job['source'],'--output',base+'/'+job['id']+'/RESULT.json'],folder)
            life.wait_exit(cid,230)
            life.save(folder/'STOP_RECEIPT.json',life.stop_all())
            result=json.loads((folder/'RESULT.json').read_text())
            assert result['status']=='EXACT_BOUNDARY_REPLAY'
            status['jobs'].append(dict(id=job['id'],seconds=result['seconds'],timings=result['timings']))
            life.save(run/'STATUS.json',status)
        status['status']='COMPLETED'
    except BaseException:
        status.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        life.save(run/'STOP_RECEIPT.json',life.stop_all())
        status.update(seconds=time.time()-start,finished_at=time.time());life.save(run/'STATUS.json',status)
        print(json.dumps(status),flush=True)


if __name__=='__main__':main()
