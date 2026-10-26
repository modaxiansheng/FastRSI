"""Run two offline action replays with the exact target image and CPU-only receipts."""
import fcntl,json,os,subprocess,time,traceback
from pathlib import Path
import orchestrate_full_os as life

def main():
    root=life.PROJECT;run=root/'runs/placement_recovery_v1_20261003'
    assert json.loads((run/'STATUS.json').read_text())['status']=='COMPLETED'
    lock=open(root/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    folder=root/'planning/placement_recovery_v1_20261003/post_geometry';folder.mkdir(exist_ok=False)
    life.deadline=time.time()+300;started=time.time();status={'status':'RUNNING'};workers=[]
    try:
        for arm in ['N','F']:
            dst=folder/arm;dst.mkdir()
            source=run/'pair0'/arm/'stream'
            old=json.loads((source/'LAUNCH_target.json').read_text())['argv']
            image=next(x for x in old if x.startswith('sha256:'));argv=old[:old.index(image)]
            argv[argv.index('--name')+1]='yuchen-fastrsi-placement-post-geometry-'+arm.lower()
            filtered=[];i=0
            while i<len(argv):
                if argv[i]=='-e' and argv[i+1].startswith('FASTRSI_'):i+=2;continue
                filtered.append(argv[i]);i+=1
            trace=str(source/'stage_trace/net_t4_i31/TRACE.jsonl').replace(str(root),'/project')
            output=str(dst/'REPLAY.json').replace(str(root),'/project')
            argv=filtered+[image,'timeout','240','python','/project/scripts/replay_placement_geometry.py',trace,output]
            cid=life.launch('replay',argv,dst);workers.append((arm,cid))
        for _,cid in workers:life.wait_exit(cid,250)
        status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        life.save(folder/'STOP_RECEIPT.json',life.stop_all());status.update(started_at=started,finished_at=time.time(),seconds=time.time()-started);life.save(folder/'STATUS.json',status);print(json.dumps(status),flush=True)
if __name__=='__main__':main()
