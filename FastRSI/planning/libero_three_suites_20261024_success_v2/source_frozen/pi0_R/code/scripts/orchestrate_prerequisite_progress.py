import fcntl,json,time,traceback
import orchestrate_full_os as life

def main():
    p=life.PROJECT;run=p/'runs/prerequisite_progress_v1_20261004'
    source=p/'runs/prerequisite_native_v2_20261004/N/stream'
    assert json.loads((source/'RESULT.json').read_text())['status']=='COMPLETED'
    lock=open(p/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/'STATUS.json').exists()
    start=time.time();life.deadline=start+260;status={'status':'RUNNING'}
    life.save(run/'STATUS.json',status)
    try:
        life.gate(run)
        old=json.loads((source/'LAUNCH_target.json').read_text())['argv'];image=next(x for x in old if x.startswith('sha256:'));argv=old[:old.index(image)]
        argv[argv.index('--name')+1]='yuchen-fastrsi-prerequisite-progress-v1'
        clean=[];i=0
        while i<len(argv):
            if argv[i]=='-e' and argv[i+1].startswith('FASTRSI_'):i+=2;continue
            clean.append(argv[i]);i+=1
        base='/project/runs/prerequisite_progress_v1_20261004'
        cid=life.launch('replay',clean+[image,'timeout','230','python',base+'/code/scripts/replay_prerequisite_progress.py','--stream',str(source).replace(str(p),'/project'),'--output',base+'/REPLAY.json'],run)
        life.wait_exit(cid,240);status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        life.save(run/'STOP_RECEIPT.json',life.stop_all());status['seconds']=time.time()-start;life.save(run/'STATUS.json',status);print(json.dumps(status))

if __name__=='__main__':main()
