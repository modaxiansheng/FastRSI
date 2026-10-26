import fcntl,json,time,traceback
import orchestrate_full_os as life

def main():
    p=life.PROJECT;run=p/'runs/prerequisite_asset_v1_20261004'
    lock=open(p/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/'STATUS.json').exists()
    start=time.time();life.deadline=start+150;status={'status':'RUNNING'}
    life.save(run/'STATUS.json',status)
    try:
        life.gate(run)
        old=json.loads((p/'runs/short_exit_v1_20261003/pair1/R/stream/LAUNCH_target.json').read_text())['argv']
        image=next(x for x in old if x.startswith('sha256:'));argv=old[:old.index(image)]
        argv[argv.index('--name')+1]='yuchen-fastrsi-prerequisite-asset-v1'
        clean=[];i=0
        while i<len(argv):
            if argv[i]=='-e' and argv[i+1].startswith('FASTRSI_'):i+=2;continue
            clean.append(argv[i]);i+=1
        base='/project/runs/prerequisite_asset_v1_20261004'
        cid=life.launch('probe',clean+[image,'timeout','120','python',base+'/code/scripts/probe_prerequisite_asset.py','--output',base+'/ASSET.json'],run)
        life.wait_exit(cid,130);status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        life.save(run/'STOP_RECEIPT.json',life.stop_all());status['seconds']=time.time()-start;life.save(run/'STATUS.json',status);print(json.dumps(status))

if __name__=='__main__':main()
