"""Exact CPU replay of two completed traces; no alternate branch or model."""
import fcntl
import hashlib
import json
import shutil
import time
import traceback
import orchestrate_full_os as life


def main():
    project=life.PROJECT;source=project/'runs/cabinet_composition_baseline_v1_20261006'
    run=project/'runs/cabinet_composition_replay_v1_20261006'
    lock=(project/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((source/'STATUS.json').read_text())['status']=='COMPLETED'
    run.mkdir(exist_ok=False);start=time.time();life.deadline=start+300
    state=dict(status='RUNNING',started_at=start)
    try:
        life.gate(run)
        for rel in ('scripts/replay_cabinet_composition_v1.py','scripts/stage_trace.py'):
            p=run/'code'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project/rel,p)
        for arm in ('N','F'):
            group=run/arm;group.mkdir()
            stream=source/arm/'stream'
            trace=next((stream/'stage_trace').glob('*/TRACE.jsonl'))
            life.save(group/'SOURCE.json',dict(trace=str(trace),sha256=hashlib.sha256(trace.read_bytes()).hexdigest()))
            old=json.loads((stream/'LAUNCH_target.json').read_text())['argv'];ix=next(i for i,v in enumerate(old) if v.startswith('sha256:'))
            argv=old[:ix];argv[argv.index('--name')+1]='yuchen-fastrsi-cabinet-replay-v1-'+arm.lower()
            clean=[];i=0
            while i<len(argv):
                if argv[i]=='-e' and argv[i+1].startswith('FASTRSI_'):i+=2;continue
                clean.append(argv[i]);i+=1
            base=str(run).replace(str(project),'/project')
            cid=life.launch('replay',clean+[old[ix],'timeout','140','python',base+'/code/scripts/replay_cabinet_composition_v1.py',
                '--stream',str(stream).replace(str(project),'/project'),'--output',base+'/'+arm+'/REPLAY.json','--cabinet-joints'],group)
            life.wait_exit(cid,145);life.save(group/'STOP_RECEIPT.json',life.stop_all())
            assert json.loads((group/'REPLAY.json').read_text())['status']=='EXACT_IMAGE_REPLAY'
        state['status']='COMPLETED'
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        life.save(run/'STOP_RECEIPT.json',life.stop_all());state.update(seconds=time.time()-start,finished_at=time.time())
        life.save(run/'STATUS.json',state);print(json.dumps(state),flush=True)


if __name__=='__main__':main()
