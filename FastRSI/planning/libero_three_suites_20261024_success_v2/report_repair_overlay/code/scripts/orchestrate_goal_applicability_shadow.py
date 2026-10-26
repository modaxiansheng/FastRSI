"""Single-card Clef compatibility/replay check, never dispatches robot tasks."""
import fcntl
import json
import time
import traceback
import orchestrate_full_os as life

NAME='goal_applicability_shadow_v1_20261005'
REV='17f0b0ad64efb65d273590632833508766b2aae6'


def main():
    run=life.PROJECT/'runs'/NAME
    lock=open(life.PROJECT/'runs/clef_validity.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (run/'STATUS.json').exists():raise RuntimeError('Already dispatched')
    started=time.time();life.CAMPAIGN=run;life.BASE='/project/runs/'+NAME;life.deadline=started+800
    state=dict(status='RUNNING',started_at=started,scope='Loading and legal unlabelled replay only')
    life.save(run/'STATUS.json',state)
    try:
        life.gate(run)
        import hashlib
        for rel,digest in json.loads((run/'CODE_MANIFEST.json').read_text()).items():
            assert hashlib.sha256((run/'code'/rel).read_bytes()).hexdigest()==digest
        model=life.PROJECT/'models/clef-flash'/REV
        assert json.loads((model/'DOWNLOAD.json').read_text())['status']=='VERIFIED'
        argv=['docker','run','-d','--name','yuchen-fastrsi-'+NAME,
            '--network','none','--cpus','8','--memory','64g','--shm-size','2g',
            '--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri',
            '--group-add','video','--group-add','render',
            '-e','HIP_VISIBLE_DEVICES=0','-e','CUDA_VISIBLE_DEVICES=0',
            '-e','PYTHONDONTWRITEBYTECODE=1','-e','HF_HUB_OFFLINE=1',
            '-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',
            '-v','/opt/hyhal:/opt/hyhal:ro','-v','/etc/hfm:/etc/hfm:ro',
            '-v',str(life.PROJECT)+':/project:ro','-v',str(run)+':/output:rw',
            life.IMAGE,'timeout','720','bash','-lc',
            'source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec /project/envs/clef-decision-v1/bin/python '+
            life.BASE+'/code/scripts/run_clef_validity_probe.py --model /project/models/clef-flash/'+REV+
            ' --inputs /project/planning/goal_applicability_shadow_v1_20261005/requests --output /output/replay']
        cid=life.launch('clef',argv,run);life.wait_exit(cid,730)
        result=json.loads((run/'replay/RESULT.json').read_text())
        state.update(status=result['status'],calls=len(result['calls']),load_seconds=result['load_seconds'],
                     verified_training_labels=0,robot_calls=0)
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        life.save(run/'STOP_RECEIPT.json',life.stop_all())
        state.update(seconds=time.time()-started,finished_at=time.time())
        life.save(run/'STATUS.json',state);print(json.dumps(state),flush=True)


if __name__=='__main__':main()
