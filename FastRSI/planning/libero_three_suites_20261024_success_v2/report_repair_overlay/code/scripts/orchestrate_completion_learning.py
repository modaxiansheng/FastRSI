"""One idle-card, bounded shadow feature/learning run; original models untouched."""
import fcntl,hashlib,json,shutil,time,traceback
import orchestrate_full_os as life
NAME='completion_learning_v1_20261004'
REV='17f0b0ad64efb65d273590632833508766b2aae6'


def main():
    run=life.PROJECT/'runs'/NAME;run.mkdir(exist_ok=False)
    lock=open(life.PROJECT/'runs/clef_validity.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();life.deadline=started+900
    state=dict(status='RUNNING',started_at=started,new_robot_episodes=0,controller_changed=False)
    life.save(run/'STATUS.json',state)
    try:
        life.gate(run)
        code=run/'code'
        for sub in ('scripts','src'):
            shutil.copytree(life.PROJECT/sub,code/sub,ignore=shutil.ignore_patterns('__pycache__'))
        inputs=run/'inputs';shutil.copytree(life.PROJECT/'planning/completion_learning_v1_20261004',inputs)
        bindings={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for base in (code,inputs) for p in base.rglob('*') if p.is_file()}
        life.save(run/'SOURCE_BINDING.json',bindings)
        command='source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec /project/envs/clef-decision-v1/bin/python /output/code/scripts/run_completion_learning.py --model /project/models/clef-flash/'+REV+' --inputs /output/inputs --output /output/experiment'
        argv=['docker','run','-d','--name','yuchen-fastrsi-'+NAME,'--network','none','--cpus','8','--memory','64g','--shm-size','2g',
            '--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri','--group-add','video','--group-add','render',
            '-e','HIP_VISIBLE_DEVICES=0','-e','CUDA_VISIBLE_DEVICES=0','-e','PYTHONDONTWRITEBYTECODE=1','-e','HF_HUB_OFFLINE=1',
            '-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4','-v','/opt/hyhal:/opt/hyhal:ro','-v','/etc/hfm:/etc/hfm:ro',
            '-v',str(life.PROJECT)+':/project:ro','-v',str(run)+':/output:rw',life.IMAGE,'timeout','780','bash','-lc',command]
        cid=life.launch('clef',argv,run);life.wait_exit(cid,790)
        result=json.loads((run/'experiment/RESULT.json').read_text())
        state.update(status=result['status'],model_calls=len(result['calls']),fit_seconds=result.get('fit_seconds'),
                     external_head_versions=2,backbone_updates=0)
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        life.save(run/'STOP_RECEIPT.json',life.stop_all());state.update(seconds=time.time()-started,finished_at=time.time())
        life.save(run/'STATUS.json',state);print(json.dumps(state),flush=True)


if __name__=='__main__':main()
