"""CPU regression for the native successful-retry experience path."""
import fcntl
import hashlib
import shutil
import time
import traceback
import orchestrate_full_os as life

NAME='recovery_admission_v1_20261004'


def main():
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=life.PROJECT/'runs'/NAME;run.mkdir(exist_ok=False)
    life.CAMPAIGN=run;life.BASE='/project/runs/'+NAME
    started=time.time();life.deadline=started+240
    status=dict(status='RUNNING',model_calls=0,robot_episodes=0,fixture_updates_only=True)
    try:
        for rel in ('scripts','src/fastrsi','tests','configs'):
            shutil.copytree(life.PROJECT/rel,run/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        life.save(run/'CODE_MANIFEST.json',{str(p.relative_to(run/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'code').rglob('*') if p.is_file()})
        life.gate(run)
        tests='test_experience_admission test_recovery_admission test_recovery_admission_native test_native_isolation'
        cmd='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest '+tests+' -v'
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',run)+[
            '-e','FASTRSI_TEST_PROJECT=/project','-e','FASTRSI_NATIVE_RUN=/tmp/recovery-admission-fixture',
            life.IMAGE,'timeout','180','bash','-lc',cmd],run)
        life.wait_exit(cid,190);status['status']='PASSED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(run/'STOP_RECEIPT.json',life.stop_all());status['seconds']=time.time()-started
        life.save(run/'STATUS.json',status);print(status,flush=True)
    if status['status']!='PASSED':raise SystemExit(1)


if __name__=='__main__':main()
