"""One paired full-OS development probe, with exact owned-worker cleanup."""
import fcntl
import hashlib
import json
import signal
import socket
import subprocess
import time
import traceback
import orchestrate_full_os as life

NAME='prefix_memory_v1_20261004'


def main():
    run=life.PROJECT/'runs'/NAME
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/'STATUS.json').exists()
    start=time.time();life.deadline=start+1000;life.CAMPAIGN=run;life.BASE='/project/runs/'+NAME
    status=dict(status='RUNNING',started_at=start);workers=[];logs=[]
    def stop(signum,frame):raise KeyboardInterrupt('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for group in (run,run/'N',run/'F'):
            for path,digest in json.loads((group/'CODE_MANIFEST.json').read_text()).items():
                assert hashlib.sha256((group/'code'/path).read_bytes()).hexdigest()==digest,path
        for path,digest in json.loads((run/'INPUT_MANIFEST.json').read_text()).items():
            assert hashlib.sha256((life.PROJECT/path).read_bytes()).hexdigest()==digest,path
        life.gate(run)
        for port in (19500,19502,19510,19520,19600,19602,19610,19620):
            with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
        protocol=json.loads((run/'PROTOCOL.json').read_text())
        protocol['absolute_deadline']=life.deadline
        life.save(run/'EXECUTION_PROTOCOL.json',protocol);life.save(run/'PROTOCOL.json',protocol)
        cpu=run/'cpu';cpu.mkdir()
        cmd='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest test_prefix_memory test_prerequisite_component test_session_flow test_feedback_window -v'
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',cpu)+[
            '-e','FASTRSI_NATIVE_RUN=/tmp/prefix-cpu',life.IMAGE,'timeout','120','bash','-lc',cmd],cpu)
        life.wait_exit(cid,130);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());status['cpu']='PASSED'
        life.gate(run)
        for arm in ('N','F'):
            log=(run/arm/'controller.log').open('x');logs.append(log)
            worker=subprocess.Popen(['python3',str(run/'code/scripts/run_placement_worker.py'),NAME,arm],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append((arm,worker))
        life.save(run/'DISPATCH.json',dict(pids={a:w.pid for a,w in workers},started_at=time.time()))
        while any(w.poll() is None for _,w in workers):
            if time.time()>life.deadline:raise TimeoutError('Prefix probe deadline')
            time.sleep(2)
        status['workers']={a:json.loads((run/a/'STATUS.json').read_text()) for a,_ in workers}
        assert all(w.returncode==0 and status['workers'][a]['status']=='COMPLETED' for a,w in workers)
        status['status']='COMPLETED'
    except BaseException:
        status.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        for arm,worker in workers:
            if worker.poll() is None:
                worker.send_signal(signal.SIGTERM)
                try:worker.wait(timeout=45)
                except subprocess.TimeoutExpired:status.setdefault('cleanup_attention',[]).append(arm)
        for log in logs:log.close()
        life.save(run/'STOP_RECEIPT.json',life.stop_all())
        status.update(seconds=time.time()-start,finished_at=time.time())
        life.save(run/'STATUS.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':main()
