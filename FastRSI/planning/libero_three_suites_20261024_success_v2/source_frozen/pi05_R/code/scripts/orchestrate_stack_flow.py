"""Two independent full native Agent workers on four freshly idle GPUs."""
import fcntl
import hashlib
import json
import signal
import subprocess
import time
import traceback
import orchestrate_full_os as life

NAME='stack_flow_v1_20261004'


def main():
    run=life.PROJECT/'runs'/NAME
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/'STATUS.json').exists()
    start=time.time();life.deadline=start+1500;life.CAMPAIGN=run
    workers=[];logs=[];status=dict(status='RUNNING',started_at=start)
    def interrupt(n,f):raise KeyboardInterrupt('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        for rel,h in json.loads((run/'CODE_MANIFEST.json').read_text()).items():
            assert hashlib.sha256((run/'code'/rel).read_bytes()).hexdigest()==h,rel
        bindings=json.loads((run/'TASKS.json').read_text())
        for entry in bindings['sources']:
            assert hashlib.sha256(__import__('pathlib').Path(entry['remote_path']).read_bytes()).hexdigest()==entry['sha256']
        for entry in bindings['tasks']:
            assert hashlib.sha256(__import__('pathlib').Path(entry['init_path']).read_bytes()).hexdigest()==entry['init_sha256']
        life.gate(run)
        life.BASE='/project/runs/'+NAME
        cpu=run/'cpu';cpu.mkdir()
        command='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest test_development_suite test_session_flow test_native_isolation test_feedback_window -v'
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',cpu)+['-e','FASTRSI_NATIVE_RUN=/tmp/stack-flow-cpu',life.IMAGE,'timeout','120','bash','-lc',command],cpu)
        life.wait_exit(cid,130);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());status['cpu']='PASSED'
        for task in ('task63','task64'):
            campaign=run/task
            for group in (campaign,campaign/'N'):
                for rel,h in json.loads((group/'CODE_MANIFEST.json').read_text()).items():
                    assert hashlib.sha256((group/'code'/rel).read_bytes()).hexdigest()==h
            protocol=json.loads((campaign/'PROTOCOL.json').read_text());protocol['absolute_deadline']=life.deadline
            life.save(campaign/'PROTOCOL.json',protocol)
            log=(campaign/'controller.log').open('x');logs.append(log)
            process=subprocess.Popen(['python3',str(campaign/'code/scripts/run_placement_worker.py'),NAME+'/'+task,'N'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append((task,process))
        life.save(run/'DISPATCH.json',dict(pids={t:p.pid for t,p in workers},deadline=life.deadline))
        life.save(run/'STATUS.json',status)
        while any(p.poll() is None for _,p in workers):
            if time.time()>life.deadline:raise TimeoutError('Campaign bound')
            time.sleep(2)
        status['workers']={t:json.loads((run/t/'N/STATUS.json').read_text()) for t,_ in workers}
        assert all(p.returncode==0 and status['workers'][t]['status']=='COMPLETED' for t,p in workers)
        status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        for t,p in workers:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
                try:p.wait(timeout=45)
                except subprocess.TimeoutExpired:status.setdefault('cleanup_attention',[]).append(t)
        for log in logs:log.close()
        life.save(run/'STOP_RECEIPT.json',life.stop_all())
        status.update(seconds=time.time()-start,finished_at=time.time());life.save(run/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)


if __name__=='__main__':main()
