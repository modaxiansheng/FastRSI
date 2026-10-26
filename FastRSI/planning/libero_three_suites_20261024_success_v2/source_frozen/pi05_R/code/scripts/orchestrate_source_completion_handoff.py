"""One same-init N/F full-Agent comparison on four freshly idle cards."""
import fcntl,hashlib,json,signal,socket,subprocess,time,traceback
import orchestrate_full_os as life
NAME='source_completion_handoff_v1_20261005'
def main():
    run=life.PROJECT/'runs'/NAME
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/'STATUS.json').exists()
    start=time.time();life.deadline=start+1500;life.CAMPAIGN=run;life.BASE='/project/runs/'+NAME
    status={'status':'RUNNING','started_at':start};workers=[];logs=[]
    def stop(n,f):raise KeyboardInterrupt('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for group in (run,run/'N',run/'F'):
            for path,digest in json.loads((group/'CODE_MANIFEST.json').read_text()).items():
                assert hashlib.sha256((group/'code'/path).read_bytes()).hexdigest()==digest,path
        protocol=json.loads((run/'PROTOCOL.json').read_text())
        assert hashlib.sha256((run/'GOAL_SKILL.json').read_bytes()).hexdigest()==protocol['goal_skill_probe']['sha256']
        life.gate(run)
        for port in (24500,24502,24510,24520,24600,24602,24610,24620):
            with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
        protocol=json.loads((run/'PROTOCOL.json').read_text());protocol['absolute_deadline']=life.deadline
        life.save(run/'EXECUTION_PROTOCOL.json',protocol);life.save(run/'PROTOCOL.json',protocol)
        cpu=run/'cpu';cpu.mkdir()
        cmd='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest test_goal_skill test_goal_skill_boundary test_development_suite test_native_isolation test_feedback_window test_boundary_worker -v'
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',cpu)+['-e','FASTRSI_NATIVE_RUN=/tmp/goal-skill-cpu',life.IMAGE,'timeout','120','bash','-lc',cmd],cpu)
        life.wait_exit(cid,130);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());status['cpu']='PASSED'
        life.gate(run)
        for arm in ('N','F'):
            log=(run/arm/'controller.log').open('x');logs.append(log)
            worker=subprocess.Popen(['python3',str(run/'code/scripts/run_placement_worker.py'),NAME,arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append((arm,worker))
        life.save(run/'DISPATCH.json',dict(pids={a:p.pid for a,p in workers}));life.save(run/'STATUS.json',status)
        while any(p.poll() is None for _,p in workers):
            if time.time()>life.deadline:raise TimeoutError('Campaign bound')
            time.sleep(2)
        status['workers']={a:json.loads((run/a/'STATUS.json').read_text()) for a,_ in workers}
        assert all(p.returncode==0 and status['workers'][a]['status']=='COMPLETED' for a,p in workers)
        status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        for arm,p in workers:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
                try:p.wait(timeout=45)
                except subprocess.TimeoutExpired:status.setdefault('cleanup_attention',[]).append(arm)
        for log in logs:log.close()
        life.save(run/'STOP_RECEIPT.json',life.stop_all());status.update(seconds=time.time()-start,finished_at=time.time())
        life.save(run/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
