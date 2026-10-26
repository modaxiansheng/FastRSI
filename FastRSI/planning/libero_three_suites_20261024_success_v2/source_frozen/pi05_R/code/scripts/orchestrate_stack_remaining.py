"""One native-OS capability diagnosis on an unmodified prerequisite task."""
import fcntl,hashlib,json,signal,socket,subprocess,time,traceback
import orchestrate_full_os as life

NAME='stack_remaining_v1_20261004'

def main():
    run=life.PROJECT/'runs'/NAME
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/'STATUS.json').exists()
    start=time.time();life.deadline=start+1000;life.CAMPAIGN=run;life.BASE='/project/runs/'+NAME
    status={'status':'RUNNING','started_at':start};worker=None
    def stop(signum,frame):raise KeyboardInterrupt('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for group in (run,run/'F'):
            for path,digest in json.loads((group/'CODE_MANIFEST.json').read_text()).items():
                assert hashlib.sha256((group/'code'/path).read_bytes()).hexdigest()==digest, path
        for rel,h in json.loads((run/'SOURCE_MANIFEST.json').read_text()).items():
            assert hashlib.sha256((life.PROJECT/rel).read_bytes()).hexdigest()==h,rel
        life.gate(run)
        for port in (23600,23602,23610,23620):
            with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
        protocol=json.loads((run/'PROTOCOL.json').read_text());protocol['absolute_deadline']=life.deadline
        life.save(run/'EXECUTION_PROTOCOL.json',protocol)
        # Worker consumes the explicitly dated execution protocol.
        life.save(run/'PROTOCOL.json',protocol)
        cpu=run/'cpu';cpu.mkdir()
        cmd='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest test_development_suite test_boundary_probe test_stack_boundary_probe test_boundary_worker -v'
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',cpu)+['-e','FASTRSI_NATIVE_RUN=/tmp/prerequisite-cpu',life.IMAGE,'timeout','120','bash','-lc',cmd],cpu)
        life.wait_exit(cid,130);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());status['cpu']='PASSED'
        life.gate(run)
        with (run/'F/controller.log').open('x') as log:
            worker=subprocess.Popen(['python3',str(run/'code/scripts/run_placement_worker.py'),NAME,'F'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        life.save(run/'DISPATCH.json',{'pid':worker.pid,'started_at':time.time()})
        while worker.poll() is None:
            if time.time()>life.deadline:raise TimeoutError('Native diagnosis deadline')
            time.sleep(2)
        status['worker']=json.loads((run/'F/STATUS.json').read_text())
        assert worker.returncode==0 and status['worker']['status']=='COMPLETED'
        status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        if worker is not None and worker.poll() is None:
            worker.send_signal(signal.SIGTERM)
            try:worker.wait(timeout=45)
            except subprocess.TimeoutExpired:status['cleanup_attention']='Worker still running; inspect owned containers'
        life.save(run/'STOP_RECEIPT.json',life.stop_all())
        status.update(seconds=time.time()-start,finished_at=time.time());life.save(run/'STATUS.json',status);print(json.dumps(status),flush=True)

if __name__=='__main__':main()
