"""One owner, two full-Agent workers plus an independent six-request Qwen audit."""
import fcntl,hashlib,json,signal,socket,subprocess,time,traceback
import orchestrate_full_os as life
from native_service_config import configure_qwen
from orchestrate_resident_services import configure_context,gpu,check_ports
NAME='boundary_consequence_v1_20261004'

def read(p):return json.loads(p.read_text())

def main():
    assert socket.gethostname()=='j03r4n13'
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=life.PROJECT/'runs'/NAME;entity=run/'ENTITY';stream=entity/'stream';stream.mkdir(exist_ok=False)
    assert not (run/'STATUS.json').exists()
    start=time.time();deadline=start+1300;configure_context(entity,deadline)
    workers=[];logs=[];status={'status':'RUNNING','started_at':start,'deadline':deadline}
    def stop(n,f):raise KeyboardInterrupt(str(n))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for group in (run,run/'N',run/'F',entity):
            for rel,h in read(group/'CODE_MANIFEST.json').items():
                assert hashlib.sha256((group/'code'/rel).read_bytes()).hexdigest()==h,rel
        for rel,h in read(run/'SOURCE_MANIFEST.json').items():
            assert hashlib.sha256((life.PROJECT/rel).read_bytes()).hexdigest()==h,rel
        life.gate(run);check_ports([22500,22502,22510,22520,22600,22602,22610,22620])
        cpu=run/'cpu';cpu.mkdir()
        tests='test_boundary_probe test_boundary_worker test_entity_identity_replay test_planning_entry_binding test_task_snapshot_worker test_resident_worker'
        cmd='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest '+tests+' -v'
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-boundary-v1-cpu',cpu)+[life.IMAGE,'timeout','120','bash','-lc',cmd],cpu)
        life.wait_exit(cid,130);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());status['cpu']='PASSED';life.gate(run)
        protocol=read(run/'PROTOCOL.json');protocol['absolute_deadline']=deadline;life.save(run/'PROTOCOL.json',protocol)
        life.save(entity/'DISPATCH.json',{'deadline':deadline,'max_model_calls':6,'gpu':4,'port':49510})
        for arm in ('N','F'):
            log=(run/arm/'controller.log').open('x');logs.append(log)
            proc=subprocess.Popen(['python3',str(run/'code/scripts/run_placement_worker.py'),NAME,arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append((arm,proc))
        life.save(run/'DISPATCH.json',{'pids':{a:p.pid for a,p in workers},'deadline':deadline})
        load=time.time();argv=read(life.OLD/'LAUNCH_qwen.json')['argv']
        argv=[x.replace(str(life.OLD),str(stream)) for x in argv]
        argv[argv.index('--name')+1]='yuchen-fastrsi-boundary-v1-entity-qwen'
        argv=gpu(configure_qwen(argv,read(entity/'MODEL_CONFIG.json'),life),4)
        ix=argv.index(life.IMAGE);assert argv[ix+1]=='timeout';argv[ix+2]=str(int(deadline-time.time()))
        life.launch('qwen',argv,stream);life.save(run/'STATUS.json',status)
        life.health(49510,limit=300);status['entity_startup_seconds']=time.time()-load
        cmd='source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/entity_identity_replay.py --run '+life.BASE
        cid=life.launch('entity_replay',life.cpu_args('yuchen-fastrsi-boundary-v1-entity-replay',stream)+[life.IMAGE,'timeout','600','bash','-lc',cmd],stream)
        life.wait_exit(cid,610)
        result=read(stream/'RESULT.json');assert result['status']=='COMPLETED'
        count=len(list((stream/'service_calls').glob('*/REQUEST.json')));assert count==6,count
        status.update(entity_calls=count,entity_seconds=result['seconds'],entity_full_wall_seconds=time.time()-load)
        life.save(entity/'STOP_RECEIPT.json',life.stop_all());life.save(run/'STATUS.json',status)
        while any(p.poll() is None for _,p in workers):
            if time.time()>deadline:raise TimeoutError('Bounded experiment expired')
            time.sleep(2)
        status['workers']={a:read(run/a/'STATUS.json') for a,p in workers}
        assert all(p.returncode==0 and status['workers'][a]['status']=='COMPLETED' for a,p in workers)
        status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        for arm,proc in workers:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:proc.wait(timeout=45)
                except subprocess.TimeoutExpired:status.setdefault('cleanup_attention',[]).append(arm)
        life.save(run/'STOP_RECEIPT.json',life.stop_all())
        for log in logs:log.close()
        status.update(seconds=time.time()-start,finished_at=time.time());life.save(run/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()
