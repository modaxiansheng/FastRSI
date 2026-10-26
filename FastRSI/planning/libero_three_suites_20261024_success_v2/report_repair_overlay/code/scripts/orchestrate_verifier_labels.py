"""CPU protocol checks then one Qwen service, four saved evidence calls, no robot."""
import fcntl,hashlib,json,shutil,signal,sys,time,traceback
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env

def main():
    name=sys.argv[1];assert name.startswith('verifier_labels_') and name.replace('_','').isalnum()
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=life.PROJECT/'runs'/name;run.mkdir(exist_ok=False);life.CAMPAIGN=run;life.BASE='/project/runs/'+name
    start=time.time();life.deadline=start+900;status={'status':'RUNNING','started_at':start,'scope':'Public evidence image mapping replay; no robot or learning'}
    def interrupt(n,f):raise RuntimeError('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    for rel in ['scripts','src/fastrsi','tests','configs']:shutil.copytree(life.PROJECT/rel,run/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(run/'CODE_MANIFEST.json',{str(p.relative_to(run/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'code').rglob('*') if p.is_file()})
    prior=life.PROJECT/'runs/session_flow_v1_20261002'
    for rel in ['MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json']:shutil.copyfile(prior/rel,run/rel)
    cfg=json.loads((run/'MODEL_CONFIG.json').read_text())
    tests=['test_session_image_labels.py','test_verifier_schema_contract.py','test_experience_admission.py','test_native_isolation.py']
    life.save(run/'PROTOCOL.json',{'order':['CPU regression','A original','A labelled','B original','B labelled'],'max_wall_seconds':900,'maximum_model_calls':4,'robot_episodes':0,'training':False,'experience_updates':False,'tests':tests,'change':'Explicit phase/path metadata preceding each existing image; original text, image bytes, goal, generation config unchanged','not_claimed':'Representative semantic quality, new episodes, learning or speedup'})
    cpu=run/'cpu';cpu.mkdir();stream=run/'stream';stream.mkdir();life.save(run/'STATUS.json',status)
    try:
        life.gate(cpu)
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-cpu',cpu)+['-e','FASTRSI_NATIVE_RUN=/tmp/fastrsi-label-cpu/stream',life.IMAGE,'timeout','180','bash','-lc','source /opt/dtk/env.sh && '+' && '.join('/project/envs/os-v1/bin/python '+life.BASE+'/code/tests/'+t for t in tests)]
        cid=life.launch('cpu',argv,cpu);life.wait_exit(cid,190);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());status['cpu_status']='PASSED';life.save(run/'STATUS.json',status)
        life.gate(stream)
        argv=json.loads((life.OLD/'LAUNCH_qwen.json').read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-qwen'
        argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',life.BASE+'/stream') for x in argv]
        life.launch('qwen',configure_qwen(argv,cfg,life),stream);life.health(49510)
        status['startup_seconds']=time.time()-start;life.save(run/'STATUS.json',status)
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-runner',stream)+model_env(cfg)+['-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream',life.IMAGE,'timeout','600','bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/replay_session_image_labels.py']
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,610)
        result=json.loads((stream/'RESULT.json').read_text());assert result['status']=='COMPLETED'
        assert len(list(stream.glob('service_calls/*/REQUEST.json')))==4
        status.update(status='COMPLETED',controller_seconds=result['seconds'])
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-start);life.save(run/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()
