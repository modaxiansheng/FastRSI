"""CPU gate plus one saved-session report replay; no robot or learning service."""
import fcntl,hashlib,json,shutil,signal,sys,time,traceback
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env

def main():
    name=sys.argv[1];assert name.startswith('terminal_read_') and name.replace('_','').isalnum()
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    master=life.PROJECT/'runs/representative_native_v1_20261002';batch=master/'batch_02_s0'
    assert json.loads((batch/'STATUS.json').read_text())['status']=='COMPLETED'
    global_deadline=json.loads((master/'PROTOCOL.json').read_text())['campaign_absolute_deadline']
    start=time.time();life.deadline=min(start+600,global_deadline);assert life.deadline-start>550
    run=life.PROJECT/'runs'/name;run.mkdir(exist_ok=False);life.CAMPAIGN=run;life.BASE='/project/runs/'+name
    source=batch/'stream/coverage_s0_t4_i0'
    for rel in ['scripts','src/fastrsi','tests','configs']:shutil.copytree(life.PROJECT/rel,run/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(run/'CODE_MANIFEST.json',{str(p.relative_to(run/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'code').rglob('*') if p.is_file()})
    for rel in ['MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json']:shutil.copyfile(master/rel,run/rel)
    tests=['test_paged_reads.py','test_session_flow.py','test_native_workspace.py','test_native_isolation.py','test_terminal_results.py']
    life.save(run/'PROTOCOL.json',{'scope':'Common current-record coverage repair; original batch not rescored','source_case':str(source),
      'maximum_wall_seconds':life.deadline-start,'deadline':life.deadline,'tests':tests,'physical_executions':0,'reverification':False,
      'learning':'Fixed evaluation guard through close/background; saved task context restored in memory only; no admission',
      'selection':'Last scheduled case with largest saved SESSIONS.md, before replay outcomes','costs':'Complete CPU, model, failure and lifecycle bill, separate from original failed read acceptance; no speedup claim'})
    status={'status':'RUNNING','started_at':start};life.save(run/'STATUS.json',status)
    def interrupt(n,f):raise RuntimeError('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    cpu=run/'cpu';cpu.mkdir();stream=run/'stream';stream.mkdir()
    try:
        life.gate(cpu)
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-cpu',cpu)+['-e','FASTRSI_NATIVE_RUN=/tmp/fastrsi-terminal-read/stream',life.IMAGE,'timeout','180','bash','-lc','source /opt/dtk/env.sh && '+' && '.join('/project/envs/os-v1/bin/python '+life.BASE+'/code/tests/'+t for t in tests)]
        cid=life.launch('cpu',argv,cpu);life.wait_exit(cid,190);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());status['cpu_status']='PASSED';life.save(run/'STATUS.json',status)
        life.gate(stream);cfg=json.loads((run/'MODEL_CONFIG.json').read_text())
        argv=json.loads((life.OLD/'LAUNCH_qwen.json').read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-qwen'
        argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',life.BASE+'/stream') for x in argv]
        life.launch('qwen',configure_qwen(argv,cfg,life),stream);life.health(49510)
        remaining=int(life.deadline-time.time()-20);assert remaining>60
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-runner',stream)+model_env(cfg)+['-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream','-e','FASTRSI_REPORT_SOURCE='+str(source).replace(str(life.PROJECT),'/project'),life.IMAGE,'timeout',str(remaining),'bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/replay_terminal_read.py']
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,remaining+5)
        result=json.loads((stream/'RESULT.json').read_text());assert result['status']=='COMPLETED'
        assert result['fixed_experience_unchanged'] and result['canonical_unchanged'] and result['source_unchanged']
        status.update(status='COMPLETED',controller_seconds=result['seconds'],physical_executions=0)
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-start);life.save(run/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()
