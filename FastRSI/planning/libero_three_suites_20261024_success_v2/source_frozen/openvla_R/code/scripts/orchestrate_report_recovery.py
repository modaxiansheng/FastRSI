"""Bounded post-mortem report protocol check; original trial remains failed."""
import fcntl,hashlib,json,shutil,signal,sys,time,traceback
from pathlib import Path
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env

def main():
    name=sys.argv[1];assert name.startswith('report_path_recovery_') and name.replace('_','').isalnum()
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    master=life.PROJECT/'runs/representative_native_v1_20261002';batch=master/'batch_00_s0';source=batch/'stream/coverage_s0_t8_i10'
    state=json.loads((master/'STATE.json').read_text());protocol=json.loads((master/'PROTOCOL.json').read_text());old=json.loads((batch/'STATUS.json').read_text())
    assert state['status']=='FAILED' and old['status']=='FAILED'
    assert state.get('technical_phase_retries',0)<10
    deadline=min(protocol['campaign_absolute_deadline'],time.time()+600)
    assert deadline-time.time()>150,'Original budget insufficient; no timer reset'
    run=life.PROJECT/'runs'/name;run.mkdir(exist_ok=False);life.CAMPAIGN=run;life.BASE='/project/runs/'+name;life.deadline=deadline
    start=time.time();status={'status':'RUNNING','started_at':start,'source_batch':str(batch),'scope':'Post-mortem terminal protocol check; original failed S1 trial never rescored','deadline':deadline}
    state['technical_phase_retries']=state.get('technical_phase_retries',0)+1;state['status']='RECOVERING';life.save(master/'STATE.json',state)
    def interrupt(n,f):raise RuntimeError('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    for rel in ['scripts','src/fastrsi','tests','configs']:shutil.copytree(life.PROJECT/rel,run/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(run/'CODE_MANIFEST.json',{str(p.relative_to(run/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'code').rglob('*') if p.is_file()})
    for rel in ['MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json']:shutil.copyfile(master/rel,run/rel)
    life.save(run/'PROTOCOL.json',{'robot_executions':0,'reverification':False,'source_case':str(source),'max_seconds':deadline-start,'original_batch_deadline':old['started_at']+2700,'native_context':'Restored same Session key from copied persisted history; native AgentLoop unchanged','change':'Explicit absolute report destination and actionable rejection; write permission unchanged','original_trial_status':'FAILED remains FAILED; this component check is outside its completed/terminated batch, not a re-scored S1 root','costs':'Original failed batch plus this bounded component check both included in study/master bill. Global deadline unchanged; no new physical sample or learning benefit.'})
    stream=run/'stream';stream.mkdir();cpu=run/'cpu';cpu.mkdir();life.save(run/'STATUS.json',status)
    try:
        life.gate(cpu)
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-cpu',cpu)+[life.IMAGE,'timeout','90','bash','-lc','source /opt/dtk/env.sh && /project/envs/os-v1/bin/python '+life.BASE+'/code/tests/test_terminal_results.py']
        cid=life.launch('cpu',argv,cpu);life.wait_exit(cid,100);life.save(cpu/'STOP_RECEIPT.json',life.stop_all());status['cpu_status']='PASSED';life.save(run/'STATUS.json',status)
        life.gate(stream);cfg=json.loads((run/'MODEL_CONFIG.json').read_text())
        argv=json.loads((life.OLD/'LAUNCH_qwen.json').read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-qwen';argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',life.BASE+'/stream') for x in argv]
        life.launch('qwen',configure_qwen(argv,cfg,life),stream);life.health(49510)
        remaining=int(deadline-time.time()-20);assert remaining>30
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-runner',stream)+model_env(cfg)+['-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream','-e','FASTRSI_REPORT_SOURCE='+str(source).replace(str(life.PROJECT),'/project'),life.IMAGE,'timeout',str(remaining),'bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/recover_native_report.py']
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,remaining+5)
        result=json.loads((stream/'RESULT.json').read_text());assert result['status']=='COMPLETED'
        state['streams']['0'].update(offset=3,accepted_source=result['next_source'].replace('/project',str(life.PROJECT),1))
        state['status']='RECOVERY_COMPLETE_NEEDS_PROTOCOL_FIX';state['report_recovery_run']=name
        status.update(status='COMPLETED',controller_seconds=result['seconds'],physical_executions=0)
    except BaseException:status.update(status='FAILED',error=traceback.format_exc());state['status']='FAILED'
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-start);life.save(run/'STATUS.json',status)
        state['elapsed_master_wall_seconds']=time.time()-state['started_at'];state['recovery_status']=status['status'];life.save(master/'STATE.json',state);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()
