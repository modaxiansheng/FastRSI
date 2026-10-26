"""Bounded 4 planning probes + 2 full native evaluations; shared fixed resources."""
import argparse,fcntl,hashlib,json,shutil,signal,socket,time,traceback
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args();name=args.run_id
    assert name.startswith('workflow_reuse_') and all(c.isalnum() or c=='_' for c in name)
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign=life.PROJECT/'runs'/name;campaign.mkdir(exist_ok=False);life.CAMPAIGN=campaign;life.BASE='/project/runs/'+name
    started=time.time();life.deadline=started+2100;status={'status':'PREFLIGHT','started_at':started}
    def interrupt(signum,frame):raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,campaign/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(campaign/'CODE_MANIFEST.json',{str(p.relative_to(campaign/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()})
    cfg=json.loads((life.PROJECT/'configs/native_agent_qwen3_v2.json').read_text());life.save(campaign/'MODEL_CONFIG.json',cfg)
    life.save(campaign/'MODEL_BINDING.json',json.loads((life.PROJECT/'runs/native_long_v1_20261002/MODEL_BINDING.json').read_text()))
    life.save(campaign/'CORE_SOURCE_BINDING.json',{'agent_loop_sha256':hashlib.sha256((life.CORE/'PhyAgentOS/agent/loop.py').read_bytes()).hexdigest()})
    tasks=[{k:v for k,v in c.items() if k in ('suite','task_id','init_state_id','task_description','max_steps')} for c in json.loads((life.PROJECT/'runs/native_long_v1_20261002/ROSTER.json').read_text()) if c['task_id'] in (6,9)]
    for t in tasks:t['init_state_id']=9
    protocol={'scope':'Warm-start fixed workflow engineering diagnostic from one admitted development trace. Not cold-start learning, AWM reproduction, ABC or statistical noninferiority.',
        'source':'/project/runs/native_admission_v1_20261002/stream/recovered_clean/accepted_experience',
        'trace':'/project/runs/native_long_v1_20261002/stream/native_long_t0_i8',
        'planning_cases':[{'label':f'plan_t{t["task_id"]}_{arm}','arm':arm,'task':t} for t,order in zip(tasks,[['A','B'],['B','A']]) for arm in order],
        'full_cases':[{'label':'full_t6_i9_'+arm,'arm':arm,'task':tasks[0]} for arm in ['B','A']],
        'max_full_episodes':6,'max_steps_per_episode':520,'max_runner_seconds':1500,'max_campaign_seconds':2100,
        'same_tools':True,'fixed_router':'Compile source read-set into a read-only expansion of initial benchmarking skill read; original Agent checks and queues; no learned selection','same_model_limits':True,'experience':'independent clones of same sealed source; B adds fixed procedure, no persistent evaluation updates',
        'generation':cfg['common_generation'],'memory_threshold':65536,'forced_consolidation':False,'training':False,'laya':False,
        'full_gate':'All planning contracts and isolation pass, B actually consumes procedure and performs fewer model calls across two pairs',
        'acquisition_cost':'Compilation measured separately; historical source episode is paid prior development and must be charged separately for any end-to-end payback claim',
        'reuse_horizon_full_tasks':1,'stop':'Any chain/protocol/isolation failure or fixed budget; preserve failed inputs and all costs'}
    life.save(campaign/'PROTOCOL.json',protocol)
    stream=campaign/'stream';stream.mkdir();life.save(campaign/'STATUS.json',status)
    try:
        life.gate(stream)
        command=' && '.join('/project/envs/os-v1/bin/python '+life.BASE+'/code/tests/'+n for n in ['test_workflow_reuse.py','test_native_workspace.py','test_terminal_results.py','test_native_isolation.py'])
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-tests',stream)+[life.IMAGE,'timeout','180','bash','-lc','source /opt/dtk/env.sh && '+command]
        cid=life.launch('tests',argv,stream);life.wait_exit(cid,180)
        status['stage']='starting_services';life.save(campaign/'STATUS.json',status)
        shutil.copytree(life.PROJECT/'models/openpi_data',stream/'tokenizer_cache')
        for role,receipt in [('pi05','LAUNCH_pi05_v2.json'),('qwen','LAUNCH_qwen.json'),('target','LAUNCH_target.json')]:
            argv=json.loads((life.OLD/receipt).read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-'+role
            argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',life.BASE+'/stream') for x in argv]
            if role=='pi05':argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(stream/'tokenizer_cache')) for x in argv]
            if role=='target':argv=[x.replace('/project/scripts/instrument_target.py',life.BASE+'/code/scripts/instrument_target.py').replace(':/project/scripts',':'+life.BASE+'/code/scripts') for x in argv]
            if role=='qwen':argv=configure_qwen(argv,cfg,life)
            life.launch(role,argv,stream)
        life.health(49500);life.health(49510)
        with socket.create_connection(('127.0.0.1',49502),timeout=5):pass
        status.update(status='RUNNING',stage='paired_diagnostic_then_full_chain',startup_seconds=time.time()-started);life.save(campaign/'STATUS.json',status)
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-runner',stream)+model_env(cfg)+[
            '-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream','-e','FASTRSI_CONSUME_TERMINAL=1','-e','FASTRSI_EXPERIENCE_ADMISSION=guarded_v1',
            life.IMAGE,'timeout','1500','bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/probe_workflow_reuse.py']
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,1500)
        result=json.loads((stream/'RESULT.json').read_text());assert result['status'] in ('COMPLETED','DIAGNOSTIC_ONLY')
        status.update(status='COMPLETED',result_status=result['status'])
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-started)
        life.save(campaign/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']=='FAILED':raise SystemExit(1)

if __name__=='__main__':main()
