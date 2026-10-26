"""One native A/B pair with fresh identically seeded services per arm."""
import fcntl,hashlib,json,shutil,signal,socket,time,traceback
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env

def main():
    name='workflow_visible_v3_20261002';lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=life.PROJECT/'runs/workflow_reuse_v3_20261002'
    assert json.loads((previous/'STATUS.json').read_text())['status']=='COMPLETED'
    tested=json.loads((previous/'CODE_MANIFEST.json').read_text())
    for rel in ['src/fastrsi/terminal_results.py','scripts/native_service_config.py','scripts/qwen_agent_server.py']:
        assert hashlib.sha256((life.PROJECT/rel).read_bytes()).hexdigest()==tested[rel],rel
    campaign=life.PROJECT/'runs'/name;campaign.mkdir(exist_ok=False);life.CAMPAIGN=campaign;life.BASE='/project/runs/'+name
    started=time.time();life.deadline=started+1800;status={'status':'PREFLIGHT','started_at':started,'groups':[]}
    def interrupt(signum,frame):raise RuntimeError('Interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,campaign/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(campaign/'CODE_MANIFEST.json',{str(p.relative_to(campaign/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()})
    cfg=json.loads((life.PROJECT/'configs/native_agent_qwen3_v2.json').read_text());life.save(campaign/'MODEL_CONFIG.json',cfg)
    life.save(campaign/'MODEL_BINDING.json',json.loads((previous/'MODEL_BINDING.json').read_text()))
    life.save(campaign/'CORE_SOURCE_BINDING.json',{'agent_loop_sha256':hashlib.sha256((life.CORE/'PhyAgentOS/agent/loop.py').read_bytes()).hexdigest()})
    task={'task_id':6,'init_state_id':9,'suite':'libero_10','max_steps':520,
        'task_description':'put the white mug on the plate and put the chocolate pudding to the right of the plate',
        'id':'matched_t6_i9','mode':'evaluation','require_discovery_reads':True,'require_clean_model_calls':True,'read_files_tool':True}
    source='/project/runs/native_admission_v1_20261002/stream/recovered_clean/accepted_experience'
    artifact='/project/runs/workflow_reuse_v3_20261002/stream/workflow'
    life.save(campaign/'PROTOCOL.json',{'scope':'Development fixed-workflow A/B engineering pair, not learned C or statistical noninferiority',
        'order':['A','B'],'task':task,'initial_experience':source,'B_workflow':artifact,'B_visible_reads':True,'read_definition':'Both arms use identical updated read_files schema; only B enables context-visible references. Actual native reads always run.',
        'policy_rng':'Fresh pi05 process for EACH arm with identical --torch-seed 7; no previous sampling before either task',
        'other_services':'Fresh target and Qwen per arm, identical pinned model/config/resources',
        'evaluation':'All arms read cloned fixed experience; no persistent updates, native AgentLoop/feedback/verification retained',
        'max_cases':2,'max_attempts':6,'max_seconds':1800,'training':False,'laya':False,
        'tests':'25 CPU regressions plus saved duplicate-input replay must pass before GPU services start',
        'quality':'Compare actual goals, initial RGB hashes, environment outcomes, verdicts and steps; preserve mismatch/failure',
        'cost':'Measure each startup and controller separately; do not extrapolate one pair or omit earlier failed development costs'})
    life.save(campaign/'STATUS.json',status)
    try:
        testdir=campaign/'checks';testdir.mkdir();life.gate(testdir)
        commands=['/project/envs/os-v1/bin/python '+life.BASE+'/code/tests/'+n for n in ['test_visible_reads.py','test_workflow_reuse.py','test_native_workspace.py','test_terminal_results.py','test_native_isolation.py']]
        commands.append('/project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/replay_visible_reads.py')
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-checks',testdir)+['-e','FASTRSI_REPLAY_OUTPUT='+life.BASE+'/checks/REPLAY.json',life.IMAGE,'timeout','180','bash','-lc','source /opt/dtk/env.sh && '+' && '.join(commands)]
        cid=life.launch('checks',argv,testdir);life.wait_exit(cid,180)
        life.save(testdir/'STOP_RECEIPT.json',life.stop_all())
        for arm in ['A','B']:
            group=campaign/arm;stream=group/'stream';stream.mkdir(parents=True);group_start=time.time()
            base=life.BASE+'/'+arm+'/stream'
            case=dict(task)
            if arm=='B':case.update(workflow_artifact=artifact,workflow_macro=True,visible_reads=True)
            life.save(group/'ROSTER.json',[case]);life.save(group/'MODEL_CONFIG.json',cfg)
            g={'arm':arm,'status':'RUNNING','started_at':group_start};status['current_arm']=arm;status['status']='RUNNING';life.save(campaign/'STATUS.json',status)
            try:
                life.gate(stream);shutil.copytree(life.PROJECT/'models/openpi_data',stream/'tokenizer_cache')
                for role,receipt in [('pi05','LAUNCH_pi05_v2.json'),('qwen','LAUNCH_qwen.json'),('target','LAUNCH_target.json')]:
                    argv=json.loads((life.OLD/receipt).read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-'+arm.lower()+'-'+role
                    argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',base) for x in argv]
                    if role=='pi05':
                        argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(stream/'tokenizer_cache')) for x in argv]
                        assert '--torch-seed 7' in argv[-1]
                    if role=='target':argv=[x.replace('/project/scripts/instrument_target.py',life.BASE+'/code/scripts/instrument_target.py').replace(':/project/scripts',':'+life.BASE+'/code/scripts') for x in argv]
                    if role=='qwen':argv=configure_qwen(argv,cfg,life)
                    life.launch(role,argv,stream)
                life.health(49500);life.health(49510)
                with socket.create_connection(('127.0.0.1',49502),timeout=5):pass
                g['startup_seconds']=time.time()-group_start
                argv=life.cpu_args('yuchen-fastrsi-'+name+'-'+arm.lower()+'-runner',stream)+model_env(cfg)+[
                    '-e','FASTRSI_NATIVE_RUN='+base,'-e','FASTRSI_CONSUME_TERMINAL=1','-e','FASTRSI_EXPERIENCE_ADMISSION=guarded_v1',
                    '-e','FASTRSI_INITIAL_EXPERIENCE='+source,life.IMAGE,'timeout','600','bash','-lc',
                    'source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/run_native_baseline.py']
                cid=life.launch('runner',argv,stream);life.wait_exit(cid,600)
                result=json.loads((stream/'RESULT.json').read_text());assert result['status']=='COMPLETED'
                row=result['cases'][0];assert row['chain_status']=='PASSED'
                assert row['planned_session']['task_description']==task['task_description'],'Goal changed; not a matched policy comparison'
                assert row['experience_before']==row['experience_after'],'Fixed experience changed'
                g.update(status='COMPLETED',controller_seconds=result['seconds'],case_seconds=row['seconds'],environment_success=row['environment_success'],verdicts=[o['verdict'] for o in row['outcomes']])
            finally:
                life.save(stream/'STOP_RECEIPT.json',life.stop_all());g.update(finished_at=time.time(),full_wall_seconds=time.time()-group_start)
                life.save(group/'STATUS.json',g);status['groups'].append(g);life.save(campaign/'STATUS.json',status)
        status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        status['remaining_stopped']=life.stop_all();life.save(campaign/'STOP_RECEIPT.json',status['remaining_stopped']);status.update(finished_at=time.time(),full_wall_seconds=time.time()-started)
        life.save(campaign/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()
