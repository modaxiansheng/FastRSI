"""Two predeclared independent development pairs; static engineering, full bill."""
import fcntl,hashlib,json,shutil,signal,socket,sys,time,traceback
from pathlib import Path
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env
sys.path.insert(0,str(life.PROJECT/'src'))
from fastrsi.session_flow import artifact
from fastrsi.experience_admission import validate_snapshot


def main():
    name=sys.argv[1] if len(sys.argv)>1 else 'shared_flow_net_v1_20261002'
    assert name.replace('_','').isalnum()
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    master=life.PROJECT/'runs/representative_native_v1_20261002'
    state=json.loads((master/'STATE.json').read_text());assert state['status']=='READY' and state['active_protocol_revision']=='terminal_reads_v3'
    plan_path=life.PROJECT/'planning/representative_coverage_v1/SHARED_FLOW_NET_CHECK_PLAN_V2.json'
    plan=json.loads(plan_path.read_text());started=time.time()
    absolute=json.loads((master/'PROTOCOL.json').read_text())['campaign_absolute_deadline'];assert absolute-started>=1800
    assert not state.get('auxiliary_flow_net_run'), 'No automatic repeat or budget reset'
    for run_id in ['terminal_read_v1_20261002','session_flow_pagination_cpu_v1_20261002']:
        assert json.loads((life.PROJECT/'runs'/run_id/'STATUS.json').read_text())['status']=='COMPLETED'
    tested=life.PROJECT/'runs/terminal_read_v1_20261002/code'
    tested_rels=['src/fastrsi/session_flow.py','src/fastrsi/paged_reads.py','src/fastrsi/terminal_results.py','scripts/run_native_baseline.py']
    for rel in tested_rels:assert (life.PROJECT/rel).read_bytes()==(tested/rel).read_bytes(), 'Untested source: '+rel
    campaign=life.PROJECT/'runs'/name;campaign.mkdir(exist_ok=False);life.CAMPAIGN=campaign;life.BASE='/project/runs/'+name
    life.deadline=min(started+1800,absolute)
    life.save(campaign/'MASTER_STATE_BEFORE.json',state);shutil.copyfile(plan_path,campaign/'PLAN.json')
    state.update(status='AUXILIARY_RUNNING',auxiliary_flow_net_run=name,auxiliary_roots_reserved=4)
    assert state['roots_reserved']+4<=60
    life.save(master/'STATE.json',state)
    status={'status':'RUNNING','started_at':started,'groups':[],'scope':'Public engineering structural pilot, not learned FastRSI benefit'}
    def interrupt(n,f):raise RuntimeError('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,campaign/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(campaign/'CODE_MANIFEST.json',{str(p.relative_to(campaign/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()})
    prior=master
    for rel in ['MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json']:shutil.copyfile(prior/rel,campaign/rel)
    cfg=json.loads((campaign/'MODEL_CONFIG.json').read_text())
    development=life.PROJECT/'runs/instruction_trial_v3_20261002/dev_A/stream/instruction_t0_i8'
    seal=validate_snapshot(development/'admission/accepted_experience')
    assert json.loads((development/'RESULT.json').read_text())['case']['mode']=='development'
    flow=artifact((development/'workspace/skills/benchmarking/SKILL.md').read_bytes(),hashlib.sha256((development/'AGENT_EVENTS.jsonl').read_bytes()).hexdigest())
    flow['source_case']=seal['source_case'];flow['source_admission_sha256']=hashlib.sha256((development/'admission/ADMISSION.json').read_bytes()).hexdigest()
    life.save(campaign/'FLOW.json',flow);flowsha=hashlib.sha256((campaign/'FLOW.json').read_bytes()).hexdigest()
    initial=master/'initial_experience';validate_snapshot(initial)
    shutil.copytree(initial,campaign/'initial_experience');initial=campaign/'initial_experience'
    initial_hashes={str(p.relative_to(initial)):hashlib.sha256(p.read_bytes()).hexdigest() for p in initial.rglob('*') if p.is_file()}
    life.save(campaign/'EXPERIENCE_BINDING.json',initial_hashes)
    core=json.loads((campaign/'CORE_SOURCE_BINDING.json').read_text())
    for rel,h in core.items():assert hashlib.sha256((life.CORE/'PhyAgentOS'/rel).read_bytes()).hexdigest()==h
    life.save(campaign/'PROTOCOL.json',{'scope':'Static shared engineering N/F, no learned benefit',
        'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'order':['N','F'],
        'wall_limit_seconds':1800,'absolute_deadline':life.deadline,'master_deadline':absolute,
        'root_episodes':4,'max_environment_executions':12,'flow_sha256':flowsha,
        'cases':plan['cases'],'initial_experience':str(initial),'learning_updates':False,
        'lifecycle':'Fresh services per arm, resident two cases; same seeds and 8 CPU quotas',
        'tested_sources':{rel:hashlib.sha256((tested/rel).read_bytes()).hexdigest() for rel in tested_rels},
        'costs':plan['costs'],'stop':'Protocol/isolation/CPU failure aborts; no automatic retries'})
    life.save(campaign/'STATUS.json',status)
    try:
        cpu=campaign/'cpu';cpu.mkdir();life.gate(cpu)
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-cpu',cpu)+['-e','FASTRSI_NATIVE_RUN=/tmp/fastrsi-flow-cpu/stream','-e','FASTRSI_EXPERIENCE_ADMISSION=guarded_v1',life.IMAGE,'timeout','240','bash','-lc',
            'source /opt/dtk/env.sh && '+ ' && '.join('/project/envs/os-v1/bin/python '+life.BASE+'/code/tests/'+t for t in ['test_paged_reads.py','test_session_flow.py','test_native_workspace.py','test_native_isolation.py','test_terminal_results.py'])]
        cid=life.launch('cpu',argv,cpu)
        try:life.wait_exit(cid,250)
        finally:life.save(cpu/'STOP_RECEIPT.json',life.stop_all())
        status['cpu_status']='PASSED';life.save(campaign/'STATUS.json',status)
        for arm in ['N','F']:
            group=campaign/arm;stream=group/'stream';stream.mkdir(parents=True);gstart=time.time();base=str(stream).replace(str(life.PROJECT),'/project')
            shutil.copyfile(campaign/'MODEL_CONFIG.json',group/'MODEL_CONFIG.json')
            cases=[]
            for item in plan['cases']:
                case={**item,'id':f"net_t{item['task_id']}_i{item['init_state_id']}",'session_intent':True,
                      'require_discovery_reads':True,'require_clean_model_calls':True,'read_files_tool':True}
                if arm=='F':case.update(session_flow_artifact=life.BASE+'/FLOW.json',session_flow_sha256=flowsha)
                cases.append(case)
            life.save(group/'ROSTER.json',cases);g={'arm':arm,'status':'RUNNING','started_at':gstart};status['current_arm']=arm;life.save(campaign/'STATUS.json',status)
            try:
                life.gate(stream);shutil.copytree(life.PROJECT/'models/openpi_data',stream/'tokenizer_cache')
                for role,receipt in [('pi05','LAUNCH_pi05_v2.json'),('qwen','LAUNCH_qwen.json'),('target','LAUNCH_target.json')]:
                    argv=json.loads((life.OLD/receipt).read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-'+arm.lower()+'-'+role
                    argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',base) for x in argv]
                    if role=='pi05':
                        argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(stream/'tokenizer_cache')) for x in argv];assert '--torch-seed 7' in argv[-1]
                    if role=='qwen':argv=configure_qwen(argv,cfg,life)
                    if role=='target':argv=[x.replace('/project/scripts/instrument_target.py',life.BASE+'/code/scripts/instrument_target.py').replace(':/project/scripts',':'+life.BASE+'/code/scripts') for x in argv]
                    life.launch(role,argv,stream)
                life.health(49510);life.health(49500)
                with socket.create_connection(('127.0.0.1',49502),timeout=5):pass
                g['startup_seconds']=time.time()-gstart
                source=str(initial).replace(str(life.PROJECT),'/project')
                remaining=int(life.deadline-time.time()-30);assert remaining>60
                argv=life.cpu_args('yuchen-fastrsi-'+name+'-'+arm.lower()+'-runner',stream)+model_env(cfg)+['-e','FASTRSI_NATIVE_RUN='+base,'-e','FASTRSI_CONSUME_TERMINAL=1','-e','FASTRSI_EXPERIENCE_ADMISSION=guarded_v1','-e','FASTRSI_INITIAL_EXPERIENCE='+source,
                     life.IMAGE,'timeout',str(remaining),'bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/run_native_baseline.py']
                cid=life.launch('runner',argv,stream);life.wait_exit(cid,remaining+5)
                report=json.loads((stream/'RESULT.json').read_text())
                assert report['status']=='COMPLETED' and len(report['cases'])==2
                assert sum(len(row['outcomes']) for row in report['cases'])<=6
                for row in report['cases']:
                    assert row['chain_status']=='PASSED' and row['experience_before']==row['experience_after']
                    assert row['terminal_report']['read_coverage']['complete']
                    folder=stream/row['case']['id'];assert (folder/'workspace/.FASTRSI_EVALUATION_CLOSED.json').exists()
                g.update(status='COMPLETED',controller_seconds=report['seconds'],
                         outcomes=[{'case':row['case']['id'],'environment_success':row['environment_success'],
                                    'verdicts':[o.get('verdict') for o in row['outcomes']]} for row in report['cases']])
            except BaseException:g['status']='FAILED';raise
            finally:
                life.save(stream/'STOP_RECEIPT.json',life.stop_all());g.update(finished_at=time.time(),full_wall_seconds=time.time()-gstart);life.save(group/'STATUS.json',g);status['groups'].append(g);life.save(campaign/'STATUS.json',status)
        assert hashlib.sha256((campaign/'FLOW.json').read_bytes()).hexdigest()==flowsha
        assert initial_hashes=={str(p.relative_to(initial)):hashlib.sha256(p.read_bytes()).hexdigest() for p in initial.rglob('*') if p.is_file()}
        status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(campaign/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-started);life.save(campaign/'STATUS.json',status);print(json.dumps(status),flush=True)
        state.update(status='READY' if status['status']=='COMPLETED' else 'AUXILIARY_FAILED',auxiliary_flow_net_status=status['status']);life.save(master/'STATE.json',state)
    if status['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()
