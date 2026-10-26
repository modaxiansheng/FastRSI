"""One public structural pilot. All costs counted; no learning or default promotion."""
import fcntl,hashlib,json,shutil,signal,socket,sys,time,traceback
from pathlib import Path
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env
from fastrsi.session_flow import artifact
from fastrsi.experience_admission import validate_snapshot


def main():
    name=sys.argv[1] if len(sys.argv)>1 else 'session_flow_v1_20261002'
    assert name.replace('_','').isalnum()
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign=life.PROJECT/'runs'/name;campaign.mkdir(exist_ok=False);life.CAMPAIGN=campaign;life.BASE='/project/runs/'+name
    started=time.time();life.deadline=started+2700
    status={'status':'RUNNING','started_at':started,'groups':[],'scope':'Public engineering structural pilot, not learned FastRSI benefit'}
    def interrupt(n,f):raise RuntimeError('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,campaign/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(campaign/'CODE_MANIFEST.json',{str(p.relative_to(campaign/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()})
    prior=life.PROJECT/'runs/instruction_followup_v2_20261002'
    for rel in ['MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json']:shutil.copyfile(prior/rel,campaign/rel)
    cfg=json.loads((campaign/'MODEL_CONFIG.json').read_text())
    development=life.PROJECT/'runs/instruction_trial_v3_20261002/dev_A/stream/instruction_t0_i8'
    seal=validate_snapshot(development/'admission/accepted_experience')
    assert json.loads((development/'RESULT.json').read_text())['case']['mode']=='development'
    flow=artifact((development/'workspace/skills/benchmarking/SKILL.md').read_bytes(),hashlib.sha256((development/'AGENT_EVENTS.jsonl').read_bytes()).hexdigest())
    flow['source_case']=seal['source_case'];flow['source_admission_sha256']=hashlib.sha256((development/'admission/ADMISSION.json').read_bytes()).hexdigest()
    life.save(campaign/'FLOW.json',flow);flowsha=hashlib.sha256((campaign/'FLOW.json').read_bytes()).hexdigest()
    life.save(campaign/'PROTOCOL.json',{'scope':'Static shared Session constructor structural pilot; not A/B/C learning','order':['CPU protocol regression','native intent A','public deterministic flow B'],
        'wall_limit_seconds':2700,'root_episodes':2,'max_attempts':6,'init_state_id':17,'task_id':3,'suite':'libero_10','max_steps':520,'updates':False,
        'candidate_acquisition':'Deterministic engineering extraction from admitted development trace; charged to this campaign','flow_sha256':flowsha,
        'costs':'Full campaign including compilation, CPU checks, startup, model calls, robot execution, verifier, native report, failure, fallback and shutdown. No learning savings claim.',
        'matched':'Same intent, exact original goal, initial experience, seed 7, policy/Qwen/target, limits, necessary evidence and verification; fresh services per arm',
        'stop':'Any CPU regression failure prevents model/robot launch; protocol failure aborts remaining arms'})
    life.save(campaign/'STATUS.json',status)
    try:
        cpu=campaign/'cpu';cpu.mkdir();life.gate(cpu)
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-cpu',cpu)+['-e','FASTRSI_NATIVE_RUN=/tmp/fastrsi-flow-cpu/stream','-e','FASTRSI_EXPERIENCE_ADMISSION=guarded_v1',life.IMAGE,'timeout','240','bash','-lc',
            'source /opt/dtk/env.sh && '+ ' && '.join('/project/envs/os-v1/bin/python '+life.BASE+'/code/tests/'+t for t in ['test_session_flow.py','test_native_workspace.py','test_file_append.py','test_file_patch.py','test_native_isolation.py'])]
        cid=life.launch('cpu',argv,cpu)
        try:life.wait_exit(cid,250)
        finally:life.save(cpu/'STOP_RECEIPT.json',life.stop_all())
        status['cpu_status']='PASSED';life.save(campaign/'STATUS.json',status)
        for arm in ['A','B']:
            group=campaign/arm;stream=group/'stream';stream.mkdir(parents=True);gstart=time.time();base=str(stream).replace(str(life.PROJECT),'/project')
            shutil.copyfile(campaign/'MODEL_CONFIG.json',group/'MODEL_CONFIG.json')
            case={'id':'structural_t3_i17','mode':'evaluation','suite':'libero_10','task_id':3,'init_state_id':17,'max_steps':520,
                  'task_description':'put the black bowl in the bottom drawer of the cabinet and close it',
                  'session_intent':True,'require_discovery_reads':True,'require_clean_model_calls':True,'read_files_tool':True}
            if arm=='B':case.update(session_flow_artifact=life.BASE+'/FLOW.json',session_flow_sha256=flowsha)
            life.save(group/'ROSTER.json',[case]);g={'arm':arm,'status':'RUNNING','started_at':gstart};status['current_arm']=arm;life.save(campaign/'STATUS.json',status)
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
                source='/project/runs/native_admission_v1_20261002/stream/recovered_clean/accepted_experience'
                argv=life.cpu_args('yuchen-fastrsi-'+name+'-'+arm.lower()+'-runner',stream)+model_env(cfg)+['-e','FASTRSI_NATIVE_RUN='+base,'-e','FASTRSI_CONSUME_TERMINAL=1','-e','FASTRSI_EXPERIENCE_ADMISSION=guarded_v1','-e','FASTRSI_INITIAL_EXPERIENCE='+source,
                     life.IMAGE,'timeout','720','bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/run_native_baseline.py']
                cid=life.launch('runner',argv,stream);life.wait_exit(cid,730)
                report=json.loads((stream/'RESULT.json').read_text());row=report['cases'][0]
                assert report['status']=='COMPLETED' and row['chain_status']=='PASSED'
                assert row['experience_before']==row['experience_after']
                events=[json.loads(x) for x in (stream/case['id']/'AGENT_EVENTS.jsonl').read_text().splitlines()]
                if arm=='B':assert any(e.get('event')=='queue_accepted' and e.get('origin')=='public_session_flow' for e in events),'Flow did not replace planning'
                g.update(status='COMPLETED',controller_seconds=report['seconds'],environment_success=row['environment_success'],verdicts=[o.get('verdict') for o in row['outcomes']])
            except BaseException:g['status']='FAILED';raise
            finally:
                life.save(stream/'STOP_RECEIPT.json',life.stop_all());g.update(finished_at=time.time(),full_wall_seconds=time.time()-gstart);life.save(group/'STATUS.json',g);status['groups'].append(g);life.save(campaign/'STATUS.json',status)
        assert hashlib.sha256((campaign/'FLOW.json').read_bytes()).hexdigest()==flowsha
        status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(campaign/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-started);life.save(campaign/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()
