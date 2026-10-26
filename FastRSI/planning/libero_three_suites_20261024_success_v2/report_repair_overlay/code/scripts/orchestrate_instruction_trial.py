"""One paid instruction pair, native transfer probe, conditional frozen eval pair."""
import fcntl,hashlib,json,shutil,signal,socket,sys,time,traceback
from pathlib import Path
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env

def main():
    name='instruction_trial_v3_20261002'
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign=life.PROJECT/'runs'/name;campaign.mkdir(exist_ok=False);life.CAMPAIGN=campaign;life.BASE='/project/runs/'+name
    start=time.time();life.deadline=start+2400;status={'status':'PREFLIGHT','started_at':start,'groups':[]}
    def interrupt(n,f):raise RuntimeError('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(life.PROJECT/rel,campaign/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    manifest={str(p.relative_to(campaign/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()};life.save(campaign/'CODE_MANIFEST.json',manifest)
    sys.path.insert(0,str(campaign/'code/src'));sys.path.insert(0,str(campaign/'code/scripts'))
    from fastrsi.instruction_memory import proposal,GOAL
    from update_instruction_memory import make_update
    cfg=json.loads((life.PROJECT/'configs/native_agent_qwen3_v2.json').read_text());life.save(campaign/'MODEL_CONFIG.json',cfg)
    cfgsha=hashlib.sha256((campaign/'MODEL_CONFIG.json').read_bytes()).hexdigest()
    source=life.PROJECT/'runs/native_admission_v1_20261002/stream/recovered_clean/accepted_experience'
    source_c=str(source).replace(str(life.PROJECT),'/project')
    t=time.perf_counter();state=proposal(source,cfgsha);life.save(campaign/'PROPOSED.json',state)
    life.save(campaign/'EXTRACTION.json',{'seconds':time.perf_counter()-t,'model_calls':0,'origin':state['origin']})
    prior=life.PROJECT/'runs/workflow_visible_v3_20261002'
    life.save(campaign/'MODEL_BINDING.json',json.loads((prior/'MODEL_BINDING.json').read_text()))
    life.save(campaign/'CORE_SOURCE_BINDING.json',{'agent_loop_sha256':hashlib.sha256((life.CORE/'PhyAgentOS/agent/loop.py').read_bytes()).hexdigest()})
    task={'suite':'libero_10','task_id':0,'max_steps':520,'task_description':GOAL,'require_discovery_reads':True,'require_clean_model_calls':True,'read_files_tool':True}
    life.save(campaign/'PROTOCOL.json',{'scope':'Paid development control-point diagnostic, not main A/B/C or broad instruction optimization',
        'task':task,'source':source_c,'order':['dev_A','dev_B','new_session_dry_transfer','eval_A_if_gate','eval_B_if_gate'],
        'development_init':8,'held_development_validation_init':9,'candidate':state,'max_root_episodes':4,'max_attempts':12,'max_seconds':2400,
        'model_iterations':10,'candidate_count':1,'continue_gate':'Both development chains, feedback, actual prompt and initial images pass; candidate controller cost at least 5% lower. Otherwise no second robot pair.',
        'fairness':'Fresh pi05 seed 7 and fresh model/target per arm. Same goal, budgets, original Agent, validation and reporting. Intent request instead of complete Session for BOTH arms.',
        'learning':'Both paid development actions are charged as acquisition of method evidence. Baseline native triggers unchanged. Eval state frozen; do not ingest evaluation outcomes.',
        'cost':'All failures, model calls, candidate creation, state update, transfer probe and service start/stop included. Compare stage spans without adding overlaps.'})
    life.save(campaign/'STATUS.json',status)
    def execute_runner(group,stream,script,extra,role='runner',timeout=660):
        base=str(stream).replace(str(life.PROJECT),'/project')
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-'+group+'-'+role,stream)+model_env(cfg)+['-e','FASTRSI_NATIVE_RUN='+base,
            '-e','FASTRSI_CONSUME_TERMINAL=1','-e','FASTRSI_EXPERIENCE_ADMISSION=guarded_v1']+extra+[
            life.IMAGE,'timeout',str(timeout),'bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/'+script]
        cid=life.launch(role,argv,stream);life.wait_exit(cid,timeout)
    try:
        checks=campaign/'checks';checks.mkdir();life.gate(checks)
        cmd='source /opt/dtk/env.sh && '+' && '.join('/project/envs/os-v1/bin/python '+life.BASE+'/code/tests/'+n for n in ['test_instruction_memory.py','test_native_workspace.py','test_native_isolation.py','test_experience_admission.py'])
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-checks',checks)+[life.IMAGE,'timeout','240','bash','-lc',cmd]
        cid=life.launch('checks',argv,checks);life.wait_exit(cid,240);life.save(checks/'STOP_RECEIPT.json',life.stop_all())
        for group_name in ['dev_A','dev_B','eval_A','eval_B']:
            development=group_name.startswith('dev');arm=group_name[-1]
            if not development and state['preference']!='reuse':
                status['eval_gate']='SKIPPED_NO_VALIDATED_COST_ADVANTAGE';break
            group=campaign/group_name;stream=group/'stream';stream.mkdir(parents=True);gstart=time.time()
            base=str(stream).replace(str(life.PROJECT),'/project')
            artifact=campaign/('PROPOSED.json' if development else 'LEARNED.json')
            case={**task,'id':'instruction_t0_i'+str(8 if development else 9),'init_state_id':8 if development else 9,
                  'mode':'development' if development else 'evaluation','instruction_experience':str(artifact).replace(str(life.PROJECT),'/project'),
                  'instruction_experience_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}
            if development or arm=='A':case['instruction_probe']='native' if arm=='A' else 'candidate'
            accepted=source_c if development else life.BASE+'/dev_'+arm+'/stream/instruction_t0_i8/admission/accepted_experience'
            life.save(group/'ROSTER.json',[case]);life.save(group/'MODEL_CONFIG.json',cfg)
            g={'arm':group_name,'status':'RUNNING','started_at':gstart};status.update(status='RUNNING',current_arm=group_name);life.save(campaign/'STATUS.json',status)
            try:
                life.gate(stream);shutil.copytree(life.PROJECT/'models/openpi_data',stream/'tokenizer_cache')
                for role,receipt in [('pi05','LAUNCH_pi05_v2.json'),('qwen','LAUNCH_qwen.json'),('target','LAUNCH_target.json')]:
                    argv=json.loads((life.OLD/receipt).read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-'+group_name.lower()+'-'+role
                    argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',base) for x in argv]
                    if role=='pi05':
                        argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(stream/'tokenizer_cache')) for x in argv];assert '--torch-seed 7' in argv[-1]
                    if role=='qwen':argv=configure_qwen(argv,cfg,life)
                    if role=='target':argv=[x.replace('/project/scripts/instrument_target.py',life.BASE+'/code/scripts/instrument_target.py').replace(':/project/scripts',':'+life.BASE+'/code/scripts') for x in argv]
                    life.launch(role,argv,stream)
                life.health(49500);life.health(49510)
                with socket.create_connection(('127.0.0.1',49502),timeout=5):pass
                g['startup_seconds']=time.time()-gstart
                execute_runner(group_name,stream,'run_native_baseline.py',['-e','FASTRSI_INITIAL_EXPERIENCE='+accepted])
                result=json.loads((stream/'RESULT.json').read_text());row=result['cases'][0]
                assert result['status']=='COMPLETED' and row['chain_status']=='PASSED','Pipeline did not finish'
                if not development:assert row['experience_before']==row['experience_after'],'Evaluation changed experience'
                g.update(status='COMPLETED',controller_seconds=result['seconds'],environment_success=row['environment_success'],verdicts=[o['verdict'] for o in row['outcomes']])
                if group_name=='dev_B':
                    state=make_update(campaign)
                    transfer=campaign/'transfer';trstream=transfer/'stream';trstream.mkdir(parents=True);life.save(transfer/'MODEL_CONFIG.json',cfg)
                    artifact=campaign/'LEARNED.json'
                    transfer_case={**task,'id':'next_session_decision','init_state_id':9,'mode':'evaluation','planning_only':True,
                        'instruction_experience':life.BASE+'/LEARNED.json','instruction_experience_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}
                    next_source=row.get('experience_admission',{}).get('next_source') or source_c
                    life.save(transfer/'TRANSFER.json',{'case':transfer_case,'source':next_source})
                    execute_runner('transfer',trstream,'probe_instruction_transfer.py',[],role='transfer',timeout=240)
                    status['policy_preference']=state['preference'];status['transfer']=json.loads((trstream/'RESULT.json').read_text())
            finally:
                life.save(stream/'STOP_RECEIPT.json',life.stop_all());g.update(finished_at=time.time(),full_wall_seconds=time.time()-gstart)
                life.save(group/'STATUS.json',g);status['groups'].append(g);life.save(campaign/'STATUS.json',status)
        status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(campaign/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-start)
        life.save(campaign/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
