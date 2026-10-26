"""Reuse paid development evidence: Qwen-only transfer, then conditional held-instance pair."""
import fcntl,hashlib,json,shutil,signal,socket,sys,time,traceback
from pathlib import Path
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env

def main():
    name=sys.argv[1] if len(sys.argv)>1 else 'instruction_followup_v1_20261002'
    assert name.replace('_','').isalnum()
    lock=open(life.PROJECT/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    campaign=life.PROJECT/'runs'/name;campaign.mkdir(exist_ok=False);life.CAMPAIGN=campaign;life.BASE='/project/runs/'+name
    started=time.time();life.deadline=started+1800;status={'status':'RUNNING','started_at':started,'groups':[]}
    def interrupt(n,f):raise RuntimeError('Interrupted '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    for rel in ['scripts','src/fastrsi','tests','configs']:shutil.copytree(life.PROJECT/rel,campaign/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    life.save(campaign/'CODE_MANIFEST.json',{str(p.relative_to(campaign/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (campaign/'code').rglob('*') if p.is_file()})
    prior=life.PROJECT/'runs/instruction_trial_v3_20261002';cfg=json.loads((prior/'MODEL_CONFIG.json').read_text())
    for rel in ['MODEL_CONFIG.json','MODEL_BINDING.json','CORE_SOURCE_BINDING.json','LEARNED.json','POLICY_UPDATE.json']:shutil.copyfile(prior/rel,campaign/rel)
    transfer=json.loads((prior/'transfer/TRANSFER.json').read_text())
    life.save(campaign/'PROTOCOL.json',{'source_run':prior.name,'scope':'Reuse paid development state; Qwen-only new Session before held-development-instance pair, not main A/B/C',
       'order':['Qwen-only new Session','eval_A original','eval_B learned choice'],'max_seconds':1800,'max_new_root_episodes':2,'max_attempts':6,'max_iterations':10,
       'state_sha256':hashlib.sha256((campaign/'LEARNED.json').read_bytes()).hexdigest(),'init_state_id':9,
       'no_training':True,'evaluation_updates':False,'resource_fairness':'Fresh seed 7 pi05 and Qwen and target per robot arm; same model/config/limits and common queue fixes',
       'gate':'Require actual full experience in rendered model prompt, accepted unique queue and post-close persistent hashes unchanged before any robot evaluation.'})
    life.save(campaign/'STATUS.json',status)
    def runner(stream,script,extra,timeout):
        base=str(stream).replace(str(life.PROJECT),'/project')
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-'+stream.parent.name+'-runner',stream)+model_env(cfg)+['-e','FASTRSI_NATIVE_RUN='+base,'-e','FASTRSI_CONSUME_TERMINAL=1','-e','FASTRSI_EXPERIENCE_ADMISSION=guarded_v1']+extra+[life.IMAGE,'timeout',str(timeout),'bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/'+script]
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,timeout+5)
    try:
        for group_name in ['probe','eval_A','eval_B']:
            group=campaign/group_name;stream=group/'stream';stream.mkdir(parents=True);gstart=time.time();base=str(stream).replace(str(life.PROJECT),'/project')
            shutil.copyfile(campaign/'MODEL_CONFIG.json',group/'MODEL_CONFIG.json')
            g={'arm':group_name,'status':'RUNNING','started_at':gstart};status['current_arm']=group_name;life.save(campaign/'STATUS.json',status)
            try:
                life.gate(stream)
                if group_name=='probe':life.save(group/'TRANSFER.json',transfer)
                else:
                    case={**transfer['case'],'id':'held_t0_i9','planning_only':False,'instruction_experience':life.BASE+'/LEARNED.json'}
                    if group_name=='eval_A':case['instruction_probe']='native'
                    life.save(group/'ROSTER.json',[case]);shutil.copytree(life.PROJECT/'models/openpi_data',stream/'tokenizer_cache')
                roles=[('qwen','LAUNCH_qwen.json')] if group_name=='probe' else [('pi05','LAUNCH_pi05_v2.json'),('qwen','LAUNCH_qwen.json'),('target','LAUNCH_target.json')]
                for role,receipt in roles:
                    argv=json.loads((life.OLD/receipt).read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-'+group_name.lower()+'-'+role
                    argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',base) for x in argv]
                    if role=='pi05':
                        argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(stream/'tokenizer_cache')) for x in argv];assert '--torch-seed 7' in argv[-1]
                    if role=='qwen':argv=configure_qwen(argv,cfg,life)
                    if role=='target':argv=[x.replace('/project/scripts/instrument_target.py',life.BASE+'/code/scripts/instrument_target.py').replace(':/project/scripts',':'+life.BASE+'/code/scripts') for x in argv]
                    life.launch(role,argv,stream)
                life.health(49510)
                if group_name!='probe':
                    life.health(49500)
                    with socket.create_connection(('127.0.0.1',49502),timeout=5):pass
                g['startup_seconds']=time.time()-gstart
                if group_name=='probe':
                    runner(stream,'probe_instruction_transfer.py',[],300)
                    row=json.loads((stream/'RESULT.json').read_text());assert row['status']=='PASSED' and row['experience_unchanged'] and row['actual_experience_prompt_proofs']
                    g['probe']=row
                else:
                    source='/project/runs/'+prior.name+'/dev_'+group_name[-1]+'/stream/instruction_t0_i8/admission/accepted_experience'
                    runner(stream,'run_native_baseline.py',['-e','FASTRSI_INITIAL_EXPERIENCE='+source],600)
                    report=json.loads((stream/'RESULT.json').read_text());row=report['cases'][0]
                    assert report['status']=='COMPLETED' and row['chain_status']=='PASSED';assert row['experience_before']==row['experience_after']
                    g.update(controller_seconds=report['seconds'],environment_success=row['environment_success'],verdicts=[o['verdict'] for o in row['outcomes']])
                g['status']='COMPLETED'
            except BaseException:g['status']='FAILED';raise
            finally:
                life.save(stream/'STOP_RECEIPT.json',life.stop_all());g.update(finished_at=time.time(),full_wall_seconds=time.time()-gstart);life.save(group/'STATUS.json',g);status['groups'].append(g);life.save(campaign/'STATUS.json',status)
        status['status']='COMPLETED'
    except BaseException:status.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(campaign/'STOP_RECEIPT.json',life.stop_all());status.update(finished_at=time.time(),full_wall_seconds=time.time()-started);life.save(campaign/'STATUS.json',status);print(json.dumps(status),flush=True)
    if status['status']!='COMPLETED':raise SystemExit(1)
if __name__=='__main__':main()
